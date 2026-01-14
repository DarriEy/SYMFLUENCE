"""Population-Based Distributed Dimensioned Search (PBDDS) Optimizer.

Population-based parallel implementation of DDS (Dynamically Dimensioned Search)
for efficient calibration on multi-processor systems. Maintains a full population
and generates trials from each member independently.

Algorithm Characteristics:
    - Population-based: N independent search threads (one per processor ideally)
    - Synchronous iteration: All population members evaluated before update
    - Self-adaptive: Each population member generates its own trials
    - Independent lineages: Each member can have different parent solutions
    - Competitive: Members with worse fitness displaced by trial solutions

Parallelization Strategy:
    - Population size: Typically 10-50 (auto-sized as 3*n_params, bounded [10,50])
    - Synchronous: Waits for all evaluations in generation before updating
    - Load balancing: Distributes population members across processors
    - Scalability: Linear scaling to N processors (if population_size >= N)

Population Management:
    - Initialization: Random population + parallel evaluation
    - Generation: Each member generates one trial via DDS mutation
    - Update: Member replaced if trial score higher (greedy selection)
    - Diversity: Independent evolution lines maintain population diversity

DDS Mutation (per-member):
    - Probability: P(vary) = 1 - log(T) / log(T_max) decreases over time
    - Minimum: At least 1 parameter varies per trial
    - Distribution: Normal(parent_value, sigma=DDS_R=0.2)
    - Bounds: Clipped to [0,1] normalized space

Advantages vs Async DDS:
    - Simpler synchronization (full generation before update)
    - Better load balancing (all members evaluated per generation)
    - Easier debugging (deterministic generation order)
    - More stable convergence (less variance in search)

Disadvantages vs Async DDS:
    - Slower if evaluations have variable runtime
    - Wastes processors if population_size < num_processors
    - Higher latency per iteration

Configuration Parameters:
    DDS_R: Mutation step size (normal distribution std dev, default: 0.2)
    POPULATION_SIZE: Population size (auto-sized if not specified)
    MAX_ITERATIONS: Generations to run (default: 100)

References:
    - Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search
      algorithm for computationally efficient watershed model calibration.
      Water Resources Research, 43, W01413.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class PopulationDDSOptimizer(BaseOptimizer):
    """Population-Based DDS (Dynamically Dimensioned Search) optimizer.

    Implements population-based parallel DDS for efficient calibration on
    multi-core and HPC systems. Each population member independently evolves
    using DDS mutations, enabling synchronous parallel evaluation.

    Algorithm Flow:
        1. Initialize population with random solutions
        2. For each generation:
           a. Generate trial solution for each population member (DDS mutation)
           b. Evaluate all trials in parallel
           c. Replace member if trial better (greedy 1:1 replacement)
           d. Track best solution found across population
        3. Repeat until max_iterations

    Population Size Determination:
        - Config override: POPULATION_SIZE if specified
        - Auto-sizing: 3 * n_parameters (typical genetic algorithm rule)
        - Bounds: [10, 50] (minimum for diversity, maximum for cost)
        - Recommendation: Match or exceed num_processors for good scaling

    Population Evolution:
        - Each member: Independent DDS trial generation
        - Selection: Greedy (replace if trial > current member)
        - Diversity: Maintained by independent evolution paths
        - Convergence: Aggregate best across population

    DDS Mutation Formula:
        - Probability: P(vary_param) = 1 - log(iteration) / log(max_iterations)
        - Decreases over time: Starts broad, focuses toward end
        - Minimum: At least 1 parameter always varies
        - Perturbation: Normal(0, DDS_R) where DDS_R = 0.2 (typical)

    Advantages:
        - Synchronous evaluation (easier debugging, deterministic)
        - Good load balancing across processors
        - Simple to understand and implement
        - Stable convergence behavior

    Disadvantages:
        - Slower with variable evaluation times (waits for slowest)
        - Less efficient if population_size << num_processors
        - Higher latency per iteration vs batch methods

    Configuration:
        DDS_R: Mutation step size (default: 0.2)
        POPULATION_SIZE: Population size (auto-determined if not set)
        MAX_ITERATIONS: Generations to run (default: 100)
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize Population DDS optimizer.

        Configures population size, mutation step size, and iteration parameters
        for population-based parallel DDS execution.

        Args:
            config: Configuration dictionary with DDS parameters
            logger: Logger instance for diagnostic output
        """
        super().__init__(config, logger)
        self.dds_r = self._cfg('DDS_R', default=0.2)
        self.population_size = self._determine_population_size()
        self.population: Optional[np.ndarray] = None
        self.population_scores: Optional[np.ndarray] = None
        self.current_generation = 0

    def get_algorithm_name(self) -> str:
        """Return algorithm identifier for results and logging."""
        return "PopulationDDS"

    def _determine_population_size(self) -> int:
        config_pop_size = self._cfg('POPULATION_SIZE', default=None)
        if config_pop_size: return config_pop_size
        param_count = len(self.parameter_manager.all_param_names)
        return max(10, min(3 * param_count, 50))

    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        self._initialize_population()
        while self.current_generation < self.max_iterations:
            tasks = self._generate_population_trials()
            results = self._run_parallel_evaluations(tasks)
            improvements = self._update_population_with_trials(results)
            self._record_generation_statistics(improvements, 0.0)
            self.current_generation += 1
        return self._extract_final_results()

    def _initialize_population(self) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)

        assert self.population is not None
        assert self.population_scores is not None

        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"p_init_{i}"} for i in range(self.population_size)]
        results = self._run_parallel_evaluations(tasks)
        for res in results: self.population_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
        self.best_score = np.nanmax(self.population_scores)
        self.best_params = self.parameter_manager.denormalize_parameters(self.population[np.nanargmax(self.population_scores)])

    def _generate_population_trials(self) -> List[Dict]:
        assert self.population is not None
        tasks = []
        for i in range(self.population_size):
            trial = self._generate_dds_trial(self.population[i], i)
            tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(trial), 'proc_id': i % self.num_processes, 'evaluation_id': f"p_gen_{self.current_generation}_{i}"})
        return tasks

    def _generate_dds_trial(self, parent, idx):
        param_count = len(self.parameter_manager.all_param_names)
        prob = 1.0 - np.log(self.current_generation * self.population_size + idx + 1) / np.log(self.max_iterations * self.population_size) if self.max_iterations > 1 else 0.5
        prob = max(prob, 1.0 / param_count)
        trial = parent.copy()
        variables = np.random.random(param_count) < prob
        if not variables.any(): variables[np.random.randint(param_count)] = True
        for i in range(param_count):
            if variables[i]: trial[i] = np.clip(parent[i] + np.random.normal(0, self.dds_r), 0, 1)
        return trial

    def _update_population_with_trials(self, results):
        assert self.population_scores is not None
        assert self.population is not None
        assert self.best_score is not None

        improvements = 0
        for res in results:
            idx = res['individual_id']
            if res['score'] is not None and res['score'] > self.population_scores[idx]:
                self.population[idx] = self.parameter_manager.normalize_parameters(res['params'])
                self.population_scores[idx] = res['score']
                improvements += 1
                if res['score'] > self.best_score:
                    self.best_score = res['score']
                    self.best_params = res['params'].copy()
        return improvements

    def _record_generation_statistics(self, improvements, duration):
        assert self.population_scores is not None
        valid = self.population_scores[~np.isnan(self.population_scores)]
        self.iteration_history.append({
            'generation': self.current_generation, 'algorithm': 'PopulationDDS', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_score': np.mean(valid) if len(valid) > 0 else None, 'valid_individuals': len(valid)
        })

    def _extract_final_results(self):
        return self.best_params, self.best_score, self.iteration_history
