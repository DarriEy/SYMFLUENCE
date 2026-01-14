"""
Differential Evolution (DE) optimizer implementation.

DE is a population-based evolutionary algorithm using vector differences
for mutation, effective for continuous optimization problems.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class DEOptimizer(BaseOptimizer):
    """Differential Evolution (DE) Optimizer for SYMFLUENCE.

    DE is a population-based stochastic optimization algorithm that uses vector
    differences between population members to guide the search. Unlike PSO which
    uses velocity-based movement, DE creates trial solutions by mutating existing
    population members with scaled differences of other members. This self-adaptive
    approach makes DE particularly effective for continuous optimization.

    Algorithm Overview:
        DE solves: maximize score
        1. Initialize population: N random solutions in [0,1]
        2. For each generation:
           a. For each population member i:
              - Select three distinct random members r1, r2, r3 (r1 ≠ i)
              - Create mutant: v = x_r1 + F * (x_r2 - x_r3)
                where F is scaling factor controlling differential variation
              - Apply clip to [0,1]
              - Binomial crossover: for each parameter:
                * With probability CR: inherit from mutant
                * Otherwise: keep from original
              - Ensure at least one parameter from mutant (j_rand)
           b. Evaluate all trial solutions
           c. Greedy selection: if trial better than current, accept
           d. Track best solution found so far

    Key Features:
        - Population-based (explores multiple solutions in parallel)
        - Derivative-free (works without gradient information)
        - Self-adaptive: population diversity guides mutation strength
        - Effective on both unimodal and multimodal landscapes
        - Robust to noise (multiple trial comparisons reduce noise sensitivity)
        - Mutation via differences: F*(x2-x3) creates adaptive step sizes
        - Fewer parameters than GA: only F and CR need tuning

    Why DE Works:
        - Vector differences encode local geometry: (x2-x3) points toward promising regions
        - Scaling factor F adapts search neighborhood automatically
        - Crossover creates diversity while exploiting good components
        - Greedy selection ensures progress (hill climbing component)
        - Population maintains multiple search directions simultaneously
        - Often outperforms PSO on noisy objectives or high-dimensional problems

    Strategy Notation: DE/rand/1/bin
        - DE: Differential Evolution base
        - rand: Base vector selected randomly (not best, reduces bias)
        - 1: Single difference vector (x_r2 - x_r3)
        - bin: Binomial (per-parameter) crossover

    Configuration Parameters:
        DE_SCALING_FACTOR (F): Differential weight (default: 0.5)
                  Controls mutation step size: v = r1 + F*(r2-r3)
                  Typical range: 0.4-1.0
                  F < 0.5: Conservative, local refinement
                  F = 0.5-0.8: Balanced exploration/exploitation
                  F > 0.8: Aggressive global exploration
                  Too high F: Instability, premature divergence
                  Too low F: Slow convergence, stagnation
        DE_CROSSOVER_RATE (CR): Probability parameter inheritance (default: 0.9)
                  Probability each parameter comes from mutant vs original
                  Typical range: 0.5-0.95
                  CR < 0.5: More original components, slow adaptation
                  CR = 0.7-0.9: Typical, good convergence
                  CR > 0.95: Rapid change, may lose promising solutions
        POPULATION_SIZE: Number of trial solutions (default: auto 4*n_params, 15-50 bounds)
                  Larger populations: more robust, slower per generation
                  Smaller populations: faster per generation, less reliable
        NUMBER_OF_ITERATIONS: Total generations (= max evaluations / population_size)

    Comparison with Other Algorithms:
        vs PSO: DE uses difference vectors (geometry), PSO uses velocity (momentum)
                DE more robust on noisy objectives, PSO converges faster on smooth
                DE better for high-dimensional problems
        vs DDS: DE population-based, DDS single-solution
                DE global search, DDS for local refinement
                DE scales better with dimensionality
        vs NSGA2: DE for single-objective, NSGA2 for multi-objective
                  DE simpler, fewer parameters

    Attributes:
        population_size: Number of candidate solutions in population
        F: Differential scaling factor (mutation step control)
        CR: Crossover probability (parameter inheritance from mutant)
        population: Current population (N, n_params) of normalized solutions
        population_scores: Fitness values for each population member

    References:
        Storn, R., & Price, K. (1997). Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces.
        Journal of Global Optimization, 11, 341-359.

        Price, K. V., Storn, R. M., & Lampinen, J. A. (2005). The Differential
        Evolution Algorithm: A Practical Approach to Global Optimization.
        Springer Berlin Heidelberg.

        Das, S., & Suganthan, P. N. (2011). Differential Evolution: A Survey of
        the State-of-the-Art. IEEE Transactions on Evolutionary Computation,
        15(1), 4-31.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize DE optimizer.

        Reads configuration for:
        - DE_SCALING_FACTOR: Mutation control (default 0.5)
        - DE_CROSSOVER_RATE: Parameter inheritance probability (default 0.9)
        - POPULATION_SIZE: Optional population size (auto-determined if not specified)

        Population size auto-determination:
            If POPULATION_SIZE not specified, uses formula: 4 * n_parameters
            Bounded to [15, 50] to prevent too small or too large populations.
            This heuristic balances exploration robustness with computational cost.

        Args:
            config: Configuration dictionary with DE parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.population_size = self._determine_population_size()
        self.F = self._cfg('DE_SCALING_FACTOR', default=0.5)
        self.CR = self._cfg('DE_CROSSOVER_RATE', default=0.9)
        self.population: Optional[np.ndarray] = None
        self.population_scores: Optional[np.ndarray] = None

    def get_algorithm_name(self) -> str:
        """Return algorithm name.

        Returns:
            str: "DE" identifier for Differential Evolution
        """
        return "DE"

    def _determine_population_size(self) -> int:
        """Determine population size from config or auto-calculate.

        Priority order:
        1. Explicit POPULATION_SIZE in config (if provided)
        2. Auto-calculated: 4 * n_parameters (recommended heuristic)
        3. Bounds: Minimum 15 (avoid tiny populations), maximum 50 (control cost)

        Rationale for Auto-Sizing:
            - Smaller problems (5 params): population ~20 (4*5=20, within bounds)
            - Larger problems (10 params): population ~40 (4*10=40, within bounds)
            - Very large problems (15+ params): population ~50 (cap to prevent overhead)

        Returns:
            int: Population size for this optimization run
        """
        config_pop_size = self._cfg('POPULATION_SIZE', default=None)
        if config_pop_size: return config_pop_size
        total_params = len(self.parameter_manager.all_param_names)
        return max(15, min(4 * total_params, 50))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Execute DE optimization algorithm.

        Workflow:
        1. Get initial parameter values (if available from prior runs/defaults)
        2. Initialize random population, optionally seeding best member with initial params
        3. Run main DE loop for max_iterations generations
        4. Return best solution found

        Returns:
            Tuple of (best_params, best_score, iteration_history)
        """
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        return self._run_de_algorithm()
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize the DE population with random individuals.

        Workflow:
        1. Create population_size random individuals in normalized [0,1] space
        2. If initial_params provided, seed population[0] with normalized initial values
           (warm start to accelerate convergence from good initial guess)
        3. Evaluate all population members to establish initial fitness scores
        4. Identify and record best member from initial population
        5. Record generation 0 for history tracking

        Warm Start Benefit:
            Seeding first member with good initial guess (e.g., from previous run
            or expert knowledge) can significantly accelerate convergence.

        Args:
            initial_params: Optional dictionary of initial parameter values to seed
                           population[0]; None for pure random initialization
        """
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        assert self.population is not None
        assert self.population_scores is not None

        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        self._evaluate_population()
        best_idx = np.nanargmax(self.population_scores)
        if not np.isnan(self.population_scores[best_idx]):
            self.best_score = self.population_scores[best_idx]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
        
        self._record_generation(0)
    
    def _run_de_algorithm(self) -> Tuple[Dict, float, List]:
        assert self.population is not None
        assert self.population_scores is not None

        """Execute main DE optimization loop.

        Main Generation Loop (generations 1 to max_iterations):
        1. Create trial population by mutating and crossing over current population
           - Uses DE/rand/1/bin strategy for each member
           - All population members participate in mutation (no elite protection)
        2. Evaluate all trial solutions
        3. Greedy selection: for each member i:
           - If trial[i] score > current[i] score: accept trial (replace)
           - Otherwise: keep current member unchanged
           - NaN trials (crashes) are rejected
        4. Track best solution found across all generations
        5. Record generation statistics

        Why Greedy Selection Works:
            - Guarantees monotonic improvement (best score never decreases)
            - Simple and effective: always accept better, never accept worse
            - Particularly robust on noisy objectives
            - Prevents population degradation

        Steady-State Selection:
            Unlike generational GA which replaces entire population,
            DE can replace individuals as soon as better solutions found.
            Provides balance between exploitation (keep good) and
            exploration (generate new via mutation).

        Returns:
            Tuple of (best_params, best_score, iteration_history) where:
            - best_params: Dictionary of best parameter values found
            - best_score: Highest objective value achieved
            - iteration_history: Convergence records per generation
        """
        assert self.population_scores is not None
        assert self.best_score is not None
        assert self.population is not None

        for generation in range(1, self.max_iterations + 1):
            trial_population = self._create_trial_population()
            trial_scores = self._evaluate_trial_population(trial_population)

            improvements = 0
            for i in range(self.population_size):
                if not np.isnan(trial_scores[i]) and trial_scores[i] > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    improvements += 1
                    if trial_scores[i] > self.best_score:
                        self.best_score = trial_scores[i]
                        self.best_params = self.parameter_manager.denormalize_parameters(trial_population[i])

            self._record_generation(generation)
        return self.best_params, self.best_score, self.iteration_history
    
    def _create_trial_population(self) -> np.ndarray:
        """Create trial population using DE/rand/1/bin mutation and crossover.

        DE/rand/1/bin Strategy:
            For each population member i (target):
            1. Mutation - create mutant via vector difference
               - Select three distinct random members r1, r2, r3 (all ≠ i)
               - Mutant: v = x_r1 + F * (x_r2 - x_r3)
               - Clip to [0,1] to maintain bounds
            2. Binomial Crossover - create trial from target and mutant
               - For each parameter j: with probability CR inherit from mutant
               - Otherwise keep from original target
               - Ensure at least one parameter from mutant (j_rand, mandatory crossover)

        Why Vector Differences?
            - (x_r2 - x_r3) encodes local geometry: direction from one population
              member toward another
            - Scaling F adapts step size to population diversity (self-adaptive)
            - Adding to random member r1 creates diverse trial points
            - Result: naturally explores regions populated by good solutions

        Crossover Purpose:
            - CR ≈ 1.0: Aggressive change, mostly from mutant (high exploration)
            - CR ≈ 0.5: Balanced mixing, half from each (moderate change)
            - CR ≈ 0.0: Conservative, mostly original (slow adaptation)
            - j_rand ensures at least one component differs from target

        Mathematical Formulation:
            For member i with parameters (x_i1, x_i2, ..., x_in):
            1. v_j = x_r1,j + F * (x_r2,j - x_r3,j)  ∀ j
            2. For each j: if rand(0,1) < CR or j == j_rand:
                  u_ij = v_ij  (inherit from mutant)
               else:
                  u_ij = x_ij   (keep original)

        Returns:
            Trial population: array shape (population_size, n_params)
            All values bounded to [0,1] via np.clip
        """
        assert self.population is not None
        trial_population = np.zeros_like(self.population)
        param_count = len(self.parameter_manager.all_param_names)
        for i in range(self.population_size):
            candidates = list(range(self.population_size)); candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = np.clip(self.population[r1] + self.F * (self.population[r2] - self.population[r3]), 0, 1)
            trial = self.population[i].copy()
            j_rand = np.random.randint(param_count)
            for j in range(param_count):
                if np.random.random() < self.CR or j == j_rand: trial[j] = mutant[j]
            trial_population[i] = trial
        return trial_population
    
    def _evaluate_population(self) -> None:
        """Evaluate current population, skipping already-evaluated members.

        Dispatches to parallel or sequential evaluation based on configuration.
        Optimization: only evaluates members with NaN scores (typically only on
        first generation; subsequent generations only evaluate trials).

        See Also:
            _evaluate_population_sequential(): Sequential single-threaded evaluation
            _evaluate_population_parallel(): Parallel multi-process evaluation
        """
        if self.use_parallel: self._evaluate_population_parallel()
        else: self._evaluate_population_sequential()

    def _evaluate_trial_population(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population solutions.

        Evaluates all trials in parallel (if configured) or sequentially.
        Trials are always evaluated (unlike population which only re-evaluates NaN).

        Args:
            trial_population: Candidate solutions to evaluate

        Returns:
            np.ndarray: Fitness scores for each trial
        """
        if self.use_parallel: return self._evaluate_trial_population_parallel(trial_population)
        return np.array([self._evaluate_individual(ind) for ind in trial_population])

    def _evaluate_population_sequential(self) -> None:
        """Evaluate population members sequentially (used in initialization).

        Skips members already evaluated (non-NaN scores). Typically called only
        on first generation when population is initialized with NaN scores.

        Subsequent calls do nothing (population scores are updated via selection).
        """
        assert self.population is not None
        assert self.population_scores is not None

        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                self.population_scores[i] = self._evaluate_individual(self.population[i])

    def _evaluate_population_parallel(self) -> None:
        """Evaluate unevaluated population members in parallel.

        Collects members with NaN scores and submits them for parallel evaluation.
        Updates population_scores array with results, using round-robin process
        assignment for load balancing.

        Typically used only on first generation; subsequent generations don't need
        this as population updates use selection in _run_de_algorithm.
        """
        assert self.population is not None
        assert self.population_scores is not None

        tasks = []
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"pop_{i}"})
        if tasks:
            for res in self._run_parallel_evaluations(tasks):
                self.population_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')

    def _evaluate_trial_population_parallel(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate all trial solutions in parallel.

        Submits all trials to process pool with round-robin process assignment.
        Used each generation to evaluate entire trial population created by mutation/crossover.

        Args:
            trial_population: Trial solutions to evaluate (shape: population_size x n_params)

        Returns:
            np.ndarray: Fitness scores for each trial, shape (population_size,)
        """
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(trial_population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"trial_{i}"} for i in range(self.population_size)]
        results = self._run_parallel_evaluations(tasks)
        scores = np.full(self.population_size, np.nan)
        for res in results: scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
        return scores
    
    def _record_generation(self, generation: int) -> None:
        """Record convergence metrics for this generation.

        Stores population statistics including:
            - generation: Iteration number (0 to max_iterations)
            - algorithm: Always "DE"
            - best_score: Current best fitness found
            - best_params: Denormalized parameters of best solution
            - mean_score: Average fitness of valid population members
            - valid_individuals: Number of members with valid scores (not NaN)

        History records enable:
            - Convergence monitoring (plot best vs generation)
            - Population diversity analysis (mean_score trends)
            - Stagnation detection (no improvement over generations)
            - Failure diagnosis (valid_individuals drops)
            - Algorithm performance analysis and parameter tuning

        Args:
            generation: Current generation number

        Note:
            NaN scores (from model crashes) are excluded from mean calculation.
            If all individuals have NaN scores, mean_score is recorded as None.
        """
        assert self.population_scores is not None
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'DE', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores)
        })
