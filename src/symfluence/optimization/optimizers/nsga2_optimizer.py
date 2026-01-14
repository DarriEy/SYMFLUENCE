"""
NSGA-II multi-objective optimizer implementation.

Non-dominated Sorting Genetic Algorithm II for Pareto-optimal calibration
with multiple objectives (e.g., streamflow, snow, ET simultaneously).
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer
from symfluence.optimization.calibration_targets import (
    StreamflowTarget, SnowTarget, GroundwaterTarget, ETTarget, SoilMoistureTarget, TWSTarget
)

class NSGA2Optimizer(BaseOptimizer):
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization.

    NSGA-II is an elitist multi-objective evolutionary algorithm designed to find
    Pareto-optimal solutions when optimizing multiple conflicting objectives
    simultaneously. It balances convergence to the Pareto front with diversity
    across non-dominated solutions.

    Multi-Objective Concept:
        Traditional single-objective optimization seeks ONE best solution.
        Multi-objective optimization with conflicting objectives has NO single
        best solution. Instead, we seek a set of "Pareto-optimal" solutions
        representing different trade-offs.

        Example: Calibrating both streamflow and snow:
        - Solution A: NSE(streamflow)=0.8, KGE(snow)=0.5  (streamflow-focused)
        - Solution B: NSE(streamflow)=0.7, KGE(snow)=0.7  (balanced)
        - Solution C: NSE(streamflow)=0.6, KGE(snow)=0.85 (snow-focused)
        All three are Pareto-optimal; none dominates all others.

    Pareto Dominance:
        Solution x dominates y if:
        - x is ≥ y in ALL objectives (minimization or maximization)
        - x is > y in AT LEAST ONE objective
        Non-dominated solutions form a "Pareto front": no solution dominates others.

    Algorithm Overview:
        NSGA-II solves: maximize [obj1, obj2, ...] (multiple objectives)
        1. Initialize population: N random solutions
        2. For each generation:
           a. Tournament Selection: Select parents based on Pareto rank + crowding distance
              - Better rank (lower is better) preferred
              - Within same rank, more isolated (higher crowding distance) preferred
           b. Genetic Operators:
              - SBX Crossover: Create children near parents (controlled by eta_c)
              - Polynomial Mutation: Small perturbations (controlled by eta_m)
           c. Evaluate: Calculate all objectives for offspring
           d. Environmental Selection: Combine parent+offspring (2N solutions)
              - Non-dominated sort: assign Pareto ranks
              - Keep best N solutions: full fronts + partial front ranked by crowding
           e. Elitism: Best solutions guaranteed to survive (monotonic progress)

    Key Features:
        - Elitist selection: best solutions never disappear
        - Fast non-dominated sorting: O(N²) instead of O(N³)
        - Crowding distance: preserves diversity in objective space
        - SBX/Polynomial operators: real-valued continuous optimization
        - Self-adaptive: population diversity guides search
        - Maintains Pareto front approximation

    Why NSGA-II Works:
        - Pareto dominance naturally balances multiple objectives
        - Non-dominated sorting provides multi-dimensional quality ranking
        - Crowding distance prevents premature convergence to single attractor
        - Elitism guarantees progress (best never lost)
        - Genetic operators explore via crossover, exploit via mutation
        - Population maintains multiple trade-off solutions simultaneously

    NSGA-II vs Single-Objective:
        - Single-objective: optimizes one metric, returns one solution
        - NSGA-II: finds trade-off surface, returns population of solutions
        - Allows decision-maker to select solution matching specific priorities
        - Better for problems with conflicting objectives (nearly all calibration)

    Modes:
        1. Single-Target Mode (default):
           - Optimizes NSE and KGE for same variable (e.g., streamflow)
           - Objectives: maximize NSE, maximize KGE (both measure fit quality)
           - Trade-off: NSE emphasizes peaks, KGE emphasizes overall shape
        2. Multi-Target Mode:
           - Optimizes different variables simultaneously
           - Example: streamflow (KGE) vs groundwater depth (KGE)
           - Each target has its own calibration metrics

    Configuration Parameters:
        NSGA2_CROSSOVER_RATE (CR): Probability of SBX crossover (default: 0.9)
                  Range: [0, 1]; Higher = more recombination
                  Typical: 0.7-0.95
        NSGA2_MUTATION_RATE (MR): Probability of polynomial mutation per gene (default: 0.1)
                  Range: [0, 1]; Often 1/n_parameters for balance
                  Typical: 0.05-0.2
        NSGA2_ETA_C: SBX distribution index (default: 20)
                  Higher eta_c → children closer to parents (less disruption)
                  Typical range: 10-30
                  eta_c=20 is common default
        NSGA2_ETA_M: Polynomial mutation distribution index (default: 20)
                  Higher eta_m → mutations closer to original (smaller steps)
                  Typical range: 10-30
        NSGA2_MULTI_TARGET: Enable multi-target mode (default: False)
        NSGA2_PRIMARY_TARGET: First objective target (streamflow, snow, gw_depth, et, sm, tws)
        NSGA2_SECONDARY_TARGET: Second objective target (defaults to gw_depth)
        NSGA2_PRIMARY_METRIC: First objective metric (NSE, KGE, etc.)
        NSGA2_SECONDARY_METRIC: Second objective metric
        POPULATION_SIZE: Number of individuals (default: auto 8*n_params, bounded 50-100)
                  Multi-objective typically uses larger populations than single-objective

    Genetic Operators:
        SBX (Simulated Binary Crossover):
            Mimics binary crossover in binary GAs but for real variables.
            Children distributed near parents with probability controlled by eta_c.
            Smaller eta_c → wider distribution (exploration)
            Larger eta_c → narrower distribution (exploitation)

        Polynomial Mutation:
            Applies bounded polynomial perturbations to genes.
            Magnitude controlled by eta_m (distribution index).
            Ensures mutations stay in [0,1] bounds.

    Attributes:
        population_size: Number of individuals maintained
        crossover_rate: SBX crossover probability
        mutation_rate: Polynomial mutation probability per gene
        eta_c, eta_m: SBX and mutation distribution indices
        multi_target_mode: Whether optimizing multiple objectives
        num_objectives: Number of objectives (currently fixed at 2)
        population_ranks: Pareto front rank for each individual
        population_crowding_distances: Crowding distance values
        pareto_front: Solutions on first non-dominated front (Pareto-optimal)

    References:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

        Deb, K. (2001). Multi-Objective Optimization using Evolutionary
        Algorithms. Wiley, Chichester, UK.

        Deb, K., & Agrawal, R. B. (1994). Simulated binary crossover for
        continuous search space. Complex Systems, 9, 1-15.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.population_size = self._determine_population_size()
        self.crossover_rate = self._cfg('NSGA2_CROSSOVER_RATE', default=0.9)
        self.mutation_rate = self._cfg('NSGA2_MUTATION_RATE', default=0.1)
        self.eta_c = self._cfg('NSGA2_ETA_C', default=20)
        self.eta_m = self._cfg('NSGA2_ETA_M', default=20)

        # Auto-detect multi-target mode from config
        # Either explicit NSGA2_MULTI_TARGET or OPTIMIZATION_TARGET='multivariate'
        self.multi_target_mode = self._cfg('NSGA2_MULTI_TARGET', default=False)
        optimization_target = self._cfg('OPTIMIZATION_TARGET', default='streamflow').lower()
        if optimization_target == 'multivariate':
            self.multi_target_mode = True

        if self.multi_target_mode: self._setup_multi_target_objectives()
        else:
            self.objectives = ['NSE', 'KGE']; self.objective_names = ['NSE', 'KGE']
            self.num_objectives = 2; self.primary_target = self.calibration_target
            self.secondary_target = None; self.primary_metric = 'NSE'; self.secondary_metric = 'KGE'
        
        self.population: Optional[np.ndarray] = None
        self.population_objectives: Optional[np.ndarray] = None
        self.population_ranks: Optional[np.ndarray] = None
        self.population_crowding_distances: Optional[np.ndarray] = None
        self.pareto_front: Optional[np.ndarray] = None
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None

    def _setup_multi_target_objectives(self) -> None:
        # Check if using multivariate mode (OBJECTIVE_WEIGHTS defined)
        objective_weights = self._cfg('OBJECTIVE_WEIGHTS', default=None)
        objective_metrics = self._cfg('OBJECTIVE_METRICS', default={})

        if objective_weights:
            # Multivariate mode: extract targets from OBJECTIVE_WEIGHTS keys
            target_list = list(objective_weights.keys()) if isinstance(objective_weights, dict) else []
            if len(target_list) >= 2:
                primary_target_type = target_list[0]
                secondary_target_type = target_list[1]
                # Get metrics, trying both with and without uppercase
                self.primary_metric = (objective_metrics.get(primary_target_type) or
                                      objective_metrics.get(primary_target_type.upper()) or 'KGE')
                self.secondary_metric = (objective_metrics.get(secondary_target_type) or
                                        objective_metrics.get(secondary_target_type.upper()) or 'KGE')
                self.logger.info(f"[NSGA2] Multivariate mode: {primary_target_type} ({self.primary_metric}) + {secondary_target_type} ({self.secondary_metric})")
            else:
                # Fallback if not enough targets in weights
                self.logger.warning(f"[NSGA2] Multivariate specified but only {len(target_list)} targets in OBJECTIVE_WEIGHTS, defaulting to streamflow + gw_depth")
                primary_target_type = 'streamflow'
                secondary_target_type = 'gw_depth'
                self.primary_metric = self._cfg('OPTIMIZATION_METRIC', default='KGE')
                self.secondary_metric = self._cfg('OPTIMIZATION_METRIC2', default='KGE')
        else:
            # Multi-target explicit mode (NSGA2_PRIMARY_TARGET / NSGA2_SECONDARY_TARGET)
            primary_target_type = self._cfg('NSGA2_PRIMARY_TARGET', default=self._cfg('OPTIMIZATION_TARGET', default='streamflow'))
            secondary_target_type = self._cfg('NSGA2_SECONDARY_TARGET',
                                              default=self._cfg('OPTIMIZATION_TARGET2', default='gw_depth'))
            self.primary_metric = self._cfg('NSGA2_PRIMARY_METRIC',
                                            default=self._cfg('OPTIMIZATION_METRIC', default='KGE'))
            self.secondary_metric = self._cfg('NSGA2_SECONDARY_METRIC',
                                              default=self._cfg('OPTIMIZATION_METRIC2', default='KGE'))

        # Store target types as instance variables for use in task building
        self.primary_target_type = primary_target_type
        self.secondary_target_type = secondary_target_type

        self.primary_target = self._create_calibration_target_by_type(primary_target_type)
        self.secondary_target = self._create_calibration_target_by_type(secondary_target_type)
        self.calibration_target = self.primary_target
        self.objectives = [f"{primary_target_type}_{self.primary_metric}", f"{secondary_target_type}_{self.secondary_metric}"]
        self.objective_names = [f"{primary_target_type.upper()}_{self.primary_metric}", f"{secondary_target_type.upper()}_{self.secondary_metric}"]
        self.num_objectives = 2

    def _create_calibration_target_by_type(self, target_type: str):
        target_type = target_type.lower()
        if target_type in ['streamflow', 'flow', 'discharge']: return StreamflowTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['swe', 'sca', 'snow_depth', 'snow']: return SnowTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['gw_depth', 'gw_grace', 'groundwater', 'gw']: return GroundwaterTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['et', 'latent_heat', 'evapotranspiration']: return ETTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn', 'soil_moisture', 'sm']: return SoilMoistureTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['tws', 'grace', 'grace_tws', 'total_storage', 'stor_grace']: return TWSTarget(self.config, self.project_dir, self.logger)
        else: raise ValueError(f"Unknown target type: {target_type}")

    def get_algorithm_name(self) -> str:
        """Return algorithm identifier for results and logging."""
        return "NSGA2"

    def _determine_population_size(self) -> int:
        config_pop_size = self._cfg('POPULATION_SIZE', default=None)
        if config_pop_size: return config_pop_size
        total_params = len(self.parameter_manager.all_param_names)
        return max(50, min(8 * total_params, 100))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        return self._run_nsga2_algorithm()

    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_objectives = np.full((self.population_size, self.num_objectives), np.nan)
        if initial_params:
            self.population[0] = np.clip(self.parameter_manager.normalize_parameters(initial_params), 0, 1)
        self._evaluate_population_multiobjective()
        self._perform_nsga2_selection()
        self._update_representative_solution()
        self._record_generation(0)
    
    def _run_nsga2_algorithm(self) -> Tuple[Dict, float, List]:
        assert self.population is not None
        assert self.population_objectives is not None

        for generation in range(1, self.max_iterations + 1):
            offspring = self._generate_offspring()
            offspring_objectives = self._evaluate_offspring(offspring)
            combined_population = np.vstack([self.population, offspring])
            combined_objectives = np.vstack([self.population_objectives, offspring_objectives])
            selected_indices = self._environmental_selection(combined_population, combined_objectives)
            self.population = combined_population[selected_indices]
            self.population_objectives = combined_objectives[selected_indices]
            self._perform_nsga2_selection()
            self._update_representative_solution()
            self._record_generation(generation)
        
        assert self.best_params is not None
        assert self.best_score is not None
        return self.best_params, self.best_score, self.iteration_history
    
    def _evaluate_population_multiobjective(self) -> None:
        assert self.population_objectives is not None
        assert self.population is not None
        if self.use_parallel: self._evaluate_population_parallel_multiobjective()
        else:
            for i in range(self.population_size):
                if np.any(np.isnan(self.population_objectives[i])):
                    self.population_objectives[i] = self._evaluate_individual_multiobjective(self.population[i])
    
    def _evaluate_individual_multiobjective(self, normalized_params: np.ndarray) -> np.ndarray:
        try:
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            if not self._apply_parameters(params): return np.array([-1.0, -1.0])
            if not self.model_executor.run_models(self.summa_sim_dir, self.mizuroute_sim_dir, self.optimization_settings_dir): return np.array([-1.0, -1.0])
            if self.multi_target_mode:
                obj1 = self._extract_specific_metric(
                    self.primary_target.calculate_metrics(
                        self.summa_sim_dir,
                        mizuroute_dir=self.mizuroute_sim_dir
                    ),
                    self.primary_metric
                )
                obj2 = self._extract_specific_metric(
                    self.secondary_target.calculate_metrics(
                        self.summa_sim_dir,
                        mizuroute_dir=self.mizuroute_sim_dir
                    ),
                    self.secondary_metric
                )
            else:
                metrics = self.calibration_target.calculate_metrics(
                    self.summa_sim_dir,
                    mizuroute_dir=self.mizuroute_sim_dir
                )
                obj1 = self._extract_specific_metric(metrics, 'NSE'); obj2 = self._extract_specific_metric(metrics, 'KGE')
            return np.array([obj1 or -1.0, obj2 or -1.0])
        except Exception: return np.array([-1.0, -1.0])

    def _extract_specific_metric(self, metrics: Dict[str, float], metric_name: str) -> Optional[float]:
        if metrics is None: return None
        if metric_name in metrics: return metrics[metric_name]
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics: return metrics[calib_key]
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"): return value
        return None

    def _evaluate_population_parallel_multiobjective(self) -> None:
        assert self.population_objectives is not None
        assert self.population is not None
        tasks = []
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                task = {
                    'individual_id': i,
                    'params': self.parameter_manager.denormalize_parameters(self.population[i]),
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"nsga2_pop_{i}",
                    'multiobjective': True,
                    'multi_target_mode': self.multi_target_mode,  # Pass multi-target flag
                    'objective_names': self.objective_names  # Pass objective names for result extraction
                }
                if self.multi_target_mode:
                    # For multi-target mode, pass target type information
                    task['primary_target_type'] = self.primary_target_type
                    task['secondary_target_type'] = self.secondary_target_type
                    task['primary_metric'] = self.primary_metric
                    task['secondary_metric'] = self.secondary_metric
                tasks.append(task)
        if tasks:
            results = self._run_parallel_evaluations(tasks)
            for res in results:
                self.population_objectives[res['individual_id']] = np.array(res.get('objectives') or [-1.0, -1.0])

    def _evaluate_offspring(self, offspring: np.ndarray) -> np.ndarray:
        offspring_objectives = np.full((len(offspring), self.num_objectives), np.nan)
        if self.use_parallel:
            tasks = []
            for i in range(len(offspring)):
                task = {
                    'individual_id': i,
                    'params': self.parameter_manager.denormalize_parameters(offspring[i]),
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"nsga2_off_{i}",
                    'multiobjective': True,
                    'multi_target_mode': self.multi_target_mode,  # Pass multi-target flag
                    'objective_names': self.objective_names  # Pass objective names for result extraction
                }
                if self.multi_target_mode:
                    # For multi-target mode, pass target type information
                    task['primary_target_type'] = self.primary_target_type
                    task['secondary_target_type'] = self.secondary_target_type
                    task['primary_metric'] = self.primary_metric
                    task['secondary_metric'] = self.secondary_metric
                tasks.append(task)
            results = self._run_parallel_evaluations(tasks)
            for res in results: offspring_objectives[res['individual_id']] = np.array(res.get('objectives') or [-1.0, -1.0])
        else:
            for i in range(len(offspring)): offspring_objectives[i] = self._evaluate_individual_multiobjective(offspring[i])
        return offspring_objectives

    def _generate_offspring(self) -> np.ndarray:
        assert self.population is not None
        offspring = np.zeros_like(self.population)
        for i in range(0, self.population_size, 2):
            p1, p2 = self.population[self._tournament_selection()], self.population[self._tournament_selection()]
            if np.random.random() < self.crossover_rate: c1, c2 = self._sbx_crossover(p1, p2)
            else: c1, c2 = p1.copy(), p2.copy()
            offspring[i] = self._polynomial_mutation(c1)
            if i + 1 < self.population_size: offspring[i + 1] = self._polynomial_mutation(c2)
        return offspring
    
    def _tournament_selection(self) -> int:
        assert self.population_ranks is not None
        assert self.population_crowding_distances is not None
        candidates = np.random.choice(self.population_size, 2, replace=False)
        best_idx = candidates[0]
        for candidate in candidates[1:]:
            if (self.population_ranks[candidate] < self.population_ranks[best_idx] or
                (self.population_ranks[candidate] == self.population_ranks[best_idx] and
                 self.population_crowding_distances[candidate] > self.population_crowding_distances[best_idx])):
                best_idx = candidate
        return best_idx
    
    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) for continuous real-valued optimization.

        SBX is the continuous analog of single-point crossover in binary genetic
        algorithms. It creates offspring near parents rather than arbitrary points,
        preserving good building blocks while exploring their neighborhood.

        Algorithm:
            For each parameter independently:
            1. With probability 0.5, apply crossover (skip if parents identical)
            2. Generate random u ∈ [0,1]
            3. Calculate spreading factor β based on u and eta_c:
               - If u ≤ 0.5: β = (2u)^(1/(eta_c+1))  (spread toward smaller parent)
               - If u > 0.5: β = (1/(2(1-u)))^(1/(eta_c+1))  (spread toward larger parent)
            4. Create children:
               c1 = 0.5 * ((1+β)*p1 + (1-β)*p2)
               c2 = 0.5 * ((1-β)*p1 + (1+β)*p2)
            5. Clip to [0,1] bounds

        Why SBX Works:
            - eta_c controls distribution: Higher eta_c → children closer to parents
            - Creates smooth interpolation, not random crossing
            - Exploration: β < 1 explores between parents
            - Exploitation: β > 1 explores outside (extrapolation)
            - Self-adaptive: diversity in population drives exploration rate

        Distribution Index (eta_c):
            - eta_c = 5: Very wide spread (lots of exploration)
            - eta_c = 20: Standard default (balanced exploration/exploitation)
            - eta_c = 50: Narrow spread (mostly near parents, fine-tuning)

        Crossover Probability (handled elsewhere):
            With probability CR (typically 0.9), crossover is applied.
            Otherwise, children are exact copies of parents.

        Args:
            p1, p2: Parent individuals (normalized parameter vectors in [0,1])

        Returns:
            Tuple of (child1, child2): Two offspring clipped to [0,1]

        Mathematical Details:
            β represents the spread magnitude. Values:
            - β ∈ [0, 1): Children between parents (interpolation)
            - β = 1: Children equal to parents (no change)
            - β > 1: Children outside parent range (extrapolation)
        """
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if np.random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                u = np.random.random()
                beta = (2 * u)**(1/(self.eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(self.eta_c+1))
                c1[i], c2[i] = 0.5*((1+beta)*p1[i]+(1-beta)*p2[i]), 0.5*((1-beta)*p1[i]+(1+beta)*p2[i])
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation for real-valued genetic algorithms.

        Applies bounded polynomial perturbations to genes, controlled by eta_m
        (distribution index). Unlike simple Gaussian mutation, polynomial mutation
        provides varying step sizes: small steps near current value (fine-tuning),
        larger steps available but less probable.

        Algorithm:
            For each parameter independently:
            1. With probability mutation_rate (typically 0.1):
               a. Generate random u ∈ [0,1]
               b. Calculate delta (perturbation) based on eta_m:
                  - If u < 0.5: δ = (2u)^(1/(eta_m+1)) - 1  (perturb downward)
                  - If u ≥ 0.5: δ = 1 - (2(1-u))^(1/(eta_m+1))  (perturb upward)
               c. new_value = current + δ
               d. Clip to [0,1] bounds

        Perturbation Properties:
            - Most perturbations are small (near original value)
            - Larger perturbations possible but less probable
            - Distribution controlled by eta_m (higher = narrower)
            - Smooth exploration around current position

        Distribution Index (eta_m):
            - eta_m = 5: Wide distribution (aggressive mutations, lots of exploration)
            - eta_m = 20: Standard default (balanced exploration/exploitation)
            - eta_m = 50: Narrow distribution (conservative, fine-tuning focus)

        Why Polynomial Mutation:
            - Self-adaptive: diversity in population guides exploration
            - Balances global and local search naturally
            - Works well with crossover (exploration/exploitation balance)
            - Boundary-aware: never forces solutions outside [0,1]

        Mutation Probability:
            - mutation_rate ≈ 1/n_parameters: One gene mutated per individual on average
            - mutation_rate = 0.1: 10% chance per gene (typical for small populations)
            - Lower mutation_rate: convergence focus, risk of premature convergence
            - Higher mutation_rate: exploration focus, slower convergence

        Args:
            individual: Individual to mutate (normalized parameter vector in [0,1])

        Returns:
            Mutated individual: New vector with mutations applied, clipped to [0,1]

        Mathematical Details:
            δ calculation uses polynomial distribution with index eta_m.
            The polynomial provides smooth probability density, concentrating
            mutations near zero while allowing larger jumps with lower probability.
            This is superior to fixed-magnitude or Gaussian mutations.
        """
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                u = np.random.random()
                delta = (2*u)**(1/(self.eta_m+1))-1 if u < 0.5 else 1-(2*(1-u))**(1/(self.eta_m+1))
                mutated[i] = individual[i] + delta
        return np.clip(mutated, 0, 1)
    
    def _environmental_selection(self, combined_population: np.ndarray, combined_objectives: np.ndarray) -> np.ndarray:
        fronts = self._non_dominated_sorting(combined_objectives)
        selected_indices: list[Any] = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size: selected_indices.extend(front)
            else:
                remaining = self.population_size - len(selected_indices)
                distances = self._calculate_crowding_distance(combined_objectives[front])
                front_sorted = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                selected_indices.extend([x[0] for x in front_sorted[:remaining]])
                break
        return np.array(selected_indices)
    
    def _perform_nsga2_selection(self) -> None:
        assert self.population_objectives is not None
        fronts = self._non_dominated_sorting(self.population_objectives)
        self.population_ranks = np.zeros(self.population_size)
        for rank, front in enumerate(fronts): self.population_ranks[front] = rank
        self.population_crowding_distances = np.zeros(self.population_size)
        for front in fronts:
            if len(front) > 0: self.population_crowding_distances[front] = self._calculate_crowding_distance(self.population_objectives[front])
    
    def _non_dominated_sorting(self, objectives: np.ndarray) -> List[List[int]]:
        """Perform fast non-dominated sorting to assign Pareto ranks.

        This is the core NSGA-II ranking mechanism. Individuals are assigned to
        Pareto fronts based on dominance: solutions that don't dominate each other
        belong to the same front.

        Domination Definition:
            Solution x dominates solution y if:
            - x is ≥ y in ALL objectives (or ≤ if minimizing)
            - x is > y in AT LEAST ONE objective (strict inequality in at least one)

            Example (maximization):
            - x = [0.8, 0.6] dominates y = [0.7, 0.5]  (better in both)
            - x = [0.8, 0.5] dominates y = [0.7, 0.6]  (better in obj1)
            - x = [0.8, 0.4] and y = [0.7, 0.6] DON'T dominate (mixed)

        Algorithm (Fast Non-dominated Sort):
            1. Initialize:
               - domination_counts[i] = # of solutions that dominate i
               - dominated_solutions[i] = list of solutions i dominates
               - front[0] = solutions with domination_count = 0 (not dominated)
            2. For each solution pair:
               - If i dominates j: add j to i's dominated list
               - If j dominates i: increment i's domination count
            3. Iteratively find subsequent fronts:
               - Each solution i in current front "frees" its dominated solutions
               - When a solution's domination count reaches 0, it enters next front
               - Repeat until all solutions assigned

        Complexity:
            - O(N²) for comparison phase (N solutions compared pairwise)
            - O(N) for domination counting
            - Total: O(N²) vs O(N³) for naive approach

        Pareto Fronts:
            - Front 0: Best solutions (Pareto-optimal), nothing dominates them
            - Front 1: Second best, dominated only by front 0
            - Front k: Dominated by fronts 0...k-1
            - Later fronts progressively worse

        Usage in NSGA-II:
            - Higher-ranked fronts preferred in selection (lower rank number)
            - Fronts fill population sequentially until capacity reached
            - Last incomplete front ranked by crowding distance

        Args:
            objectives: Fitness array, shape (n_individuals, n_objectives)
                       Each row is one solution's objective values

        Returns:
            List of fronts where each front is list of individual indices.
            Example: [[3, 7, 9], [1, 4, 8, 12], [2, 5, 6, 10, 11]]
                     means 3 individuals in front 0, 4 in front 1, 5 in front 2
        """
        n = len(objectives); domination_counts = np.zeros(n); dominated_solutions: list[list[int]] = [[] for _ in range(n)]; fronts: list[list[int]] = [[]]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(objectives[i] >= objectives[j]) and np.any(objectives[i] > objectives[j]): dominated_solutions[i].append(j)
                    elif np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]): domination_counts[i] += 1
            if domination_counts[i] == 0: fronts[0].append(i)
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0: next_front.append(j)
            current_front += 1; fronts.append(next_front)
        return fronts[:-1]
        
    def _calculate_crowding_distance(self, front_fvals: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for diversity preservation.

        Crowding distance is a NSGA-II innovation measuring how isolated each
        solution is from its neighbors in objective space. It prevents premature
        convergence by preferring diverse solutions, enabling approximation of
        the full Pareto front rather than clustering around few points.

        Concept:
            Imagine solutions plotted in objective space. Each solution has a
            "neighborhood": distance to nearest neighbors. Solutions far from
            others (isolated) have high crowding distance. Dense regions have
            low crowding distance. We prefer isolated solutions to spread the
            population across the Pareto front.

        Algorithm:
            1. Set all solutions' distances to 0 (sum of contributions)
            2. For each objective independently:
               a. Sort solutions by that objective's value
               b. Set boundary solutions (minimum/maximum) to infinite distance
                  (boundary solutions always valuable for exploring front edges)
               c. For interior solutions i: contribution = (next_val - prev_val) / range
                  (larger gaps to neighbors = higher contribution)
               d. Add contribution to total distance
            3. Sum contributions across all objectives

        Why This Works:
            - Boundaries: extreme solutions get infinity (always preferred)
            - Gaps: isolated solutions have large neighbor distances
            - Scaling: each objective normalized by its range
            - Aggregation: sums across objectives (multi-dimensional isolation)

        Benefits:
            - Preserves boundary solutions (explore front edges)
            - Avoids clustering (isolated solutions preferred)
            - Fair comparison across objectives (each normalized)
            - Cheap computation: O(N log N) per objective due to sort

        Mathematical Formulation:
            For each solution in front, crowding distance = Σ_j (next_j - prev_j)
            where next_j and prev_j are neighboring solutions in objective j,
            normalized by range_j (max_j - min_j).

        Edge Cases:
            - Small fronts (≤2 solutions): all assigned infinite distance (preserve all)
            - NaN values: replaced with median of column (robust to crashes)
            - Zero range: solutions with identical objective value in one dimension

        Args:
            front_fvals: Objective values of individuals in a Pareto front
                        Shape: (n_solutions_in_front, n_objectives)

        Returns:
            Array of crowding distances: (n_solutions_in_front,)
            Each value ≥ 0, with boundary solutions = np.inf
        """
        if front_fvals is None or len(front_fvals) == 0: return np.array([])
        N, M = front_fvals.shape; distances = np.zeros(N, dtype=float)
        if N <= 2: distances[:] = np.inf; return distances
        for j in range(M):
            vals = front_fvals[:, j].astype(float)
            if np.isnan(vals).any(): vals[np.isnan(vals)] = np.nanmedian(vals) or 0.0
            order = np.argsort(vals); sorted_vals = vals[order]
            distances[order[0]] = np.inf; distances[order[-1]] = np.inf
            obj_range = sorted_vals[-1] - sorted_vals[0]
            if obj_range > 0: distances[order[1:-1]] += (sorted_vals[2:] - sorted_vals[:-2]) / obj_range
        return distances

    def _normalize_objectives_for_selection(self, objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objectives to [0, 1] range for proper multi-objective selection.

        This is critical for multi-target optimization where objectives may have
        different scales (e.g., streamflow KGE [-inf, 1] vs TWS KGE [-inf, 1] but
        with different typical ranges).

        Uses robust normalization that handles extreme values and ensures both
        objectives contribute meaningfully to selection pressure.
        """
        if objectives is None or len(objectives) == 0:
            return objectives

        normalized = np.zeros_like(objectives, dtype=float)

        for j in range(objectives.shape[1]):
            col = objectives[:, j].copy()

            # Replace NaN with column minimum
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                normalized[:, j] = 0.5
                continue

            # Clip extreme negative values (KGE/NSE can go to -inf)
            # Use -1 as practical minimum for skill metrics
            col = np.clip(col, -1.0, 1.0)

            min_val = np.nanmin(col)
            max_val = np.nanmax(col)

            if max_val > min_val:
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, j] = 0.5

        return normalized

    def _update_representative_solution(self) -> None:
        assert self.population_ranks is not None
        assert self.population_objectives is not None
        assert self.population is not None
        front_1_indices = np.where(self.population_ranks == 0)[0]
        if len(front_1_indices) > 0:
            objs = self.population_objectives[front_1_indices]
            # Use proper normalization for multi-objective selection
            # This ensures both objectives contribute equally regardless of scale
            normalized_objs = self._normalize_objectives_for_selection(objs)
            composite_scores = np.sum(normalized_objs, axis=1)
            best_idx = front_1_indices[np.argmax(composite_scores)]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
            self.best_score = np.max(composite_scores)
        else:
            self.best_params = None; self.best_score = -2.0
        
    def _record_generation(self, generation: int) -> None:
        assert self.population_objectives is not None
        valid_obj1 = self.population_objectives[:, 0][~np.isnan(self.population_objectives[:, 0])]
        valid_obj2 = self.population_objectives[:, 1][~np.isnan(self.population_objectives[:, 1])]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'NSGA-II', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_nse': np.mean(valid_obj1) if len(valid_obj1) > 0 else None,
            'mean_kge': np.mean(valid_obj2) if len(valid_obj2) > 0 else None,
            'valid_individuals': len(valid_obj1)
        })
