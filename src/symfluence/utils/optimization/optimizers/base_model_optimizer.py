"""
Base Model Optimizer

Abstract base class for model-specific optimizers (FUSE, NGEN, SUMMA).
Uses mixins for shared functionality and provides template methods for
algorithm implementations.
"""

import logging
import random
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime

from ..mixins import (
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
    GradientOptimizationMixin
)
from ..workers.base_worker import BaseWorker, WorkerTask, WorkerResult


class BaseModelOptimizer(
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
    GradientOptimizationMixin,
    ABC
):
    """
    Abstract base class for model-specific optimizers.

    Provides shared infrastructure for optimization including:
    - Parallel processing (via ParallelExecutionMixin)
    - Results tracking (via ResultsTrackingMixin)
    - Retry logic (via RetryExecutionMixin)
    - Gradient optimization (via GradientOptimizationMixin)

    Subclasses must implement:
    - _get_model_name(): Return model name (e.g., 'FUSE', 'NGEN')
    - _create_parameter_manager(): Create model-specific parameter manager
    - _create_calibration_target(): Create model-specific calibration target
    - _create_worker(): Create model-specific worker

    Provides algorithm implementations:
    - run_dds(): Dynamical Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization
    """

    # Default algorithm parameters
    DEFAULT_ITERATIONS = 100
    DEFAULT_POPULATION_SIZE = 30
    DEFAULT_PENALTY_SCORE = -999.0

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the model optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager

        # Setup paths
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        self.domain_name = config.get('DOMAIN_NAME', 'default')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID', 'optimization')

        # Optimization settings directory
        if optimization_settings_dir is not None:
            self.optimization_settings_dir = Path(optimization_settings_dir)
        else:
            model_name = self._get_model_name()
            self.optimization_settings_dir = (
                self.project_dir / 'settings' / model_name
            )

        # Results directory
        self.results_dir = (
            self.project_dir / 'optimization' /
            f"{self._get_model_name().lower()}_{self.experiment_id}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results tracking
        self.__init_results_tracking__()

        # Create model-specific components
        self.param_manager = self._create_parameter_manager()
        self.calibration_target = self._create_calibration_target()
        self.worker = self._create_worker()

        # Algorithm parameters
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', self.DEFAULT_ITERATIONS)
        self.population_size = config.get('POPULATION_SIZE', self.DEFAULT_POPULATION_SIZE)
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')

        # Random seed
        self.random_seed = config.get('RANDOM_SEED')
        if self.random_seed is not None and self.random_seed != 'None':
            self._set_random_seeds(int(self.random_seed))

        # Parallel processing state
        self.parallel_dirs = {}
        if self.use_parallel:
            self._setup_parallel_dirs()

    def _visualize_progress(self, algorithm: str) -> None:
        """Helper to visualize optimization progress if reporting manager available."""
        if self.reporting_manager:
            calibration_variable = self.config.get("CALIBRATION_VARIABLE", "streamflow")
            self.reporting_manager.visualize_optimization_progress(
                self._iteration_history, 
                self.results_dir.parent / f"{algorithm.lower()}_{self.experiment_id}", # Matches results_dir logic or pass results_dir
                calibration_variable, 
                self.target_metric
            )
            
            if self.config.get('CALIBRATE_DEPTH', False):
                self.reporting_manager.visualize_optimization_depth_parameters(
                    self._iteration_history, 
                    self.results_dir.parent / f"{algorithm.lower()}_{self.experiment_id}"
                )

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        Return the model name (e.g., 'FUSE', 'NGEN', 'SUMMA').

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    def _create_parameter_manager(self):
        """
        Create the model-specific parameter manager.

        Returns:
            Parameter manager instance
        """
        pass

    @abstractmethod
    def _create_calibration_target(self):
        """
        Create the model-specific calibration target.

        Returns:
            Calibration target instance
        """
        pass

    @abstractmethod
    def _create_worker(self) -> BaseWorker:
        """
        Create the model-specific worker.

        Returns:
            Worker instance
        """
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel processing directories."""
        base_dir = self.project_dir / 'parallel_processing'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            self._get_model_name(),
            self.experiment_id
        )

    # =========================================================================
    # Evaluation methods
    # =========================================================================

    def _evaluate_solution(
        self,
        normalized_params: np.ndarray,
        proc_id: int = 0
    ) -> float:
        """
        Evaluate a normalized parameter set.

        Args:
            normalized_params: Normalized parameters [0, 1]
            proc_id: Process ID for parallel execution

        Returns:
            Fitness score
        """
        # Denormalize parameters
        params = self.param_manager.denormalize_parameters(normalized_params)

        # Create task
        dirs = self.parallel_dirs.get(proc_id, {})
        task = WorkerTask(
            individual_id=0,
            params=params,
            proc_id=proc_id,
            config=self.config,
            settings_dir=dirs.get('settings_dir', self.optimization_settings_dir),
            output_dir=dirs.get('output_dir', self.results_dir),
            sim_dir=dirs.get('sim_dir'),
        )

        # Evaluate
        result = self.worker.evaluate(task)

        return result.score if result.score is not None else self.DEFAULT_PENALTY_SCORE

    def _evaluate_population(
        self,
        population: np.ndarray,
        iteration: int = 0
    ) -> np.ndarray:
        """
        Evaluate a population of solutions.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            iteration: Current iteration number

        Returns:
            Array of fitness scores
        """
        n_individuals = len(population)
        fitness = np.full(n_individuals, self.DEFAULT_PENALTY_SCORE)

        if self.use_parallel and n_individuals > 1:
            # Parallel evaluation
            tasks = []
            for i, params_normalized in enumerate(population):
                params = self.param_manager.denormalize_parameters(params_normalized)
                proc_id = i % self.num_processes
                dirs = self.parallel_dirs.get(proc_id, {})

                task = WorkerTask(
                    individual_id=i,
                    params=params,
                    proc_id=proc_id,
                    config=self.config,
                    settings_dir=dirs.get('settings_dir', self.optimization_settings_dir),
                    output_dir=dirs.get('output_dir', self.results_dir),
                    sim_dir=dirs.get('sim_dir'),
                    iteration=iteration,
                )
                tasks.append(task.to_legacy_dict())

            # Execute batch
            worker_func = self.worker.evaluate_worker_function
            results = self.execute_batch(tasks, worker_func)

            # Extract scores
            for result in results:
                idx = result.get('individual_id', 0)
                score = result.get('score')
                if score is not None and not np.isnan(score):
                    fitness[idx] = score
        else:
            # Sequential evaluation
            for i, params_normalized in enumerate(population):
                fitness[i] = self._evaluate_solution(params_normalized, proc_id=0)

        return fitness

    # =========================================================================
    # Algorithm implementations
    # =========================================================================

    def run_dds(self) -> Path:
        """
        Run Dynamically Dimensioned Search (DDS) optimization.

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting DDS optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)

        # Initialize with random starting point
        x_best = np.random.uniform(0, 1, n_params)
        f_best = self._evaluate_solution(x_best)

        self.record_iteration(0, f_best, self.param_manager.denormalize_parameters(x_best))
        self.update_best(f_best, self.param_manager.denormalize_parameters(x_best), 0)

        # DDS main loop
        for iteration in range(1, self.max_iterations + 1):
            # Calculate probability of perturbation
            p = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            p = max(1.0 / n_params, p)  # Ensure at least one parameter is perturbed

            # Select parameters to perturb
            perturb_mask = np.random.random(n_params) < p

            # Ensure at least one parameter is perturbed
            if not perturb_mask.any():
                perturb_mask[np.random.randint(n_params)] = True

            # Generate candidate
            x_new = x_best.copy()
            r = 0.2  # Perturbation range

            for i in range(n_params):
                if perturb_mask[i]:
                    perturbation = r * np.random.standard_normal()
                    x_new[i] = x_best[i] + perturbation

                    # Reflect at boundaries
                    if x_new[i] < 0:
                        x_new[i] = -x_new[i]
                    if x_new[i] > 1:
                        x_new[i] = 2 - x_new[i]

                    # Clip to bounds
                    x_new[i] = np.clip(x_new[i], 0, 1)

            # Evaluate candidate
            f_new = self._evaluate_solution(x_new)

            # Update if better
            if f_new > f_best:
                x_best = x_new
                f_best = f_new

            # Record results
            params_dict = self.param_manager.denormalize_parameters(x_best)
            self.record_iteration(iteration, f_best, params_dict)
            self.update_best(f_best, params_dict, iteration)

            if iteration % 10 == 0:
                self.logger.info(
                    f"DDS iteration {iteration}/{self.max_iterations}: "
                    f"best={f_best:.4f}"
                )

        # Save results
        results_path = self.save_results('DDS', standard_filename=True)
        self.save_best_params('DDS')
        self._visualize_progress('DDS')

        self.logger.info(f"DDS completed in {self.format_elapsed_time()}")
        return results_path

    def run_pso(self) -> Path:
        """
        Run Particle Swarm Optimization (PSO).

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting PSO optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)
        n_particles = self.population_size

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        v_max = 0.2  # Maximum velocity

        # Initialize swarm
        positions = np.random.uniform(0, 1, (n_particles, n_params))
        velocities = np.random.uniform(-v_max, v_max, (n_particles, n_params))

        # Evaluate initial population
        fitness = self._evaluate_population(positions, iteration=0)

        # Initialize personal and global bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        global_best_idx = np.argmax(fitness)
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = fitness[global_best_idx]

        # Record initial best
        self.record_iteration(0, global_best_fit, self.param_manager.denormalize_parameters(global_best_pos))
        self.update_best(global_best_fit, self.param_manager.denormalize_parameters(global_best_pos), 0)

        # PSO main loop
        for iteration in range(1, self.max_iterations + 1):
            # Update velocities
            r1 = np.random.random((n_particles, n_params))
            r2 = np.random.random((n_particles, n_params))

            cognitive = c1 * r1 * (personal_best_pos - positions)
            social = c2 * r2 * (global_best_pos - positions)

            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, 0, 1)

            # Evaluate
            fitness = self._evaluate_population(positions, iteration=iteration)

            # Update personal bests
            improved = fitness > personal_best_fit
            personal_best_pos[improved] = positions[improved]
            personal_best_fit[improved] = fitness[improved]

            # Update global best
            if np.max(fitness) > global_best_fit:
                global_best_idx = np.argmax(fitness)
                global_best_pos = positions[global_best_idx].copy()
                global_best_fit = fitness[global_best_idx]

            # Record results
            params_dict = self.param_manager.denormalize_parameters(global_best_pos)
            self.record_iteration(iteration, global_best_fit, params_dict)
            self.update_best(global_best_fit, params_dict, iteration)

            if iteration % 10 == 0:
                self.logger.info(
                    f"PSO iteration {iteration}/{self.max_iterations}: "
                    f"best={global_best_fit:.4f}"
                )

        # Save results
        results_path = self.save_results('PSO', standard_filename=True)
        self.save_best_params('PSO')
        self._visualize_progress('PSO')

        self.logger.info(f"PSO completed in {self.format_elapsed_time()}")
        return results_path

    def run_de(self) -> Path:
        """
        Run Differential Evolution (DE) optimization.

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting DE optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)
        pop_size = self.population_size

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.uniform(0, 1, (pop_size, n_params))
        fitness = self._evaluate_population(population, iteration=0)

        # Record initial best
        best_idx = np.argmax(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        self.record_iteration(0, best_fit, self.param_manager.denormalize_parameters(best_pos))
        self.update_best(best_fit, self.param_manager.denormalize_parameters(best_pos), 0)

        # DE main loop
        for iteration in range(1, self.max_iterations + 1):
            for i in range(pop_size):
                # Select three random individuals (not i)
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, 0, 1)

                # Crossover
                cross_points = np.random.random(n_params) < CR
                if not cross_points.any():
                    cross_points[np.random.randint(n_params)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = self._evaluate_solution(trial, proc_id=i % self.num_processes)

                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness > best_fit:
                        best_pos = trial.copy()
                        best_fit = trial_fitness

            # Record results
            params_dict = self.param_manager.denormalize_parameters(best_pos)
            self.record_iteration(iteration, best_fit, params_dict)
            self.update_best(best_fit, params_dict, iteration)

            if iteration % 10 == 0:
                self.logger.info(
                    f"DE iteration {iteration}/{self.max_iterations}: "
                    f"best={best_fit:.4f}"
                )

        # Save results
        results_path = self.save_results('DE', standard_filename=True)
        self.save_best_params('DE')
        self._visualize_progress('DE')

        self.logger.info(f"DE completed in {self.format_elapsed_time()}")
        return results_path

    def run_sce(self) -> Path:
        """
        Run Shuffled Complex Evolution (SCE-UA) optimization.

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting SCE-UA optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)

        # SCE-UA parameters
        n_complexes = max(2, self.population_size // 10)
        n_per_complex = 2 * n_params + 1
        pop_size = n_complexes * n_per_complex

        # Initialize population
        population = np.random.uniform(0, 1, (pop_size, n_params))
        fitness = self._evaluate_population(population, iteration=0)

        # Sort by fitness (descending for maximization)
        sorted_idx = np.argsort(-fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        best_pos = population[0].copy()
        best_fit = fitness[0]

        self.record_iteration(0, best_fit, self.param_manager.denormalize_parameters(best_pos))
        self.update_best(best_fit, self.param_manager.denormalize_parameters(best_pos), 0)

        # SCE-UA main loop
        for iteration in range(1, self.max_iterations + 1):
            # Partition into complexes
            for complex_idx in range(n_complexes):
                complex_members = list(range(complex_idx, pop_size, n_complexes))
                sub_complex = population[complex_members]
                sub_fitness = fitness[complex_members]

                # Evolve sub-complex (simplified CCE step)
                for _ in range(n_per_complex):
                    # Select simplex
                    simplex_size = n_params + 1
                    simplex_idx = np.random.choice(
                        len(complex_members), simplex_size, replace=False
                    )

                    # Generate new point (reflection)
                    worst_idx = simplex_idx[np.argmin(sub_fitness[simplex_idx])]
                    others = [i for i in simplex_idx if i != worst_idx]
                    centroid = np.mean(sub_complex[others], axis=0)

                    # Reflection
                    new_point = 2 * centroid - sub_complex[worst_idx]
                    new_point = np.clip(new_point, 0, 1)

                    # Evaluate
                    new_fitness = self._evaluate_solution(new_point)

                    if new_fitness > sub_fitness[worst_idx]:
                        sub_complex[worst_idx] = new_point
                        sub_fitness[worst_idx] = new_fitness

                # Update main population
                population[complex_members] = sub_complex
                fitness[complex_members] = sub_fitness

            # Shuffle and sort
            sorted_idx = np.argsort(-fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update best
            if fitness[0] > best_fit:
                best_pos = population[0].copy()
                best_fit = fitness[0]

            # Record results
            params_dict = self.param_manager.denormalize_parameters(best_pos)
            self.record_iteration(iteration, best_fit, params_dict)
            self.update_best(best_fit, params_dict, iteration)

            if iteration % 10 == 0:
                self.logger.info(
                    f"SCE-UA iteration {iteration}/{self.max_iterations}: "
                    f"best={best_fit:.4f}"
                )

        # Save results
        results_path = self.save_results('SCE-UA', standard_filename=True)
        self.save_best_params('SCE-UA')
        self._visualize_progress('SCE-UA')

        self.logger.info(f"SCE-UA completed in {self.format_elapsed_time()}")
        return results_path

    def run_adam(self, steps: int = 100, lr: float = 0.01) -> Path:
        """
        Run Adam gradient-based optimization.

        Args:
            steps: Number of optimization steps
            lr: Learning rate

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting Adam optimization for {self._get_model_name()}")

        def evaluate_func(x):
            return self._evaluate_solution(x)

        best_x, best_fitness, history = self._run_adam(
            evaluate_func,
            steps=steps,
            lr=lr
        )

        # Record history
        for record in history:
            self.record_iteration(
                record['step'],
                record['fitness'],
                self.param_manager.denormalize_parameters(best_x)
            )

        self.update_best(best_fitness, self.param_manager.denormalize_parameters(best_x), steps)

        # Save results
        results_path = self.save_results('ADAM', standard_filename=True)
        self.save_best_params('ADAM')
        self._visualize_progress('ADAM')

        self.logger.info(f"Adam completed in {self.format_elapsed_time()}")
        return results_path

    def run_lbfgs(self, steps: int = 50, lr: float = 0.1) -> Path:
        """
        Run L-BFGS gradient-based optimization.

        Args:
            steps: Maximum number of steps
            lr: Initial step size

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting L-BFGS optimization for {self._get_model_name()}")

        def evaluate_func(x):
            return self._evaluate_solution(x)

        best_x, best_fitness, history = self._run_lbfgs(
            evaluate_func,
            steps=steps,
            lr=lr
        )

        # Record history
        for record in history:
            self.record_iteration(
                record['step'],
                record['fitness'],
                self.param_manager.denormalize_parameters(best_x)
            )

        self.update_best(best_fitness, self.param_manager.denormalize_parameters(best_x), len(history))

        # Save results
        results_path = self.save_results('LBFGS', standard_filename=True)
        self.save_best_params('LBFGS')
        self._visualize_progress('LBFGS')

        self.logger.info(f"L-BFGS completed in {self.format_elapsed_time()}")
        return results_path

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Cleanup parallel processing directories and temporary files."""
        if self.parallel_dirs:
            self.cleanup_parallel_processing(self.parallel_dirs)
