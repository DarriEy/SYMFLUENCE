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
        algorithm = config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        self.results_dir = (
            self.project_dir / 'optimization' /
            f"{algorithm}_{self.experiment_id}"
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
        self.default_sim_dir = self.results_dir  # Initialize with results_dir as fallback
        # Setup directories if MPI_PROCESSES is set, regardless of count (for isolation)
        if config.get('MPI_PROCESSES', 1) >= 1:
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
        """Return the name of the model being optimized."""
        pass

    @abstractmethod
    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run the model for final evaluation (model-specific implementation)."""
        pass

    @abstractmethod
    def _get_final_file_manager_path(self) -> Path:
        """Get path to the file manager used for final evaluation."""
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

    def _adjust_end_time_for_forcing(self, end_time_str: str) -> str:
        """
        Adjust end time to align with forcing data timestep.
        For sub-daily forcing (e.g., 3-hourly CERRA), ensures end time is a valid timestep.

        Args:
            end_time_str: End time string in format 'YYYY-MM-DD HH:MM'

        Returns:
            Adjusted end time string
        """
        try:
            forcing_timestep_seconds = self.config.get('FORCING_TIME_STEP_SIZE', 3600)

            if forcing_timestep_seconds >= 3600:  # Hourly or coarser
                # Parse the end time
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

                # Calculate the last valid hour based on timestep
                forcing_timestep_hours = forcing_timestep_seconds / 3600
                last_hour = int(24 - (24 % forcing_timestep_hours)) - forcing_timestep_hours
                if last_hour < 0:
                    last_hour = 0

                # Adjust if needed
                if end_time.hour > last_hour or (end_time.hour == 23 and last_hour < 23):
                    end_time = end_time.replace(hour=int(last_hour), minute=0)
                    adjusted_str = end_time.strftime('%Y-%m-%d %H:%M')
                    self.logger.info(f"Adjusted end time from {end_time_str} to {adjusted_str} for {forcing_timestep_hours}h forcing")
                    return adjusted_str

            return end_time_str

        except Exception as e:
            self.logger.warning(f"Could not adjust end time: {e}")
            return end_time_str

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel processing directories."""
        # Determine algorithm for directory naming
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()

        # Use algorithm-specific directory
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            self._get_model_name(),
            self.experiment_id
        )

        # For non-parallel runs, set a default output directory for fallback
        # This ensures SUMMA outputs go to the simulation directory, not the optimization results directory
        if not self.use_parallel and self.parallel_dirs:
            # Use process_0 directories as the default
            self.default_sim_dir = self.parallel_dirs[0].get('sim_dir', self.results_dir)
        else:
            self.default_sim_dir = self.results_dir

    def _create_mpi_worker_script(self, script_path: Path, tasks_file: Path, results_file: Path, worker_module: str, worker_function: str) -> None:
        """
        Create the MPI worker script file with model-specific worker function.

        This method is called by ParallelExecutionMixin to create the MPI worker script.
        The worker_module and worker_function are already determined by the mixin.

        Args:
            script_path: Path to create the worker script at
            tasks_file: Path to the tasks pickle file
            results_file: Path to the results pickle file
            worker_module: Module containing the worker function
            worker_function: Name of the worker function to call
        """
        # Calculate the correct path to the src directory
        src_path = Path(__file__).parent.parent.parent.parent

        script_content = f'''#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Silence noisy libraries
for noisy_logger in ['rasterio', 'fiona', 'boto3', 'botocore', 'matplotlib', 'urllib3', 's3transfer']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Add symphluence src to path to ensure imports work
sys.path.insert(0, r"{str(src_path)}")

try:
    from {worker_module} import {worker_function}
except ImportError as e:
    logger.error(f"Failed to import {{e}}")
    sys.exit(1)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])

    if rank == 0:
        with open(tasks_file, 'rb') as f:
            all_tasks = pickle.load(f)

        tasks_per_rank = len(all_tasks) // size
        extra_tasks = len(all_tasks) % size
        all_results = []

        for worker_rank in range(size):
            start_idx = worker_rank * tasks_per_rank + min(worker_rank, extra_tasks)
            end_idx = start_idx + tasks_per_rank + (1 if worker_rank < extra_tasks else 0)

            if worker_rank == 0:
                my_tasks = all_tasks[start_idx:end_idx]
            else:
                comm.send(all_tasks[start_idx:end_idx], dest=worker_rank, tag=1)

        for task in my_tasks:
            all_results.append({worker_function}(task))

        for worker_rank in range(1, size):
            all_results.extend(comm.recv(source=worker_rank, tag=2))

        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
    else:
        my_tasks = comm.recv(source=0, tag=1)
        my_results = [{worker_function}(t) for t in my_tasks]
        comm.send(my_results, dest=0, tag=2)

if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w') as f:
            f.write(script_content)

    # =========================================================================
    # Evaluation methods
    # =========================================================================

    def log_iteration_progress(
        self,
        algorithm_name: str,
        iteration: int,
        best_score: float,
        n_improved: Optional[int] = None,
        population_size: Optional[int] = None
    ) -> None:
        """
        Log optimization progress in a consistent format across all algorithms.

        Args:
            algorithm_name: Name of the algorithm (e.g., 'DDS', 'PSO', 'DE')
            iteration: Current iteration number
            best_score: Current best score
            n_improved: Optional number of improved individuals (for population-based)
            population_size: Optional total population size
        """
        progress_pct = (iteration / self.max_iterations) * 100
        elapsed = self.format_elapsed_time()

        msg_parts = [
            f"{algorithm_name} {iteration}/{self.max_iterations} ({progress_pct:.0f}%)",
            f"Best: {best_score:.4f}"
        ]

        if n_improved is not None and population_size is not None:
            msg_parts.append(f"Improved: {n_improved}/{population_size}")

        msg_parts.append(f"Elapsed: {elapsed}")

        self.logger.info(" | ".join(msg_parts))

    def log_initial_population(
        self,
        algorithm_name: str,
        population_size: int,
        best_score: float
    ) -> None:
        """
        Log initial population evaluation completion.

        Args:
            algorithm_name: Name of the algorithm
            population_size: Size of the population
            best_score: Best score from initial population
        """
        self.logger.info(
            f"{algorithm_name} initial population ({population_size} individuals) "
            f"complete | Best score: {best_score:.4f}"
        )

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
            output_dir=dirs.get('sim_dir', self.default_sim_dir),
            sim_dir=dirs.get('sim_dir', self.default_sim_dir),
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

                # Create task with all required fields (matching v0.5.0)
                # Use correct parallel_dirs keys: 'settings_dir', 'sim_dir', 'output_dir'
                settings_dir = Path(dirs.get('settings_dir', self.optimization_settings_dir))
                task_dict = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': proc_id,
                    'evaluation_id': f"pop_eval_{i:03d}",
                    'config': self.config,
                    'target_metric': self.target_metric,
                    'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                    'domain_name': self.domain_name,
                    'project_dir': str(self.project_dir),
                    'proc_settings_dir': str(settings_dir),
                    'proc_output_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                    'proc_sim_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                    'summa_settings_dir': str(settings_dir),
                    'mizuroute_settings_dir': str(dirs.get('root', self.project_dir) / 'settings' / 'mizuRoute') if dirs else '',
                    'summa_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                    'mizuroute_dir': str(Path(dirs.get('sim_dir', self.default_sim_dir)).parent / 'mizuRoute') if dirs and dirs.get('sim_dir') else str(Path(self.default_sim_dir).parent / 'mizuRoute'),
                    'mizuroute_settings_dir': str(dirs.get('root', self.project_dir) / 'settings' / 'mizuRoute') if dirs else '',
                    'file_manager': str(settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')),
                    'summa_exe': str(self.summa_exe_path) if hasattr(self, 'summa_exe_path') else '',
                    'original_depths': self.param_manager.original_depths.tolist() if hasattr(self.param_manager, 'original_depths') and self.param_manager.original_depths is not None else None,
                }
                if hasattr(self, 'random_seed') and self.random_seed is not None:
                    task_dict['random_seed'] = self.random_seed + i + 1000

                tasks.append(task_dict)

            # Execute batch
            # Dynamic resolution of module-level worker function (required for MPI import)
            model_name_lower = self._get_model_name().lower()
            worker_func = None
            
            try:
                # Get the module where the worker is defined
                import sys
                worker_module_name = self.worker.__class__.__module__
                worker_module = sys.modules.get(worker_module_name)
                
                # Look for naming convention function: _evaluate_<model>_parameters_worker
                func_name = f"_evaluate_{model_name_lower}_parameters_worker"
                if worker_module and hasattr(worker_module, func_name):
                    worker_func = getattr(worker_module, func_name)
                    self.logger.debug(f"Resolved MPI worker function from loaded module: {worker_module_name}.{func_name}")
            except Exception as e:
                self.logger.debug(f"Dynamic worker resolution failed: {e}")
                
            if worker_func is None:
                # Fallback to class static method (may fail under MPI if not module-level)
                worker_func = self.worker.evaluate_worker_function
                
            results = self.execute_batch(tasks, worker_func)

            # Extract scores
            valid_count = 0
            for result in results:
                idx = result.get('individual_id', 0)
                score = result.get('score')
                error = result.get('error')
                if error:
                    error_str = str(error)
                    # Log full error at debug level, truncated at info level
                    self.logger.debug(f"Task {idx} full error: {error_str}")
                    self.logger.warning(f"Task {idx} error: {error_str[:500] if len(error_str) > 500 else error_str}")
                if score is not None and not np.isnan(score):
                    fitness[idx] = score
                    if score != self.DEFAULT_PENALTY_SCORE:
                        valid_count += 1
                else:
                    self.logger.warning(f"Task {idx} returned score={score}")
            self.logger.debug(f"Batch results: {len(results)} returned, {valid_count} valid scores")
        else:
            # Sequential evaluation
            for i, params_normalized in enumerate(population):
                # In sequential mode, proc_id is always 0, but we pass it for consistency
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

            # Log progress
            self.log_iteration_progress('DDS', iteration, f_best)

        # Save results
        results_path = self.save_results('DDS', standard_filename=True)
        self.save_best_params('DDS')
        self._visualize_progress('DDS')

        self.logger.info(f"DDS completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(x_best)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'DDS')

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
        self.logger.info(f"Evaluating initial population ({n_particles} particles)...")
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

        self.log_initial_population('PSO', n_particles, global_best_fit)

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
            n_improved = np.sum(improved)
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

            # Log progress
            self.log_iteration_progress('PSO', iteration, global_best_fit, n_improved, n_particles)

        # Save results
        results_path = self.save_results('PSO', standard_filename=True)
        self.save_best_params('PSO')
        self._visualize_progress('PSO')

        self.logger.info(f"PSO completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(global_best_pos)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'PSO')

        return results_path

    def run_de(self) -> Path:
        """
        Run Differential Evolution (DE) optimization.
        
        Returns:
            Path to the results CSV file
        """
        self.logger.info(f"Starting DE optimization for {self._get_model_name()}")
        
        # DE requires at least 4 individuals to sample 3 distinct ones (r1, r2, r3) plus current i
        if self.population_size < 4:
            self.logger.warning(f"Population size {self.population_size} is too small for DE. Increasing to 4.")
            self.population_size = 4

        n_params = len(self.param_manager.all_param_names)
        pop_size = self.population_size

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Initialize population
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        population = np.random.uniform(0, 1, (pop_size, n_params))
        fitness = self._evaluate_population(population, iteration=0)

        # Record initial best
        best_idx = np.argmax(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        self.record_iteration(0, best_fit, self.param_manager.denormalize_parameters(best_pos))
        self.update_best(best_fit, self.param_manager.denormalize_parameters(best_pos), 0)

        self.log_initial_population('DE', pop_size, best_fit)

        # DE main loop
        for iteration in range(1, self.max_iterations + 1):
            # Generate all trial solutions for this iteration
            trials = []
            trial_indices = []
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
                trials.append(trial)
                trial_indices.append(i)

            # Batch evaluate all trials in parallel - pass as separate evaluations with correct indices
            if self.use_parallel and len(trials) > 1:
                # Parallel batch evaluation
                tasks = []
                for idx, trial in enumerate(trials):
                    params = self.param_manager.denormalize_parameters(trial)
                    proc_id = trial_indices[idx] % self.num_processes
                    dirs = self.parallel_dirs.get(proc_id, {})

                    # Create task with all required fields (matching v0.5.0)
                    # Use correct parallel_dirs keys: 'settings_dir', 'sim_dir', 'output_dir'
                    settings_dir = Path(dirs.get('settings_dir', self.optimization_settings_dir))
                    task_dict = {
                        'individual_id': trial_indices[idx],
                        'params': params,
                        'proc_id': proc_id,
                        'evaluation_id': f"trial_{iteration:03d}_{trial_indices[idx]:03d}",
                        'config': self.config,
                        'target_metric': self.target_metric,
                        'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                        'domain_name': self.domain_name,
                        'project_dir': str(self.project_dir),
                        'proc_settings_dir': str(settings_dir),
                        'proc_output_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                        'proc_sim_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                        'summa_settings_dir': str(settings_dir),
                        'mizuroute_settings_dir': str(dirs.get('root', self.project_dir) / 'settings' / 'mizuRoute') if dirs else '',
                        'summa_dir': str(dirs.get('sim_dir', self.default_sim_dir)),
                        'mizuroute_dir': str(Path(dirs.get('sim_dir', self.default_sim_dir)).parent / 'mizuRoute') if dirs and dirs.get('sim_dir') else str(Path(self.default_sim_dir).parent / 'mizuRoute'),
                        'file_manager': str(settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')),
                        'summa_exe': str(self.summa_exe_path) if hasattr(self, 'summa_exe_path') else '',
                        'original_depths': self.param_manager.original_depths.tolist() if hasattr(self.param_manager, 'original_depths') and self.param_manager.original_depths is not None else None,
                    }
                    if hasattr(self, 'random_seed') and self.random_seed is not None:
                        task_dict['random_seed'] = self.random_seed + trial_indices[idx] + 1000

                    tasks.append(task_dict)

                # Execute batch
                # Dynamic resolution of module-level worker function (required for MPI import)
                model_name_lower = self._get_model_name().lower()
                worker_func = None
                
                try:
                    import sys
                    worker_module_name = self.worker.__class__.__module__
                    worker_module = sys.modules.get(worker_module_name)
                    
                    func_name = f"_evaluate_{model_name_lower}_parameters_worker"
                    if worker_module and hasattr(worker_module, func_name):
                        worker_func = getattr(worker_module, func_name)
                except Exception:
                    pass
                    
                if worker_func is None:
                    worker_func = self.worker.evaluate_worker_function
                    
                results = self.execute_batch(tasks, worker_func)

                # Extract trial fitness scores
                trial_fitness = np.full(pop_size, self.DEFAULT_PENALTY_SCORE)
                for result in results:
                    idx = result.get('individual_id', 0)
                    score = result.get('score')
                    error = result.get('error')
                    if error:
                        error_str = str(error)
                        self.logger.debug(f"Task {idx} full error: {error_str}")
                        self.logger.warning(f"Task {idx} error: {error_str[:500] if len(error_str) > 500 else error_str}")
                    if score is not None and not np.isnan(score):
                        # Find which trial this corresponds to
                        if idx in trial_indices:
                            trial_idx = trial_indices.index(idx)
                            trial_fitness[trial_idx] = score
            else:
                # Sequential evaluation fallback
                trial_fitness = np.array([
                    self._evaluate_solution(trial, proc_id=trial_indices[i] % self.num_processes)
                    for i, trial in enumerate(trials)
                ])

            # Selection - update population based on trial results
            n_improved = 0
            for i in range(pop_size):
                if trial_fitness[i] > fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
                    n_improved += 1

                    if trial_fitness[i] > best_fit:
                        best_pos = trials[i].copy()
                        best_fit = trial_fitness[i]

            # Record results
            params_dict = self.param_manager.denormalize_parameters(best_pos)
            self.record_iteration(iteration, best_fit, params_dict)
            self.update_best(best_fit, params_dict, iteration)

            # Log progress
            self.log_iteration_progress('DE', iteration, best_fit, n_improved, pop_size)

        # Save results
        results_path = self.save_results('DE', standard_filename=True)
        self.save_best_params('DE')
        self._visualize_progress('DE')

        self.logger.info(f"DE completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_pos)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'DE')

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
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
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

        self.log_initial_population('SCE-UA', pop_size, best_fit)

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

            # Log progress
            self.log_iteration_progress('SCE-UA', iteration, best_fit)

        # Save results
        results_path = self.save_results('SCE-UA', standard_filename=True)
        self.save_best_params('SCE-UA')
        self._visualize_progress('SCE-UA')

        self.logger.info(f"SCE-UA completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_pos)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'SCE-UA')

        return results_path

    def run_async_dds(self) -> Path:
        """
        Run Asynchronous Parallel DDS optimization.

        Asynchronous DDS maintains a pool of best solutions and generates
        batches of trials by selecting from the pool. Much more efficient
        for parallel execution than synchronous DDS.

        Returns:
            Path to results file
        """
        import random
        import time

        self.start_timing()
        self.logger.info(f"Starting Async DDS optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)

        # Async DDS parameters
        dds_r = self.config.get('DDS_R', 0.2)
        pool_size = self.config.get('ASYNC_DDS_POOL_SIZE', min(20, self.num_processes * 2))
        batch_size = self.config.get('ASYNC_DDS_BATCH_SIZE', self.num_processes)
        max_stagnation = self.config.get('MAX_STAGNATION_BATCHES', 10)

        # Calculate target evaluations
        total_target_evaluations = self.max_iterations * self.num_processes
        target_batches = total_target_evaluations // batch_size

        # Solution pool tracking
        solution_pool = []  # List of (solution, score, batch_num) tuples
        pool_scores = []
        total_evaluations = 0
        stagnation_counter = 0
        last_improvement_batch = 0
        best_score = float('-inf')  # Start with -inf so any valid score is considered better
        best_solution = None

        self.logger.info(f"Async DDS configuration:")
        self.logger.info(f"  Pool size: {pool_size}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Target batches: {target_batches}")
        self.logger.info(f"  MPI processes: {self.num_processes}")

        # Initialize solution pool
        self.logger.info(f"Evaluating initial pool ({pool_size} solutions)...")
        initial_population = np.random.uniform(0, 1, (pool_size, n_params))
        initial_fitness = self._evaluate_population(initial_population, iteration=0)

        # Log initial pool scores for debugging
        valid_initial = [s for s in initial_fitness if s is not None and s != self.DEFAULT_PENALTY_SCORE]
        if valid_initial:
            self.logger.info(f"Initial pool scores: min={min(valid_initial):.4f}, max={max(valid_initial):.4f}, all={[f'{s:.4f}' for s in valid_initial]}")
        else:
            self.logger.warning("Initial pool: No valid scores!")

        for i, (solution, score) in enumerate(zip(initial_population, initial_fitness)):
            if score is not None and score != self.DEFAULT_PENALTY_SCORE:
                solution_pool.append((solution.copy(), score, 0))
                pool_scores.append(score)
                total_evaluations += 1

                if score > best_score:
                    best_score = score
                    best_solution = solution.copy()

        # Sort pool by score (best first)
        if solution_pool:
            combined = list(zip(solution_pool, pool_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            solution_pool = [item[0] for item in combined]
            pool_scores = [item[1] for item in combined]

        if not solution_pool:
            self.logger.error("No valid solutions in initial pool - all evaluations failed")
            return self.save_results('AsyncDDS', standard_filename=True)

        # Record initial best
        self.record_iteration(0, best_score, self.param_manager.denormalize_parameters(best_solution))
        self.update_best(best_score, self.param_manager.denormalize_parameters(best_solution), 0)
        self.logger.info(f"Initial pool complete | Best score: {best_score:.4f}")

        # Main batch loop
        for batch_num in range(1, target_batches + 1):
            # Check convergence
            if total_evaluations >= total_target_evaluations:
                self.logger.info(f"Reached target evaluations: {total_evaluations}")
                break
            if stagnation_counter >= max_stagnation:
                self.logger.info(f"Stopping due to stagnation ({stagnation_counter} batches)")
                break

            # Generate batch of trials from pool
            trials = []
            for i in range(batch_size):
                # Tournament selection from pool
                tournament_size = min(3, len(solution_pool))
                candidates = random.sample(range(len(solution_pool)), tournament_size)
                parent_idx = min(candidates, key=lambda idx: -pool_scores[idx])  # Best wins
                parent = solution_pool[parent_idx][0].copy()

                # DDS perturbation
                prob_select = max(1.0 - np.log(total_evaluations + i + 1) / np.log(total_target_evaluations),
                                 1.0 / n_params)

                trial = parent.copy()
                perturb_mask = np.random.random(n_params) < prob_select
                if not perturb_mask.any():
                    perturb_mask[np.random.randint(n_params)] = True

                for j in range(n_params):
                    if perturb_mask[j]:
                        perturbation = np.random.normal(0, dds_r)
                        trial[j] = parent[j] + perturbation

                        # Reflect at bounds
                        if trial[j] < 0:
                            trial[j] = -trial[j]
                        elif trial[j] > 1:
                            trial[j] = 2.0 - trial[j]
                        trial[j] = np.clip(trial[j], 0, 1)

                trials.append(trial)

            # Evaluate batch
            trial_population = np.array(trials)
            trial_fitness = self._evaluate_population(trial_population, iteration=batch_num)

            # Debug: Log all returned scores to trace score tracking
            valid_scores = [s for s in trial_fitness if s is not None and s != self.DEFAULT_PENALTY_SCORE]
            if valid_scores:
                max_batch_score = max(valid_scores)
                min_batch_score = min(valid_scores)
                self.logger.info(f"Batch {batch_num} scores: min={min_batch_score:.4f}, max={max_batch_score:.4f}, current_best={best_score:.4f}")
                if max_batch_score > best_score:
                    self.logger.info(f"NEW BEST FOUND in batch {batch_num}: {max_batch_score:.4f} > {best_score:.4f}")
            else:
                self.logger.warning(f"Batch {batch_num}: No valid scores returned!")

            # Update pool with batch results
            improvements = 0
            for trial, score in zip(trials, trial_fitness):
                if score is None or score == self.DEFAULT_PENALTY_SCORE:
                    continue

                total_evaluations += 1

                # Check for improvement BEFORE adding to pool
                is_improvement = False
                if score > best_score:
                    best_score = score
                    best_solution = trial.copy()
                    last_improvement_batch = batch_num
                    stagnation_counter = 0
                    is_improvement = True
                elif len(solution_pool) < pool_size:
                    # Pool not full, this is an improvement
                    is_improvement = True
                elif pool_scores and score > min(pool_scores):
                    # Better than worst in pool, this is an improvement
                    is_improvement = True

                if is_improvement:
                    improvements += 1

                # Add to pool (after checking)
                solution_pool.append((trial.copy(), score, batch_num))
                pool_scores.append(score)

            # Trim pool to size
            if len(solution_pool) > pool_size:
                combined = list(zip(solution_pool, pool_scores))
                combined.sort(key=lambda x: x[1], reverse=True)
                solution_pool = [item[0] for item in combined[:pool_size]]
                pool_scores = [item[1] for item in combined[:pool_size]]

            if improvements == 0:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Record results
            params_dict = self.param_manager.denormalize_parameters(best_solution)
            self.record_iteration(batch_num, best_score, params_dict)
            self.update_best(best_score, params_dict, batch_num)

            # Log progress
            self.log_iteration_progress('AsyncDDS', batch_num, best_score, improvements, batch_size)

        # Save results
        results_path = self.save_results('AsyncDDS', standard_filename=True)
        self.save_best_params('AsyncDDS')
        self._visualize_progress('AsyncDDS')

        self.logger.info(f"AsyncDDS completed in {self.format_elapsed_time()}")
        self.logger.info(f"Total evaluations: {total_evaluations}")
        self.logger.info(f"Final pool size: {len(solution_pool)}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_solution)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'AsyncDDS')

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

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_x)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'ADAM')

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

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_x)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'LBFGS')

        return results_path

    def run_nsga2(self, num_objectives: int = 2) -> Path:
        """
        Run NSGA-II (Non-dominated Sorting Genetic Algorithm II) multi-objective optimization.

        Args:
            num_objectives: Number of objectives (default: 2 for NSE and KGE)

        Returns:
            Path to results file
        """
        self.start_timing()
        self.logger.info(f"Starting NSGA-II optimization for {self._get_model_name()}")

        n_params = len(self.param_manager.all_param_names)
        pop_size = self.population_size

        # NSGA-II parameters
        # Note: Using more exploratory defaults than traditional NSGA-II
        # to match the aggressive exploration of PSO/DDS for hydrological models
        crossover_rate = self.config.get('NSGA2_CROSSOVER_RATE', 0.9)
        mutation_rate = self.config.get('NSGA2_MUTATION_RATE', 0.5)  # Higher rate for better exploration
        eta_c = self.config.get('NSGA2_ETA_C', 15)  # Crossover distribution index
        eta_m = self.config.get('NSGA2_ETA_M', 10)  # Lower = larger perturbations for exploration

        # Initialize population
        self.logger.info(f"Initializing population ({pop_size} individuals)...")
        population = np.random.uniform(0, 1, (pop_size, n_params))
        objectives = np.full((pop_size, num_objectives), np.nan)

        # Evaluate initial population using parallel batch execution
        fitness = self._evaluate_population(population, iteration=0)
        # For NSGA-II, we need two objectives - for now use same fitness for both
        # This can be extended to calculate different metrics (NSE, KGE, etc.)
        objectives[:, 0] = fitness  # Primary objective
        objectives[:, 1] = fitness  # Secondary objective (can be customized)

        # Perform NSGA-II selection (ranking and crowding distance)
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        # Find representative solution (best on first objective)
        best_idx = np.argmax(objectives[:, 0])
        best_solution = population[best_idx].copy()
        best_fitness = objectives[best_idx, 0]

        # Record initial best
        self.record_iteration(0, best_fitness, self.param_manager.denormalize_parameters(best_solution))
        self.update_best(best_fitness, self.param_manager.denormalize_parameters(best_solution), 0)
        self.logger.info(f"Initial population complete | Best obj1: {best_fitness:.4f}")

        # Main NSGA-II loop
        for generation in range(1, self.max_iterations + 1):
            # Generate offspring through selection, crossover, and mutation
            offspring = np.zeros_like(population)
            for i in range(0, pop_size, 2):
                # Tournament selection
                p1_idx = self._tournament_selection_nsga2(ranks, crowding_distances, pop_size)
                p2_idx = self._tournament_selection_nsga2(ranks, crowding_distances, pop_size)
                p1, p2 = population[p1_idx], population[p2_idx]

                # Crossover
                if np.random.random() < crossover_rate:
                    c1, c2 = self._sbx_crossover(p1, p2, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                offspring[i] = self._polynomial_mutation(c1, eta_m, mutation_rate)
                if i + 1 < pop_size:
                    offspring[i + 1] = self._polynomial_mutation(c2, eta_m, mutation_rate)

            # Evaluate offspring using parallel batch execution
            offspring_fitness = self._evaluate_population(offspring, iteration=generation)
            # For NSGA-II, we need two objectives - for now use same fitness for both
            offspring_objectives = np.full((pop_size, num_objectives), np.nan)
            offspring_objectives[:, 0] = offspring_fitness  # Primary objective
            offspring_objectives[:, 1] = offspring_fitness  # Secondary objective

            # Combine parent and offspring populations
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_objectives])

            # Environmental selection (select best pop_size individuals)
            selected_indices = self._environmental_selection_nsga2(combined_obj, pop_size)
            population = combined_pop[selected_indices]
            objectives = combined_obj[selected_indices]

            # Update ranks and crowding distances
            ranks = self._fast_non_dominated_sort(objectives)
            crowding_distances = self._calculate_crowding_distance(objectives, ranks)

            # Update best solution (using first objective as primary)
            current_best_idx = np.argmax(objectives[:, 0])
            if objectives[current_best_idx, 0] > best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = objectives[current_best_idx, 0]

            # Record results
            params_dict = self.param_manager.denormalize_parameters(best_solution)
            self.record_iteration(generation, best_fitness, params_dict)
            self.update_best(best_fitness, params_dict, generation)

            # Log progress
            self.log_iteration_progress('NSGA-II', generation, best_fitness)

        # Save results
        results_path = self.save_results('NSGA-II', standard_filename=True)
        self.save_best_params('NSGA-II')
        self._visualize_progress('NSGA-II')

        self.logger.info(f"NSGA-II completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        best_params_dict = self.param_manager.denormalize_parameters(best_solution)
        final_result = self.run_final_evaluation(best_params_dict)

        if final_result:
            # Save final evaluation results
            self._save_final_evaluation_results(final_result, 'NSGA-II')

        return results_path

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> np.ndarray:
        """Fast non-dominated sorting for NSGA-II."""
        pop_size = len(objectives)
        ranks = np.zeros(pop_size, dtype=int)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]

        # Find domination relationships
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Assign ranks
        current_front = np.where(domination_count == 0)[0]
        rank = 0
        while len(current_front) > 0:
            ranks[current_front] = rank
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = np.array(next_front)
            rank += 1

        return ranks

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (maximization)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _calculate_crowding_distance(self, objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for each solution."""
        pop_size = len(objectives)
        num_objectives = objectives.shape[1]
        crowding_distance = np.zeros(pop_size)

        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(rank_indices) <= 2:
                crowding_distance[rank_indices] = np.inf
                continue

            for obj_idx in range(num_objectives):
                obj_values = objectives[rank_indices, obj_idx]
                sorted_indices = np.argsort(obj_values)
                sorted_rank_indices = rank_indices[sorted_indices]

                # Boundary solutions get infinite distance
                crowding_distance[sorted_rank_indices[0]] = np.inf
                crowding_distance[sorted_rank_indices[-1]] = np.inf

                # Calculate crowding distance for middle solutions
                obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        crowding_distance[sorted_rank_indices[i]] += (
                            (obj_values[sorted_indices[i + 1]] - obj_values[sorted_indices[i - 1]]) / obj_range
                        )

        return crowding_distance

    def _tournament_selection_nsga2(
        self,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        pop_size: int
    ) -> int:
        """Tournament selection for NSGA-II."""
        candidates = np.random.choice(pop_size, 2, replace=False)
        best_idx = candidates[0]

        for candidate in candidates[1:]:
            if (ranks[candidate] < ranks[best_idx] or
                (ranks[candidate] == ranks[best_idx] and
                 crowding_distances[candidate] > crowding_distances[best_idx])):
                best_idx = candidate

        return best_idx

    def _environmental_selection_nsga2(self, objectives: np.ndarray, target_size: int) -> np.ndarray:
        """Select best individuals for next generation."""
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        selected_indices = []
        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(selected_indices) + len(rank_indices) <= target_size:
                selected_indices.extend(rank_indices)
            else:
                # Sort by crowding distance and select best
                remaining = target_size - len(selected_indices)
                rank_crowding = crowding_distances[rank_indices]
                sorted_indices = np.argsort(rank_crowding)[::-1]
                selected_indices.extend(rank_indices[sorted_indices[:remaining]])
                break

        return np.array(selected_indices)

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        eta_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        c1, c2 = p1.copy(), p2.copy()
        n_params = len(p1)

        for i in range(n_params):
            if np.random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-9:
                if p1[i] < p2[i]:
                    y1, y2 = p1[i], p2[i]
                else:
                    y1, y2 = p2[i], p1[i]

                rand = np.random.random()
                beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta_c + 1.0)

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                c1[i] = np.clip(c1[i], 0, 1)
                c2[i] = np.clip(c2[i], 0, 1)

        return c1, c2

    def _polynomial_mutation(
        self,
        solution: np.ndarray,
        eta_m: float,
        mutation_rate: float
    ) -> np.ndarray:
        """Polynomial mutation."""
        mutated = solution.copy()
        n_params = len(solution)

        for i in range(n_params):
            if np.random.random() < mutation_rate:
                y = mutated[i]
                delta1 = y - 0.0
                delta2 = 1.0 - y

                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutated[i] = y + deltaq
                mutated[i] = np.clip(mutated[i], 0, 1)

        return mutated

    # =========================================================================
    # Final Evaluation
    # =========================================================================

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Run final evaluation with best parameters over full period.

        This evaluates the calibrated model on both calibration and evaluation periods,
        providing comprehensive performance metrics.

        Args:
            best_params: Best parameters from optimization

        Returns:
            Dictionary with final metrics for both periods, or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Update file manager for full period
            self._update_file_manager_for_final_run()

            # Update model decisions for more accurate solver if configured
            # NOTE: Temporarily disabled as it may be overwriting optimization settings
            # if self.config.get('FINAL_EVALUATION_NUMERICAL_METHOD', 'ida') == 'ida':
            #     self._update_model_decisions_for_final_run()

            # Apply best parameters directly
            if not self._apply_best_parameters_for_final(best_params):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            # Setup output directory
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Update file manager output path
            self._update_file_manager_output_path(final_output_dir)

            # Run model directly using specific hook
            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error(f"{self._get_model_name()} run failed during final evaluation")
                return None

            # Calculate metrics for both periods (calibration_only=False)
            metrics = self.calibration_target.calculate_metrics(
                final_output_dir,
                calibration_only=False
            )

            if not metrics:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            # Extract period-specific metrics
            calib_metrics = self._extract_period_metrics(metrics, 'Calib')
            eval_metrics = self._extract_period_metrics(metrics, 'Eval')

            # Log detailed results
            self._log_final_evaluation_results(calib_metrics, eval_metrics)

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            return final_result

        except Exception as e:
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Restore optimization settings
            self._restore_model_decisions_for_optimization()
            self._restore_file_manager_for_optimization()

    def _extract_period_metrics(self, all_metrics: Dict, period_prefix: str) -> Dict:
        """
        Extract metrics for a specific period (Calib or Eval).

        Args:
            all_metrics: All metrics dictionary
            period_prefix: Period prefix ('Calib' or 'Eval')

        Returns:
            Dictionary of period-specific metrics
        """
        period_metrics = {}
        for key, value in all_metrics.items():
            if key.startswith(f"{period_prefix}_"):
                # Remove prefix for cleaner reporting
                period_metrics[key.replace(f"{period_prefix}_", "")] = value
            elif period_prefix == 'Calib' and not any(key.startswith(p) for p in ['Calib_', 'Eval_']):
                # Include unprefixed metrics in calibration (backwards compatibility)
                period_metrics[key] = value
        return period_metrics

    def _log_final_evaluation_results(
        self,
        calib_metrics: Dict,
        eval_metrics: Dict
    ) -> None:
        """
        Log detailed final evaluation results.

        Args:
            calib_metrics: Calibration period metrics
            eval_metrics: Evaluation period metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("FINAL EVALUATION RESULTS")
        self.logger.info("=" * 60)

        # Calibration period
        if calib_metrics:
            self.logger.info("📊 CALIBRATION PERIOD PERFORMANCE:")
            for metric, value in sorted(calib_metrics.items()):
                if value is not None and not np.isnan(value):
                    self.logger.info(f"   {metric}: {value:.6f}")

        # Evaluation period
        if eval_metrics:
            self.logger.info("📈 EVALUATION PERIOD PERFORMANCE:")
            for metric, value in sorted(eval_metrics.items()):
                if value is not None and not np.isnan(value):
                    self.logger.info(f"   {metric}: {value:.6f}")
        else:
            self.logger.info("📈 EVALUATION PERIOD: No evaluation period configured")

        self.logger.info("=" * 60)

    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager to use full experiment period (not just calibration)."""
        file_manager_path = self._get_final_file_manager_path()
        if not file_manager_path.exists():
            self.logger.warning(f"File manager not found: {file_manager_path}")
            return

        try:
            # Get full experiment period from config
            sim_start = self.config.get('EXPERIMENT_TIME_START')
            sim_end = self.config.get('EXPERIMENT_TIME_END')

            if not sim_start or not sim_end:
                self.logger.warning("Full experiment period not configured, using current settings")
                return

            # Adjust end time to align with forcing timestep
            sim_end = self._adjust_end_time_for_forcing(sim_end)

            with open(file_manager_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if 'simStartTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simStartTime         '{sim_start}'\n")
                elif 'simEndTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simEndTime           '{sim_end}'\n")
                else:
                    updated_lines.append(line)

            with open(file_manager_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated file manager for full period: {sim_start} to {sim_end}")

        except Exception as e:
            self.logger.error(f"Failed to update file manager for final run: {e}")

    def _update_model_decisions_for_final_run(self) -> None:
        """Update modelDecisions.txt to use more accurate solver for final evaluation."""
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'
        if not model_decisions_path.exists():
            return

        try:
            with open(model_decisions_path, 'r') as f:
                lines = f.readlines()

            # Backup original if not already done
            backup_path = model_decisions_path.parent / 'modelDecisions_optimization_backup.txt'
            if not backup_path.exists():
                with open(backup_path, 'w') as f:
                    f.writelines(lines)

            updated_lines = []
            for line in lines:
                if 'soilCatTbl' in line and not line.strip().startswith('!'):
                    updated_lines.append("soilCatTbl               ROSETTA              ! soil-category dateset\n")
                elif 'vegeParTbl' in line and not line.strip().startswith('!'):
                    updated_lines.append("vegeParTbl               USGS                 ! vegetation category dataset\n")
                elif 'soilStress' in line and not line.strip().startswith('!'):
                    updated_lines.append("soilStress               NoahType             ! choice of function for the soil moisture control on stomatal resistance\n")
                elif 'stomResist' in line and not line.strip().startswith('!'):
                    updated_lines.append("stomResist               BallBerry            ! choice of function for stomatal resistance\n")
                elif 'num_method' in line and not line.strip().startswith('!'):
                    updated_lines.append("num_method               itertive             ! choice of numerical method\n")
                elif 'fDerivMeth' in line and not line.strip().startswith('!'):
                    updated_lines.append("fDerivMeth               analytic             ! choice of method to calculate flux derivatives\n")
                elif 'LAI_method' in line and not line.strip().startswith('!'):
                    updated_lines.append("LAI_method               monTable             ! choice of method to determine LAI and SAI\n")
                elif 'cIntercept' in line and not line.strip().startswith('!'):
                    updated_lines.append("cIntercept               sparseCanopy         ! choice of parameterization for canopy interception\n")
                elif 'f_Richards' in line and not line.strip().startswith('!'):
                    updated_lines.append("f_Richards               mixdform             ! choice of form of Richards' equation\n")
                elif 'groundwatr' in line and not line.strip().startswith('!'):
                    updated_lines.append("groundwatr               qTopmodl             ! choice of groundwater parameterization\n")
                elif 'hc_profile' in line and not line.strip().startswith('!'):
                    updated_lines.append("hc_profile               pow_prof             ! choice of hydraulic conductivity profile\n")
                elif 'bcUpprTdyn' in line and not line.strip().startswith('!'):
                    updated_lines.append("bcUpprTdyn               nrg flux             ! type of upper boundary condition for thermodynamics\n")
                elif 'bcLowrTdyn' in line and not line.strip().startswith('!'):
                    updated_lines.append("bcLowrTdyn               zeroFlux             ! type of lower boundary condition for thermodynamics\n")
                elif 'bcUpprSoiH' in line and not line.strip().startswith('!'):
                    updated_lines.append("bcUpprSoiH               liq_flux             ! type of upper boundary condition for soil hydrology\n")
                elif 'bcLowrSoiH' in line and not line.strip().startswith('!'):
                    updated_lines.append("bcLowrSoiH               drainage             ! type of lower boundary condition for soil hydrology\n")
                elif 'veg_traits' in line and not line.strip().startswith('!'):
                    updated_lines.append("veg_traits               CM_QJRMS1988         ! choice of parameterization for vegetation roughness length and displacement height\n")
                elif 'canopyEmis' in line and not line.strip().startswith('!'):
                    updated_lines.append("canopyEmis               difTrans             ! choice of parameterization for canopy emissivity\n")
                elif 'snowIncept' in line and not line.strip().startswith('!'):
                    updated_lines.append("snowIncept               lightSnow            ! choice of parameterization for snow interception\n")
                elif 'windPrfile' in line and not line.strip().startswith('!'):
                    updated_lines.append("windPrfile               logBelowCanopy       ! choice of wind profile through the canopy\n")
                elif 'astability' in line and not line.strip().startswith('!'):
                    updated_lines.append("astability               louisinv             ! choice of stability function\n")
                elif 'canopySrad' in line and not line.strip().startswith('!'):
                    updated_lines.append("canopySrad               CLM_2stream          ! choice of method for canopy shortwave radiation\n")
                elif 'alb_method' in line and not line.strip().startswith('!'):
                    updated_lines.append("alb_method               varDecay             ! choice of albedo representation\n")
                elif 'compaction' in line and not line.strip().startswith('!'):
                    updated_lines.append("compaction               anderson             ! choice of compaction routine\n")
                elif 'snowLayers' in line and not line.strip().startswith('!'):
                    updated_lines.append("snowLayers               CLM_2010             ! choice of method to combine and sub-divide snow layers\n")
                elif 'thCondSnow' in line and not line.strip().startswith('!'):
                    updated_lines.append("thCondSnow               jrdn1991             ! choice of thermal conductivity representation for snow\n")
                elif 'thCondSoil' in line and not line.strip().startswith('!'):
                    updated_lines.append("thCondSoil               funcSoilWet          ! choice of thermal conductivity representation for soil\n")
                elif 'spatial_gw' in line and not line.strip().startswith('!'):
                    updated_lines.append("spatial_gw               localColumn          ! choice of spatial representation of groundwater\n")
                elif 'subRouting' in line and not line.strip().startswith('!'):
                    updated_lines.append("subRouting               timeDlay             ! choice of method for sub-grid routing\n")
                else:
                    updated_lines.append(line)

            with open(model_decisions_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug("Updated model decisions for final evaluation (ida solver)")

        except Exception as e:
            self.logger.error(f"Error updating model decisions for final run: {e}")

    def _restore_model_decisions_for_optimization(self) -> None:
        """Restore model decisions to optimization settings."""
        backup_path = self.optimization_settings_dir / 'modelDecisions_optimization_backup.txt'
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'

        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    lines = f.readlines()
                with open(model_decisions_path, 'w') as f:
                    f.writelines(lines)
                self.logger.debug("Restored model decisions to optimization settings")
            except Exception as e:
                self.logger.error(f"Error restoring model decisions: {e}")

    def _restore_file_manager_for_optimization(self) -> None:
        """Restore file manager to calibration period settings."""
        file_manager_path = self._get_final_file_manager_path()
        if not file_manager_path.exists():
            return

        try:
            calib_start = self.config.get('CALIBRATION_START_DATE')
            calib_end = self.config.get('CALIBRATION_END_DATE')

            if not calib_start or not calib_end:
                return

            with open(file_manager_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if 'simStartTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simStartTime         '{calib_start}'\n")
                elif 'simEndTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simEndTime           '{calib_end}'\n")
                else:
                    updated_lines.append(line)

            with open(file_manager_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Restored file manager to calibration period")

        except Exception as e:
            self.logger.error(f"Failed to restore file manager: {e}")

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best parameters for final evaluation."""
        try:
            return self.worker.apply_parameters(
                best_params,
                self.optimization_settings_dir,
                config=self.config
            )
        except Exception as e:
            self.logger.error(f"Error applying parameters for final evaluation: {e}")
            return False

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Update file manager with final evaluation output path."""
        file_manager_path = self._get_final_file_manager_path()
        if not file_manager_path.exists():
            return

        try:
            with open(file_manager_path, 'r') as f:
                lines = f.readlines()

            # Ensure path ends with slash
            output_path_str = str(output_dir)
            if not output_path_str.endswith('/'):
                output_path_str += '/'

            updated_lines = []
            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"outputPath '{output_path_str}' \n")
                else:
                    updated_lines.append(line)

            with open(file_manager_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated output path to: {output_path_str}")

        except Exception as e:
            self.logger.error(f"Failed to update output path: {e}")

    def _run_summa_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run SUMMA for final evaluation."""
        try:
            import subprocess

            # Get SUMMA executable path
            summa_install_path = self.config.get('SUMMA_INSTALL_PATH', 'default')
            summa_exe_name = self.config.get('SUMMA_EXE', 'summa_sundials.exe')

            if summa_install_path == 'default':
                data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', '.'))
                summa_exe = data_dir / 'installs' / 'summa' / 'bin' / summa_exe_name
            else:
                # If install path is provided, combine it with executable name
                summa_exe = Path(summa_install_path) / summa_exe_name

            summa_fm_name = self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
            file_manager = self.optimization_settings_dir / summa_fm_name

            if not summa_exe.exists():
                self.logger.error(f"SUMMA executable not found: {summa_exe}")
                return False

            if not file_manager.exists():
                self.logger.error(f"File manager not found: {file_manager}")
                return False

            # Run SUMMA with proper environment variables
            cmd = [str(summa_exe), '-m', str(file_manager)]
            self.logger.info(f"Running SUMMA: {' '.join(cmd)}")

            # Set environment for single-threaded execution and disable file locking
            import os
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'VECLIB_MAXIMUM_THREADS': '1',
                'NETCDF_DISABLE_LOCKING': '1'
            })

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(output_dir),
                env=env,
                timeout=self.config.get('SUMMA_TIMEOUT', 3600)  # 1 hour default for final run
            )

            if result.returncode != 0:
                self.logger.error(f"SUMMA failed with exit code {result.returncode}")
                self.logger.error(f"STDOUT (full): {result.stdout}")
                self.logger.error(f"STDERR (full): {result.stderr}")

                # Also save to file for detailed debugging
                error_log = output_dir / 'summa_error.log'
                with open(error_log, 'w') as f:
                    f.write("SUMMA EXECUTION ERROR\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Exit code: {result.returncode}\n\n")
                    f.write("STDOUT:\n" + "=" * 80 + "\n")
                    f.write(result.stdout + "\n\n")
                    f.write("STDERR:\n" + "=" * 80 + "\n")
                    f.write(result.stderr + "\n")
                self.logger.error(f"Full error saved to {error_log}")
                return False

            self.logger.info("SUMMA completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("SUMMA timeout during final evaluation")
            return False
        except Exception as e:
            self.logger.error(f"Error running SUMMA for final evaluation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _save_final_evaluation_results(
        self,
        final_result: Dict[str, Any],
        algorithm: str
    ) -> None:
        """
        Save final evaluation results to JSON file.

        Args:
            final_result: Final evaluation results dictionary
            algorithm: Algorithm name (e.g., 'PSO', 'DDS')
        """
        try:
            import json

            output_file = self.results_dir / f'{self.experiment_id}_{algorithm.lower()}_final_evaluation.json'

            def _convert_to_serializable(obj):
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: _convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_convert_to_serializable(i) for i in obj]
                return obj

            # Create serializable result
            serializable_result = {
                'algorithm': algorithm,
                'experiment_id': self.experiment_id,
                'domain_name': self.domain_name,
                'calibration_metrics': _convert_to_serializable(final_result.get('calibration_metrics', {})),
                'evaluation_metrics': _convert_to_serializable(final_result.get('evaluation_metrics', {})),
                'best_params': _convert_to_serializable(final_result.get('best_params', {})),
                'timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)

            self.logger.info(f"Saved final evaluation results to {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save final evaluation results: {e}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Cleanup parallel processing directories and temporary files."""
        if self.parallel_dirs:
            self.cleanup_parallel_processing(self.parallel_dirs)
