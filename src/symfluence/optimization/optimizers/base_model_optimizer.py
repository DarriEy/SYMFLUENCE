"""
Base Model Optimizer

Abstract base class for model-specific optimizers (FUSE, NGEN, SUMMA).
Uses mixins for shared functionality and provides template methods for
algorithm implementations.

Delegates specialized operations to:
- TaskBuilder: Task dict construction for worker execution
- PopulationEvaluator: Batch evaluation of parameter populations
- NSGA2Operators: Multi-objective optimization operators
- FinalEvaluationRunner: Post-optimization evaluation
"""

import logging
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, Union, TYPE_CHECKING
from datetime import datetime

from symfluence.core import ConfigurableMixin
from ..mixins import (
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
    GradientOptimizationMixin
)
from ..workers.base_worker import BaseWorker, WorkerTask, WorkerResult
from .algorithms import get_algorithm, list_algorithms
from .evaluators import TaskBuilder, PopulationEvaluator
from .multiobjective import NSGA2Operators
from .final_evaluation import FinalEvaluationRunner, FinalResultsSaver

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelOptimizer(
    ConfigurableMixin,
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
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the model optimizer.

        Args:
            config: Configuration (typed SymfluenceConfig or legacy dict)
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        from symfluence.core.config.models import SymfluenceConfig

        # Auto-convert dict to typed config for backward compatibility
        if isinstance(config, dict):
            self._config = SymfluenceConfig(**config)
        else:
            self._config = config

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Setup paths using typed config accessors
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir, default='.'
        ))
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='default'
        )
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        # Note: experiment_id is provided by ConfigMixin property

        # Optimization settings directory
        if optimization_settings_dir is not None:
            self.optimization_settings_dir = Path(optimization_settings_dir)
        else:
            model_name = self._get_model_name()
            self.optimization_settings_dir = (
                self.project_dir / 'settings' / model_name
            )

        # Results directory
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm, default='optimization'
        ).lower()
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

        # Algorithm parameters (using typed config)
        self.max_iterations = self._get_config_value(
            lambda: self.config.optimization.iterations, default=self.DEFAULT_ITERATIONS
        )
        self.population_size = self._get_config_value(
            lambda: self.config.optimization.population_size, default=self.DEFAULT_POPULATION_SIZE
        )
        self.target_metric = self._get_config_value(
            lambda: self.config.optimization.metric, default='KGE'
        )

        # Random seed
        self.random_seed = self._get_config_value(lambda: self.config.system.random_seed)
        if self.random_seed is not None and self.random_seed != 'None':
            self._set_random_seeds(int(self.random_seed))

        # Parallel processing state
        self.parallel_dirs = {}
        self.default_sim_dir = self.results_dir  # Initialize with results_dir as fallback
        # Setup directories if MPI_PROCESSES is set, regardless of count (for isolation)
        mpi_processes = self._get_config_value(lambda: self.config.system.mpi_processes, default=1)
        if mpi_processes >= 1:
            self._setup_parallel_dirs()

        # Runtime config overrides (for algorithm-specific settings like Adam/LBFGS)
        self._runtime_overrides: Dict[str, Any] = {}

        # Lazy-initialized components
        self._task_builder = None
        self._population_evaluator = None
        self._final_evaluation_runner = None
        self._results_saver = None

    # =========================================================================
    # Lazy-initialized component properties
    # =========================================================================

    @property
    def task_builder(self) -> TaskBuilder:
        """Lazy-initialized task builder."""
        if self._task_builder is None:
            self._task_builder = TaskBuilder(
                config=self.config,
                project_dir=self.project_dir,
                domain_name=self.domain_name,
                optimization_settings_dir=self.optimization_settings_dir,
                default_sim_dir=self.default_sim_dir,
                parallel_dirs=self.parallel_dirs,
                num_processes=self.num_processes,
                target_metric=self.target_metric,
                param_manager=self.param_manager,
                logger=self.logger
            )
            if hasattr(self, 'summa_exe_path'):
                self._task_builder.set_summa_exe_path(self.summa_exe_path)
        return self._task_builder

    @property
    def population_evaluator(self) -> PopulationEvaluator:
        """Lazy-initialized population evaluator."""
        if self._population_evaluator is None:
            self._population_evaluator = PopulationEvaluator(
                task_builder=self.task_builder,
                worker=self.worker,
                execute_batch=self.execute_batch,
                use_parallel=self.use_parallel,
                num_processes=self.num_processes,
                model_name=self._get_model_name(),
                logger=self.logger
            )
        return self._population_evaluator

    @property
    def results_saver(self) -> FinalResultsSaver:
        """Lazy-initialized results saver."""
        if self._results_saver is None:
            self._results_saver = FinalResultsSaver(
                results_dir=self.results_dir,
                experiment_id=self.experiment_id,
                domain_name=self.domain_name,
                logger=self.logger
            )
        return self._results_saver

    def _visualize_progress(self, algorithm: str) -> None:
        """Helper to visualize optimization progress if reporting manager available."""
        if self.reporting_manager:
            calibration_variable = self._get_config_value(
                lambda: self.config.optimization.calibration_variable, default='streamflow'
            )
            self.reporting_manager.visualize_optimization_progress(
                self._iteration_history,
                self.results_dir.parent / f"{algorithm.lower()}_{self.experiment_id}", # Matches results_dir logic or pass results_dir
                calibration_variable,
                self.target_metric
            )

            calibrate_depth = self._get_config_value(
                lambda: self.config.model.summa.calibrate_depth, default=False
            )
            if calibrate_depth:
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

    def _get_nsga2_objective_names(self) -> List[str]:
        """Resolve NSGA-II objective metric names in priority order."""
        primary_metric = self._get_config_value(
            lambda: self.config.optimization.nsga2.primary_metric, default=self.target_metric
        )
        secondary_metric = self._get_config_value(
            lambda: self.config.optimization.nsga2.secondary_metric, default=self.target_metric
        )
        return [str(primary_metric).upper(), str(secondary_metric).upper()]

    def _log_calibration_alignment(self) -> None:
        """Log basic calibration alignment info before optimization starts."""
        try:
            if not hasattr(self.calibration_target, '_load_observed_data'):
                return

            obs = self.calibration_target._load_observed_data()
            if obs is None or obs.empty:
                self.logger.warning("Calibration check: no observed data found")
                return

            if not isinstance(obs.index, pd.DatetimeIndex):
                obs.index = pd.to_datetime(obs.index)

            calib_period = self.calibration_target._parse_date_range(
                self._get_config_value(lambda: self.config.domain.calibration_period, default='')
            )
            obs_period = obs.copy()
            if calib_period[0] and calib_period[1]:
                obs_period = obs_period[(obs_period.index >= calib_period[0]) & (obs_period.index <= calib_period[1])]

            eval_timestep = str(self._get_config_value(
                lambda: self.config.optimization.calibration_timestep, default='native'
            )).lower()
            if eval_timestep != 'native' and hasattr(self.calibration_target, '_resample_to_timestep'):
                obs_period = self.calibration_target._resample_to_timestep(obs_period, eval_timestep)

            sim_start = self._get_config_value(lambda: self.config.domain.time_start)
            sim_end = self._get_config_value(lambda: self.config.domain.time_end)
            overlap = obs_period
            if sim_start and sim_end:
                sim_start = pd.Timestamp(sim_start)
                sim_end = pd.Timestamp(sim_end)
                overlap = obs_period[(obs_period.index >= sim_start) & (obs_period.index <= sim_end)]

            self.logger.info(
                "Calibration data check | timestep=%s | obs=%d | calib_window=%d | overlap_with_sim=%d",
                eval_timestep,
                len(obs),
                len(obs_period),
                len(overlap)
            )
        except Exception as e:
            self.logger.debug(f"Calibration alignment check failed: {e}")

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
            forcing_timestep_seconds = self._get_config_value(
                lambda: self.config.forcing.time_step_size, default=3600
            )

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
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm, default='optimization'
        ).lower()

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

    # =========================================================================
    # Evaluation methods
    # =========================================================================

    def log_iteration_progress(
        self,
        algorithm_name: str,
        iteration: int,
        best_score: float,
        secondary_score: Optional[float] = None,
        secondary_label: Optional[str] = None,
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

        if secondary_score is not None:
            label = secondary_label or "Secondary"
            msg_parts.append(f"{label}: {secondary_score:.4f}")

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
        return self.population_evaluator.evaluate_solution(normalized_params, proc_id)

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
        base_seed = self.random_seed if hasattr(self, 'random_seed') else None
        return self.population_evaluator.evaluate_population(
            population, iteration, base_random_seed=base_seed
        )

    def _evaluate_population_objectives(
        self,
        population: np.ndarray,
        objective_names: List[str],
        iteration: int = 0
    ) -> np.ndarray:
        """
        Evaluate a population for multiple objectives using worker metrics.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            objective_names: Ordered list of objective metric names (e.g., ['KGE', 'NSE'])
            iteration: Current iteration number

        Returns:
            Array of objective values (n_individuals x n_objectives)
        """
        base_seed = self.random_seed if hasattr(self, 'random_seed') else None
        return self.population_evaluator.evaluate_population_objectives(
            population, objective_names, iteration, base_random_seed=base_seed
        )

    # =========================================================================
    # Algorithm implementations
    # =========================================================================

    def _run_default_only(self, algorithm_name: str) -> Path:
        """
        Run a single default evaluation when no parameters are configured.
        """
        self.start_timing()
        self.logger.info(
            f"No parameters configured for {self._get_model_name()} - running default evaluation only"
        )

        score = self.DEFAULT_PENALTY_SCORE
        final_result = self.run_final_evaluation({})
        if final_result and isinstance(final_result, dict):
            metrics = final_result.get('final_metrics', {})
            score = metrics.get(self.target_metric, self.DEFAULT_PENALTY_SCORE)

        self.record_iteration(0, score, {})
        self.update_best(score, {}, 0)
        self.save_best_params(algorithm_name)
        return self.save_results(algorithm_name, standard_filename=True)

    def run_optimization(self, algorithm_name: str) -> Path:
        """
        Run optimization using a specified algorithm from the registry.

        This is the unified entry point for all optimization algorithms.
        Individual methods like run_dds(), run_pso() delegate to this method.

        Args:
            algorithm_name: Name of the algorithm ('dds', 'pso', 'de', 'sce-ua',
                          'async_dds', 'nsga2')

        Returns:
            Path to results file

        Example:
            >>> optimizer.run_optimization('dds')
            >>> optimizer.run_optimization('nsga2')
        """
        self.start_timing()
        self.logger.info(f"Starting {algorithm_name.upper()} optimization for {self._get_model_name()}")
        self._log_calibration_alignment()

        n_params = len(self.param_manager.all_param_names)
        if n_params == 0:
            return self._run_default_only(algorithm_name)

        # Get algorithm instance from registry
        algorithm = get_algorithm(algorithm_name, self.config, self.logger)

        # Prepare callbacks for the algorithm
        def evaluate_solution(normalized_params, proc_id=0):
            return self._evaluate_solution(normalized_params, proc_id)

        def evaluate_population(population, iteration=0):
            return self._evaluate_population(population, iteration)

        def denormalize_params(normalized):
            return self.param_manager.denormalize_parameters(normalized)

        def record_iteration(iteration, score, params, additional_metrics=None):
            self.record_iteration(iteration, score, params, additional_metrics=additional_metrics)

        def update_best(score, params, iteration):
            self.update_best(score, params, iteration)

        def log_progress(alg_name, iteration, best_score, n_improved=None, pop_size=None, secondary_score=None, secondary_label=None):
            self.log_iteration_progress(
                alg_name, iteration, best_score,
                secondary_score=secondary_score, secondary_label=secondary_label,
                n_improved=n_improved, population_size=pop_size
            )

        # Additional callbacks for specific algorithms
        kwargs = {
            'log_initial_population': self.log_initial_population,
            'num_processes': self.num_processes if hasattr(self, 'num_processes') else 1,
        }

        # For NSGA-II, add multi-objective support
        if algorithm_name.lower() in ['nsga2', 'nsga-ii']:
            kwargs['evaluate_population_objectives'] = self._evaluate_population_objectives
            kwargs['objective_names'] = self._get_nsga2_objective_names()
            kwargs['multiobjective'] = bool(self._get_config_value(
                lambda: self.config.optimization.nsga2.multi_target, default=False
            ))

        # Run the algorithm
        result = algorithm.optimize(
            n_params=n_params,
            evaluate_solution=evaluate_solution,
            evaluate_population=evaluate_population,
            denormalize_params=denormalize_params,
            record_iteration=record_iteration,
            update_best=update_best,
            log_progress=log_progress,
            **kwargs
        )

        # Save results
        results_path = self.save_results(algorithm.name, standard_filename=True)
        self.save_best_params(algorithm.name)
        self._visualize_progress(algorithm.name)

        self.logger.info(f"{algorithm.name} completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        if result.get('best_params'):
            final_result = self.run_final_evaluation(result['best_params'])
            if final_result:
                self._save_final_evaluation_results(final_result, algorithm.name)

        return results_path

    # =========================================================================
    # Algorithm convenience methods - delegate to run_optimization()
    # =========================================================================

    def run_dds(self) -> Path:
        """Run Dynamically Dimensioned Search (DDS) optimization."""
        return self.run_optimization('dds')

    def run_pso(self) -> Path:
        """Run Particle Swarm Optimization (PSO)."""
        return self.run_optimization('pso')

    def run_de(self) -> Path:
        """Run Differential Evolution (DE) optimization."""
        return self.run_optimization('de')

    def run_sce(self) -> Path:
        """Run Shuffled Complex Evolution (SCE-UA) optimization."""
        return self.run_optimization('sce-ua')

    def run_async_dds(self) -> Path:
        """Run Asynchronous Parallel DDS optimization."""
        return self.run_optimization('async_dds')

    def run_nsga2(self) -> Path:
        """Run NSGA-II multi-objective optimization."""
        return self.run_optimization('nsga2')

    def run_adam(self, steps: int = 100, lr: float = 0.01) -> Path:
        """
        Run Adam gradient-based optimization.

        Args:
            steps: Number of optimization steps (passed via config ADAM_STEPS)
            lr: Learning rate (passed via config ADAM_LR)

        Returns:
            Path to results file
        """
        # Store parameters in runtime overrides for the algorithm to use
        self._runtime_overrides['ADAM_STEPS'] = steps
        self._runtime_overrides['ADAM_LR'] = lr
        return self.run_optimization('adam')

    def run_lbfgs(self, steps: int = 50, lr: float = 0.1) -> Path:
        """
        Run L-BFGS gradient-based optimization.

        Args:
            steps: Maximum number of steps (passed via config LBFGS_STEPS)
            lr: Initial step size (passed via config LBFGS_LR)

        Returns:
            Path to results file
        """
        # Store parameters in runtime overrides for the algorithm to use
        self._runtime_overrides['LBFGS_STEPS'] = steps
        self._runtime_overrides['LBFGS_LR'] = lr
        return self.run_optimization('lbfgs')

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
        return FinalResultsSaver.extract_period_metrics(all_metrics, period_prefix)

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
        self.results_saver.log_results(calib_metrics, eval_metrics)

    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager to use full experiment period (not just calibration)."""
        file_manager_path = self._get_final_file_manager_path()
        if not file_manager_path.exists():
            self.logger.warning(f"File manager not found: {file_manager_path}")
            return

        try:
            # Get full experiment period from config
            sim_start = self._get_config_value(lambda: self.config.domain.time_start)
            sim_end = self._get_config_value(lambda: self.config.domain.time_end)

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
            calib_start = self._get_config_value(lambda: self.config.domain.calibration_start_date)
            calib_end = self._get_config_value(lambda: self.config.domain.calibration_end_date)

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
        self.results_saver.save_results(final_result, algorithm)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Cleanup parallel processing directories and temporary files."""
        if self.parallel_dirs:
            self.cleanup_parallel_processing(self.parallel_dirs)
