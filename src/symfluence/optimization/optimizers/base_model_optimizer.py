"""Base Model Optimizer

Abstract base class providing unified optimization infrastructure for all hydrological
models (SUMMA, FUSE, NGEN, GR4J, etc.). Implements template method pattern to delegate
model-specific operations while centralizing algorithm execution, parallel processing,
results tracking, and final evaluation workflows.

Architecture:
    The BaseModelOptimizer uses multiple mixins to provide comprehensive optimization
    capabilities without tight coupling to specific algorithms:

    - ConfigurableMixin: Typed configuration object (SymfluenceConfig) with fallback
      to legacy dict-based configs for backward compatibility. Provides _get_config_value()
      utility for safe nested attribute access with defaults.

    - ParallelExecutionMixin: Parallel processing infrastructure including setup_parallel_processing()
      for MPI directory structures and execute_batch() for task distribution across processors.

    - ResultsTrackingMixin: Tracks optimization history across iterations, manages best solution,
      and provides iteration history recording via record_iteration() and update_best().

    - RetryExecutionMixin: Fault tolerance with configurable retry logic (e.g., if worker crashes).

    - GradientOptimizationMixin: Shared utilities for gradient-based optimizers (gradient
      caching, step size management).

Algorithm Execution Flow:
    1. User calls run_*() convenience method (run_dds(), run_pso(), etc.) or run_optimization()
    2. Optimizer validates parameter count and logs calibration alignment
    3. Algorithm retrieved from registry (algorithms module) with model config
    4. Callbacks bound to BaseModelOptimizer methods:
       - evaluate_solution(): Single parameter set evaluation via population_evaluator
       - evaluate_population(): Batch evaluation via TaskBuilder → worker
       - denormalize_params(): Transform [0,1] normalized → physical units
       - record_iteration(): Log iteration history
       - update_best(): Track best solution across generations
       - log_progress(): Consistent logging format across all algorithms
    5. Algorithm.optimize() runs until convergence or max_iterations
    6. Results saved via results_saver to JSON/CSV
    7. Final evaluation runs best params over full simulation period (if implemented)

Parameter Normalization:
    Optimization operates in normalized space [0, 1] to:
    - Provide common numerical scale across diverse parameter ranges
    - Allow identical algorithm implementations across models
    - Simplify bounds checking and mutation operators

    Workflow:
    - Physical → Normalized: normalize_parameters() clips to [0,1] after scaling
    - Normalized → Physical: denormalize_parameters() applies bounds and units
    - Example: Parameter "x" with bounds [100, 500]:
      * Physical x=300 → Normalized (300-100)/(500-100) = 0.5
      * Normalized 0.5 → Physical 0.5*(500-100)+100 = 300

Population Evaluation:
    TaskBuilder generates task dicts containing parameter sets and metadata:
    - individual_id: Index in population (maps to results array)
    - params: Denormalized parameters for worker
    - proc_id: Processor assignment for parallel execution
    - evaluation_id: Unique ID for logging/debugging

    PopulationEvaluator batches tasks and submits to:
    - Parallel execution: Tasks distributed via execute_batch() across MPI processes
    - Sequential execution: Tasks evaluated one-by-one (fallback for debugging)

Results Tracking:
    Each iteration records:
    - generation: Iteration number
    - algorithm: Algorithm name (e.g., 'DDS', 'PSO')
    - best_score: Best fitness found to date
    - best_params: Associated parameter values
    - mean_score: Population mean (if available)
    - additional_metrics: Algorithm-specific data (e.g., n_improved for GA)

Final Evaluation:
    After optimization completes, run_final_evaluation():
    1. Updates file manager to full experiment period (not just calibration)
    2. Applies best parameters to model configuration
    3. Runs model once over full period (calibration + evaluation windows)
    4. Calculates metrics for both periods separately
    5. Saves final results and generates progress visualizations
    6. Restores settings to optimization configuration for reproducibility

Supported Algorithms (via registry):
    - Single-objective: DDS, PSO, DE, SCE-UA, ASYNC_DDS, ADAM, LBFGS
    - Multi-objective: NSGA-II

    Algorithm selection by calling:
    run_dds(), run_pso(), run_de(), run_sce(), run_async_dds(), run_nsga2(),
    run_adam(), run_lbfgs()

    Or directly via: run_optimization('algorithm_name')

Subclass Implementation Requirements:
    REQUIRED (1 abstract method):
    1. _get_model_name() → str
       Returns model identifier ('SUMMA', 'FUSE', 'NGEN', 'GR', etc.)

    OPTIONAL (override for custom behavior):
    2. _create_parameter_manager() → ParameterManager
       Default: Registry-based discovery via _create_parameter_manager_default()
       Override when: Non-standard constructor signature (FUSE, SUMMA, GR)
       Handles: Parameter bounds, normalization, file writing

    3. _create_calibration_target() → CalibrationTarget
       Default: Uses create_calibration_target() factory with registry lookup
       Override when: Rarely needed (factory handles all complexity)
       Handles: Output extraction, observed data, metric calculation

    4. _create_worker() → BaseWorker
       Default: Registry-based discovery via _create_worker_default()
       Override when: Rarely needed (all workers use same signature)
       Handles: Model execution, parameter application, output parsing

    5. _run_model_for_final_evaluation(output_dir) → bool
       Custom final evaluation logic if standard workflow insufficient

    6. _get_final_file_manager_path() → Path
       Path to file manager for final evaluation time period updates

    7. _get_settings_directory() → Path
       Override default {project_dir}/settings/{MODEL}/ convention

Examples:
    >>> # Create subclass
    >>> class SUMMAOptimizer(BaseModelOptimizer):
    ...     def _get_model_name(self): return 'SUMMA'
    ...     def _create_parameter_manager(self): return SUMMAParameterManager(...)
    ...     def _create_calibration_target(self): return StreamflowEvaluator(...)
    ...     def _create_worker(self): return SUMMAWorker(...)

    >>> # Run optimization
    >>> optimizer = SUMMAOptimizer(config, logger)
    >>> results_path = optimizer.run_dds()
    >>> results_path = optimizer.run_pso()
    >>> results_path = optimizer.run_adam(steps=100, lr=0.01)

Configuration Parameters:
    Core Optimization:
        optimization.algorithm: Algorithm name (e.g., 'dds')
        optimization.iterations: Max generations/batches (default: 100)
        optimization.population_size: Population size for GA methods (default: 30)
        optimization.metric: Primary metric to maximize (e.g., 'KGE')
        optimization.calibration_timestep: Timestep for evaluation ('native', 'daily', etc.)

    Domain:
        domain.name: Domain identifier (e.g., 'Bow_at_Banff')
        domain.calibration_start_date: Calibration window start
        domain.calibration_end_date: Calibration window end
        domain.time_start: Full experiment start
        domain.time_end: Full experiment end
        domain.calibration_period: Alternative period specification

    System:
        system.data_dir: Root data directory
        system.random_seed: Random seed for reproducibility
        system.mpi_processes: Number of parallel processes

References:
    - Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search
      algorithm for computationally efficient watershed model calibration.
      Water Resources Research, 43, W01413.
    - Kingma, D. K., & Ba, J. (2015). Adam: A method for stochastic optimization.
      ICLR.
    - Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage.
      Mathematics of Computation, 35(151), 773-782.
"""

import logging
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TYPE_CHECKING
from datetime import datetime

from symfluence.core import ConfigurableMixin
from symfluence.core.constants import ModelDefaults
from symfluence.optimization.registry import OptimizerRegistry
from ..mixins import (
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
    GradientOptimizationMixin
)
from ..workers.base_worker import BaseWorker
from .algorithms import get_algorithm, ALGORITHM_REGISTRY
from .evaluators import TaskBuilder, PopulationEvaluator
from .final_evaluation import FinalResultsSaver

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
    """Abstract base class for model-specific optimizers.

    Implements the template method pattern to provide unified optimization
    infrastructure across all hydrological models (SUMMA, FUSE, NGEN, GR4J, etc.)
    while allowing model-specific parameter and worker implementations.

    Mixin Components:
        ConfigurableMixin:
            - Typed configuration access via self.config (SymfluenceConfig)
            - Legacy dict config fallback for backward compatibility
            - Safe nested access via _get_config_value(lambda: ..., default=...)

        ParallelExecutionMixin:
            - setup_parallel_processing(): Create MPI directory structure
            - execute_batch(): Distribute tasks across MPI processes
            - Supports both parallel (MPI) and sequential execution modes
            - Via self.use_parallel and self.num_processes properties

        ResultsTrackingMixin:
            - record_iteration(): Log generation results to history
            - update_best(): Track best solution and score
            - Provides self.iteration_history, self.best_score, self.best_params
            - Enables post-optimization visualization and reporting

        RetryExecutionMixin:
            - Configurable retry logic for worker failures
            - Useful for HPC environments with occasional job failures

        GradientOptimizationMixin:
            - Gradient computation utilities (finite differences, caching)
            - Step size management for gradient-based optimizers
            - Convergence detection (gradient norm < threshold)

    Algorithm Selection:
        All algorithms (DDS, PSO, DE, SCE-UA, ASYNC_DDS, NSGA-II, ADAM, LBFGS)
        retrieved from algorithms.py registry. Delegates to algorithm.optimize()
        with unified callback interface. See run_optimization() for details.

    Lazy Initialization:
        Several heavy components initialized on first use to reduce startup time:
        - task_builder: Creates task dicts for population evaluation
        - population_evaluator: Runs tasks via worker
        - results_saver: Saves results to disk
        - final_evaluation_runner: Post-optimization evaluation

    Abstract Methods (Must Implement in Subclass):
        1. _get_model_name() → str
           Example: 'SUMMA', 'FUSE', 'NGEN', 'GR'

        2. _create_parameter_manager() → ParameterManager
           Creates bounds, handles normalization, writes parameter files

        3. _create_calibration_target() → CalibrationTarget
           Loads observations, extracts simulated data, calculates metrics

        4. _create_worker() → BaseWorker
           Runs model, applies parameters, extracts outputs

    Optional Abstract Methods:
        5. _run_model_for_final_evaluation(output_dir) → bool
           Override to customize final evaluation run

        6. _get_final_file_manager_path() → Path
           Override to specify where file manager lives for time period updates

    Attributes:
        config: Typed configuration object (SymfluenceConfig)
        logger: Logging instance
        data_dir: Root data directory (from config.system.data_dir)
        domain_name: Domain identifier (from config.domain.name)
        project_dir: Project root (data_dir / f"domain_{domain_name}")
        results_dir: Algorithm-specific results directory (auto-created)
        optimization_settings_dir: Model settings directory
        param_manager: Parameter manager instance (model-specific)
        calibration_target: Evaluation/metric calculator (model-specific)
        worker: Worker instance (model-specific)
        max_iterations: Max generations/batches (from config)
        population_size: Population size for GA methods (from config)
        target_metric: Primary objective (e.g., 'KGE', from config)
        random_seed: Seed for reproducibility (from config.system.random_seed)
        use_parallel: Whether to use MPI execution
        num_processes: Number of parallel processes
        parallel_dirs: MPI directory structure {proc_id: {'sim_dir': ..., ...}}
        iteration_history: List of dicts recording each generation

    Examples:
        Basic usage:

        >>> class MyOptimizer(BaseModelOptimizer):
        ...     def _get_model_name(self):
        ...         return 'MYMODEL'
        ...
        ...     def _create_parameter_manager(self):
        ...         return MyParameterManager(self.config)
        ...
        ...     def _create_calibration_target(self):
        ...         return MyEvaluator(self.config, self.project_dir)
        ...
        ...     def _create_worker(self):
        ...         return MyWorker(self.config)

        >>> optimizer = MyOptimizer(config, logger)
        >>> results_path = optimizer.run_dds()  # Run DDS
        >>> results_path = optimizer.run_nsga2()  # Run NSGA-II
        >>> results_path = optimizer.run_adam(steps=100, lr=0.01)  # Run ADAM

    References:
        - Tolson & Shoemaker (2007): DDS algorithm
        - Kingma & Ba (2015): ADAM optimizer
        - Deb et al. (2002): NSGA-II multi-objective optimization
    """

    # Default algorithm parameters
    DEFAULT_ITERATIONS = 100
    DEFAULT_POPULATION_SIZE = 30
    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

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
        self.parallel_dirs: Dict[int, Dict[str, Any]] = {}
        self.default_sim_dir = self.results_dir  # Initialize with results_dir as fallback
        # Setup directories if MPI_PROCESSES is set, regardless of count (for isolation)
        mpi_processes = self._get_config_value(lambda: self.config.system.mpi_processes, default=1)
        if mpi_processes >= 1:
            self._setup_parallel_dirs()

        # Runtime config overrides (for algorithm-specific settings like Adam/LBFGS)
        self._runtime_overrides: Dict[str, Any] = {}

        # Algorithm registry
        self._registry = ALGORITHM_REGISTRY

        # Lazy-initialized components
        self._task_builder: Optional[TaskBuilder] = None
        self._population_evaluator: Optional[PopulationEvaluator] = None
        self._final_evaluation_runner: Optional[Any] = None
        self._results_saver: Optional[FinalResultsSaver] = None

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
        assert self._task_builder is not None
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
        assert self._population_evaluator is not None
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
        assert self._results_saver is not None
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
    # Default factory methods using registry-based discovery
    # =========================================================================

    def _create_parameter_manager_default(self):
        """
        Default factory for parameter managers using registry discovery.

        Uses convention-over-configuration:
        1. Gets model name from _get_model_name()
        2. Looks up parameter manager class from OptimizerRegistry
        3. Determines settings directory using standard convention
        4. Instantiates with standard signature (config, logger, settings_dir)

        Override _create_parameter_manager() if non-standard constructor needed.

        Returns:
            ParameterManager instance for the model

        Raises:
            RuntimeError: If parameter manager not registered
        """
        model_name = self._get_model_name()

        # Look up parameter manager class from registry
        param_manager_cls = OptimizerRegistry.get_parameter_manager(model_name)

        if param_manager_cls is None:
            raise RuntimeError(
                f"No parameter manager registered for model '{model_name}'. "
                f"Ensure the parameter manager is decorated with "
                f"@OptimizerRegistry.register_parameter_manager('{model_name}')"
            )

        # Determine settings directory using convention
        settings_dir = self._get_settings_directory()

        self.logger.debug(
            f"Creating parameter manager: {param_manager_cls.__name__} "
            f"for {model_name} at {settings_dir}"
        )

        return param_manager_cls(self.config, self.logger, settings_dir)

    def _create_worker_default(self) -> BaseWorker:
        """
        Default factory for workers using registry discovery.

        Uses convention-over-configuration:
        1. Gets model name from _get_model_name()
        2. Looks up worker class from OptimizerRegistry
        3. Instantiates with standard signature (config, logger)

        All workers use the same constructor signature, so overriding is rarely needed.

        Returns:
            BaseWorker instance for the model

        Raises:
            RuntimeError: If worker not registered
        """
        model_name = self._get_model_name()

        # Look up worker class from registry
        worker_cls = OptimizerRegistry.get_worker(model_name)

        if worker_cls is None:
            raise RuntimeError(
                f"No worker registered for model '{model_name}'. "
                f"Ensure the worker is decorated with "
                f"@OptimizerRegistry.register_worker('{model_name}')"
            )

        self.logger.debug(f"Creating worker: {worker_cls.__name__} for {model_name}")

        return worker_cls(self.config, self.logger)

    def _create_calibration_target_default(self):
        """
        Default factory for calibration targets using centralized factory.

        Uses the existing create_calibration_target() factory which:
        1. Checks OptimizerRegistry for registered targets
        2. Falls back to model-specific target mappings
        3. Falls back to default (model-agnostic) targets

        This method is rarely overridden as the factory handles all complexity.

        Returns:
            CalibrationTarget instance for the model and target type
        """
        from symfluence.optimization.calibration_targets import create_calibration_target

        model_name = self._get_model_name()

        # Get target type from config (supports both typed and dict configs)
        target_type = self._get_config_value(
            lambda: self.config.optimization.target,
            default=self._get_config_value(lambda: self.config.optimization.target, default='streamflow', dict_key='OPTIMIZATION_TARGET')
        ) if hasattr(self, '_get_config_value') else self._get_config_value(lambda: self.config.optimization.target, default='streamflow', dict_key='OPTIMIZATION_TARGET')

        target_type = str(target_type).lower()

        return create_calibration_target(
            model_name=model_name,
            target_type=target_type,
            config=self.config,
            project_dir=self.project_dir,
            logger=self.logger
        )

    def _get_settings_directory(self) -> Path:
        """
        Get the model-specific settings directory using convention.

        Convention: {project_dir}/settings/{MODEL_NAME}/

        Override this method if:
        - Non-standard settings directory location
        - Settings directory determined dynamically

        Returns:
            Path to model settings directory
        """
        model_name = self._get_model_name()
        return self.project_dir / 'settings' / model_name

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


    def _create_parameter_manager(self):
        """
        Create the model-specific parameter manager.

        Default implementation uses registry-based discovery via
        _create_parameter_manager_default(). Override if:
        - Non-standard constructor signature needed
        - Pre-initialization logic required
        - Custom path resolution needed

        Examples of when to override:
        - FUSE: Needs fuse_sim_dir computed before super().__init__()
        - SUMMA: Uses summa_settings_dir instead of generic settings_dir
        - GR: Uses gr_setup_dir instead of generic settings_dir

        Returns:
            Parameter manager instance
        """
        return self._create_parameter_manager_default()

    def _create_calibration_target(self):
        """
        Create the model-specific calibration target.

        Default implementation uses centralized factory via
        _create_calibration_target_default(). Override rarely needed as
        the factory handles registry lookup and fallbacks.

        Returns:
            Calibration target instance
        """
        return self._create_calibration_target_default()

    def _create_worker(self) -> BaseWorker:
        """
        Create the model-specific worker.

        Default implementation uses registry-based discovery via
        _create_worker_default(). Override rarely needed as all workers
        use the same constructor signature.

        Returns:
            Worker instance
        """
        return self._create_worker_default()

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
            # Check if this is a multivariate target
            if hasattr(self.calibration_target, 'variables') and self.calibration_target.variables:
                # Multivariate calibration: check each variable's observed data
                from symfluence.evaluation.registry import EvaluationRegistry

                all_found = True
                for var in self.calibration_target.variables:
                    evaluator = EvaluationRegistry.get_evaluator(
                        var, self.config, self.logger, self.project_dir, target=var
                    )
                    if evaluator and hasattr(evaluator, '_load_observed_data'):
                        obs = evaluator._load_observed_data()
                        if obs is None or obs.empty:
                            self.logger.warning(f"Calibration check: no observed data found for {var}")
                            all_found = False
                        else:
                            self.logger.info(f"Calibration check: {var} has {len(obs)} observed data points")

                if not all_found:
                    self.logger.warning("Some variables in multivariate calibration lack observed data")
                return

            # Single-target calibration
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
        """Log optimization progress in consistent format across all algorithms.

        Provides unified progress logging for all algorithms (DDS, PSO, DE, etc.)
        to enable easy comparison and debugging. Logs both essential information
        (best score, iteration) and optional algorithm-specific metrics.

        Log Format:
            "{ALGORITHM} {iter}/{max_iter} ({%}%) | Best: {best:.4f} | [optional fields] | Elapsed: {time}"

        Algorithm-Specific Metrics:
            - Single-objective (DDS, PSO, DE): Reports best score and iteration count
            - Population-based (GA, PSO, DE): Additionally reports improved individuals
            - Multi-objective (NSGA-II): Reports objectives separately
            - Gradient-based (Adam, LBFGS): Reports gradient norm or step size

        Examples of Log Output:
            >>> # DDS single-objective
            >>> "DDS 25/100 (25%) | Best: 0.7325 | Elapsed: 00:15:30"

            >>> # PSO with improvements
            >>> "PSO 50/100 (50%) | Best: 0.8420 | Improved: 12/30 | Elapsed: 00:31:45"

            >>> # NSGA-II multi-objective
            >>> "NSGA2 30/100 (30%) | Best: 0.7890 | NSE: 0.7654 | Improved: 8/20 | Elapsed: 00:10:20"

            >>> # Adam gradient-based
            >>> "ADAM 100/200 (50%) | Best: 0.8765 | Gradient: 1.2e-04 | Elapsed: 01:45:00"

        Args:
            algorithm_name: Algorithm name for logging
                - Examples: 'DDS', 'PSO', 'DE', 'SCE', 'ASYNC_DDS', 'ADAM', 'LBFGS', 'NSGA2'
                - Should match algorithm.name for consistency

            iteration: Current generation or step number (0-indexed or 1-indexed)
                - Example: 25 for iteration 25

            best_score: Best fitness/objective value found to date
                - Units depend on metric (e.g., 0.85 for KGE)
                - Always maximized (higher = better)

            secondary_score: Optional second objective value (NSGA-II, multi-metric)
                - Example: 0.75 for NSE when primary is KGE
                - Default None (not printed if None)

            secondary_label: Label for secondary score
                - Example: 'NSE' or 'KGE_Eval'
                - Ignored if secondary_score is None

            n_improved: Number of individuals improved this generation (GA methods)
                - Example: 12 improvements out of 30 population
                - Printed as "Improved: {n_improved}/{population_size}"
                - Default None (not printed if None)

            population_size: Total population size (GA methods)
                - Required if n_improved specified
                - Used to compute improvement percentage

        Side Effects:
            - Logs single info message via self.logger.info()
            - Uses elapsed time from self.format_elapsed_time()
            - Progress percentage calculated as (iteration / max_iterations) * 100

        Examples:
            >>> # Single-objective, no improvements
            >>> optimizer.log_iteration_progress(
            ...     algorithm_name='DDS',
            ...     iteration=25,
            ...     best_score=0.7325
            ... )
            >>> # Output: "DDS 25/100 (25%) | Best: 0.7325 | Elapsed: 00:15:30"

            >>> # Population-based with improvements
            >>> optimizer.log_iteration_progress(
            ...     algorithm_name='PSO',
            ...     iteration=50,
            ...     best_score=0.8420,
            ...     n_improved=12,
            ...     population_size=30
            ... )
            >>> # Output: "PSO 50/100 (50%) | Best: 0.8420 | Improved: 12/30 | Elapsed: 00:31:45"

            >>> # Multi-objective NSGA-II
            >>> optimizer.log_iteration_progress(
            ...     algorithm_name='NSGA2',
            ...     iteration=30,
            ...     best_score=0.7890,
            ...     secondary_score=0.7654,
            ...     secondary_label='NSE',
            ...     n_improved=8,
            ...     population_size=20
            ... )
            >>> # Output: "NSGA2 30/100 (30%) | Best: 0.7890 | NSE: 0.7654 | Improved: 8/20 | Elapsed: 00:10:20"

        See Also:
            log_initial_population(): Log initial population results
            format_elapsed_time(): Get elapsed time since start_timing()
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

    def _create_gradient_callback(self) -> Optional[Callable]:
        """
        Create native gradient callback if worker supports autodiff.

        This enables gradient-based optimizers (Adam, L-BFGS) to use native
        gradients from autodiff-capable models (e.g., HBV with JAX) instead
        of finite differences, providing ~N times speedup where N is parameter count.

        The callback handles:
        1. Denormalization: [0,1] normalized space → physical parameters
        2. Worker gradient computation via evaluate_with_gradient()
        3. Gradient chain rule: transform from physical to normalized space

        Returns:
            Callable with signature (x_normalized: np.ndarray) -> Tuple[loss, gradient]
            where loss is the objective value (for minimization) and gradient is
            in normalized [0,1] space. Returns None if worker doesn't support
            native gradients.

        Note:
            The returned callback computes gradients for MINIMIZATION (loss).
            The optimization algorithms handle the sign conversion for maximization.
        """
        # Check if worker supports native gradients
        if not hasattr(self, 'worker') or self.worker is None:
            return None

        if not hasattr(self.worker, 'supports_native_gradients'):
            return None

        if not self.worker.supports_native_gradients():
            return None

        # Get optimization metric from config
        # Uses OPTIMIZATION_METRIC first, then CALIBRATION_METRIC, matching _extract_primary_score
        # in base_worker.py to ensure FD and native gradient paths optimize the same objective
        metric = self.config_dict.get(
            'OPTIMIZATION_METRIC',
            self.config_dict.get('CALIBRATION_METRIC', 'KGE')
        ).lower()

        # Get parameter names and bounds for gradient transformation
        param_names = self.param_manager.all_param_names
        bounds = self.param_manager.get_parameter_bounds()

        # Compute scale factors for gradient chain rule
        # d(loss)/d(x_norm) = d(loss)/d(x_phys) * d(x_phys)/d(x_norm)
        # where d(x_phys)/d(x_norm) = (upper - lower) for linear scaling
        scale_factors = np.array([
            bounds[name]['max'] - bounds[name]['min']
            for name in param_names
        ])

        def gradient_callback(x_normalized: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Compute loss and gradient for normalized parameters.

            Args:
                x_normalized: Parameters in [0,1] normalized space

            Returns:
                Tuple of (loss, gradient_normalized) where:
                - loss: Scalar loss value (negative of metric, for minimization)
                - gradient_normalized: Gradient in normalized [0,1] space
            """
            # Denormalize to physical parameters
            params_dict = self.param_manager.denormalize_parameters(x_normalized)

            # Call worker's evaluate_with_gradient
            loss, grad_dict = self.worker.evaluate_with_gradient(params_dict, metric)

            if grad_dict is None:
                raise RuntimeError(
                    f"Worker returned None gradient despite supporting native gradients. "
                    f"Check {self.worker.__class__.__name__}.evaluate_with_gradient() implementation."
                )

            # Convert gradient dict to array (same order as param_names)
            grad_physical = np.array([grad_dict[name] for name in param_names])

            # Transform gradient from physical to normalized space via chain rule
            grad_normalized = grad_physical * scale_factors

            return loss, grad_normalized

        self.logger.info(
            f"Native gradient callback created for {self._get_model_name()} "
            f"({len(param_names)} parameters)"
        )
        return gradient_callback

    def _get_gradient_mode(self) -> str:
        """
        Get gradient computation mode from configuration.

        Returns:
            One of: 'auto', 'native', 'finite_difference'
            - 'auto': Use native gradients if available, else finite differences
            - 'native': Require native gradients (error if unavailable)
            - 'finite_difference': Always use FD (useful for comparison/debugging)
        """
        return self._get_config_value(
            lambda: self.config.optimization.gradient_mode,
            default='auto',
            dict_key='GRADIENT_MODE'
        )

    def _evaluate_population(
        self,
        population: np.ndarray,
        iteration: int = 0
    ) -> np.ndarray:
        """Evaluate a population of solutions in parallel.

        Bulk evaluation interface for population-based algorithms (GA, PSO, DE, etc.).
        Converts normalized parameters to physical units, batches task creation, and
        distributes evaluation across MPI processes for efficiency.

        Parameter Format:
            Normalized space [0, 1] for algorithm simplicity and uniform scaling.
            Converts to physical units via param_manager.denormalize_parameters() for
            each individual in the population.

        Evaluation Workflow:
            1. PopulationEvaluator.evaluate_population() called with population array
            2. TaskBuilder creates task dicts for each individual:
               {
                   'individual_id': i,  # Index in population array
                   'params': {...},     # Denormalized parameters for worker
                   'proc_id': i % num_processes,  # Processor assignment
                   'evaluation_id': f"{iteration}_{i}"  # Unique ID
               }
            3. execute_batch() distributes tasks across MPI processes
            4. Worker executes model for each task, extracts output
            5. CalibrationTarget calculates metric (KGE, NSE, etc.)
            6. Results collected and scores returned in original order

        Parallel Execution:
            - execute_batch() handles MPI distribution, not evaluate_population()
            - Each process runs 1+ individuals depending on load balancing
            - Waits for all processes to complete (synchronous)
            - Useful for load-balanced work when tasks have variable runtime

        Args:
            population: Array shape (n_individuals, n_parameters)
                - Values in [0, 1] normalized space
                - Each row is one individual's parameters
                - Row indices must match individual_id in results

            iteration: Current generation/batch number (for logging/seeding)
                - Used to seed random number generator if reproducibility enabled
                - Passed to task building for unique evaluation IDs

        Returns:
            Array shape (n_individuals,) with fitness scores
            - Scores in physical units (e.g., 0.7 for KGE=0.7)
            - Index i corresponds to population[i]
            - Invalid evaluations marked as -inf or small penalty

        Examples:
            >>> # Evaluate 30 individuals with 5 parameters each
            >>> population = np.random.random((30, 5))
            >>> scores = optimizer._evaluate_population(population, iteration=0)
            >>> assert scores.shape == (30,)
            >>> best_idx = np.argmax(scores)
            >>> print(f"Best fitness: {scores[best_idx]:.4f}")

            >>> # Used by algorithms via callback
            >>> def my_callback(pop_array):
            ...     return optimizer._evaluate_population(pop_array)
            >>> algorithm.optimize(..., evaluate_population=my_callback)

        See Also:
            _evaluate_solution(): Evaluate single parameter set
            population_evaluator: PopulationEvaluator instance
            TaskBuilder: Creates task dicts
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
        """Evaluate population for multiple objectives (NSGA-II).

        Multi-objective evaluation enables simultaneous optimization of multiple metrics
        (e.g., KGE for calibration + NSE for evaluation) using Pareto dominance ranking.
        Different from single-objective which returns scalar score.

        Objectives:
            NSGA-II evaluates each individual on multiple objectives simultaneously:
            - Each individual gets a vector of scores [obj1, obj2, ...]
            - Pareto ranking identifies non-dominated solutions
            - Front 1 (best) contains solutions dominating all others
            - Front 2 contains solutions dominated by only Front 1, etc.

        Typical Objectives (for hydrological models):
            Primary: KGE (Kling-Gupta Efficiency) - balanced metric
            Secondary: NSE (Nash-Sutcliffe), RMSE, Bias
            Hybrid: 'KGE_Calib' vs 'KGE_Eval' (train vs validation)

        Example Configuration (config.yaml):
            optimization:
              algorithm: nsga2
              nsga2:
                multi_target: true
                primary_metric: KGE      # Maximize KGE
                secondary_metric: NSE    # Simultaneously maximize NSE

        Evaluation Workflow:
            1. Same as _evaluate_population() for parameter conversion and task creation
            2. Worker executes model, returns all available metrics
            3. PopulationEvaluator extracts requested objectives from all metrics
            4. Returns array shape (n_individuals, n_objectives)

        Args:
            population: Array shape (n_individuals, n_parameters)
                - Normalized [0, 1] like _evaluate_population()

            objective_names: List of metric names to extract from worker results
                - Examples: ['KGE', 'NSE'], ['KGE_Calib', 'KGE_Eval']
                - Order matters: results[i, 0] = objective_names[0] score
                - Must exist in worker.get_metrics() output

            iteration: Current generation number (for logging/seeding)

        Returns:
            Array shape (n_individuals, n_objectives) with objective values
            - Each row is one individual's objective vector
            - Column j is objective_names[j] score
            - Example for ['KGE', 'NSE']:
                [[0.70, 0.65],  # Individual 0: KGE=0.70, NSE=0.65
                 [0.75, 0.72],  # Individual 1: KGE=0.75, NSE=0.72
                 ...]

        Examples:
            >>> # Multi-objective for calibration vs evaluation
            >>> objectives = optimizer._evaluate_population_objectives(
            ...     population,
            ...     objective_names=['KGE_Calib', 'KGE_Eval'],
            ...     iteration=0
            ... )
            >>> assert objectives.shape == (population.shape[0], 2)
            >>> print(f"Mean calib KGE: {objectives[:, 0].mean():.3f}")
            >>> print(f"Mean eval KGE: {objectives[:, 1].mean():.3f}")

        See Also:
            _evaluate_population(): Single-objective version
            _get_nsga2_objective_names(): Resolve objectives from config
            NSGA2Operators: Pareto ranking and selection
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
        # Save results
        results_path = self.save_results(algorithm_name, standard_filename=True)
        assert results_path is not None
        return results_path

    def run_optimization(self, algorithm_name: str) -> Path:
        """Run optimization using a specified algorithm from the registry.

        This is the unified entry point for all optimization algorithms.
        Individual methods like run_dds(), run_pso() delegate to this method.

        Execution Workflow:
            1. Validate parameter count and log calibration alignment
            2. Retrieve algorithm from registry with configuration
            3. Bind callbacks to BaseModelOptimizer methods:
               - evaluate_solution(): Single-point evaluation
               - evaluate_population(): Batch evaluation
               - denormalize_params(): [0,1] → physical units
               - record_iteration(): Log generation results
               - update_best(): Track best solution
               - log_progress(): Consistent progress logging
            4. Run algorithm.optimize() with callback interface
            5. Save results to JSON/CSV files
            6. Generate progress visualizations
            7. Run final evaluation on best parameters

        Args:
            algorithm_name: Algorithm name (case-insensitive)

        Returns:
            Path to results JSON file
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

        # Handle initial guess - only for GR models (others benefit from random initialization)
        if self._get_model_name().upper() == 'GR':
            try:
                initial_params_dict = self.param_manager.get_initial_parameters()
                if initial_params_dict:
                    initial_guess = self.param_manager.normalize_parameters(initial_params_dict)
                    kwargs['initial_guess'] = initial_guess
                    self.logger.info("Using initial parameter guess for optimization seeding")
            except Exception as e:
                self.logger.warning(f"Failed to prepare initial parameter guess: {e}")

        # For NSGA-II, add multi-objective support
        if algorithm_name.lower() in ['nsga2', 'nsga-ii']:
            kwargs['evaluate_population_objectives'] = self._evaluate_population_objectives
            kwargs['objective_names'] = self._get_nsga2_objective_names()
            kwargs['multiobjective'] = bool(self._get_config_value(
                lambda: self.config.optimization.nsga2.multi_target, default=False
            ))

        # For gradient-based algorithms (Adam, L-BFGS), add native gradient support
        if algorithm_name.lower() in ['adam', 'lbfgs']:
            gradient_callback = self._create_gradient_callback()
            gradient_mode = self._get_gradient_mode()

            if gradient_callback is not None:
                kwargs['compute_gradient'] = gradient_callback
                self.logger.info(
                    f"Native gradient support enabled for {algorithm_name.upper()} "
                    f"(mode: {gradient_mode})"
                )
            else:
                self.logger.info(
                    f"Using finite-difference gradients for {algorithm_name.upper()} "
                    f"(native gradients not available for {self._get_model_name()})"
                )

            kwargs['gradient_mode'] = gradient_mode

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

        return results_path  # type: ignore[return-value]

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
        """Run final evaluation with best parameters over full experiment period.

        Final evaluation verifies model performance on both calibration and evaluation
        windows using best parameters found during optimization. This separates training
        (calibration) and validation (evaluation) performance to detect overfitting.

        Execution Workflow:
            1. Update file manager (SUMMA) to use full experiment period:
               - simStartTime ← config.domain.time_start
               - simEndTime ← config.domain.time_end
               - Adjusts end time for sub-daily forcing (e.g., CERRA 3-hourly)

            2. Apply best parameters to model configuration files

            3. Run model once over full period via _run_model_for_final_evaluation()

            4. Extract outputs and calculate metrics for:
               - Calibration period: Using config.domain.calibration_start_date/end_date
               - Evaluation period: time_start to end (minus calibration window)
               - Full period: Aggregated metrics

            5. Generate progress visualizations if ReportingManager available

            6. Restore model configuration for reproducibility:
               - Restore modelDecisions to optimization settings
               - Restore file manager to calibration period

        Window Definitions:
            Calibration Window: Time period used to optimize parameters
                - From: config.domain.calibration_start_date
                - To: config.domain.calibration_end_date
                - Metric names: *_Calib (e.g., KGE_Calib, NSE_Calib)

            Evaluation Window: Period not used for calibration (validation)
                - From: config.domain.time_start to calibration_start_date
                         OR calibration_end_date to time_end
                - Metric names: *_Eval (e.g., KGE_Eval, NSE_Eval)

            Full Period: Entire simulation window
                - From: config.domain.time_start
                - To: config.domain.time_end

        Metric Calculation:
            Model output extracted via calibration_target.extract_simulated_data()
            Metrics calculated via calibration_target.calculate_metrics() with
            calibration_only=False to get all periods.

        Args:
            best_params: Best parameters from optimization (Dict[str, float])
                - Keys: Parameter names (model-specific)
                - Values: Physical units (not normalized)
                - Example: {'PARAM1': 100.5, 'PARAM2': 0.25}

        Returns:
            Dict with structure:
                {
                    'final_metrics': {...},  # All metrics for all periods
                    'calibration_metrics': {...},  # *_Calib metrics only
                    'evaluation_metrics': {...},  # *_Eval metrics only
                    'success': True,
                    'best_params': best_params
                }
            Or None if any step fails (logged to logger.error)

        Raises:
            (Caught internally, returns None instead):
            - FileNotFoundError: File manager not found or output not generated
            - ValueError: Metric calculation fails
            - RuntimeError: Model execution fails
            - Exception: Any unexpected error

        Side Effects:
            - Creates results_dir/final_evaluation/ directory
            - Modifies file manager temp file (restored after run)
            - Generates CSV/PNG outputs via ReportingManager if available
            - Logs detailed results to logger.info

        Examples:
            >>> optimizer = MyOptimizer(config, logger)
            >>> best_params = optimizer.run_dds()  # Returns best params
            >>> final_result = optimizer.run_final_evaluation(best_params)
            >>> if final_result:
            ...     print(f"Calibration KGE: {final_result['calibration_metrics']['KGE_Calib']}")
            ...     print(f"Evaluation KGE: {final_result['evaluation_metrics']['KGE_Eval']}")

        See Also:
            _update_file_manager_for_final_run(): Updates time period
            _apply_best_parameters_for_final(): Applies parameter values
            _extract_period_metrics(): Extracts period-specific metrics
            _restore_file_manager_for_optimization(): Restores settings
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

            self.logger.debug("Restored file manager to calibration period")

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
