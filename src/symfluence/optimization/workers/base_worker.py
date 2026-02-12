"""Base worker infrastructure for model evaluation during optimization.

Provides abstract base class and data structures for optimization workers that execute
parameter set evaluations through a standardized workflow: parameter application → model
execution → metric calculation. Implements Template Method pattern with retry logic for
transient failures, backward compatibility with legacy dictionary formats, and support
for model-specific customization via abstract methods.

Architecture:
    The base worker module implements the Factory and Template Method patterns to enable
    extensible worker implementations for different hydrological models:

    1. Data Structures (Dataclasses):
       - WorkerTask: Encapsulates task specification (parameters, paths, config)
       - WorkerResult: Encapsulates evaluation results (score, metrics, errors)
       Both support legacy dictionary conversion for backward compatibility

    2. WorkerTask:
       Represents a single model evaluation task to be executed by a worker.
       Attributes:
           individual_id: Unique identifier for this evaluation (from optimizer)
           params: Dictionary mapping parameter names to values (to be applied to model)
           proc_id: Process ID for parallel execution (used for directory isolation)
           config: Configuration dictionary with simulation settings
           settings_dir: Path to model settings/configuration directory
           output_dir: Path for model output files
           sim_dir: Optional path for simulation working files
           iteration: Optional iteration number (from optimizer)
           additional_data: Extension point for model-specific task data

       Backward Compatibility:
           from_legacy_dict(): Convert from old dictionary format
           to_legacy_dict(): Convert to old dictionary format
           Handles multiple naming conventions (proc_id vs process_id, params vs parameters)

    3. WorkerResult:
       Encapsulates evaluation results returned by a worker.
       Attributes:
           individual_id: Identifier matching the task
           params: Parameter values that were evaluated
           score: Objective score (fitness/KGE/NSE), None if evaluation failed
           metrics: Dictionary of all calculated metrics (KGE, NSE, RMSE, etc.)
           error: Error message if evaluation failed
           runtime: Execution time in seconds
           iteration: Iteration number if applicable
           additional_data: Extension point for model-specific result data

       Properties:
           success: True if evaluation succeeded (score != None and error == None)
           valid_score: True if score is valid (not NaN, not penalty, not None)

       Factory Methods:
           failure(): Create failure result with penalty score
           from_legacy_dict(): Convert from old result format
           to_legacy_dict(): Convert to old result format

    4. BaseWorker (Abstract Base Class):
       Abstract base class implementing the Template Method pattern.
       Each concrete subclass implements model-specific behavior:

       Abstract Methods (must be implemented by subclasses):
           apply_parameters(params, settings_dir, **kwargs)
               - Apply parameter values to model config files
               - Model-specific: SUMMA, FUSE, GR, HYPE, etc.

           run_model(config, settings_dir, output_dir, **kwargs)
               - Execute model simulation with configured parameters
               - Model-specific: Invoke executables, manage processes

           calculate_metrics(output_dir, config, **kwargs)
               - Extract and calculate objective metrics from model outputs
               - Model-specific: Parse output files, unit conversions

       Template Method:
           evaluate(task) → WorkerResult
               - Orchestrates full evaluation workflow
               - Implements retry logic for transient failures
               - Records runtime and error handling

Workflow:

    1. Optimizer creates WorkerTask:
       task = WorkerTask(
           individual_id=1,
           params={'param1': 0.5, 'param2': 1.2},
           proc_id=0,
           config=config_dict,
           settings_dir=Path('settings/proc_0'),
           output_dir=Path('outputs/proc_0')
       )

    2. Worker evaluates task via template method:
       result = worker.evaluate(task)

       Template method orchestrates:
       a) Retry loop with exponential backoff for transient failures
       b) apply_parameters() - Apply param values to config files
       c) run_model() - Execute model simulation
       d) calculate_metrics() - Extract metrics from outputs
       e) Return WorkerResult with score and all metrics

    3. Retry Logic:
       - For transient errors (file handle, permission denied, etc.):
         Retry up to max_retries times with exponential backoff
       - For fatal errors: Fail immediately with penalty score
       - All errors logged with traceback for debugging

    4. Metric Extraction:
       - Extracts primary score from metrics using CALIBRATION_METRIC config
       - Handles case-insensitive and variant naming (KGE, Calib_KGE, kge)
       - Falls back to common alternatives (NSE, RMSE, etc.)
       - Returns penalty score if metric not found

Error Handling:

    Transient Errors (Retried):
        - 'stale file handle' (NFS issues)
        - 'resource temporarily unavailable' (System overload)
        - 'no such file or directory' (Timing issue)
        - 'permission denied' (Transient file lock)
        - 'connection refused' (Network hiccup)
        - 'broken pipe' (Process communication)

    Retry Strategy:
        Exponential backoff: delay = base_delay * (2 ** attempt)
        Config: WORKER_MAX_RETRIES (default 3), WORKER_BASE_DELAY (default 0.5s)

    Fatal Errors:
        - Any error not in TRANSIENT_ERRORS list
        - Errors after max_retries exceeded
        - Returns WorkerResult with penalty score

Configuration Parameters:

    WORKER_MAX_RETRIES: int (default 3)
        Maximum number of retry attempts for transient failures

    WORKER_BASE_DELAY: float (default 0.5)
        Base delay in seconds for exponential backoff calculation

    PENALTY_SCORE: float (default -999.0)
        Score assigned to failed evaluations

    CALIBRATION_METRIC: str (default 'KGE')
        Name of metric to use as primary optimization score

Supported Hydrological Models (Subclasses):
    - SUMMAWorker: Spectral Matching Input Model for Land Surface
    - FUSEWorker: Flexible Utility Splitter for Evapotranspiration
    - GRWorker: GR4J/GR6J lumped rainfall-runoff
    - HYPEWorker: Hydrological Predictions for the Environment
    - RHESSysWorker: Regional Hydro-Ecological Simulation System
    - NGENWorker: NextGen National Water Model
    - MESHWorker: MESH (Canadian Arctic model)
    - LSTMWorker: LSTM neural network surrogate model

Example Implementation:

    >>> class MyModelWorker(BaseWorker):
    ...     def apply_parameters(self, params, settings_dir, **kwargs):
    ...         # Model-specific parameter application
    ...         config_file = settings_dir / 'config.txt'
    ...         # Write params to config file
    ...         return True
    ...
    ...     def run_model(self, config, settings_dir, output_dir, **kwargs):
    ...         # Model-specific execution
    ...         # subprocess.run(['/path/to/model/executable', ...])
    ...         return True
    ...
    ...     def calculate_metrics(self, output_dir, config, **kwargs):
    ...         # Model-specific metric calculation
    ...         # Parse output files, calculate KGE, NSE, etc.
    ...         return {'KGE': 0.85, 'NSE': 0.82, ...}

    >>> # Usage:
    >>> worker = MyModelWorker(config, logger)
    >>> task = WorkerTask(...)
    >>> result = worker.evaluate(task)
    >>> if result.success:
    ...     print(f"Score: {result.score}, Metrics: {result.metrics}")

References:
    - Template Method Pattern: Gang of Four design patterns
    - Factory Pattern: Gang of Four design patterns
    - Exponential Backoff: Standard error recovery strategy
    - Objective Metrics: Kling-Gupta Efficiency, Nash-Sutcliffe Efficiency

See Also:
    - BaseModelOptimizer: Uses workers to evaluate parameter sets
    - ParallelExecutionMixin: Manages parallel worker execution
    - PopulationEvaluator: Batch evaluation of worker results
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.exceptions import RetryExhaustedError
from symfluence.evaluation.metric_transformer import MetricTransformer

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """
    Data structure representing a task to be executed by a worker.

    Attributes:
        individual_id: Unique identifier for this evaluation
        params: Dictionary mapping parameter names to values
        proc_id: Process ID for parallel execution
        config: Configuration dictionary
        settings_dir: Path to model settings directory
        output_dir: Path for model outputs
        sim_dir: Optional path for simulation files
        iteration: Optional iteration number
        additional_data: Optional additional data for the task
    """
    individual_id: int
    params: Dict[str, float]
    proc_id: int
    config: Dict[str, Any]
    settings_dir: Path
    output_dir: Path
    sim_dir: Optional[Path] = None
    iteration: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.settings_dir, str):
            self.settings_dir = Path(self.settings_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.sim_dir, str):
            self.sim_dir = Path(self.sim_dir)

    @classmethod
    def from_legacy_dict(cls, task_data: Dict[str, Any]) -> 'WorkerTask':
        """
        Create a WorkerTask from a legacy dictionary format.

        This maintains backward compatibility with existing worker functions.

        Args:
            task_data: Dictionary in legacy format

        Returns:
            WorkerTask instance
        """
        # Handle various key naming conventions
        individual_id = task_data.get('individual_id', task_data.get('task_id', 0))
        params = task_data.get('params', task_data.get('parameters', {}))
        proc_id = task_data.get('proc_id', task_data.get('process_id', 0))

        # Settings directory - various key names used
        settings_dir = (
            task_data.get('proc_settings_dir') or
            task_data.get('settings_dir') or
            task_data.get('optimization_settings_dir') or
            Path('.')
        )

        # Output directory
        output_dir = (
            task_data.get('proc_output_dir') or
            task_data.get('output_dir') or
            Path('.')
        )

        # Simulation directory
        sim_dir = (
            task_data.get('proc_sim_dir') or
            task_data.get('sim_dir') or
            None
        )

        # Config - could be nested or at top level
        config = task_data.get('config', {})

        # Convert Pydantic model to dict if needed
        if hasattr(config, 'model_dump'):
            config = config.model_dump()
        elif hasattr(config, 'dict'):
            config = config.dict()

        if not config:
            # Extract config keys from task_data itself
            config_keys = [
                'EXPERIMENT_ID', 'DOMAIN_NAME', 'ROOT_PATH', 'HYDROLOGICAL_MODEL',
                'ROUTING_MODEL', 'CALIBRATION_METRIC', 'CALIBRATION_PERIOD',
                'EVALUATION_PERIOD', 'NUM_PROCESSES', 'DOMAIN_DEFINITION_METHOD'
            ]
            config = {k: task_data[k] for k in config_keys if k in task_data}

        # Additional data
        additional_keys = set(task_data.keys()) - {
            'individual_id', 'task_id', 'params', 'parameters', 'proc_id',
            'process_id', 'proc_settings_dir', 'settings_dir',
            'optimization_settings_dir', 'proc_output_dir', 'output_dir',
            'proc_sim_dir', 'sim_dir', 'config', 'iteration'
        } - set(config.keys())
        additional_data = {k: task_data[k] for k in additional_keys}

        return cls(
            individual_id=individual_id,
            params=params,
            proc_id=proc_id,
            config=config,
            settings_dir=settings_dir,
            output_dir=output_dir,
            sim_dir=sim_dir,
            iteration=task_data.get('iteration'),
            additional_data=additional_data
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format.

        Returns:
            Dictionary in legacy format
        """
        result = {
            'individual_id': self.individual_id,
            'params': self.params,
            'proc_id': self.proc_id,
            'proc_settings_dir': str(self.settings_dir),
            'proc_output_dir': str(self.output_dir),
            'config': self.config,
        }
        if self.sim_dir:
            result['proc_sim_dir'] = str(self.sim_dir)
        if self.iteration is not None:
            result['iteration'] = self.iteration
        result.update(self.additional_data)
        return result


@dataclass
class WorkerResult:
    """
    Data structure representing the result of a worker evaluation.

    Attributes:
        individual_id: Identifier matching the task
        params: Parameter values that were evaluated
        score: Objective score (fitness), None if evaluation failed
        metrics: Dictionary of all calculated metrics
        error: Error message if evaluation failed
        runtime: Execution time in seconds
        iteration: Iteration number if applicable
        additional_data: Optional additional result data
    """
    individual_id: int
    params: Dict[str, float]
    score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    runtime: Optional[float] = None
    iteration: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the evaluation was successful."""
        return self.error is None and self.score is not None

    @property
    def valid_score(self) -> bool:
        """Check if the score is valid (not NaN, not penalty value)."""
        if self.score is None:
            return False
        if np.isnan(self.score):
            return False
        if self.score <= -900:  # Common penalty value
            return False
        return True

    @classmethod
    def failure(
        cls,
        individual_id: int,
        params: Dict[str, float],
        error: str,
        penalty_score: float = ModelDefaults.PENALTY_SCORE
    ) -> 'WorkerResult':
        """
        Create a failure result.

        Args:
            individual_id: Task identifier
            params: Parameters that were attempted
            error: Error message
            penalty_score: Penalty score to assign

        Returns:
            WorkerResult indicating failure
        """
        return cls(
            individual_id=individual_id,
            params=params,
            score=penalty_score,
            error=error
        )

    @classmethod
    def from_legacy_dict(cls, result_data: Dict[str, Any]) -> 'WorkerResult':
        """
        Create a WorkerResult from a legacy dictionary format.

        Args:
            result_data: Dictionary in legacy format

        Returns:
            WorkerResult instance
        """
        individual_id = result_data.get('individual_id', result_data.get('task_id', 0))
        params = result_data.get('params', result_data.get('parameters', {}))

        # Score may be under various keys
        score = (
            result_data.get('score') or
            result_data.get('fitness') or
            result_data.get('objective') or
            result_data.get('kge') or
            result_data.get('nse')
        )

        # Handle metrics
        metrics = result_data.get('metrics', {})
        if not metrics:
            # Look for common metric keys
            metric_keys = ['kge', 'nse', 'rmse', 'mae', 'bias', 'correlation']
            metrics = {k: result_data[k] for k in metric_keys if k in result_data}

        return cls(
            individual_id=individual_id,
            params=params,
            score=score,
            metrics=metrics,
            error=result_data.get('error'),
            runtime=result_data.get('runtime'),
            iteration=result_data.get('iteration'),
            additional_data={
                k: v for k, v in result_data.items()
                if k not in {'individual_id', 'task_id', 'params', 'parameters',
                            'score', 'fitness', 'objective', 'metrics', 'error',
                            'runtime', 'iteration', 'kge', 'nse', 'rmse', 'mae',
                            'bias', 'correlation'}
            }
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format.

        Returns:
            Dictionary in legacy format
        """
        result = {
            'individual_id': self.individual_id,
            'params': self.params,
            'score': self.score,
            'metrics': self.metrics,
        }
        if self.error:
            result['error'] = self.error
        if self.runtime is not None:
            result['runtime'] = self.runtime
        if self.iteration is not None:
            result['iteration'] = self.iteration
        # Flatten common metrics for backward compatibility
        for key in ['kge', 'nse', 'rmse', 'mae', 'bias']:
            if key in self.metrics:
                result[key] = self.metrics[key]
        result.update(self.additional_data)
        return result


class BaseWorker(ABC):
    """Abstract base class for model evaluation workers in optimization.

    Central orchestrator for parameter set evaluation, implementing the Template Method
    pattern to coordinate model-specific parameter application, execution, and metric
    extraction. Provides robust error handling with exponential backoff retry logic for
    transient failures, comprehensive logging, and support for both sequential and
    parallel execution contexts.

    This class defines the contract that all model-specific workers must implement:
    apply_parameters(), run_model(), and calculate_metrics(). The evaluate() template
    method orchestrates these three steps with integrated retry logic, timing, and
    error handling.

    Key Responsibilities:

        1. Workflow Orchestration (Template Method):
           - Manages retry loop for transient failures
           - Coordinates parameter application, model execution, metric calculation
           - Records execution time and captures errors
           - Returns standardized WorkerResult with score and all metrics

        2. Error Resilience:
           - Distinguishes transient errors (retry) vs fatal errors (fail)
           - Implements exponential backoff for retry delays
           - Logs all errors with traceback for debugging
           - Returns penalty score for failed evaluations

        3. Extensibility Points:
           - Abstract methods for model-specific implementation
           - Pre/post evaluation hooks for customization
           - Additional task/result data support for extensions

        4. Configuration Management:
           - Reads retry settings, delays, and penalty from config
           - Supports customizable calibration metric selection
           - Handles multiple metric naming conventions

        5. Result Processing:
           - Extracts primary optimization score from metrics
           - Handles metric name variations (KGE, Calib_KGE, kge)
           - Falls back to alternative metrics if configured metric not found
           - Validates score values (NaN, penalty, None checks)

    Abstract Methods (must be implemented by model-specific subclasses):

        apply_parameters(params, settings_dir, **kwargs) → bool:
            Apply parameter values to model configuration files.
            Params:
                params: Dict mapping parameter names → values (e.g., {'a': 0.5, 'b': 1.2})
                settings_dir: Path to model settings directory
                **kwargs: Model-specific arguments (output_dir, config, etc.)
            Returns: True if successful, False otherwise
            Model-specific: SUMMA modifies fileManager.txt, FUSE modifies control file, etc.

        run_model(config, settings_dir, output_dir, **kwargs) → bool:
            Execute model simulation with configured parameters.
            Params:
                config: Configuration dictionary with model, paths, time periods
                settings_dir: Path to model settings directory
                output_dir: Path for model output files
                **kwargs: Model-specific arguments (sim_dir, proc_id, params, etc.)
            Returns: True if successful, False otherwise
            Model-specific: Invokes SUMMA/FUSE/GR executable, manages processes/jobs

        calculate_metrics(output_dir, config, **kwargs) → Dict[str, float]:
            Extract and calculate objective metrics from model outputs.
            Params:
                output_dir: Path to model output files
                config: Configuration dictionary
                **kwargs: Model-specific arguments (sim_dir, proc_id, etc.)
            Returns: Dict mapping metric names → values (e.g., {'KGE': 0.85, 'NSE': 0.82})
            Model-specific: Parse output format (NetCDF, binary, text), unit conversions, calculations

    Template Method (Evaluation Workflow):

        evaluate(task) → WorkerResult:
            Main entry point for parameter set evaluation.
            Orchestrates:
                1. _evaluate_with_retry() - Handles retry loop
                2. _evaluate_once() - Single evaluation attempt
                3. apply_parameters() - Parameter application (model-specific)
                4. run_model() - Model execution (model-specific)
                5. calculate_metrics() - Metric extraction (model-specific)
                6. _extract_primary_score() - Score extraction
            Records execution time and captures all errors
            Returns: WorkerResult with score, metrics, errors, runtime

    Attributes:

        config (Dict[str, Any]): Configuration dictionary with:
            - NUM_PROCESSES: Number of parallel processes
            - CALIBRATION_METRIC: Primary objective metric name (default 'KGE')
            - PENALTY_SCORE: Score for failed evaluations (default -999.0)
            - WORKER_MAX_RETRIES: Max retry attempts (default 3)
            - WORKER_BASE_DELAY: Base delay for backoff (default 0.5 seconds)

        logger (logging.Logger): Logger instance for execution logging

        TRANSIENT_ERRORS (ClassVar[Tuple]): Error messages indicating transient failures:
            - 'stale file handle': NFS/network filesystem issues
            - 'resource temporarily unavailable': System resource conflicts
            - 'no such file or directory': Timing issue (directory/file not yet accessible)
            - 'permission denied': Transient file lock
            - 'connection refused': Network/communication failure
            - 'broken pipe': Process communication broken

    Properties:

        max_retries (int): Maximum retry attempts for transient failures.
            From config['WORKER_MAX_RETRIES'], default 3.

        base_delay (float): Base delay in seconds for exponential backoff.
            From config['WORKER_BASE_DELAY'], default 0.5.
            Retry delays: 0.5s, 1.0s, 2.0s, 4.0s for attempt 0,1,2,3...

        penalty_score (float): Score assigned to failed evaluations.
            From config['PENALTY_SCORE'], default -999.0.

    Retry Logic:

        Transient Error Handling:
            - Detects transient errors by string matching in exception message
            - Retries up to max_retries times with exponential backoff
            - Delay = base_delay * (2^attempt_number)

            Example:
                Attempt 0 fails: wait 0.5s, retry
                Attempt 1 fails: wait 1.0s, retry
                Attempt 2 fails: wait 2.0s, retry
                Attempt 3 fails: return penalty score

        Fatal Error Handling:
            - Non-transient errors fail immediately
            - Errors after max_retries exceeded treated as fatal
            - All errors logged with full traceback

    Metric Extraction:

        Primary Score Selection:
            1. Look for exact match: config['CALIBRATION_METRIC'] in metrics
            2. Look for Calib_ prefix: f"Calib_{metric_name}" in metrics
            3. Case-insensitive search
            4. Try common alternatives: ['kge', 'nse', 'score', 'fitness', 'objective']
            5. Return penalty_score if no metric found (with warning log)

        Example:
            If CALIBRATION_METRIC = 'KGE' and metrics = {'Calib_KGE': 0.85, ...}:
                Returns 0.85

    Customization Hooks:

        pre_evaluation(task) → None:
            Called before evaluation begins. Override for setup/validation.

        post_evaluation(task, result) → None:
            Called after evaluation completes. Override for cleanup/logging.

    Configuration Example:

        config = {
            'CALIBRATION_METRIC': 'KGE',          # Primary objective metric
            'PENALTY_SCORE': -999.0,                # Score for failed runs
            'WORKER_MAX_RETRIES': 3,                # Retry attempts
            'WORKER_BASE_DELAY': 0.5,               # Base delay (seconds)
            'DOMAIN_NAME': 'site_01',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'ROUTING_MODEL': 'MIZUROUTE',
            ...
        }

    Example Subclass Implementation:

        >>> class SUMMAWorker(BaseWorker):
        ...     def apply_parameters(self, params, settings_dir, **kwargs):
        ...         # Read fileManager template
        ...         fm_path = settings_dir / 'fileManager.txt'
        ...         content = fm_path.read_text()
        ...         # Update with parameter values
        ...         for param, value in params.items():
        ...             content = content.replace(f'{{{param}}}', str(value))
        ...         fm_path.write_text(content)
        ...         return True
        ...
        ...     def run_model(self, config, settings_dir, output_dir, **kwargs):
        ...         # Execute SUMMA
        ...         result = subprocess.run(
        ...             ['/path/to/summa.exe', str(settings_dir / 'fileManager.txt')],
        ...             cwd=str(output_dir),
        ...             capture_output=True,
        ...             timeout=3600
        ...         )
        ...         return result.returncode == 0
        ...
        ...     def calculate_metrics(self, output_dir, config, **kwargs):
        ...         # Parse SUMMA outputs
        ...         output_file = output_dir / 'SUMMA_outputs.nc'
        ...         with xr.open_dataset(output_file) as ds:
        ...             streamflow_sim = ds['routedRunoff'].values
        ...         # Load observations
        ...         obs = load_observations(config)
        ...         # Calculate metrics
        ...         return calculate_metrics(streamflow_sim, obs)

    Parallel Execution Support:

        - proc_id parameter specifies process ID for directory isolation
        - proc_output_dir parameter passes process-specific output directory
        - Each parallel process has isolated settings/output directories
        - Task distribution managed by ParallelExecutionMixin

    Performance Considerations:

        - Parameter application: ~10-100ms (model-specific)
        - Model execution: minutes to hours (depends on model complexity)
        - Metric calculation: ~100-1000ms (depends on output file size)
        - Retry overhead: Minimal if errors are rare

    Error Reporting:

        WorkerResult contains:
        - score: Primary objective score (or penalty if failed)
        - metrics: All calculated metrics (empty if failed)
        - error: Error message (None if successful)
        - runtime: Total execution time in seconds
        - individual_id: Identifier for tracing

    References:

        - Template Method Pattern: Gang of Four
        - Retry Pattern: Exponential backoff strategy (RFC 7231, etc.)
        - Objective Metrics: Kling-Gupta Efficiency, Nash-Sutcliffe Efficiency

    See Also:

        - WorkerTask: Task specification dataclass
        - WorkerResult: Result specification dataclass
        - BaseModelOptimizer: Uses workers for parameter evaluation
        - ParallelExecutionMixin: Manages parallel worker execution
        - SUMMAWorker, FUSEWorker, GRWorker: Model-specific implementations
    """

    # Default retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 0.5
    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    # Transient errors that warrant retry
    TRANSIENT_ERRORS = (
        'stale file handle',
        'resource temporarily unavailable',
        'no such file or directory',
        'permission denied',
        'connection refused',
        'broken pipe',
    )

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self.config.get('WORKER_MAX_RETRIES', self.DEFAULT_MAX_RETRIES)

    @property
    def base_delay(self) -> float:
        """Base delay for exponential backoff."""
        return self.config.get('WORKER_BASE_DELAY', self.DEFAULT_BASE_DELAY)

    @property
    def penalty_score(self) -> float:
        """Penalty score for failed evaluations."""
        return self.config.get('PENALTY_SCORE', self.DEFAULT_PENALTY_SCORE)

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to model configuration files.

        Args:
            params: Dictionary of parameter names to values
            settings_dir: Path to settings directory
            **kwargs: Additional model-specific arguments

        Returns:
            True if parameters were applied successfully, False otherwise
        """
        pass

    @abstractmethod
    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run the model simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Path to settings directory
            output_dir: Path for outputs
            **kwargs: Additional model-specific arguments

        Returns:
            True if model ran successfully, False otherwise
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate objective metrics from model outputs.

        Args:
            output_dir: Path to model outputs
            config: Configuration dictionary
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary of metric names to values
        """
        pass

    # =========================================================================
    # Template method - main evaluation logic
    # =========================================================================

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        """
        Evaluate a parameter set by running the full workflow.

        This is the template method that orchestrates:
        1. Parameter application
        2. Model execution
        3. Metric calculation

        Includes retry logic for transient failures.

        Args:
            task: WorkerTask containing parameters and paths

        Returns:
            WorkerResult with score and metrics
        """
        start_time = time.time()

        try:
            result = self._evaluate_with_retry(task)
            result.runtime = time.time() - start_time
            return result

        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error(
                f"Evaluation failed for individual {task.individual_id}: {e}"
            )
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error=str(e),
                penalty_score=self.penalty_score
            )

    def _evaluate_with_retry(self, task: WorkerTask) -> WorkerResult:
        """
        Execute evaluation with retry logic for transient failures.

        Args:
            task: WorkerTask to evaluate

        Returns:
            WorkerResult from evaluation
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._evaluate_once(task)

            except (ValueError, RuntimeError, IOError, TimeoutError) as e:
                last_error = e

                if attempt >= self.max_retries:
                    raise

                if not self._is_transient_error(e):
                    raise

                delay = self.base_delay * (2 ** attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RetryExhaustedError("Evaluation failed: retry loop completed without success or captured error")

    def _evaluate_once(self, task: WorkerTask) -> WorkerResult:
        """
        Execute a single evaluation attempt.

        Args:
            task: WorkerTask to evaluate

        Returns:
            WorkerResult from evaluation
        """
        # Step 1: Apply parameters
        # Note: proc_output_dir, sim_dir, proc_id must be passed explicitly because
        # they're excluded from additional_data (since they're primary fields in WorkerTask)
        if not self.apply_parameters(
            task.params,
            task.settings_dir,
            config=task.config,
            proc_output_dir=task.output_dir,
            output_dir=task.output_dir,
            sim_dir=task.sim_dir,
            proc_id=task.proc_id,
            **task.additional_data
        ):
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error="Failed to apply parameters",
                penalty_score=self.penalty_score
            )

        # Step 2: Run model
        if not self.run_model(
            task.config,
            task.settings_dir,
            task.output_dir,
            sim_dir=task.sim_dir,
            proc_id=task.proc_id,
            params=task.params,
            **task.additional_data
        ):
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error="Model execution failed",
                penalty_score=self.penalty_score
            )

        # Step 3: Calculate metrics
        metrics = self.calculate_metrics(
            task.output_dir,
            task.config,
            sim_dir=task.sim_dir,
            proc_id=task.proc_id,
            **task.additional_data
        )

        # Determine primary score
        score = self._extract_primary_score(metrics, task.config)

        return WorkerResult(
            individual_id=task.individual_id,
            params=task.params,
            score=score,
            metrics=metrics,
            iteration=task.iteration
        )

    def _extract_primary_score(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> float:
        """
        Extract the primary optimization score from metrics.

        Automatically transforms the score to maximization convention using
        MetricTransformer, so optimization algorithms can always maximize.

        Args:
            metrics: Dictionary of calculated metrics
            config: Configuration dictionary

        Returns:
            Primary score for optimization (transformed for maximization)
        """
        # Get configured metric name - check OPTIMIZATION_METRIC first, then CALIBRATION_METRIC
        # This ensures consistency with gradient-based optimization which uses optimization.metric
        metric_name = config.get(
            'OPTIMIZATION_METRIC',
            config.get('CALIBRATION_METRIC', 'KGE')
        )

        # Debug log to trace metric configuration
        self.logger.debug(
            f"Primary metric extraction: OPTIMIZATION_METRIC={config.get('OPTIMIZATION_METRIC')}, "
            f"CALIBRATION_METRIC={config.get('CALIBRATION_METRIC')}, using metric_name={metric_name}"
        )

        raw_value = None

        # Check for exact match first
        if metric_name in metrics:
            raw_value = metrics[metric_name]
        # Check for Calib_ prefix
        elif f"Calib_{metric_name}" in metrics:
            raw_value = metrics[f"Calib_{metric_name}"]
        else:
            # Case-insensitive search
            metric_lower = metric_name.lower()
            for k, v in metrics.items():
                if k.lower() == metric_lower or k.lower() == f"calib_{metric_lower}":
                    raw_value = v
                    break

            # Try common alternatives if still not found
            if raw_value is None:
                alternatives = ['kge', 'nse', 'score', 'fitness', 'objective']
                for alt in alternatives:
                    if alt in metrics:
                        raw_value = metrics[alt]
                        metric_name = alt  # Update metric_name for correct transformation
                        break
                    for k, v in metrics.items():
                        if k.lower() == alt:
                            raw_value = v
                            metric_name = k
                            break
                    if raw_value is not None:
                        break

        # Return penalty if no metric found
        if raw_value is None:
            self.logger.warning(
                f"Could not find metric '{metric_name}' in results. "
                f"Available keys: {list(metrics.keys())}"
            )
            return self.penalty_score

        # Transform to maximization convention
        # This ensures RMSE, MAE, PBIAS etc. are properly handled
        transformed_score = MetricTransformer.transform_for_maximization(
            metric_name, raw_value
        )

        # Debug: trace score transformation to identify any sign issues
        self.logger.debug(
            f"Score extraction: metric={metric_name}, raw={raw_value:.4f}, "
            f"direction={MetricTransformer.get_direction(metric_name)}, "
            f"transformed={transformed_score:.4f}"
        )

        return transformed_score if transformed_score is not None else self.penalty_score

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Check if an error is likely transient and worth retrying.

        Args:
            error: The exception to check

        Returns:
            True if the error is likely transient
        """
        error_str = str(error).lower()
        return any(te in error_str for te in self.TRANSIENT_ERRORS)

    # =========================================================================
    # Hooks for subclass customization
    # =========================================================================

    def pre_evaluation(self, task: WorkerTask) -> None:
        """
        Hook called before evaluation begins.

        Subclasses can override to perform setup tasks.

        Args:
            task: WorkerTask about to be evaluated
        """
        pass

    def post_evaluation(self, task: WorkerTask, result: WorkerResult) -> None:
        """
        Hook called after evaluation completes.

        Subclasses can override to perform cleanup or logging.

        Args:
            task: WorkerTask that was evaluated
            result: WorkerResult from evaluation
        """
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    def setup_worker_isolation(self) -> None:
        """
        Setup process isolation for worker.

        Sets environment variables to prevent thread contention
        and file locking issues in parallel execution.
        """
        import os

        env_vars = {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'NETCDF_DISABLE_LOCKING': '1',
            'HDF5_USE_FILE_LOCKING': 'FALSE',
            'HDF5_DISABLE_VERSION_CHECK': '1',
        }

        for key, value in env_vars.items():
            os.environ[key] = value

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static method for use with process pools.

        This is a template that subclasses should override to provide
        a picklable worker function.

        Args:
            task_data: Dictionary with task parameters

        Returns:
            Dictionary with results
        """
        raise NotImplementedError(
            "Subclasses must implement evaluate_worker_function"
        )

    # =========================================================================
    # Native Gradient Support (Optional)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """
        Check if this worker supports native gradient computation.

        Native gradients (e.g., via JAX autodiff) can be significantly more
        efficient than finite-difference gradients for gradient-based optimization.
        When supported, gradient computation requires only ~2 model evaluations
        (forward + backward pass) instead of 2N+1 evaluations for N parameters.

        Override this method in subclasses that implement autodiff-capable models.

        Returns:
            True if compute_gradient() and evaluate_with_gradient() are available
            and functional. Default: False (use finite differences).

        Example:
            >>> class HBVWorker(BaseWorker):
            ...     def supports_native_gradients(self) -> bool:
            ...         return HAS_JAX  # True if JAX is installed
        """
        return False

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to parameters using native method.

        This method should be overridden by workers that support autodiff
        (e.g., JAX, PyTorch). The gradient is computed for the loss function
        (negative of the objective metric), so for maximizing KGE, gradients
        point in the direction of decreasing KGE (increasing loss).

        Args:
            params: Dictionary mapping parameter names to current values
            metric: Objective metric to compute gradient for ('kge', 'nse', etc.)

        Returns:
            Dictionary mapping parameter names to gradient values (d(loss)/d(param)),
            or None if native gradients are not supported.

        Note:
            - Gradients are for the LOSS (negative metric), not the metric itself
            - For maximization problems, negate the gradient for gradient ascent
            - Returns None by default; override in autodiff-capable workers

        Example:
            >>> worker = HBVWorker(config, logger)
            >>> if worker.supports_native_gradients():
            ...     grads = worker.compute_gradient({'fc': 250.0, 'k1': 0.1}, 'kge')
            ...     print(grads)  # {'fc': -0.001, 'k1': 0.05, ...}
        """
        return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Evaluate loss and compute gradient in a single pass.

        This is more efficient than calling evaluate + compute_gradient separately
        when using autodiff, as the forward pass computation can be shared.
        Uses jax.value_and_grad or torch.autograd for efficient computation.

        Args:
            params: Dictionary mapping parameter names to current values
            metric: Objective metric ('kge', 'nse', etc.)

        Returns:
            Tuple of (loss_value, gradient_dict):
            - loss_value: Scalar loss (negative of metric, for minimization)
            - gradient_dict: Dictionary mapping parameter names to gradients,
              or None if native gradients not supported

        Note:
            - Returns (loss, None) by default; override in autodiff-capable workers
            - loss is NEGATIVE of metric (e.g., -KGE) for minimization
            - Subclasses should use value_and_grad for efficiency

        Example:
            >>> worker = HBVWorker(config, logger)
            >>> loss, grads = worker.evaluate_with_gradient({'fc': 250.0}, 'kge')
            >>> print(f"Loss: {loss}, Gradients: {grads}")
        """
        # Default implementation: not supported
        # Subclasses override with actual autodiff implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native gradients. "
            "Override evaluate_with_gradient() or use finite differences."
        )
