"""
Base Worker Module

Provides abstract base class and data structures for optimization workers.
Workers handle the evaluation of parameter sets by running model simulations
and calculating objective metrics.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

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
        if not config:
            # Extract config keys from task_data itself
            config_keys = [
                'EXPERIMENT_ID', 'DOMAIN_NAME', 'ROOT_PATH', 'HYDROLOGICAL_MODEL',
                'ROUTING_MODEL', 'CALIBRATION_METRIC', 'CALIBRATION_PERIOD',
                'EVALUATION_PERIOD', 'MPI_PROCESSES', 'DOMAIN_DEFINITION_METHOD'
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
        penalty_score: float = -999.0
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
    """
    Abstract base class for optimization workers.

    Workers are responsible for:
    1. Applying parameters to model configuration files
    2. Running model simulations
    3. Calculating objective metrics from model outputs

    Subclasses must implement the abstract methods for model-specific behavior.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        max_retries: Maximum retry attempts for transient failures
        base_delay: Base delay for exponential backoff
    """

    # Default retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 0.5
    DEFAULT_PENALTY_SCORE = -999.0

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
    ) -> Dict[str, float]:
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

        except Exception as e:
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

            except Exception as e:
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

        raise last_error

    def _evaluate_once(self, task: WorkerTask) -> WorkerResult:
        """
        Execute a single evaluation attempt.

        Args:
            task: WorkerTask to evaluate

        Returns:
            WorkerResult from evaluation
        """
        # Step 1: Apply parameters
        if not self.apply_parameters(
            task.params,
            task.settings_dir,
            config=task.config,
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

        Args:
            metrics: Dictionary of calculated metrics
            config: Configuration dictionary

        Returns:
            Primary score for optimization
        """
        # Get configured metric name
        metric_name = config.get('CALIBRATION_METRIC', 'KGE')
        
        # Check for exact match first
        if metric_name in metrics:
            return metrics[metric_name]
            
        # Check for Calib_ prefix
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics:
            return metrics[calib_key]

        # Case-insensitive search
        metric_lower = metric_name.lower()
        for k, v in metrics.items():
            if k.lower() == metric_lower:
                return v
            if k.lower() == f"calib_{metric_lower}":
                return v

        # Try common alternatives
        alternatives = ['kge', 'nse', 'score', 'fitness', 'objective']
        for alt in alternatives:
            # Check exact and lower
            if alt in metrics:
                return metrics[alt]
            for k, v in metrics.items():
                if k.lower() == alt:
                    return v

        # Return penalty if no metric found
        self.logger.warning(f"Could not find metric '{metric_name}' in results. Available keys: {list(metrics.keys())}")
        return self.penalty_score

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
