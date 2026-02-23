"""
Custom exception hierarchy for SYMFLUENCE.

This module defines a hierarchy of exceptions that provide clear, specific
error types for different failure modes throughout the SYMFLUENCE framework.
"""

import logging
from contextlib import contextmanager
from typing import Optional


class SYMFLUENCEError(Exception):
    """
    Base exception for all SYMFLUENCE-specific errors.

    All custom exceptions in SYMFLUENCE should inherit from this class.
    This allows catching all SYMFLUENCE errors with a single except clause.
    """
    pass


class ConfigurationError(SYMFLUENCEError):
    """
    Configuration-related errors.

    Raised when:
    - Required configuration keys are missing
    - Configuration values are invalid
    - Configuration file cannot be loaded or parsed
    """
    pass


class ModelExecutionError(SYMFLUENCEError):
    """
    Model execution failures.

    Raised when:
    - Model preprocessor fails
    - Model runner encounters execution errors
    - Model postprocessor fails
    - Model binary not found or not executable
    """
    pass


class DataAcquisitionError(SYMFLUENCEError):
    """
    Data download/processing failures.

    Raised when:
    - Forcing data download fails
    - Observation data cannot be retrieved
    - Data conversion/processing fails
    - Remote data service unavailable
    """
    pass


class OptimizationError(SYMFLUENCEError):
    """
    Calibration/optimization failures.

    Raised when:
    - Optimizer fails to converge
    - Parameter bounds are invalid
    - Objective function evaluation fails
    - Optimization algorithm encounters errors
    """
    pass


class WorkerExecutionError(OptimizationError):
    """
    Worker execution failures during optimization.

    Raised when:
    - Worker fails to execute model evaluation
    - Parameter application fails
    - Model run fails within worker context
    - Metric calculation fails
    """
    pass


class RetryExhaustedError(OptimizationError):
    """
    All retry attempts have been exhausted.

    Raised when:
    - A retriable operation has failed all retry attempts
    - Transient errors persist beyond the retry limit
    - The retry loop completes without success or captured error
    """
    pass


class GeospatialError(SYMFLUENCEError):
    """
    Geospatial processing failures.

    Raised when:
    - Shapefile processing fails
    - Coordinate transformation errors
    - DEM processing fails
    - Spatial intersection errors
    """
    pass


class ValidationError(SYMFLUENCEError):
    """
    Data or parameter validation failures.

    Raised when:
    - Input data fails validation checks
    - Parameter values are out of acceptable range
    - File format validation fails
    """
    pass


class FileOperationError(SYMFLUENCEError):
    """
    File I/O operation failures.

    Raised when:
    - Required file not found
    - File cannot be read or written
    - Directory creation fails
    - File permissions are insufficient
    """
    pass


class DiscretizationError(GeospatialError):
    """
    Domain discretization failures.

    Raised when:
    - HRU creation fails
    - Geometry operations fail during discretization
    - Raster-to-vector conversion errors
    - Invalid discretization method specified
    """
    pass


class ShapefileError(GeospatialError):
    """
    Shapefile I/O failures.

    Raised when:
    - Shapefile cannot be read or written
    - Required columns missing from shapefile
    - CRS transformation fails
    - Geometry validation fails
    """
    pass


class RasterProcessingError(GeospatialError):
    """
    Raster processing failures.

    Raised when:
    - Raster mask extraction fails
    - Zonal statistics calculation fails
    - Raster resampling or reprojection fails
    - Invalid raster data or metadata
    """
    pass


class CodeAnalysisError(SYMFLUENCEError):
    """
    Agent code analysis failures.

    Raised when:
    - Code search subprocess fails
    - File parsing errors occur
    - Repository analysis fails
    """
    pass


class ConfigValidationError(ConfigurationError):
    """
    Configuration validation failures.

    Raised when:
    - Model configuration fails schema validation
    - Required configuration fields have invalid values
    - Cross-field validation constraints are violated
    """
    pass


class EvaluationError(SYMFLUENCEError):
    """
    Model evaluation and analysis failures.

    Raised when:
    - Benchmarking analysis fails
    - Sensitivity analysis encounters errors
    - Decision analysis fails
    - Metric calculation errors
    """
    pass


class ReportingError(SYMFLUENCEError):
    """
    Visualization and reporting failures.

    Raised when:
    - Plot generation fails
    - Report creation encounters errors
    - Visualization data is missing or invalid
    """
    pass


# =============================================================================
# Validation Helpers
# =============================================================================

from typing import TypeVar

T = TypeVar('T')


def require(condition: bool, message: str, error_type: type = None) -> None:
    """
    Validate a condition, raising an exception if it fails.

    This replaces assert statements with proper validation that cannot be
    disabled with python -O.

    Args:
        condition: The condition that must be True
        message: Error message if condition is False
        error_type: Exception type to raise (default: ValidationError)

    Raises:
        ValidationError (or specified error_type) if condition is False

    Example:
        >>> require(len(params) > 0, "Parameters cannot be empty")
        >>> require(value >= 0, "Value must be non-negative", ValueError)
    """
    if error_type is None:
        error_type = ValidationError
    if not condition:
        raise error_type(message)


def require_not_none(value: Optional[T], name: str, error_type: type = None) -> T:
    """
    Validate that a value is not None, returning it if valid.

    This replaces the pattern:
        assert x is not None
        return x

    With:
        return require_not_none(x, "x")

    Args:
        value: The value to check
        name: Name of the value (for error message)
        error_type: Exception type to raise (default: ValidationError)

    Returns:
        The value if it is not None

    Raises:
        ValidationError (or specified error_type) if value is None

    Example:
        >>> task_builder = require_not_none(self._task_builder, "task_builder")
    """
    if error_type is None:
        error_type = ValidationError
    if value is None:
        raise error_type(f"{name} must not be None")
    return value


@contextmanager
def symfluence_error_handler(
    operation: str,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    error_type: type = SYMFLUENCEError
):
    """
    Context manager for standardized error handling.

    Provides consistent error handling across SYMFLUENCE, with logging,
    error type conversion, and optional re-raising.

    Args:
        operation: Description of the operation being performed (for logging)
        logger: Logger instance for error messages. If None, errors are not logged.
        reraise: Whether to re-raise the exception after handling (default: True)
        error_type: SYMFLUENCE exception type to convert generic exceptions to

    Raises:
        The original exception if it's already a SYMFLUENCEError, or the
        specified error_type if reraise=True

    Example:
        >>> with symfluence_error_handler("model preprocessing", logger, error_type=ModelExecutionError):
        ...     preprocessor.run_preprocessing()

        >>> with symfluence_error_handler("data download", logger, reraise=False):
        ...     downloader.fetch_data()  # Logs error but doesn't raise
    """
    try:
        yield
    except SYMFLUENCEError:
        # Already a SYMFLUENCE error, just re-raise as-is
        if logger:
            logger.error(f"Error during {operation}", exc_info=True)
        if reraise:
            raise
    except Exception as e:
        # Convert generic exception to SYMFLUENCE exception
        if logger:
            logger.error(f"Error during {operation}: {e}", exc_info=True)
        if reraise:
            raise error_type(f"Failed during {operation}: {e}") from e


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base
    'SYMFLUENCEError',
    # Domain exceptions
    'ConfigurationError',
    'ConfigValidationError',
    'ModelExecutionError',
    'DataAcquisitionError',
    'OptimizationError',
    'WorkerExecutionError',
    'RetryExhaustedError',
    'GeospatialError',
    'ValidationError',
    'FileOperationError',
    'DiscretizationError',
    'ShapefileError',
    'RasterProcessingError',
    'CodeAnalysisError',
    'EvaluationError',
    'ReportingError',
    # Helpers
    'require',
    'require_not_none',
    'symfluence_error_handler',
]
