"""
Custom exception hierarchy for SYMFLUENCE.

This module defines a hierarchy of exceptions that provide clear, specific
error types for different failure modes throughout the SYMFLUENCE framework.
"""

from contextlib import contextmanager
from typing import Optional, Any
import logging


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


from symfluence.core.validation import (
    validate_config_keys,
    validate_file_exists,
    validate_directory_exists
)


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


