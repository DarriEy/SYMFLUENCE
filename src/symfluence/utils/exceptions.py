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


def validate_config_keys(
    config: dict,
    required_keys: list,
    operation: str = "configuration validation"
) -> None:
    """
    Validate that all required configuration keys are present.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names
        operation: Description of operation requiring these keys (for error message)

    Raises:
        ConfigurationError: If any required keys are missing

    Example:
        >>> validate_config_keys(
        ...     config,
        ...     ['DOMAIN_NAME', 'FORCING_DATASET'],
        ...     "model preprocessing"
        ... )
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration keys for {operation}: "
            f"{', '.join(missing_keys)}"
        )


def validate_file_exists(
    file_path,
    file_description: str = "file"
) -> None:
    """
    Validate that a file exists and is readable.

    Args:
        file_path: Path to file (str or Path object)
        file_description: Human-readable description of the file

    Raises:
        FileOperationError: If file doesn't exist or isn't readable

    Example:
        >>> validate_file_exists(config_file, "configuration file")
    """
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise FileOperationError(
            f"Required {file_description} not found: {file_path}"
        )

    if not path.is_file():
        raise FileOperationError(
            f"{file_description} is not a file: {file_path}"
        )

    # Check if readable
    try:
        with open(path, 'r') as f:
            pass
    except PermissionError:
        raise FileOperationError(
            f"Permission denied reading {file_description}: {file_path}"
        )
    except Exception as e:
        raise FileOperationError(
            f"Cannot read {file_description} {file_path}: {e}"
        )


def validate_directory_exists(
    dir_path,
    dir_description: str = "directory",
    create_if_missing: bool = False
) -> None:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory (str or Path object)
        dir_description: Human-readable description of the directory
        create_if_missing: If True, create directory if it doesn't exist

    Raises:
        FileOperationError: If directory doesn't exist and create_if_missing=False

    Example:
        >>> validate_directory_exists(output_dir, "output directory", create_if_missing=True)
    """
    from pathlib import Path

    path = Path(dir_path)

    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileOperationError(
                    f"Cannot create {dir_description} {dir_path}: {e}"
                )
        else:
            raise FileOperationError(
                f"Required {dir_description} not found: {dir_path}"
            )

    if not path.is_dir():
        raise FileOperationError(
            f"{dir_description} is not a directory: {dir_path}"
        )
