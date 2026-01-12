"""
Validation utilities for SYMFLUENCE.

Provides standardized validation helpers for configuration, files, and directories.
"""

from pathlib import Path
from typing import Dict, List, Any, Union
from symfluence.core.exceptions import ConfigurationError, FileOperationError


def validate_config_keys(
    config: Dict[str, Any],
    required_keys: List[str],
    operation: str = "configuration validation"
) -> None:
    """
    Validate that all required configuration keys are present.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names
        operation: Description of operation requiring these keys

    Raises:
        ConfigurationError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration keys for {operation}: "
            f"{', '.join(missing_keys)}"
        )


def validate_file_exists(
    file_path: Union[str, Path],
    file_description: str = "file"
) -> Path:
    """
    Validate that a file exists and is readable.

    Args:
        file_path: Path to file
        file_description: Human-readable description of the file

    Returns:
        Path object if valid

    Raises:
        FileOperationError: If file doesn't exist or isn't a file
    """
    path = Path(file_path)

    if not path.exists():
        raise FileOperationError(
            f"Required {file_description} not found: {file_path}"
        )

    if not path.is_file():
        raise FileOperationError(
            f"{file_description} is not a file: {file_path}"
        )

    return path


def validate_directory_exists(
    dir_path: Union[str, Path],
    dir_description: str = "directory"
) -> Path:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory
        dir_description: Human-readable description of the directory

    Returns:
        Path object if valid

    Raises:
        FileOperationError: If directory doesn't exist or isn't a directory
    """
    path = Path(dir_path)

    if not path.exists():
        raise FileOperationError(
            f"Required {dir_description} not found: {dir_path}"
        )

    if not path.is_dir():
        raise FileOperationError(
            f"{dir_description} is not a directory: {dir_path}"
        )

    return path