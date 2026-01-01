"""
Validation utilities for SYMFLUENCE CLI arguments.

This module contains validation functions for various argument types used
across the SYMFLUENCE CLI commands.
"""

from pathlib import Path
from typing import Tuple, Optional


def validate_coordinates(coord_string: str) -> Tuple[bool, Optional[str]]:
    """
    Validate coordinate string format.

    Args:
        coord_string: Coordinate string in format "lat/lon"

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.

    Example:
        >>> validate_coordinates("51.1722/-115.5717")
        (True, None)
        >>> validate_coordinates("invalid")
        (False, "Expected format: lat/lon")
    """
    try:
        parts = coord_string.split('/')
        if len(parts) != 2:
            return False, "Expected format: lat/lon"

        lat, lon = float(parts[0]), float(parts[1])

        # Latitude range validation
        if not (-90 <= lat <= 90):
            return False, f"Latitude {lat} out of range [-90, 90]"

        # Longitude range validation
        if not (-180 <= lon <= 180):
            return False, f"Longitude {lon} out of range [-180, 180]"

        return True, None
    except (ValueError, IndexError):
        return False, "Coordinates must be numeric in format: lat/lon"


def validate_bounding_box(bbox_string: str) -> Tuple[bool, Optional[str]]:
    """
    Validate bounding box coordinate string format.

    Args:
        bbox_string: Bounding box string in format "lat_max/lon_min/lat_min/lon_max"

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.

    Example:
        >>> validate_bounding_box("55.0/10.0/45.0/20.0")
        (True, None)
    """
    try:
        parts = bbox_string.split('/')
        if len(parts) != 4:
            return False, "Expected format: lat_max/lon_min/lat_min/lon_max"

        lat_max, lon_min, lat_min, lon_max = map(float, parts)

        # Latitude range and logic validation
        if not (-90 <= lat_min <= 90):
            return False, f"lat_min {lat_min} out of range [-90, 90]"
        if not (-90 <= lat_max <= 90):
            return False, f"lat_max {lat_max} out of range [-90, 90]"
        if lat_min >= lat_max:
            return False, f"lat_min ({lat_min}) must be less than lat_max ({lat_max})"

        # Longitude range and logic validation
        if not (-180 <= lon_min <= 180):
            return False, f"lon_min {lon_min} out of range [-180, 180]"
        if not (-180 <= lon_max <= 180):
            return False, f"lon_max {lon_max} out of range [-180, 180]"
        if lon_min >= lon_max:
            return False, f"lon_min ({lon_min}) must be less than lon_max ({lon_max})"

        return True, None
    except (ValueError, IndexError):
        return False, "Bounding box coordinates must be numeric in format: lat_max/lon_min/lat_min/lon_max"


def validate_config_exists(config_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a configuration file exists.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    path = Path(config_path)
    if not path.exists():
        return False, f"Config file not found: {config_path}"
    if not path.is_file():
        return False, f"Config path is not a file: {config_path}"
    return True, None


def validate_file_exists(file_path: str, file_type: str = "File") -> Tuple[bool, Optional[str]]:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Description of file type for error messages (e.g., "Template", "Script")

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"{file_type} not found: {file_path}"
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    return True, None


def validate_directory_exists(dir_path: str, dir_type: str = "Directory") -> Tuple[bool, Optional[str]]:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory
        dir_type: Description of directory type for error messages

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    path = Path(dir_path)
    if not path.exists():
        return False, f"{dir_type} not found: {dir_path}"
    if not path.is_dir():
        return False, f"Path is not a directory: {dir_path}"
    return True, None
