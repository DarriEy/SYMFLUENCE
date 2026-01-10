"""
Validation utilities for SYMFLUENCE CLI arguments.

This module contains validation functions for various argument types used
across the SYMFLUENCE CLI commands. All validators return Result[T] for
consistent error handling.
"""

from pathlib import Path
from typing import Tuple

from symfluence.core.result import Result, ValidationError


# Type aliases for validated values
Coordinates = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]


def validate_coordinates(coord_string: str) -> Result[Coordinates]:
    """
    Validate coordinate string format.

    Args:
        coord_string: Coordinate string in format "lat/lon"

    Returns:
        Result containing (lat, lon) tuple if valid, or ValidationError if invalid.

    Example:
        >>> result = validate_coordinates("51.1722/-115.5717")
        >>> if result.is_ok:
        ...     lat, lon = result.unwrap()
    """
    try:
        parts = coord_string.split('/')
        if len(parts) != 2:
            return Result.err(ValidationError(
                field="coordinates",
                message="Expected format: lat/lon",
                value=coord_string,
                suggestion="Use format like: 51.1722/-115.5717",
            ))

        lat, lon = float(parts[0]), float(parts[1])

        # Latitude range validation
        if not (-90 <= lat <= 90):
            return Result.err(ValidationError(
                field="latitude",
                message=f"Latitude {lat} out of range [-90, 90]",
                value=lat,
            ))

        # Longitude range validation
        if not (-180 <= lon <= 180):
            return Result.err(ValidationError(
                field="longitude",
                message=f"Longitude {lon} out of range [-180, 180]",
                value=lon,
            ))

        return Result.ok((lat, lon))
    except (ValueError, IndexError):
        return Result.err(ValidationError(
            field="coordinates",
            message="Coordinates must be numeric in format: lat/lon",
            value=coord_string,
        ))


def validate_bounding_box(bbox_string: str) -> Result[BoundingBox]:
    """
    Validate bounding box coordinate string format.

    Args:
        bbox_string: Bounding box string in format "lat_max/lon_min/lat_min/lon_max"

    Returns:
        Result containing (lat_max, lon_min, lat_min, lon_max) tuple if valid.

    Example:
        >>> result = validate_bounding_box("55.0/10.0/45.0/20.0")
        >>> if result.is_ok:
        ...     lat_max, lon_min, lat_min, lon_max = result.unwrap()
    """
    try:
        parts = bbox_string.split('/')
        if len(parts) != 4:
            return Result.err(ValidationError(
                field="bounding_box",
                message="Expected format: lat_max/lon_min/lat_min/lon_max",
                value=bbox_string,
                suggestion="Use format like: 55.0/10.0/45.0/20.0",
            ))

        lat_max, lon_min, lat_min, lon_max = map(float, parts)

        # Latitude range and logic validation
        if not (-90 <= lat_min <= 90):
            return Result.err(ValidationError(
                field="lat_min",
                message=f"lat_min {lat_min} out of range [-90, 90]",
                value=lat_min,
            ))
        if not (-90 <= lat_max <= 90):
            return Result.err(ValidationError(
                field="lat_max",
                message=f"lat_max {lat_max} out of range [-90, 90]",
                value=lat_max,
            ))
        if lat_min >= lat_max:
            return Result.err(ValidationError(
                field="bounding_box",
                message=f"lat_min ({lat_min}) must be less than lat_max ({lat_max})",
                value=bbox_string,
            ))

        # Longitude range and logic validation
        if not (-180 <= lon_min <= 180):
            return Result.err(ValidationError(
                field="lon_min",
                message=f"lon_min {lon_min} out of range [-180, 180]",
                value=lon_min,
            ))
        if not (-180 <= lon_max <= 180):
            return Result.err(ValidationError(
                field="lon_max",
                message=f"lon_max {lon_max} out of range [-180, 180]",
                value=lon_max,
            ))
        if lon_min >= lon_max:
            return Result.err(ValidationError(
                field="bounding_box",
                message=f"lon_min ({lon_min}) must be less than lon_max ({lon_max})",
                value=bbox_string,
            ))

        return Result.ok((lat_max, lon_min, lat_min, lon_max))
    except (ValueError, IndexError):
        return Result.err(ValidationError(
            field="bounding_box",
            message="Bounding box coordinates must be numeric in format: lat_max/lon_min/lat_min/lon_max",
            value=bbox_string,
        ))


def validate_config_exists(config_path: str) -> Result[Path]:
    """
    Validate that a configuration file exists.

    Args:
        config_path: Path to configuration file

    Returns:
        Result containing Path if valid, or ValidationError if not found.
    """
    path = Path(config_path)
    if not path.exists():
        return Result.err(ValidationError(
            field="config",
            message=f"Config file not found: {config_path}",
            value=config_path,
        ))
    if not path.is_file():
        return Result.err(ValidationError(
            field="config",
            message=f"Config path is not a file: {config_path}",
            value=config_path,
        ))
    return Result.ok(path)


def validate_file_exists(file_path: str, file_type: str = "File") -> Result[Path]:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Description of file type for error messages (e.g., "Template", "Script")

    Returns:
        Result containing Path if valid, or ValidationError if not found.
    """
    path = Path(file_path)
    if not path.exists():
        return Result.err(ValidationError(
            field=file_type.lower(),
            message=f"{file_type} not found: {file_path}",
            value=file_path,
        ))
    if not path.is_file():
        return Result.err(ValidationError(
            field=file_type.lower(),
            message=f"Path is not a file: {file_path}",
            value=file_path,
        ))
    return Result.ok(path)


def validate_directory_exists(dir_path: str, dir_type: str = "Directory") -> Result[Path]:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory
        dir_type: Description of directory type for error messages

    Returns:
        Result containing Path if valid, or ValidationError if not found.
    """
    path = Path(dir_path)
    if not path.exists():
        return Result.err(ValidationError(
            field=dir_type.lower(),
            message=f"{dir_type} not found: {dir_path}",
            value=dir_path,
        ))
    if not path.is_dir():
        return Result.err(ValidationError(
            field=dir_type.lower(),
            message=f"Path is not a directory: {dir_path}",
            value=dir_path,
        ))
    return Result.ok(path)
