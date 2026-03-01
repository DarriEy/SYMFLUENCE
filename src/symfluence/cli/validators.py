# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

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


def validate_identifier(value: str) -> Result[str]:
    """
    Validate an identifier (alphanumeric with underscores).

    Identifiers must start with a letter or underscore and contain
    only alphanumeric characters and underscores.

    Args:
        value: String to validate

    Returns:
        Result containing the value if valid, or ValidationError if invalid.

    Example:
        >>> result = validate_identifier("my_domain_1")
        >>> if result.is_ok:
        ...     print(f"Valid: {result.unwrap()}")
    """
    import re

    if not value:
        return Result.err(ValidationError(
            field="identifier",
            message="Identifier cannot be empty",
            value=value,
        ))

    # Must start with letter or underscore, then alphanumeric/underscore
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    if not re.match(pattern, value):
        return Result.err(ValidationError(
            field="identifier",
            message="Invalid identifier format",
            value=value,
            suggestion="Use only letters, numbers, and underscores. Must start with a letter or underscore.",
        ))

    return Result.ok(value)


def validate_date(date_string: str) -> Result[str]:
    """
    Validate date string in YYYY-MM-DD format.

    Args:
        date_string: Date string to validate

    Returns:
        Result containing the date string if valid, or ValidationError if invalid.

    Example:
        >>> result = validate_date("2020-01-15")
        >>> if result.is_ok:
        ...     print(f"Valid date: {result.unwrap()}")
    """
    from datetime import datetime

    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return Result.ok(date_string)
    except ValueError:
        return Result.err(ValidationError(
            field="date",
            message="Invalid date format",
            value=date_string,
            suggestion="Use YYYY-MM-DD format (e.g., 2020-01-15)",
        ))


def validate_date_range(start_date: str, end_date: str) -> Result[Tuple[str, str]]:
    """
    Validate that end date is after start date.

    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format

    Returns:
        Result containing (start_date, end_date) tuple if valid.

    Example:
        >>> result = validate_date_range("2010-01-01", "2020-12-31")
        >>> if result.is_ok:
        ...     start, end = result.unwrap()
    """
    from datetime import datetime

    # First validate individual dates
    start_result = validate_date(start_date)
    if start_result.is_err:
        return Result.err(ValidationError(
            field="start_date",
            message=f"Invalid start date: {start_date}",
            value=start_date,
        ))

    end_result = validate_date(end_date)
    if end_result.is_err:
        return Result.err(ValidationError(
            field="end_date",
            message=f"Invalid end date: {end_date}",
            value=end_date,
        ))

    # Parse and compare
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    if end_dt <= start_dt:
        return Result.err(ValidationError(
            field="date_range",
            message="End date must be after start date",
            value=f"{start_date} to {end_date}",
            suggestion="Ensure the end date is later than the start date",
        ))

    return Result.ok((start_date, end_date))
