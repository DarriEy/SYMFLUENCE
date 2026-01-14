"""
Validation utilities for SYMFLUENCE.

Provides standardized validation helpers for configuration, files, directories,
geospatial data, NetCDF files, and numeric parameters.
"""

from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Sequence
import logging

from symfluence.core.exceptions import (
    ConfigurationError,
    FileOperationError,
    ValidationError,
    GeospatialError
)


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


# =============================================================================
# Bounding Box Validation
# =============================================================================

def validate_bounding_box(
    bbox: Dict[str, float],
    context: str = "bounding box validation",
    allow_global: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Validate a geographic bounding box with comprehensive checks.

    Args:
        bbox: Dictionary with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
        context: Description of where this bbox is used (for error messages)
        allow_global: If True, allows very large bounding boxes
        logger: Optional logger for warnings

    Returns:
        The validated bbox dictionary

    Raises:
        ValidationError: If bbox is invalid
    """
    required_keys = {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
    missing = required_keys - set(bbox.keys())
    if missing:
        raise ValidationError(f"Bounding box missing keys for {context}: {missing}")

    # Check for None/NaN values
    import math
    for key in required_keys:
        val = bbox[key]
        if val is None:
            raise ValidationError(f"Bounding box has None value for '{key}' in {context}")
        if isinstance(val, float) and math.isnan(val):
            raise ValidationError(f"Bounding box has NaN value for '{key}' in {context}")

    lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']

    # Validate latitude range
    if lat_min < -90 or lat_min > 90:
        raise ValidationError(f"lat_min ({lat_min}) out of range [-90, 90] in {context}")
    if lat_max < -90 or lat_max > 90:
        raise ValidationError(f"lat_max ({lat_max}) out of range [-90, 90] in {context}")
    if lat_min >= lat_max:
        raise ValidationError(f"lat_min ({lat_min}) >= lat_max ({lat_max}) in {context}")

    # Validate longitude range (allow both -180/180 and 0/360 conventions)
    if lon_min < -180 or lon_min > 360:
        raise ValidationError(f"lon_min ({lon_min}) out of valid range in {context}")
    if lon_max < -180 or lon_max > 360:
        raise ValidationError(f"lon_max ({lon_max}) out of valid range in {context}")

    # Check for very small bounding boxes
    lat_extent = lat_max - lat_min
    lon_extent = lon_max - lon_min
    if lon_extent < 0:
        lon_extent += 360  # Handle wraparound

    if lat_extent < 0.001 or lon_extent < 0.001:
        if logger:
            logger.warning(
                f"Bounding box is very small ({lat_extent:.4f}° x {lon_extent:.4f}°) "
                f"in {context}. May not cover any data grid cells."
            )

    # Check for very large bounding boxes
    if not allow_global and (lat_extent > 90 or lon_extent > 180):
        if logger:
            logger.warning(
                f"Bounding box is very large ({lat_extent:.1f}° x {lon_extent:.1f}°) "
                f"in {context}. This may result in large data downloads."
            )

    return bbox


# =============================================================================
# Numeric Range Validation
# =============================================================================

def validate_numeric_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    param_name: str = "parameter",
    context: str = "validation"
) -> Union[int, float]:
    """
    Validate a numeric value is within specified bounds.

    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive), None for no lower bound
        max_val: Maximum allowed value (inclusive), None for no upper bound
        param_name: Name of the parameter (for error messages)
        context: Context where validation is happening

    Returns:
        The validated value

    Raises:
        ValidationError: If value is out of range or invalid
    """
    import math

    if value is None:
        raise ValidationError(f"{param_name} cannot be None in {context}")

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValidationError(f"{param_name} has invalid value ({value}) in {context}")

    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{param_name} ({value}) is below minimum ({min_val}) in {context}"
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{param_name} ({value}) exceeds maximum ({max_val}) in {context}"
        )

    return value


def validate_positive(
    value: Union[int, float],
    param_name: str = "parameter",
    context: str = "validation",
    allow_zero: bool = False
) -> Union[int, float]:
    """
    Validate a value is positive (or non-negative).

    Args:
        value: The value to validate
        param_name: Name of the parameter
        context: Context for error messages
        allow_zero: If True, zero is allowed

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not positive/non-negative
    """
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{param_name} ({value}) must be non-negative in {context}")
    else:
        if value <= 0:
            raise ValidationError(f"{param_name} ({value}) must be positive in {context}")
    return value


# =============================================================================
# Date/Time Validation
# =============================================================================

def validate_date_range(
    start_date: Any,
    end_date: Any,
    context: str = "date validation",
    max_span_days: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> tuple:
    """
    Validate a date range.

    Args:
        start_date: Start date (string or datetime-like)
        end_date: End date (string or datetime-like)
        context: Context for error messages
        max_span_days: Optional maximum allowed span in days
        logger: Optional logger for warnings

    Returns:
        Tuple of (parsed_start, parsed_end) as pandas Timestamps

    Raises:
        ValidationError: If dates are invalid
    """
    import pandas as pd

    if start_date is None or end_date is None:
        raise ValidationError(f"Start and end dates are required for {context}")

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValidationError(f"Could not parse dates for {context}: {e}") from e

    if pd.isna(start) or pd.isna(end):
        raise ValidationError(
            f"Invalid date values for {context}: start={start_date}, end={end_date}"
        )

    if start >= end:
        raise ValidationError(
            f"Start date ({start}) must be before end date ({end}) in {context}"
        )

    span_days = (end - start).days
    if max_span_days is not None and span_days > max_span_days:
        raise ValidationError(
            f"Date range ({span_days} days) exceeds maximum ({max_span_days} days) "
            f"in {context}"
        )

    # Warn about very long spans
    if logger and span_days > 36500:  # ~100 years
        logger.warning(f"Date range is very long ({span_days} days) in {context}")

    return start, end


# =============================================================================
# NetCDF Validation
# =============================================================================

def validate_netcdf_variables(
    dataset: Any,
    required_vars: Sequence[str],
    context: str = "NetCDF validation",
    any_of: bool = False
) -> List[str]:
    """
    Validate that required variables exist in a NetCDF dataset.

    Args:
        dataset: xarray Dataset or netCDF4 Dataset
        required_vars: List of required variable names
        context: Context for error messages
        any_of: If True, require at least one variable; if False, require all

    Returns:
        List of present variables from required_vars

    Raises:
        ValidationError: If required variables are missing
    """
    # Handle both xarray and netCDF4
    if hasattr(dataset, 'data_vars'):
        available = set(dataset.data_vars)
    elif hasattr(dataset, 'variables'):
        available = set(dataset.variables.keys())
    else:
        raise ValidationError(f"Unknown dataset type in {context}")

    present = [v for v in required_vars if v in available]
    missing = [v for v in required_vars if v not in available]

    if any_of:
        if not present:
            raise ValidationError(
                f"NetCDF missing all expected variables in {context}. "
                f"Expected at least one of: {list(required_vars)}. "
                f"Available: {sorted(available)}"
            )
    else:
        if missing:
            raise ValidationError(
                f"NetCDF missing required variables in {context}: {missing}. "
                f"Available: {sorted(available)}"
            )

    return present


def validate_netcdf_dimensions(
    dataset: Any,
    required_dims: Sequence[str],
    context: str = "NetCDF validation"
) -> None:
    """
    Validate that required dimensions exist in a NetCDF dataset.

    Args:
        dataset: xarray Dataset or netCDF4 Dataset
        required_dims: List of required dimension names
        context: Context for error messages

    Raises:
        ValidationError: If required dimensions are missing
    """
    # Handle both xarray and netCDF4
    if hasattr(dataset, 'dims'):
        available = set(dataset.dims)
    elif hasattr(dataset, 'dimensions'):
        available = set(dataset.dimensions.keys())
    else:
        raise ValidationError(f"Unknown dataset type in {context}")

    missing = [d for d in required_dims if d not in available]

    if missing:
        raise ValidationError(
            f"NetCDF missing required dimensions in {context}: {missing}. "
            f"Available: {sorted(available)}"
        )


def validate_netcdf_coordinates(
    dataset: Any,
    lat_name: str = 'latitude',
    lon_name: str = 'longitude',
    context: str = "NetCDF validation"
) -> None:
    """
    Validate coordinate arrays in a NetCDF dataset.

    Args:
        dataset: xarray Dataset
        lat_name: Name of latitude coordinate
        lon_name: Name of longitude coordinate
        context: Context for error messages

    Raises:
        ValidationError: If coordinates are invalid
    """
    import numpy as np

    # Check coordinates exist
    if lat_name not in dataset.coords:
        raise ValidationError(f"Missing latitude coordinate '{lat_name}' in {context}")
    if lon_name not in dataset.coords:
        raise ValidationError(f"Missing longitude coordinate '{lon_name}' in {context}")

    lat = dataset[lat_name].values
    lon = dataset[lon_name].values

    # Check not empty
    if len(lat) == 0:
        raise ValidationError(f"Empty latitude array in {context}")
    if len(lon) == 0:
        raise ValidationError(f"Empty longitude array in {context}")

    # Check for NaN values
    if np.any(np.isnan(lat)):
        raise ValidationError(f"Latitude array contains NaN values in {context}")
    if np.any(np.isnan(lon)):
        raise ValidationError(f"Longitude array contains NaN values in {context}")

    # Check latitude bounds
    if np.min(lat) < -90 or np.max(lat) > 90:
        raise ValidationError(
            f"Latitude values out of range [-90, 90] in {context}: "
            f"[{np.min(lat):.2f}, {np.max(lat):.2f}]"
        )


# =============================================================================
# GeoDataFrame Validation
# =============================================================================

def validate_geodataframe(
    gdf: Any,
    required_columns: Optional[Sequence[str]] = None,
    require_crs: bool = True,
    require_valid_geometry: bool = True,
    context: str = "GeoDataFrame validation",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Validate a GeoDataFrame has required properties.

    Args:
        gdf: GeoDataFrame to validate
        required_columns: List of required column names
        require_crs: If True, require a defined CRS
        require_valid_geometry: If True, check all geometries are valid
        context: Context for error messages
        logger: Optional logger for warnings

    Raises:
        GeospatialError: If validation fails
    """
    if gdf is None:
        raise GeospatialError(f"GeoDataFrame is None in {context}")

    if gdf.empty:
        raise GeospatialError(f"GeoDataFrame is empty in {context}")

    # Check for geometry column
    if 'geometry' not in gdf.columns:
        raise GeospatialError(f"GeoDataFrame missing 'geometry' column in {context}")

    # Check required columns
    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            raise GeospatialError(
                f"GeoDataFrame missing required columns in {context}: {missing}"
            )

    # Check CRS
    if require_crs and gdf.crs is None:
        raise GeospatialError(
            f"GeoDataFrame has undefined CRS in {context}. "
            "Set CRS with gdf.set_crs() or ensure source file has CRS defined."
        )

    # Check geometry validity
    if require_valid_geometry:
        invalid_count = (~gdf.geometry.is_valid).sum()
        if invalid_count > 0:
            if logger:
                logger.warning(
                    f"GeoDataFrame has {invalid_count} invalid geometries in {context}. "
                    "Consider using gdf.geometry.make_valid()."
                )

    # Check for empty geometries
    empty_count = gdf.geometry.is_empty.sum()
    if empty_count > 0:
        if logger:
            logger.warning(f"GeoDataFrame has {empty_count} empty geometries in {context}")


def validate_shapefile_field(
    gdf: Any,
    field_name: str,
    expected_type: Optional[str] = None,
    allow_nulls: bool = False,
    check_unique: bool = False,
    context: str = "shapefile validation"
) -> None:
    """
    Validate a specific field in a GeoDataFrame/shapefile.

    Args:
        gdf: GeoDataFrame to validate
        field_name: Name of the field to validate
        expected_type: Expected dtype ('int', 'float', 'str', 'numeric')
        allow_nulls: If False, raise error if nulls are present
        check_unique: If True, verify all values are unique
        context: Context for error messages

    Raises:
        GeospatialError: If validation fails
    """
    import pandas as pd

    if field_name not in gdf.columns:
        raise GeospatialError(f"Field '{field_name}' not found in {context}")

    series = gdf[field_name]

    # Check for nulls
    if not allow_nulls and series.isna().any():
        null_count = series.isna().sum()
        raise GeospatialError(
            f"Field '{field_name}' has {null_count} null values in {context}"
        )

    # Check uniqueness
    if check_unique:
        if series.duplicated().any():
            dup_count = series.duplicated().sum()
            raise GeospatialError(
                f"Field '{field_name}' has {dup_count} duplicate values in {context}"
            )

    # Check type
    if expected_type:
        if expected_type == 'int':
            if not pd.api.types.is_integer_dtype(series):
                raise GeospatialError(
                    f"Field '{field_name}' expected int type, got {series.dtype} in {context}"
                )
        elif expected_type == 'float':
            if not pd.api.types.is_float_dtype(series):
                raise GeospatialError(
                    f"Field '{field_name}' expected float type, got {series.dtype} in {context}"
                )
        elif expected_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(series):
                raise GeospatialError(
                    f"Field '{field_name}' expected numeric type, got {series.dtype} in {context}"
                )
        elif expected_type == 'str':
            if not pd.api.types.is_string_dtype(series) and series.dtype != 'object':
                raise GeospatialError(
                    f"Field '{field_name}' expected string type, got {series.dtype} in {context}"
                )
