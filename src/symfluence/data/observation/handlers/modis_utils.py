"""
MODIS Observation Handler Utilities.

This module provides shared utilities for MODIS observation handlers
(ET, Snow Cover, etc.) including time conversion, column standardization,
and interpolation functions.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Union

# MODIS fill and invalid values
MODIS_FILL_VALUES = {200, 201, 211, 237, 239, 250, 254, 255}
"""
MODIS special values (invalid for snow cover):
- 200: missing data
- 201: no decision
- 211: night
- 237: inland water
- 239: ocean
- 250: cloud
- 254: detector saturated
- 255: fill value
"""

CLOUD_VALUE = 250
"""MODIS cloud flag value."""

VALID_SNOW_RANGE = (0, 100)
"""Valid range for NDSI snow cover percentage."""

# Standard column mappings for MODIS ET data
MODIS_ET_COLUMN_MAP = {
    'time': 'date',
    'datetime': 'date',
    'Date': 'date',
    'et': 'et_mm_day',
    'ET': 'et_mm_day',
    'et_mm': 'et_mm_day',
    'ET_500m': 'et_mm_day',
    'ET_basin_mean': 'et_mm_day',
    'mean_et_mm': 'et_8day_mm',
    'et_daily_mm': 'et_mm_day'
}


def convert_cftime_to_datetime(time_values) -> pd.DatetimeIndex:
    """
    Convert time values to pandas DatetimeIndex, handling cftime objects.

    This handles various time formats encountered in MODIS NetCDF files:
    - numpy datetime64
    - pandas Timestamp
    - cftime.datetime objects (various calendars)
    - string representations

    Args:
        time_values: Array of time values from xarray/NetCDF

    Returns:
        pandas DatetimeIndex
    """
    # Try direct conversion first (works for numpy datetime64, pandas Timestamp)
    try:
        return pd.to_datetime(time_values)
    except (TypeError, ValueError):
        pass

    # Handle cftime objects
    try:
        import cftime
        if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
            converted = []
            for t in time_values:
                try:
                    # Create standard pandas Timestamp from cftime components
                    dt = pd.Timestamp(
                        year=t.year,
                        month=t.month,
                        day=t.day,
                        hour=getattr(t, 'hour', 0),
                        minute=getattr(t, 'minute', 0),
                        second=getattr(t, 'second', 0)
                    )
                    converted.append(dt)
                except Exception:
                    # Fall back to string parsing (first 10 chars: YYYY-MM-DD)
                    converted.append(pd.to_datetime(str(t)[:10]))
            return pd.DatetimeIndex(converted)
    except ImportError:
        pass

    # Last resort: try string conversion
    return pd.to_datetime([str(t)[:10] for t in time_values])


def standardize_et_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Standardize MODIS ET DataFrame column names.

    Args:
        df: Input DataFrame
        column_map: Custom column mapping. If None, uses MODIS_ET_COLUMN_MAP.

    Returns:
        DataFrame with standardized column names and datetime index
    """
    if column_map is None:
        column_map = MODIS_ET_COLUMN_MAP

    result = df.copy()

    # Apply column renaming
    for old, new in column_map.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})

    # Ensure date is datetime
    if 'date' in result.columns:
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index('date')

    # Convert 8-day total to daily if needed
    if 'et_8day_mm' in result.columns and 'et_mm_day' not in result.columns:
        result['et_mm_day'] = result['et_8day_mm'] / 8.0

    return result


def interpolate_8day_to_daily(
    df: pd.DataFrame,
    value_column: str = 'et_mm_day',
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate 8-day composite values to daily frequency.

    Args:
        df: DataFrame with DatetimeIndex
        value_column: Column to interpolate
        method: Interpolation method:
            - 'linear': Linear interpolation (default)
            - 'nearest': Nearest neighbor
            - 'constant': Forward fill (constant over 8-day period)

    Returns:
        DataFrame with daily values
    """
    if df.empty or value_column not in df.columns:
        return df

    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            return df

    try:
        # Create daily date range
        start = df.index.min()
        end = df.index.max()
        daily_index = pd.date_range(start=start, end=end, freq='D')

        # Reindex to daily
        df_daily = df.reindex(daily_index)

        # Apply interpolation method
        if method == 'constant':
            # Forward fill (constant over 8-day period)
            df_daily = df_daily.ffill()
        elif method == 'nearest':
            df_daily = df_daily.interpolate(method='nearest')
        else:
            # Linear interpolation (default)
            df_daily = df_daily.interpolate(method='linear')

        df_daily.index.name = 'date'
        return df_daily

    except Exception:
        # Return original on failure
        return df


def apply_modis_quality_filter(
    data: xr.DataArray,
    valid_range: tuple = VALID_SNOW_RANGE,
    fill_values: set = MODIS_FILL_VALUES
) -> xr.DataArray:
    """
    Apply quality filtering to MODIS data.

    Masks out fill values and values outside the valid range.

    Args:
        data: Input DataArray
        valid_range: (min, max) tuple of valid values
        fill_values: Set of fill values to mask

    Returns:
        Filtered DataArray with invalid values as NaN
    """
    # Mask values outside valid range
    filtered = data.where(
        (data >= valid_range[0]) & (data <= valid_range[1])
    )
    return filtered


def extract_spatial_average(
    data: xr.DataArray,
    output_column: str = 'value'
) -> pd.DataFrame:
    """
    Extract spatially averaged time series from MODIS data.

    Args:
        data: DataArray with time and spatial dimensions
        output_column: Name for the output value column

    Returns:
        DataFrame with columns: output_column, valid_pixels, valid_ratio
    """
    # Get spatial dimensions (everything except time)
    spatial_dims = [d for d in data.dims if d != 'time']

    # Convert time values, handling cftime
    time_values = convert_cftime_to_datetime(data.time.values)

    if spatial_dims:
        # Count valid pixels for each timestep
        valid_counts = data.notnull().sum(dim=spatial_dims)
        total_pixels = np.prod([data.sizes[d] for d in spatial_dims])

        # Compute mean over spatial dims
        mean_values = data.mean(dim=spatial_dims, skipna=True)

        df = pd.DataFrame({
            output_column: mean_values.values,
            'valid_pixels': valid_counts.values,
            'valid_ratio': valid_counts.values / total_pixels
        }, index=time_values)
    else:
        df = pd.DataFrame({
            output_column: data.values
        }, index=time_values)

    df.index.name = 'date'
    return df


def find_variable_in_dataset(
    ds: xr.Dataset,
    priority_vars: List[str],
    fallback_pattern: Optional[str] = None
) -> Optional[str]:
    """
    Find a variable in a dataset using priority list and optional pattern matching.

    Args:
        ds: xarray Dataset
        priority_vars: List of variable names to try in order
        fallback_pattern: Optional regex pattern for fallback search

    Returns:
        Variable name if found, None otherwise
    """
    # Try priority variables first
    for var in priority_vars:
        if var in ds.data_vars:
            return var

    # Try pattern matching
    if fallback_pattern:
        import re
        for var in ds.data_vars:
            if re.search(fallback_pattern, var, re.IGNORECASE):
                return var

    return None
