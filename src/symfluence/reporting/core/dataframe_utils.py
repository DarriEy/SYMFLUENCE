# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
DataFrame and time series utility functions.

This module provides common operations for handling time series data,
including datetime index management, alignment, and resampling.
"""

from typing import Any, List, Optional, Tuple, Union

import pandas as pd


def ensure_datetime_index(
    df: Union[pd.DataFrame, pd.Series],
    time_col: Optional[str] = None
) -> Union[pd.DataFrame, pd.Series]:
    """
    Ensure DataFrame/Series has a proper DatetimeIndex.

    Args:
        df: DataFrame or Series to process
        time_col: Column name to use as datetime index (if not already indexed)

    Returns:
        DataFrame/Series with DatetimeIndex

    Example:
        >>> df = pd.DataFrame({'time': ['2020-01-01', '2020-01-02'], 'value': [1, 2]})
        >>> result = ensure_datetime_index(df, 'time')
        >>> isinstance(result.index, pd.DatetimeIndex)
        True
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Make a copy to avoid modifying original
    result = df.copy()

    # Try common time column names if not specified
    if time_col is None:
        for col in ['time', 'datetime', 'date', 'timestamp', 'Unnamed: 0']:
            if hasattr(result, 'columns') and col in result.columns:
                time_col = col
                break

    if time_col is not None and hasattr(result, 'columns') and time_col in result.columns:
        result[time_col] = pd.to_datetime(result[time_col])
        result = result.set_index(time_col)
    else:
        # Try to convert existing index
        try:
            result.index = pd.to_datetime(result.index)
        except (ValueError, TypeError):
            pass

    return result


def align_time_series(
    *datasets: Union[pd.DataFrame, pd.Series, Any],
    method: str = 'inner',
    spinup_percent: float = 0.0
) -> Tuple:
    """
    Align multiple time series to common time range.

    Args:
        *datasets: Variable number of DataFrames/Series to align
        method: Alignment method ('inner' for intersection, 'outer' for union)
        spinup_percent: Percentage of data to skip at the beginning (0-100)

    Returns:
        Tuple of aligned datasets (same order as input)

    Example:
        >>> s1 = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        >>> s2 = pd.Series([4, 5, 6], index=pd.date_range('2020-01-02', periods=3))
        >>> a1, a2 = align_time_series(s1, s2)
        >>> len(a1) == len(a2) == 2
        True
    """
    # Filter out None values but track positions
    valid_datasets = []
    valid_indices = []
    for i, ds in enumerate(datasets):
        if ds is not None and len(ds) > 0:
            valid_datasets.append(ds)
            valid_indices.append(i)

    if not valid_datasets:
        return datasets

    # Get time ranges
    start_times = []
    end_times = []

    for ds in valid_datasets:
        if hasattr(ds, 'index'):
            idx = ds.index
        elif hasattr(ds, 'time'):
            idx = ds.time.values
        else:
            continue

        start_times.append(pd.Timestamp(idx.min()))
        end_times.append(pd.Timestamp(idx.max()))

    if not start_times:
        return datasets

    # Calculate common range
    if method == 'inner':
        common_start = max(start_times)
        common_end = min(end_times)
    else:  # outer
        common_start = min(start_times)
        common_end = max(end_times)

    # Apply spinup
    if spinup_percent > 0:
        total_duration = (common_end - common_start).total_seconds()
        spinup_duration = total_duration * (spinup_percent / 100.0)
        common_start = common_start + pd.Timedelta(seconds=spinup_duration)

    # Check for valid range
    if common_start >= common_end:
        return datasets

    # Slice all datasets
    result = list(datasets)
    for i, ds in zip(valid_indices, valid_datasets):
        if hasattr(ds, 'loc'):
            result[i] = ds.loc[common_start:common_end]
        else:
            result[i] = ds

    return tuple(result)


def determine_common_time_range(
    datasets: List[Tuple[str, pd.Series]]
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Determine overlapping time range across multiple datasets.

    Args:
        datasets: List of (name, Series) tuples

    Returns:
        Tuple of (start_date, end_date)

    Raises:
        ValueError: If no datasets provided or no overlap exists
    """
    if not datasets:
        raise ValueError("No datasets provided")

    start_date = max([data.index.min() for _, data in datasets if len(data) > 0])
    end_date = min([data.index.max() for _, data in datasets if len(data) > 0])

    if start_date >= end_date:
        raise ValueError("No overlapping time range found across datasets")

    return start_date, end_date


def resample_to_daily(
    df: Union[pd.DataFrame, pd.Series],
    agg_method: str = 'mean'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Resample time series to daily frequency.

    Args:
        df: DataFrame or Series to resample
        agg_method: Aggregation method ('mean', 'sum', 'max', 'min')

    Returns:
        Resampled DataFrame/Series
    """
    df = ensure_datetime_index(df)
    return df.resample('D').agg(agg_method)


def resample_to_hourly(
    df: Union[pd.DataFrame, pd.Series],
    agg_method: str = 'mean'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Resample time series to hourly frequency.

    Args:
        df: DataFrame or Series to resample
        agg_method: Aggregation method ('mean', 'sum', 'max', 'min')

    Returns:
        Resampled DataFrame/Series
    """
    df = ensure_datetime_index(df)
    return df.resample('h').agg(agg_method)


def align_multiple_datasets(
    datasets: List[Tuple[str, pd.Series]],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Align multiple time series datasets to common time range.

    Args:
        datasets: List of (name, Series) tuples
        start_date: Optional start date for alignment
        end_date: Optional end date for alignment

    Returns:
        DataFrame with aligned datasets as columns

    Example:
        >>> s1 = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3), name='a')
        >>> s2 = pd.Series([4, 5, 6], index=pd.date_range('2020-01-01', periods=3), name='b')
        >>> result = align_multiple_datasets([('A', s1), ('B', s2)])
        >>> list(result.columns)
        ['A', 'B']
    """
    if not datasets:
        return pd.DataFrame()

    # Create DataFrame with all datasets
    df = pd.concat(
        [series for _, series in datasets],
        axis=1,
        keys=[name for name, _ in datasets]
    )

    # Filter by date range if specified
    if start_date is not None:
        df = df[df.index >= start_date]

    if end_date is not None:
        df = df[df.index <= end_date]

    return df


def skip_spinup_period(
    df: Union[pd.DataFrame, pd.Series],
    spinup_years: int = 1
) -> Union[pd.DataFrame, pd.Series]:
    """
    Skip spinup period at the beginning of a time series.

    Args:
        df: DataFrame or Series
        spinup_years: Number of years to skip (default: 1)

    Returns:
        DataFrame/Series with spinup period removed
    """
    df = ensure_datetime_index(df)

    if len(df) == 0:
        return df

    start_time = df.index[0]
    end_time = df.index[-1]
    duration_days = (end_time - start_time).days

    if duration_days > 365 * spinup_years:
        first_year = start_time.year
        start_date = pd.Timestamp(year=first_year + spinup_years, month=1, day=1)
        return df[df.index >= start_date]

    return df
