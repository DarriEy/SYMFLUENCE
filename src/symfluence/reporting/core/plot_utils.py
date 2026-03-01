# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Pure utility functions for plotting.

This module provides standalone functions for common plotting operations
like metrics calculation, data alignment, and flow duration curves.
"""

from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics import kge, kge_np, kge_prime, mae, nse, rmse


def calculate_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard hydrological performance metrics.

    Args:
        obs: Observed values
        sim: Simulated values

    Returns:
        Dictionary with metric names and values:
        - RMSE: Root Mean Square Error
        - KGE: Kling-Gupta Efficiency
        - KGEp: Modified Kling-Gupta Efficiency
        - NSE: Nash-Sutcliffe Efficiency
        - MAE: Mean Absolute Error
        - KGEnp: Non-parametric Kling-Gupta Efficiency

    Note:
        - Handles NaN values by filtering them out
        - Returns NaN for all metrics if no valid data pairs exist
        - All metrics use transfo=1 (no transformation)

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> sim = np.array([1.1, 1.9, 3.2])
        >>> metrics = calculate_metrics(obs, sim)
        >>> metrics['NSE']  # doctest: +SKIP
        0.95
    """
    # Ensure arrays
    obs = np.asarray(obs)
    sim = np.asarray(sim)

    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[valid_mask]
    sim_clean = sim[valid_mask]

    # Check if we have data
    if len(obs_clean) == 0:
        return {
            'RMSE': np.nan,
            'KGE': np.nan,
            'KGEp': np.nan,
            'NSE': np.nan,
            'MAE': np.nan,
            'KGEnp': np.nan
        }

    # Calculate metrics
    from typing import cast
    return {
        'RMSE': rmse(obs_clean, sim_clean, transfo=1),
        'KGE': cast(float, kge(obs_clean, sim_clean, transfo=1)),
        'KGEp': cast(float, kge_prime(obs_clean, sim_clean, transfo=1)),
        'NSE': nse(obs_clean, sim_clean, transfo=1),
        'MAE': mae(obs_clean, sim_clean, transfo=1),
        'KGEnp': kge_np(obs_clean, sim_clean, transfo=1)
    }


def calculate_flow_duration_curve(
    flows: np.ndarray,
    remove_zeros: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate flow duration curve (exceedance probability).

    Args:
        flows: Array of flow values
        remove_zeros: Whether to remove zero values before calculation

    Returns:
        Tuple of (exceedance_probability, sorted_flows)

    Example:
        >>> flows = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        >>> exceedance, sorted_flows = calculate_flow_duration_curve(flows)
        >>> exceedance  # doctest: +SKIP
        array([0.2, 0.4, 0.6, 0.8, 1.0])
        >>> sorted_flows  # doctest: +SKIP
        array([5.0, 4.0, 3.0, 2.0, 1.0])
    """
    flows = np.asarray(flows)

    # Remove NaN
    flows = flows[~np.isnan(flows)]

    # Optionally remove zeros
    if remove_zeros:
        flows = flows[flows > 0]

    if len(flows) == 0:
        return np.array([]), np.array([])

    # Sort in descending order
    sorted_flows = np.sort(flows)[::-1]

    # Calculate exceedance probability
    exceedance = np.arange(1, len(flows) + 1) / len(flows)

    return exceedance, sorted_flows


def align_timeseries(
    obs: pd.Series,
    sim: pd.Series,
    spinup_days: Optional[int] = None,
    spinup_percent: Optional[float] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Align two time series by common timestamps and apply spinup removal.

    Args:
        obs: Observed time series (pandas Series with datetime index)
        sim: Simulated time series (pandas Series with datetime index)
        spinup_days: Number of days to remove from the beginning
        spinup_percent: Percentage of data to remove from the beginning (0-100)

    Returns:
        Tuple of (aligned_obs, aligned_sim)

    Note:
        - If both spinup_days and spinup_percent are provided, spinup_days takes precedence
        - Returns empty series if no overlapping timestamps exist

    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=100)
        >>> obs = pd.Series(range(100), index=dates)
        >>> sim = pd.Series(range(50, 150), index=dates)
        >>> obs_aligned, sim_aligned = align_timeseries(obs, sim, spinup_days=10)
        >>> len(obs_aligned)
        90
    """
    # Merge on index (timestamps)
    aligned = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()

    if aligned.empty:
        return pd.Series(), pd.Series()

    # Apply spinup removal
    if spinup_days is not None:
        # Remove first N days
        cutoff_date = aligned.index[0] + pd.Timedelta(days=spinup_days)
        aligned = aligned[aligned.index >= cutoff_date]

    elif spinup_percent is not None:
        # Remove first X% of data
        n_remove = int(len(aligned) * spinup_percent / 100)
        aligned = aligned.iloc[n_remove:]

    return aligned['obs'], aligned['sim']


def add_north_arrow(
    ax: Any,
    position: Tuple[float, float] = (0.95, 0.95),
    size: float = 0.05,
    text: str = 'N',
    **kwargs
) -> None:
    """
    Add a north arrow to a spatial plot.

    Args:
        ax: Matplotlib axis object
        position: (x, y) position in axis coordinates (0-1)
        size: Size of the arrow relative to axis
        text: Text to display (usually 'N')
        **kwargs: Additional arguments passed to annotate()

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> add_north_arrow(ax, position=(0.9, 0.9))  # doctest: +SKIP
    """
    x, y = position

    # Default styling
    arrow_props = {
        'arrowstyle': '->',
        'lw': 2,
        'color': 'black'
    }
    arrow_props.update(kwargs.get('arrowprops', {}))

    ax.annotate(
        text,
        xy=(x, y - size),
        xytext=(x, y),
        xycoords='axes fraction',
        textcoords='axes fraction',
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='bottom',
        arrowprops=arrow_props,
        **{k: v for k, v in kwargs.items() if k != 'arrowprops'}
    )


def format_metrics_for_display(
    metrics: Dict[str, float],
    precision: int = 3,
    label: Optional[str] = None
) -> str:
    """
    Format metrics dictionary as a string for display.

    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
        label: Optional label to prepend

    Returns:
        Formatted string with metrics

    Example:
        >>> metrics = {'NSE': 0.87654, 'RMSE': 1.234}
        >>> print(format_metrics_for_display(metrics, precision=2, label='Model A'))
        Model A:
        NSE: 0.88
        RMSE: 1.23
    """
    if not metrics:
        return ""

    lines = []
    if label:
        lines.append(f"{label}:")

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            lines.append(f"{key}: {value:.{precision}f}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def resample_timeseries(
    series: pd.Series,
    freq: str = 'D',
    aggregation: str = 'mean'
) -> pd.Series:
    """
    Resample a time series to a different frequency.

    Args:
        series: Pandas Series with datetime index
        freq: Target frequency ('h' for hourly, 'D' for daily, 'M' for monthly, etc.)
        aggregation: Aggregation method ('mean', 'sum', 'min', 'max')

    Returns:
        Resampled series

    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=48, freq='h')
        >>> series = pd.Series(range(48), index=dates)
        >>> daily = resample_timeseries(series, freq='D', aggregation='mean')
        >>> len(daily)
        2
    """
    agg_methods = {
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'min': lambda x: x.min(),
        'max': lambda x: x.max(),
        'median': lambda x: x.median(),
    }

    if aggregation not in agg_methods:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Must be one of: {', '.join(agg_methods.keys())}"
        )

    return cast(pd.Series, series.resample(freq).apply(agg_methods[aggregation]))


def calculate_summary_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics for a dataset.

    Args:
        data: Array of values

    Returns:
        Dictionary with statistics:
        - mean, median, std, min, max, q25, q75

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> stats = calculate_summary_statistics(data)
        >>> stats['mean']
        3.0
    """
    data = np.asarray(data)
    data_clean = data[~np.isnan(data)]

    if len(data_clean) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan
        }

    return {
        'mean': float(np.mean(data_clean)),
        'median': float(np.median(data_clean)),
        'std': float(np.std(data_clean)),
        'min': float(np.min(data_clean)),
        'max': float(np.max(data_clean)),
        'q25': float(np.percentile(data_clean, 25)),
        'q75': float(np.percentile(data_clean, 75))
    }
