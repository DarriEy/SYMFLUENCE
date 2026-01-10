"""
Core reporting utilities.

This module provides shared utilities for the reporting subsystem.
"""

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.plot_utils import (
    calculate_metrics,
    calculate_flow_duration_curve,
    align_timeseries,
)
from symfluence.reporting.core.shapefile_helper import (
    ShapefileHelper,
    resolve_default_name,
)
from symfluence.reporting.core.dataframe_utils import (
    ensure_datetime_index,
    align_time_series,
    determine_common_time_range,
    resample_to_daily,
    resample_to_hourly,
    align_multiple_datasets,
    skip_spinup_period,
)

__all__ = [
    # Base plotter
    'BasePlotter',
    # Plot utilities
    'calculate_metrics',
    'calculate_flow_duration_curve',
    'align_timeseries',
    # Shapefile helper
    'ShapefileHelper',
    'resolve_default_name',
    # DataFrame utilities
    'ensure_datetime_index',
    'align_time_series',
    'determine_common_time_range',
    'resample_to_daily',
    'resample_to_hourly',
    'align_multiple_datasets',
    'skip_spinup_period',
]
