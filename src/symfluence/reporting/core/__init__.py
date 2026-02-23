"""
Core reporting utilities.

This module provides shared utilities for the reporting subsystem.
"""

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.dataframe_utils import (
    align_multiple_datasets,
    align_time_series,
    determine_common_time_range,
    ensure_datetime_index,
    resample_to_daily,
    resample_to_hourly,
    skip_spinup_period,
)
from symfluence.reporting.core.decorators import (
    log_visualization,
    requires_plotter,
    skip_if_not_visualizing,
)
from symfluence.reporting.core.plot_utils import (
    align_timeseries,
    calculate_flow_duration_curve,
    calculate_metrics,
)
from symfluence.reporting.core.shapefile_helper import (
    ShapefileHelper,
    resolve_default_name,
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
    # Decorators
    'skip_if_not_visualizing',
    'requires_plotter',
    'log_visualization',
]
