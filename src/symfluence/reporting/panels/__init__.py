"""
Reusable visualization panel components for model comparison plots.

This module provides composable panel classes that can be used to build
multi-panel visualizations. Each panel is responsible for rendering a
specific type of visualization onto a matplotlib axis.

Available Panels:
    - BasePanel: Abstract base class for all panels
    - TimeSeriesPanel: Time series comparison
    - MetricsTablePanel: Performance metrics table
    - FDCPanel: Flow duration curves
    - ScatterPanel: Scatter plot (obs vs sim)
    - MultiScatterPanel: Multiple scatter plots
    - MonthlyBoxplotPanel: Monthly distribution boxplots
    - ResidualAnalysisPanel: Monthly bias bar plot
    - ResidualHistogramPanel: Residual distribution histogram

Example:
    >>> from symfluence.reporting.panels import TimeSeriesPanel, FDCPanel
    >>>
    >>> ts_panel = TimeSeriesPanel(plot_config, logger)
    >>> fdc_panel = FDCPanel(plot_config, logger)
    >>>
    >>> ts_panel.render(ax1, {'results_df': df, 'obs_series': obs, 'model_cols': cols})
    >>> fdc_panel.render(ax2, {'results_df': df, 'obs_series': obs, 'model_cols': cols})
"""

from symfluence.reporting.panels.base_panel import BasePanel
from symfluence.reporting.panels.timeseries_panel import TimeSeriesPanel
from symfluence.reporting.panels.metrics_table_panel import MetricsTablePanel
from symfluence.reporting.panels.fdc_panel import FDCPanel
from symfluence.reporting.panels.scatter_panel import ScatterPanel, MultiScatterPanel
from symfluence.reporting.panels.monthly_boxplot_panel import MonthlyBoxplotPanel
from symfluence.reporting.panels.residual_analysis_panel import (
    ResidualAnalysisPanel,
    ResidualHistogramPanel,
)

__all__ = [
    # Base class
    'BasePanel',
    # Individual panels
    'TimeSeriesPanel',
    'MetricsTablePanel',
    'FDCPanel',
    'ScatterPanel',
    'MultiScatterPanel',
    'MonthlyBoxplotPanel',
    'ResidualAnalysisPanel',
    'ResidualHistogramPanel',
]
