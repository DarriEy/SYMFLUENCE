# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Time series panel for model comparison visualizations.

Renders time series comparison of observations and model outputs.
"""

from typing import Any, Dict

from symfluence.reporting.panels.base_panel import BasePanel


class TimeSeriesPanel(BasePanel):
    """Panel for time series comparison visualization.

    Renders a time series plot comparing observations (black line)
    against multiple model outputs (colored lines).

    Data Requirements:
        - results_df: DataFrame with model output columns
        - obs_series: Optional observation time series
        - model_cols: List of model column names to plot

    Example:
        panel = TimeSeriesPanel(plot_config, logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge', 'FUSE_discharge']
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render time series comparison.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - results_df: DataFrame with time index and model columns
                - obs_series: Optional observation Series
                - model_cols: List of model column names
        """
        results_df = data.get('results_df')
        obs_series = data.get('obs_series')
        model_cols = data.get('model_cols', [])

        if results_df is None or results_df.empty:
            ax.text(0.5, 0.5, 'No data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Plot observations
        if obs_series is not None:
            ax.plot(results_df.index, obs_series,
                   color='black', linewidth=1.5, label='Observed', zorder=10)

        # Plot each model
        for i, col in enumerate(model_cols):
            if col in results_df.columns:
                color = self._get_color(i)
                model_name = self._extract_model_name(col)
                ax.plot(results_df.index, results_df[col],
                       color=color, linewidth=1.0, alpha=0.8, label=model_name)

        self._apply_styling(
            ax,
            xlabel='Date',
            ylabel='Discharge (m\u00b3/s)',
            title='Time Series Comparison',
            legend=True,
            legend_loc='upper right'
        )

        # Format date axis
        self._format_date_axis(ax)

    def _format_date_axis(self, ax: Any) -> None:
        """Format the x-axis for dates.

        Args:
            ax: Matplotlib axis
        """
        try:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.tick_params(axis='x', rotation=45)
        except (ImportError, ValueError, TypeError):
            pass  # Keep default formatting if matplotlib.dates unavailable or axis incompatible
