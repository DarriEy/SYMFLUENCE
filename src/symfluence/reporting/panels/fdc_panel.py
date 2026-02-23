"""
Flow Duration Curve panel for model comparison visualizations.

Renders flow duration curves for observations and model outputs.
"""

from typing import Any, Dict

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class FDCPanel(BasePanel):
    """Panel for Flow Duration Curve visualization.

    Renders FDC (exceedance probability vs flow) on log-log scale
    for observations and all models.

    Data Requirements:
        - results_df: DataFrame with model output columns
        - obs_series: Optional observation time series
        - model_cols: List of model column names to plot

    Example:
        panel = FDCPanel(plot_config, logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge', 'FUSE_discharge']
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render flow duration curves.

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

        # Plot observed FDC
        if obs_series is not None:
            exc_obs, flows_obs = self._calculate_fdc(obs_series.values)
            if len(exc_obs) > 0:
                ax.plot(exc_obs * 100, flows_obs, color='black',
                       linewidth=2, label='Observed', zorder=10)

        # Plot model FDCs
        if results_df is not None:
            for i, col in enumerate(model_cols):
                if col in results_df.columns:
                    color = self._get_color(i)
                    exc, flows = self._calculate_fdc(results_df[col].values)
                    if len(exc) > 0:
                        model_name = self._extract_model_name(col)
                        ax.plot(exc * 100, flows, color=color, linewidth=1.5,
                               alpha=0.8, label=model_name)

        # Set log scale for y-axis
        ax.set_yscale('log')
        ax.set_xlim([0, 100])

        # Ensure positive y-axis limits
        ylim = ax.get_ylim()
        ax.set_ylim([max(ylim[0], 0.01), ylim[1]])

        self._apply_styling(
            ax,
            xlabel='Exceedance Probability (%)',
            ylabel='Discharge (m\u00b3/s)',
            title='Flow Duration Curves',
            legend=True,
            legend_loc='upper right'
        )

    def _calculate_fdc(self, values: np.ndarray) -> tuple:
        """Calculate flow duration curve.

        Args:
            values: Array of flow values

        Returns:
            Tuple of (exceedance_probabilities, sorted_flows)
        """
        # Remove NaN values
        clean_values = values[~np.isnan(values)]

        if len(clean_values) == 0:
            return np.array([]), np.array([])

        # Sort in descending order
        sorted_flows = np.sort(clean_values)[::-1]

        # Calculate exceedance probability using Weibull plotting position
        n = len(sorted_flows)
        rank = np.arange(1, n + 1)
        exceedance = rank / (n + 1)

        return exceedance, sorted_flows
