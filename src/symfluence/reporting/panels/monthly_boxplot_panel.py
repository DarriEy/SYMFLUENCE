"""
Monthly boxplot panel for model comparison visualizations.

Renders monthly distribution boxplots for observations and models.
"""

from typing import Any, Dict, List

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class MonthlyBoxplotPanel(BasePanel):
    """Panel for monthly distribution boxplot visualization.

    Renders side-by-side boxplots showing monthly distribution of
    observations and model outputs.

    Data Requirements:
        - results_df: DataFrame with DateTimeIndex and model columns
        - obs_series: Optional observation time series
        - model_cols: List of model column names

    Example:
        panel = MonthlyBoxplotPanel(plot_config, logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge']
        })
    """

    MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render monthly boxplots.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - results_df: DataFrame with DateTimeIndex and model columns
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

        # Get months from index
        months = results_df.index.month
        positions = np.arange(1, 13)

        legend_elements = []

        # Plot observed boxplots
        if obs_series is not None:
            obs_monthly = self._group_by_month(obs_series.values, months)
            bp_obs = ax.boxplot(obs_monthly, positions=positions - 0.2,
                               widths=0.15, patch_artist=True)
            for patch in bp_obs['boxes']:
                patch.set_facecolor('black')
                patch.set_alpha(0.5)

            # Create legend element for observations
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor='black', alpha=0.5, label='Observed')
            )

        # Plot model boxplots (first model only to avoid clutter)
        if model_cols and model_cols[0] in results_df.columns:
            col = model_cols[0]
            model_monthly = self._group_by_month(results_df[col].values, months)
            bp_model = ax.boxplot(model_monthly, positions=positions + 0.2,
                                 widths=0.15, patch_artist=True)
            color = self._get_color(0)
            for patch in bp_model['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            # Create legend element for model
            from matplotlib.patches import Patch
            model_name = self._extract_model_name(col)
            legend_elements.append(
                Patch(facecolor=color, alpha=0.5, label=model_name)
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(self.MONTH_LABELS)

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        self._apply_styling(
            ax,
            xlabel='Month',
            ylabel='Discharge (m\u00b3/s)',
            title='Monthly Distribution',
            legend=False  # Manual legend above
        )

    def _group_by_month(
        self,
        values: np.ndarray,
        months: np.ndarray
    ) -> List[np.ndarray]:
        """Group values by month.

        Args:
            values: Array of values
            months: Array of month numbers (1-12)

        Returns:
            List of arrays, one per month
        """
        return [values[(months == m) & ~np.isnan(values)] for m in range(1, 13)]
