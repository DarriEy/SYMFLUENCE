"""
Scatter plot panel for model comparison visualizations.

Renders scatter plots comparing observations vs simulated values.
"""

from typing import Any, Dict, List

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class ScatterPanel(BasePanel):
    """Panel for scatter plot visualization.

    Renders scatter plot of observed vs simulated values with 1:1 line
    and R-squared annotation.

    Data Requirements:
        - obs_values: Observation values array
        - sim_values: Simulated values array
        - model_name: Name of the model (for title)
        - color_index: Index for color selection (optional)

    Example:
        panel = ScatterPanel(plot_config, logger)
        panel.render(ax, {
            'obs_values': obs_array,
            'sim_values': sim_array,
            'model_name': 'SUMMA',
            'color_index': 0
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render scatter plot.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - obs_values: Observation array or Series
                - sim_values: Simulation array or Series
                - model_name: Model name for title
                - color_index: Index for color selection (default 0)
        """
        obs_values = data.get('obs_values')
        sim_values = data.get('sim_values')
        model_name = data.get('model_name', 'Model')
        color_index = data.get('color_index', 0)

        # Convert to numpy arrays if needed
        if hasattr(obs_values, 'values'):
            obs_values = obs_values.values
        if hasattr(sim_values, 'values'):
            sim_values = sim_values.values

        if obs_values is None or sim_values is None:
            ax.text(0.5, 0.5, 'No data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Get valid data
        obs_clean, sim_clean = self._get_valid_data(obs_values, sim_values)

        if len(obs_clean) < 10:
            ax.text(0.5, 0.5, 'Insufficient data',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Scatter plot
        color = self._get_color(color_index)
        ax.scatter(obs_clean, sim_clean, c=color, alpha=0.3, s=10, edgecolors='none')

        # 1:1 line
        max_val = max(np.max(obs_clean), np.max(sim_clean))
        min_val = min(np.min(obs_clean), np.min(sim_clean))
        ax.plot([min_val, max_val], [min_val, max_val],
               'k--', linewidth=1, label='1:1 line')

        # Calculate R-squared
        correlation = np.corrcoef(obs_clean, sim_clean)[0, 1]
        r_squared = correlation ** 2

        # Add R-squared annotation
        ax.text(0.05, 0.95, f'R\u00b2 = {r_squared:.3f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self._apply_styling(
            ax,
            xlabel='Observed (m\u00b3/s)',
            ylabel='Simulated (m\u00b3/s)',
            title=model_name,
            legend=False
        )


class MultiScatterPanel(BasePanel):
    """Panel for multiple scatter plots in a row.

    Renders scatter plots for multiple models side by side.

    Data Requirements:
        - results_df: DataFrame with model output columns
        - obs_series: Observation time series
        - model_cols: List of model column names to plot
        - axes: List of axes to render onto

    Example:
        panel = MultiScatterPanel(plot_config, logger)
        panel.render_multiple(axes, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge', 'FUSE_discharge']
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render scatter plot(s).

        If ax is a list of axes, renders scatter plots for multiple models.
        If ax is a single axis, renders a single scatter plot.

        Args:
            ax: Single matplotlib axis or list of axes
            data: Dictionary containing:
                - For multiple: results_df, obs_series, model_cols
                - For single: obs_values, sim_values, model_name, color_index
        """
        # If ax is a list, delegate to render_multiple
        if isinstance(ax, list):
            self.render_multiple(ax, data)
            return

        # Single scatter plot
        single_panel = ScatterPanel(self.plot_config, self.logger)
        single_panel.render(ax, data)

    def render_multiple(
        self,
        axes: List[Any],
        data: Dict[str, Any]
    ) -> None:
        """Render scatter plots for multiple models.

        Args:
            axes: List of matplotlib axes
            data: Dictionary containing:
                - results_df: DataFrame with model columns
                - obs_series: Observation Series
                - model_cols: List of model column names
        """
        results_df = data.get('results_df')
        obs_series = data.get('obs_series')
        model_cols = data.get('model_cols', [])

        if obs_series is None or results_df is None:
            return

        obs_values = obs_series.values

        for i, (ax, col) in enumerate(zip(axes, model_cols[:len(axes)])):
            if col in results_df.columns:
                sim_values = results_df[col].values
                model_name = self._extract_model_name(col)

                panel_data = {
                    'obs_values': obs_values,
                    'sim_values': sim_values,
                    'model_name': model_name,
                    'color_index': i
                }

                single_panel = ScatterPanel(self.plot_config, self.logger)
                single_panel.render(ax, panel_data)
