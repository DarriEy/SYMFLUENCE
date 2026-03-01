# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Residual analysis panel for model comparison visualizations.

Renders residual/bias analysis visualizations.
"""

from typing import Any, Dict, List

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class ResidualAnalysisPanel(BasePanel):
    """Panel for residual/bias analysis visualization.

    Renders bar plot showing monthly bias (percentage) for model vs observations.

    Data Requirements:
        - results_df: DataFrame with DateTimeIndex and model columns
        - obs_series: Observation time series
        - model_cols: List of model column names (uses first one)

    Example:
        panel = ResidualAnalysisPanel(plot_config, logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge']
        })
    """

    MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render residual analysis bar plot.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - results_df: DataFrame with DateTimeIndex and model columns
                - obs_series: Observation Series
                - model_cols: List of model column names
        """
        results_df = data.get('results_df')
        obs_series = data.get('obs_series')
        model_cols = data.get('model_cols', [])

        if obs_series is None or results_df is None or not model_cols:
            ax.text(0.5, 0.5, 'No data for residual analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            return

        col = model_cols[0]
        if col not in results_df.columns:
            ax.text(0.5, 0.5, 'Model column not found',
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            return

        # Get months from index
        months = results_df.index.month
        obs_values = obs_series.values
        sim_values = results_df[col].values

        # Calculate monthly bias
        monthly_bias = self._calculate_monthly_bias(obs_values, sim_values, months)

        # Create bar plot
        positions = np.arange(1, 13)
        colors = [self._get_color(0) if b >= 0 else '#d62728' for b in monthly_bias]
        ax.bar(positions, monthly_bias, color=colors, alpha=0.7)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(self.MONTH_LABELS)

        model_name = self._extract_model_name(col)
        self._apply_styling(
            ax,
            xlabel='Month',
            ylabel='Bias (%)',
            title=f'Monthly Bias - {model_name}',
            legend=False
        )

    def _calculate_monthly_bias(
        self,
        obs_values: np.ndarray,
        sim_values: np.ndarray,
        months: np.ndarray
    ) -> List[float]:
        """Calculate monthly percentage bias.

        Args:
            obs_values: Observation values
            sim_values: Simulated values
            months: Array of month numbers (1-12)

        Returns:
            List of monthly bias percentages
        """
        monthly_bias = []

        for m in range(1, 13):
            mask = (months == m) & ~np.isnan(obs_values) & ~np.isnan(sim_values)
            if mask.sum() > 0:
                obs_m = obs_values[mask]
                sim_m = sim_values[mask]
                mean_obs = np.mean(obs_m)
                if mean_obs != 0:
                    bias = ((np.mean(sim_m) - mean_obs) / mean_obs) * 100
                else:
                    bias = 0
                monthly_bias.append(bias)
            else:
                monthly_bias.append(0)

        return monthly_bias


class ResidualHistogramPanel(BasePanel):
    """Panel for residual histogram visualization.

    Renders histogram of residuals (sim - obs).

    Data Requirements:
        - obs_values: Observation values
        - sim_values: Simulated values
        - model_name: Model name for title

    Example:
        panel = ResidualHistogramPanel(plot_config, logger)
        panel.render(ax, {
            'obs_values': obs_array,
            'sim_values': sim_array,
            'model_name': 'SUMMA'
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render residual histogram.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - obs_values: Observation array
                - sim_values: Simulation array
                - model_name: Model name for title
        """
        obs_values = data.get('obs_values')
        sim_values = data.get('sim_values')
        model_name = data.get('model_name', 'Model')

        if obs_values is None or sim_values is None:
            ax.text(0.5, 0.5, 'No data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Convert to numpy if needed
        if hasattr(obs_values, 'values'):
            obs_values = obs_values.values
        if hasattr(sim_values, 'values'):
            sim_values = sim_values.values

        # Get valid data
        obs_clean, sim_clean = self._get_valid_data(obs_values, sim_values)

        if len(obs_clean) < 10:
            ax.text(0.5, 0.5, 'Insufficient data',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Calculate residuals
        residuals = sim_clean - obs_clean

        # Plot histogram
        ax.hist(residuals, bins=30, color=self._get_color(0),
               alpha=0.7, edgecolor='white')

        # Add mean line
        mean_residual = np.mean(residuals)
        ax.axvline(x=mean_residual, color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_residual:.2f}')

        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        self._apply_styling(
            ax,
            xlabel='Residual (m\u00b3/s)',
            ylabel='Frequency',
            title=f'Residual Distribution - {model_name}',
            legend=True,
            legend_loc='upper right'
        )
