# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Metrics table panel for model comparison visualizations.

Renders performance metrics as a formatted table.
"""

from typing import Any, Dict

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class MetricsTablePanel(BasePanel):
    """Panel for displaying performance metrics in a table format.

    Renders a table showing KGE, NSE, RMSE, and Bias for each model.

    Data Requirements:
        - metrics_dict: Dict mapping model names to their metrics

    Example:
        panel = MetricsTablePanel(plot_config, logger)
        panel.render(ax, {
            'metrics_dict': {
                'SUMMA': {'KGE': 0.75, 'NSE': 0.70, 'RMSE': 2.5, 'Bias%': -5.2},
                'FUSE': {'KGE': 0.68, 'NSE': 0.65, 'RMSE': 3.1, 'Bias%': 2.1}
            }
        })
    """

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render metrics table.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - metrics_dict: Dict mapping model names to metrics dicts
        """
        ax.axis('off')

        metrics_dict = data.get('metrics_dict', {})

        if not metrics_dict:
            ax.text(0.5, 0.5, 'No metrics available',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            return

        # Prepare table data
        headers = ['Model', 'KGE', 'NSE', 'RMSE', 'Bias%']
        cell_data = []

        for model_name, metrics in metrics_dict.items():
            row = [
                model_name,
                self._format_metric(metrics.get('KGE'), precision=3),
                self._format_metric(metrics.get('NSE'), precision=3),
                self._format_metric(metrics.get('RMSE'), precision=2),
                self._format_bias(metrics.get('Bias%'))
            ]
            cell_data.append(row)

        # Create table
        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(headers)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color code cells based on performance
        self._apply_cell_colors(table, cell_data, headers)

        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)

    def _format_metric(self, value: Any, precision: int = 3) -> str:
        """Format a metric value for display.

        Args:
            value: Metric value (can be None or NaN)
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        return f'{value:.{precision}f}'

    def _format_bias(self, value: Any) -> str:
        """Format bias value with sign and percent.

        Args:
            value: Bias value

        Returns:
            Formatted string with sign
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        return f'{value:+.1f}%'

    def _apply_cell_colors(
        self,
        table: Any,
        cell_data: list,
        headers: list
    ) -> None:
        """Apply color coding to table cells based on metric quality.

        Args:
            table: Matplotlib table object
            cell_data: Table data
            headers: Column headers
        """
        try:
            # Get column indices
            kge_idx = headers.index('KGE')
            nse_idx = headers.index('NSE')

            for row_idx, row in enumerate(cell_data):
                # KGE coloring (higher is better)
                try:
                    kge_val = float(row[kge_idx])
                    cell = table[(row_idx + 1, kge_idx)]
                    if kge_val >= 0.7:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif kge_val >= 0.5:
                        cell.set_facecolor('#FFFFE0')  # Light yellow
                    else:
                        cell.set_facecolor('#FFB6C1')  # Light red
                except (ValueError, KeyError):
                    pass

                # NSE coloring (higher is better)
                try:
                    nse_val = float(row[nse_idx])
                    cell = table[(row_idx + 1, nse_idx)]
                    if nse_val >= 0.65:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif nse_val >= 0.4:
                        cell.set_facecolor('#FFFFE0')  # Light yellow
                    else:
                        cell.set_facecolor('#FFB6C1')  # Light red
                except (ValueError, KeyError):
                    pass

        except Exception as e:  # noqa: BLE001 — reporting resilience
            self.logger.debug(f"Could not apply cell coloring: {e}")
