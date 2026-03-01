# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Plotter

Model-specific visualization for LSTM outputs including streamflow and SWE comparisons.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.plotter_registry import PlotterRegistry


@PlotterRegistry.register_plotter('LSTM')
class LSTMPlotter(BasePlotter):
    """
    Plotter for LSTM model outputs.

    Handles visualization of LSTM simulation results including:
    - Streamflow comparisons against observations
    - SWE (Snow Water Equivalent) comparisons (optional)
    """

    def plot_results(
        self,
        results_df: pd.DataFrame,
        obs_streamflow: pd.DataFrame,
        obs_snow: pd.DataFrame,
        use_snow: bool,
        output_dir: Path,
        experiment_id: str
    ) -> Optional[str]:
        """
        Visualize LSTM simulation results.

        Args:
            results_df: Simulation results dataframe
            obs_streamflow: Observed streamflow dataframe
            obs_snow: Observed snow dataframe
            use_snow: Whether snow metrics/plots are required
            output_dir: Output directory
            experiment_id: Experiment identifier

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from matplotlib.gridspec import GridSpec

        from symfluence.reporting.core.plot_utils import calculate_metrics

        try:
            sim_dates, sim_q = results_df.index, results_df['predicted_streamflow']
            obs_q = obs_streamflow.reindex(sim_dates)['streamflow']

            fig = plt.figure(figsize=self.plot_config.FIGURE_SIZE_LARGE)
            gs = GridSpec(2 if use_snow else 1, 1)
            ax1 = fig.add_subplot(gs[0])

            ax1.plot(sim_dates, sim_q, label='LSTM simulated', color='blue')
            ax1.plot(sim_dates, obs_q, label='Observed', color='red')
            self._add_metrics_text(ax1, calculate_metrics(obs_q.values, sim_q.values), label="Streamflow")
            self._apply_standard_styling(ax1, ylabel='Streamflow (m³/s)', title='Observed vs Simulated Streamflow')
            self._format_date_axis(ax1)

            if use_snow and not obs_snow.empty and 'predicted_SWE' in results_df.columns:
                ax2 = fig.add_subplot(gs[1])
                sim_swe, obs_swe = results_df['predicted_SWE'], obs_snow.reindex(sim_dates)['snw']
                ax2.plot(sim_dates, sim_swe, label='LSTM simulated', color='blue')
                ax2.plot(sim_dates, obs_swe, label='Observed', color='red')
                self._add_metrics_text(ax2, calculate_metrics(obs_swe.values, sim_swe.values), label="SWE")
                self._apply_standard_styling(ax2, ylabel='SWE (mm)', title='Observed vs Simulated SWE')
                self._format_date_axis(ax2)

            plot_file = self._ensure_output_dir('results') / f"{experiment_id}_LSTM_results.png"
            return self._save_and_close(fig, plot_file)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error in LSTMPlotter.plot_results: {str(e)}")
            return None

    def plot(self, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        required = ['results_df', 'obs_streamflow', 'obs_snow', 'use_snow', 'output_dir', 'experiment_id']
        if all(k in kwargs for k in required):
            return self.plot_results(**{k: kwargs[k] for k in required})
        return None
