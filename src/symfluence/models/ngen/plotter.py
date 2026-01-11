"""
NGEN Model Plotter

Model-specific visualization for NGEN outputs including streamflow comparisons
and flow duration curves.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from symfluence.reporting.plotter_registry import PlotterRegistry
from symfluence.reporting.core.base_plotter import BasePlotter


@PlotterRegistry.register_plotter('NGEN')
class NGENPlotter(BasePlotter):
    """
    Plotter for NGEN model outputs.

    Handles visualization of NGEN simulation results including:
    - Streamflow comparisons against observations
    - Flow duration curves
    """

    def plot_results(
        self,
        sim_df: pd.DataFrame,
        obs_df: Optional[pd.DataFrame],
        experiment_id: str,
        results_dir: Path
    ) -> Optional[str]:
        """
        Visualize NGen streamflow plots.

        Args:
            sim_df: Simulated streamflow dataframe
            obs_df: Observed streamflow dataframe (optional)
            experiment_id: Experiment identifier
            results_dir: Results directory

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_metrics, calculate_flow_duration_curve

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.plot_config.FIGURE_SIZE_LARGE)
            ax1.plot(sim_df['datetime'], sim_df['streamflow_cms'], label='NGEN Simulated', color=self.plot_config.COLOR_SIMULATED_PRIMARY)

            if obs_df is not None:
                ax1.plot(obs_df['datetime'], obs_df['streamflow_cms'], label='Observed', color=self.plot_config.COLOR_OBSERVED, alpha=0.7)
                merged = pd.merge(sim_df, obs_df, on='datetime', suffixes=('_sim', '_obs'))
                if not merged.empty:
                    self._add_metrics_text(ax1, calculate_metrics(merged['streamflow_cms_obs'].values, merged['streamflow_cms_sim'].values))

            self._apply_standard_styling(ax1, ylabel='Streamflow (cms)', title=f'NGEN Streamflow - {experiment_id}')
            self._format_date_axis(ax1, format_type='month')

            exc_sim, flows_sim = calculate_flow_duration_curve(sim_df['streamflow_cms'].values)
            ax2.semilogy(exc_sim, flows_sim, label='NGEN Simulated', color=self.plot_config.COLOR_SIMULATED_PRIMARY)

            if obs_df is not None:
                exc_obs, flows_obs = calculate_flow_duration_curve(obs_df['streamflow_cms'].values)
                ax2.semilogy(exc_obs, flows_obs, label='Observed', color=self.plot_config.COLOR_OBSERVED)

            self._apply_standard_styling(ax2, xlabel='Exceedance Probability (%)', ylabel='Streamflow (cms)', title='Flow Duration Curve')

            plot_file = self._ensure_output_dir('results') / f"ngen_streamflow_{experiment_id}.png"
            return self._save_and_close(fig, plot_file)

        except Exception as e:
            self.logger.error(f"Error in NGENPlotter.plot_results: {str(e)}")
            return None

    def plot(self, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        required = ['sim_df', 'experiment_id', 'results_dir']
        if all(k in kwargs for k in required):
            return self.plot_results(
                sim_df=kwargs['sim_df'],
                obs_df=kwargs.get('obs_df'),
                experiment_id=kwargs['experiment_id'],
                results_dir=kwargs['results_dir']
            )
        return None
