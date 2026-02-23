"""
HYPE Model Plotter

Model-specific visualization for HYPE outputs including streamflow comparisons.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.plotter_registry import PlotterRegistry


@PlotterRegistry.register_plotter('HYPE')
class HYPEPlotter(BasePlotter):
    """
    Plotter for HYPE model outputs.

    Handles visualization of HYPE simulation results including:
    - Streamflow comparisons against observations
    """

    def plot_streamflow(
        self,
        sim_flow: pd.DataFrame,
        obs_flow: pd.DataFrame,
        outlet_id: str,
        domain_name: str,
        experiment_id: str,
        project_dir: Path
    ) -> Optional[str]:
        """
        Visualize HYPE streamflow comparison.

        Args:
            sim_flow: Simulated streamflow dataframe
            obs_flow: Observed streamflow dataframe
            outlet_id: Outlet ID
            domain_name: Domain name
            experiment_id: Experiment identifier
            project_dir: Project directory

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)
            ax.plot(sim_flow.index, sim_flow['HYPE_discharge_cms'], label='Simulated', color='blue')
            ax.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red')
            self._apply_standard_styling(
                ax, ylabel='Discharge (m³/s)',
                title=f'Streamflow Comparison - {domain_name}\nOutlet ID: {outlet_id}'
            )
            self._format_date_axis(ax)

            plot_file = self._ensure_output_dir("results") / f"{experiment_id}_HYPE_comparison.png"
            return self._save_and_close(fig, plot_file)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error in HYPEPlotter.plot_streamflow: {str(e)}")
            return None

    def plot(self, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        required = ['sim_flow', 'obs_flow', 'outlet_id', 'domain_name', 'experiment_id', 'project_dir']
        if all(k in kwargs for k in required):
            return self.plot_streamflow(**{k: kwargs[k] for k in required})
        return None
