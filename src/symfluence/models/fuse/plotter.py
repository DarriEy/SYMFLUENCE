"""
FUSE Model Plotter

Model-specific visualization for FUSE outputs including streamflow comparisons.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from symfluence.reporting.plotter_registry import PlotterRegistry
from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.shapefile_helper import resolve_default_name


@PlotterRegistry.register_plotter('FUSE')
class FUSEPlotter(BasePlotter):
    """
    Plotter for FUSE model outputs.

    Handles visualization of FUSE simulation results including:
    - Streamflow comparisons against observations
    - Unit conversions from mm/day to m³/s
    """

    def plot_streamflow(
        self,
        model_outputs: List[Tuple[str, str]],
        obs_files: List[Tuple[str, str]]
    ) -> Optional[str]:
        """
        Visualize FUSE simulated streamflow against observations.

        Args:
            model_outputs: List of tuples (model_name, output_file)
            obs_files: List of tuples (obs_name, obs_file)

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr  # type: ignore
        import geopandas as gpd  # type: ignore
        from symfluence.reporting.core.plot_utils import calculate_metrics
        from symfluence.core.constants import UnitConversion

        try:
            plot_dir = self._ensure_output_dir('results')
            exp_id = self.config.get('EXPERIMENT_ID', 'FUSE')
            plot_filename = plot_dir / f"{exp_id}_FUSE_streamflow_comparison.png"

            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)

            # Handle observations
            obs_dfs = []
            for _, obs_file in obs_files:
                df = pd.read_csv(obs_file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                obs_dfs.append(df)

            # Handle FUSE output
            for model_name, output_file in model_outputs:
                if model_name.upper() == 'FUSE':
                    with xr.open_dataset(output_file) as ds:
                        sim_flow = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0).to_series()

                        # Unit conversion (mm/day to cms)
                        basin_name = resolve_default_name(
                            self.config,
                            'RIVER_BASINS_NAME',
                            '{domain}_riverBasins_delineate.shp'
                        )

                        basin_path = self.project_dir / 'shapefiles' / 'river_basins' / basin_name
                        if not basin_path.exists():
                            basin_path = Path(self.config.get('RIVER_BASINS_PATH', ''))

                        if basin_path.exists():
                            basin_gdf = gpd.read_file(basin_path)
                            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
                            sim_flow = sim_flow * area_km2 / UnitConversion.MM_DAY_TO_CMS

                        if obs_dfs:
                            start_date = max(sim_flow.index.min(), obs_dfs[0].index.min())
                            end_date = min(sim_flow.index.max(), obs_dfs[0].index.max())

                            sim_plot = sim_flow.loc[start_date:end_date]
                            obs_plot = obs_dfs[0]['discharge_cms'].loc[start_date:end_date]

                            ax.plot(sim_plot.index, sim_plot, label='FUSE', color=self.plot_config.COLOR_SIMULATED_PRIMARY)
                            ax.plot(obs_plot.index, obs_plot, label='Observed', color=self.plot_config.COLOR_OBSERVED)

                            metrics = calculate_metrics(obs_plot.values, sim_plot.values)
                            self._add_metrics_text(ax, metrics)

            self._apply_standard_styling(
                ax, xlabel='Date', ylabel='Streamflow (m³/s)',
                title='FUSE Streamflow Comparison', legend=True
            )
            self._format_date_axis(ax, format_type='month')

            plt.tight_layout()
            return self._save_and_close(fig, plot_filename)

        except Exception as e:
            self.logger.error(f"Error in FUSEPlotter.plot_streamflow: {str(e)}")
            return None

    def plot(self, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        if 'model_outputs' in kwargs and 'obs_files' in kwargs:
            return self.plot_streamflow(kwargs['model_outputs'], kwargs['obs_files'])
        return None
