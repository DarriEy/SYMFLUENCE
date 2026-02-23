"""
Snow visualization plotter.

Handles plotting of snow water equivalent (SWE) comparisons and metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore

from symfluence.reporting.core.base_plotter import BasePlotter


class SnowPlotter(BasePlotter):
    """
    Plotter for snow-related visualizations.

    Handles:
    - Snow Water Equivalent (SWE) simulations vs observations
    - Per-HRU snow analysis
    - Snow metrics calculation and table generation
    """

    def plot_snow_comparison(self, model_outputs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Visualize SWE comparison across HRUs and models.

        Args:
            model_outputs: List of tuples (model_name, output_file)

        Returns:
            Dictionary containing paths to plots and calculated metrics
        """
        import geopandas as gpd  # type: ignore
        import xarray as xr  # type: ignore
        plt, mdates = self._setup_matplotlib()
        from matplotlib.gridspec import GridSpec  # type: ignore

        results = {'plot_file': '', 'individual_plots': [], 'metrics': {}}

        try:
            # Load observation data
            # (Logic migrated from VisualizationReporter.plot_snow_simulations_vs_observations)
            snow_obs_path = Path(self._get_config_value(lambda: None, default='', dict_key='snow_processed_path')) / self._get_config_value(lambda: None, default='', dict_key='snow_processed_name')
            if not snow_obs_path.exists():
                self.logger.warning(f"Snow observation file not found: {snow_obs_path}")
                return results

            snow_obs = pd.read_csv(snow_obs_path, parse_dates=['datetime'])

            station_shp_path = Path(self._get_config_value(lambda: None, default='', dict_key='snow_station_shapefile_path')) / self._get_config_value(lambda: None, default='', dict_key='snow_station_shapefile_name')
            if not station_shp_path.exists():
                self.logger.warning(f"Snow station shapefile not found: {station_shp_path}")
                return results

            station_gdf = gpd.read_file(station_shp_path)
            snow_obs['station_id'] = snow_obs['station_id'].astype(str)
            merged_obs = pd.merge(snow_obs, station_gdf, on='station_id')

            # Load model data
            model_datasets = [(name, xr.open_dataset(file)) for name, file in model_outputs]

            unique_hrus = merged_obs['HRU_ID'].unique()
            n_hrus = len(unique_hrus)

            plot_dir = self._ensure_output_dir('snow')

            # Create main figure
            fig_all = plt.figure(figsize=(20, 10 * n_hrus + 8))
            gs = GridSpec(n_hrus + 1, 1, height_ratios=[10] * n_hrus + [2])
            fig_all.suptitle("Snow Water Equivalent Comparison", fontsize=16, fontweight='bold')

            for idx, hru_id in enumerate(unique_hrus):
                hru_obs = merged_obs[merged_obs['HRU_ID'] == hru_id]
                ax = fig_all.add_subplot(gs[idx])

                # Plot simulations
                for i, (sim_name, ds) in enumerate(model_datasets):
                    if 'scalarSWE' in ds:
                        hru_sim = ds['scalarSWE'].sel(hru=hru_id).to_series()
                        ax.plot(hru_sim.index, hru_sim, label=f'Simulated ({sim_name})',
                               color=self.plot_config.get_color_from_palette(i),
                               linestyle=self.plot_config.get_line_style(i))

                # Plot observations
                for j, station in enumerate(hru_obs['station_id'].unique()):
                    st_data = hru_obs[hru_obs['station_id'] == station].sort_values('datetime')
                    ax.scatter(st_data['datetime'], st_data['snw'], label=f'Obs (Station {station})',
                              alpha=0.6, s=30)

                self._apply_standard_styling(ax, ylabel='SWE (mm)', title=f'SWE - HRU {hru_id}', legend=True)
                self._format_date_axis(ax)

            # Save main plot
            main_plot_path = plot_dir / 'snow_comparison_all_hrus.png'
            results['plot_file'] = self._save_and_close(fig_all, main_plot_path)

            return results

        except Exception as e:
            self.logger.error(f"Error in plot_snow_comparison: {str(e)}")
            return results

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Main plot method."""
        if 'model_outputs' in kwargs:
            res = self.plot_snow_comparison(kwargs['model_outputs'])
            return res.get('plot_file')
        return None
