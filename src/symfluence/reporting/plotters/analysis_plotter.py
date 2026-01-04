"""
Analysis visualization plotter.

Handles plotting of sensitivity analysis, decision impacts, and threshold analysis.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.core.constants import UnitConversion


class AnalysisPlotter(BasePlotter):
    """
    Plotter for analysis visualizations.

    Handles:
    - Sensitivity analysis results
    - Decision impact analysis
    - Hydrograph comparisons with highlighting
    - Drop/threshold analysis
    """

    def plot_sensitivity_analysis(
        self,
        sensitivity_data: Any,
        output_file: Path,
        plot_type: str = 'single'
    ) -> Optional[str]:
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_data: Data to plot (Series or DataFrame)
            output_file: Path to save the plot
            plot_type: 'single' for one method, 'comparison' for multiple

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if plot_type == 'single':
                fig, ax = plt.subplots(
                    figsize=self.plot_config.FIGURE_SIZE_SMALL
                )
                sensitivity_data.plot(kind='bar', ax=ax)

                self._apply_standard_styling(
                    ax,
                    xlabel="Parameters",
                    ylabel="Sensitivity",
                    title="Parameter Sensitivity Analysis",
                    legend=False
                )

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

            elif plot_type == 'comparison':
                fig, ax = plt.subplots(
                    figsize=self.plot_config.FIGURE_SIZE_MEDIUM_TALL
                )
                sensitivity_data.plot(kind='bar', ax=ax)

                self._apply_standard_styling(
                    ax,
                    xlabel="Parameters",
                    ylabel="Sensitivity",
                    title="Sensitivity Analysis Comparison",
                    legend=True,
                    legend_loc='upper left'
                )

                ax.legend(
                    title="Method",
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left'
                )

                plt.tight_layout()

            else:
                self.logger.warning(f"Unknown plot_type '{plot_type}', using 'single'")
                return self.plot_sensitivity_analysis(sensitivity_data, output_file, 'single')

            return self._save_and_close(fig, output_file)

        except Exception as e:
            self.logger.error(f"Error creating sensitivity plot: {str(e)}")
            return None

    def plot_decision_impacts(
        self,
        results_file: Path,
        output_folder: Path
    ) -> Optional[Dict[str, str]]:
        """
        Visualize decision analysis impacts.

        Args:
            results_file: Path to the CSV results file
            output_folder: Folder to save plots

        Returns:
            Dictionary mapping metric names to plot paths, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            output_folder.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(results_file)
            metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']

            # Identify decision columns (exclude Iteration and metrics)
            decisions = [col for col in df.columns if col not in ['Iteration'] + metrics]

            plot_paths = {}

            for metric in metrics:
                if metric not in df.columns:
                    continue

                fig, axes = plt.subplots(
                    len(decisions), 1,
                    figsize=(12, 6 * len(decisions))
                )

                # Handle single decision case
                if len(decisions) == 1:
                    axes = [axes]

                for i, decision in enumerate(decisions):
                    impact = df.groupby(decision)[metric].mean().sort_values(ascending=False)
                    impact.plot(kind='bar', ax=axes[i])

                    axes[i].set_title(f'Impact of {decision} on {metric}')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=self.plot_config.GRID_ALPHA)

                plt.tight_layout()
                output_path = output_folder / f'{metric}_decision_impacts.png'
                self._save_and_close(fig, output_path)
                plot_paths[metric] = str(output_path)

            self.logger.info("Decision impact plots saved")
            return plot_paths

        except Exception as e:
            self.logger.error(f"Error creating decision impact plots: {str(e)}")
            return None

    def plot_hydrographs_with_highlight(
        self,
        results_file: Path,
        simulation_results: Dict,
        observed_streamflow: Any,
        decision_options: Dict,
        output_folder: Path,
        metric: str = 'kge'
    ) -> Optional[str]:
        """
        Visualize hydrographs with top performers highlighted.

        Args:
            results_file: Path to results CSV
            simulation_results: Dictionary of simulation results
            observed_streamflow: Observed streamflow series
            decision_options: Dictionary of decision options
            output_folder: Output folder
            metric: Metric to use for highlighting

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            output_folder.mkdir(parents=True, exist_ok=True)

            # Read results file
            results_df = pd.read_csv(results_file)

            # Calculate threshold for top 5%
            if metric in ['mae', 'rmse']:  # Lower is better
                threshold = results_df[metric].quantile(0.05)
                top_combinations = results_df[results_df[metric] <= threshold]
            else:  # Higher is better
                threshold = results_df[metric].quantile(0.95)
                top_combinations = results_df[results_df[metric] >= threshold]

            # Find overlapping period
            start_date = observed_streamflow.index.min()
            end_date = observed_streamflow.index.max()

            for sim in simulation_results.values():
                start_date = max(start_date, sim.index.min())
                end_date = min(end_date, sim.index.max())

            # Calculate y-axis limit from top 5%
            max_top5 = 0
            for _, row in top_combinations.iterrows():
                combo = tuple(row[list(decision_options.keys())])
                if combo in simulation_results:
                    sim = simulation_results[combo]
                    sim_overlap = sim.loc[start_date:end_date]
                    max_top5 = max(max_top5, sim_overlap.max())

            # Create plot
            fig, ax = plt.subplots(
                figsize=self.plot_config.FIGURE_SIZE_MEDIUM
            )

            ax.set_title(
                f'Hydrograph Comparison ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})\n'
                f'Top 5% combinations by {metric} metric highlighted',
                fontsize=self.plot_config.FONT_SIZE_TITLE,
                pad=20
            )

            ax.set_ylim(0, max_top5 * 1.1)

            # Plot top 5%
            for _, row in top_combinations.iterrows():
                combo = tuple(row[list(decision_options.keys())])
                if combo in simulation_results:
                    sim = simulation_results[combo]
                    sim_overlap = sim.loc[start_date:end_date]
                    ax.plot(
                        sim_overlap.index,
                        sim_overlap.values,
                        color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                        alpha=self.plot_config.ALPHA_FAINT,
                        linewidth=self.plot_config.LINE_WIDTH_THIN
                    )

            # Add legend
            ax.plot(
                [], [],
                color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                alpha=self.plot_config.ALPHA_FAINT,
                label=f'Top 5% by {metric}'
            )

            self._apply_standard_styling(
                ax,
                xlabel='Date',
                ylabel='Streamflow (m³/s)',
                legend=True
            )

            plt.tight_layout()

            # Save plot
            plot_file = output_folder / f'hydrograph_comparison_{metric}.png'
            saved_path = self._save_and_close(fig, plot_file)

            # Save summary CSV
            summary_file = output_folder / f'top_combinations_{metric}.csv'
            top_combinations.to_csv(summary_file, index=False)
            self.logger.info(f"Top combinations saved to: {summary_file}")

            return saved_path

        except Exception as e:
            self.logger.error(f"Error creating hydrograph plot: {str(e)}")
            return None

    def plot_drop_analysis(
        self,
        drop_data: List[Dict],
        optimal_threshold: float,
        project_dir: Path
    ) -> Optional[str]:
        """
        Visualize drop analysis for stream threshold selection.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics
            optimal_threshold: The selected optimal threshold
            project_dir: Project directory for saving the plot

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            thresholds = [d['threshold'] for d in drop_data]
            mean_drops = [d['mean_drop'] for d in drop_data]

            fig, ax = plt.subplots(
                figsize=self.plot_config.FIGURE_SIZE_SMALL
            )

            ax.loglog(
                thresholds,
                mean_drops,
                'bo-',
                linewidth=self.plot_config.LINE_WIDTH_THICK,
                markersize=self.plot_config.MARKER_SIZE_LARGE,
                label='Mean Drop'
            )

            ax.axvline(
                optimal_threshold,
                color=self.plot_config.COLOR_VALIDATION,
                linestyle='--',
                linewidth=self.plot_config.LINE_WIDTH_THICK,
                label=f'Optimal Threshold = {optimal_threshold:.0f}'
            )

            self._apply_standard_styling(
                ax,
                xlabel='Contributing Area Threshold (cells)',
                ylabel='Mean Stream Drop (m)',
                title='Drop Analysis for Stream Threshold Selection',
                legend=True
            )

            # Save plot
            plot_path = self._ensure_output_dir("drop_analysis") / "drop_analysis.png"
            return self._save_and_close(fig, plot_path)

        except Exception as e:
            self.logger.warning(f"Could not create drop analysis plot: {str(e)}")
            return None

    def plot_streamflow_comparison(
        self,
        model_outputs: List[Tuple[str, str]],
        obs_files: List[Tuple[str, str]],
        lumped: bool = False,
        spinup_percent: Optional[float] = None
    ) -> Optional[str]:
        """
        Visualize streamflow comparison between multiple models and observations.

        Args:
            model_outputs: List of tuples (model_name, file_path)
            obs_files: List of tuples (obs_name, file_path)
            lumped: Whether these are lumped watershed models
            spinup_percent: Percentage of data to skip as spinup

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr  # type: ignore
        from symfluence.reporting.core.plot_utils import (
            calculate_metrics, calculate_flow_duration_curve, align_timeseries
        )

        spinup_percent = spinup_percent if spinup_percent is not None else self.plot_config.SPINUP_PERCENT_DEFAULT

        try:
            plot_dir = self._ensure_output_dir('results')
            plot_filename = plot_dir / 'streamflow_comparison.png'

            # Load observations
            obs_data = []
            for obs_name, obs_file in obs_files:
                try:
                    df = pd.read_csv(obs_file, parse_dates=['datetime'])
                    df.set_index('datetime', inplace=True)
                    # Resample to hourly if needed, or daily
                    df = df['discharge_cms'].resample('h').mean()
                    obs_data.append((obs_name, df))
                except Exception as e:
                    self.logger.warning(f"Could not read observation file {obs_file}: {str(e)}")

            if not obs_data:
                self.logger.error("No observation data could be loaded")
                return None

            # Load simulations
            sim_data = []
            for sim_name, sim_file in model_outputs:
                try:
                    ds = xr.open_dataset(sim_file)
                    
                    if lumped:
                        if 'averageRoutedRunoff' in ds:
                            runoff = ds['averageRoutedRunoff'].to_series()
                            sim_data.append((sim_name, runoff))
                    else:
                        if 'IRFroutedRunoff' in ds:
                            runoff = ds['IRFroutedRunoff'].to_series()
                            sim_data.append((sim_name, runoff))
                        elif 'averageRoutedRunoff' in ds:
                            runoff = ds['averageRoutedRunoff'].to_series()
                            sim_data.append((sim_name, runoff))
                except Exception as e:
                    self.logger.warning(f"Could not read simulation file {sim_file}: {str(e)}")

            if not sim_data:
                self.logger.error("No simulation data could be loaded")
                return None

            # Create figure
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.plot_config.FIGURE_SIZE_XLARGE_TALL
            )

            # Plot time series
            for obs_name, obs in obs_data:
                ax1.plot(
                    obs.index, obs,
                    label=f'Observed ({obs_name})',
                    color=self.plot_config.COLOR_OBSERVED,
                    linewidth=self.plot_config.LINE_WIDTH_OBSERVED,
                    zorder=5
                )

            for i, (sim_name, sim) in enumerate(sim_data):
                color = self.plot_config.get_color_from_palette(i)
                style = self.plot_config.get_line_style(i)
                
                # Align and calculate metrics
                aligned_obs, aligned_sim = align_timeseries(
                    obs_data[0][1], sim, spinup_percent=spinup_percent
                )
                
                if not aligned_sim.empty:
                    ax1.plot(
                        aligned_sim.index, aligned_sim,
                        label=f'Simulated ({sim_name})',
                        color=color,
                        linestyle=style,
                        linewidth=self.plot_config.LINE_WIDTH_DEFAULT
                    )

                    metrics = calculate_metrics(aligned_obs.values, aligned_sim.values)
                    self._add_metrics_text(
                        ax1, metrics,
                        position=(0.02, 0.98 - 0.15 * i),
                        label=sim_name
                    )

            self._apply_standard_styling(
                ax1,
                xlabel='Date',
                ylabel='Streamflow (m³/s)',
                title=f'Streamflow Comparison (after {spinup_percent}% spinup)',
                legend=True
            )
            self._format_date_axis(ax1)

            # Plot FDC
            for obs_name, obs in obs_data:
                exc, flows = calculate_flow_duration_curve(obs.values)
                ax2.plot(
                    exc, flows,
                    label=f'Observed ({obs_name})',
                    color=self.plot_config.COLOR_OBSERVED,
                    linewidth=self.plot_config.LINE_WIDTH_OBSERVED
                )

            for i, (sim_name, sim) in enumerate(sim_data):
                color = self.plot_config.get_color_from_palette(i)
                style = self.plot_config.get_line_style(i)
                exc, flows = calculate_flow_duration_curve(sim.values)
                ax2.plot(
                    exc, flows,
                    label=f'Simulated ({sim_name})',
                    color=color,
                    linestyle=style,
                    linewidth=self.plot_config.LINE_WIDTH_DEFAULT
                )

            ax2.set_xscale('log')
            ax2.set_yscale('log')
            self._apply_standard_styling(
                ax2,
                xlabel='Exceedance Probability',
                ylabel='Streamflow (m³/s)',
                title='Flow Duration Curve',
                legend=True
            )

            plt.tight_layout()
            return self._save_and_close(fig, plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_streamflow_comparison: {str(e)}")
            return None

    def plot_fuse_streamflow(
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
                        # Get q_routed
                        sim_flow = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0).to_series()
                        
                        # Unit conversion (mm/day to cms)
                        basin_name = self.config.get('RIVER_BASINS_NAME', 'default')
                        if basin_name == 'default':
                            basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_delineate.shp"
                        
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
            self.logger.error(f"Error in plot_fuse_streamflow: {str(e)}")
            return None

    def plot_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """Create spatial and temporal visualizations for SUMMA output variables."""
        plt, _ = self._setup_matplotlib()
        from matplotlib import gridspec  # type: ignore
        import xarray as xr  # type: ignore
        import geopandas as gpd  # type: ignore

        plot_paths = {}
        try:
            summa_file = self.project_dir / "simulations" / experiment_id / "SUMMA" / f"{experiment_id}_day.nc"
            if not summa_file.exists():
                return {}

            plot_dir = self._ensure_output_dir('summa_outputs', experiment_id)
            ds = xr.open_dataset(summa_file)
            
            hru_name = self.config.get('CATCHMENT_SHP_NAME', 'default')
            if hru_name == 'default':
                hru_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"
            hru_path = self.project_dir / 'shapefiles' / 'catchment' / hru_name
            hru_gdf = gpd.read_file(hru_path) if hru_path.exists() else None

            skip_vars = {'hru', 'time', 'gru', 'dateId', 'latitude', 'longitude', 'hruId', 'gruId'}
            
            for var_name in ds.data_vars:
                if var_name in skip_vars or 'time' not in ds[var_name].dims:
                    continue
                
                fig = plt.figure(figsize=self.plot_config.FIGURE_SIZE_MEDIUM_TALL)
                gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
                ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
                
                var_mean = ds[var_name].mean(dim='time').compute()
                if hru_gdf is not None:
                    plot_gdf = hru_gdf.copy()
                    plot_gdf['value'] = var_mean.values
                    plot_gdf = plot_gdf.to_crs(epsg=3857)
                    vmin, vmax = np.percentile(var_mean.values, [2, 98])
                    plot_gdf.plot(column='value', ax=ax1, vmin=vmin, vmax=vmax, cmap='RdYlBu', legend=True)
                    ax1.set_axis_off()
                
                mean_ts = ds[var_name].mean(dim='hru').compute()
                ax2.plot(mean_ts.time, mean_ts, color=self.plot_config.COLOR_SIMULATED_PRIMARY)
                self._apply_standard_styling(ax2, xlabel='Date', ylabel=var_name, title=f'Mean Time Series: {var_name}', legend=False)
                self._format_date_axis(ax2)
                
                plot_file = plot_dir / f'{var_name}.png'
                self._save_and_close(fig, plot_file)
                plot_paths[var_name] = str(plot_file)
            ds.close()
        except Exception as e:
            self.logger.error(f"Error in plot_summa_outputs: {str(e)}")
        return plot_paths

    def plot_ngen_results(self, sim_df: pd.DataFrame, obs_df: Optional[pd.DataFrame], experiment_id: str, results_dir: Path) -> Optional[str]:
        """Visualize NGen streamflow plots."""
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
            self.logger.error(f"Error in plot_ngen_results: {str(e)}")
            return None

    def plot_lstm_results(self, results_df: pd.DataFrame, obs_streamflow: pd.DataFrame, obs_snow: pd.DataFrame, use_snow: bool, output_dir: Path, experiment_id: str) -> Optional[str]:
        """Visualize LSTM simulation results."""
        plt, _ = self._setup_matplotlib()
        from matplotlib.gridspec import GridSpec  # type: ignore
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
        except Exception as e:
            self.logger.error(f"Error in plot_lstm_results: {str(e)}")
            return None

    def plot_hype_results(self, sim_flow: pd.DataFrame, obs_flow: pd.DataFrame, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path) -> Optional[str]:
        """Visualize HYPE streamflow comparison."""
        plt, _ = self._setup_matplotlib()
        try:
            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)
            ax.plot(sim_flow.index, sim_flow['HYPE_discharge_cms'], label='Simulated', color='blue')
            ax.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red')
            self._apply_standard_styling(ax, ylabel='Discharge (m³/s)', title=f'Streamflow Comparison - {domain_name}\nOutlet ID: {outlet_id}')
            self._format_date_axis(ax)
            plot_file = self._ensure_output_dir("results") / f"{experiment_id}_HYPE_comparison.png"
            return self._save_and_close(fig, plot_file)
        except Exception as e:
            self.logger.error(f"Error in plot_hype_results: {str(e)}")
            return None

    def plot_timeseries_results(self, df: pd.DataFrame, experiment_id: str, domain_name: str) -> Optional[str]:
        """Create timeseries comparison plot from consolidated results DataFrame."""
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_metrics

        try:
            plot_dir = self._ensure_output_dir('results')
            plot_file = plot_dir / f'{experiment_id}_timeseries_comparison.png'

            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_LARGE)
            
            # Find models in columns
            models = [c.replace('_discharge_cms', '') for c in df.columns if '_discharge_cms' in c]
            
            # Plot models
            for i, model in enumerate(models):
                col = f"{model}_discharge_cms"
                color = self.plot_config.get_color_from_palette(i)
                style = self.plot_config.get_line_style(i)
                
                # Plot with KGE in label
                metrics = calculate_metrics(df['Observed'].values, df[col].values)
                kge = metrics.get('KGE', np.nan)
                label = f'{model} (KGE: {kge:.3f})'
                
                ax.plot(df.index, df[col], label=label, color=color, linestyle=style, alpha=0.6)

            # Plot Observed on top
            ax.plot(df.index, df['Observed'], color=self.plot_config.COLOR_OBSERVED, 
                   label='Observed', linewidth=self.plot_config.LINE_WIDTH_OBSERVED, zorder=10)

            self._apply_standard_styling(
                ax, ylabel='Discharge (m³/s)', 
                title=f'Streamflow Comparison - {domain_name}',
                legend=True
            )
            self._format_date_axis(ax)
            
            plt.tight_layout()
            return self._save_and_close(fig, plot_file)
        except Exception as e:
            self.logger.error(f"Error in plot_timeseries_results: {str(e)}")
            return None

    def plot_diagnostics(self, df: pd.DataFrame, experiment_id: str, domain_name: str) -> Optional[str]:
        """Create diagnostic plots (scatter and FDC) for each model."""
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_metrics, calculate_flow_duration_curve

        try:
            plot_dir = self._ensure_output_dir('results')
            plot_file = plot_dir / f'{experiment_id}_diagnostic_plots.png'

            models = [c.replace('_discharge_cms', '') for c in df.columns if '_discharge_cms' in c]
            n_models = len(models)
            if n_models == 0: return None

            fig = plt.figure(figsize=(15, 5 * n_models))
            gs = plt.GridSpec(n_models, 2)

            for i, model in enumerate(models):
                col = f"{model}_discharge_cms"
                color = self.plot_config.get_color_from_palette(i)
                
                # Scatter plot
                ax_scatter = fig.add_subplot(gs[i, 0])
                ax_scatter.scatter(df['Observed'], df[col], alpha=0.5, s=10, color=color)
                
                # 1:1 line
                max_val = max(df['Observed'].max(), df[col].max())
                ax_scatter.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                
                metrics = calculate_metrics(df['Observed'].values, df[col].values)
                self._add_metrics_text(ax_scatter, metrics)
                
                self._apply_standard_styling(ax_scatter, xlabel='Observed', ylabel='Simulated', title=f'{model} - Scatter')
                
                # FDC
                ax_fdc = fig.add_subplot(gs[i, 1])
                exc_obs, f_obs = calculate_flow_duration_curve(df['Observed'].values)
                exc_sim, f_sim = calculate_flow_duration_curve(df[col].values)
                
                ax_fdc.plot(exc_obs, f_obs, 'k-', label='Observed')
                ax_fdc.plot(exc_sim, f_sim, color=color, label=model)
                
                ax_fdc.set_xscale('log')
                ax_fdc.set_yscale('log')
                self._apply_standard_styling(ax_fdc, xlabel='Exceedance', ylabel='Discharge', title=f'{model} - FDC', legend=True)

            plt.tight_layout()
            return self._save_and_close(fig, plot_file)
        except Exception as e:
            self.logger.error(f"Error in plot_diagnostics: {str(e)}")
            return None

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        if 'sensitivity_data' in kwargs and 'output_file' in kwargs:
            return self.plot_sensitivity_analysis(
                kwargs['sensitivity_data'],
                kwargs['output_file'],
                kwargs.get('plot_type', 'single')
            )
        elif 'drop_data' in kwargs and 'optimal_threshold' in kwargs:
            return self.plot_drop_analysis(
                kwargs['drop_data'],
                kwargs['optimal_threshold'],
                kwargs.get('project_dir', Path('.'))
            )
        return None
