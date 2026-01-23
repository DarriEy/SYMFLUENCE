"""
Model comparison plotter for creating multi-panel overview visualizations.

Creates comprehensive comparison sheets showing observations vs all models,
including time series, flow duration curves, scatter plots, metrics, and
monthly/residual analysis. Based on Camille Gautier's overview_model_comparison.

Reference: https://github.com/camille-gautier/overview_model_comparison
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.plot_utils import (
    calculate_metrics,
    calculate_flow_duration_curve,
)


class ModelComparisonPlotter(BasePlotter):
    """Creates model comparison overview sheets for obs + all models.

    Generates comprehensive multi-panel comparison plots showing:
    - Time series comparison (observations vs all models)
    - Flow duration curves (log-log scale)
    - Scatter plots (obs vs sim per model with 1:1 line)
    - Performance metrics table (KGE, NSE, RMSE, Bias)
    - Monthly boxplots for seasonal analysis
    - Residual analysis (histogram/bias bars)

    Visualization Layout:
        +------------------------------------------------------------------+
        |                    Model Comparison Overview                      |
        +------------------------------------------------------------------+
        |  TIME SERIES COMPARISON (full width)             | METRICS TABLE  |
        |  - Black: Observations                           | Model | KGE    |
        |  - Colors: Each model                            | SUMMA | 0.72   |
        |                                                  | FUSE  | 0.68   |
        +------------------------------------------+-------+----------------+
        |  FLOW DURATION CURVES                    |  MONTHLY AGGREGATION   |
        |  (log-log, all models + obs)             |  (box plots by month)  |
        +------------------------------------------+------------------------+
        |  SCATTER PLOTS (obs vs sim per model)    |  RESIDUAL ANALYSIS     |
        |  - With 1:1 line, R² in corner           |  (histogram/bias bars) |
        +------------------------------------------+------------------------+

    Data Sources:
        Primary: project_dir/results/{experiment_id}_results.csv
        Fallback obs: project_dir/observations/streamflow/preprocessed/
                      {domain}_streamflow_processed.csv

    Output:
        project_dir/reporting/model_comparison/{experiment_id}_comparison_overview.png
    """

    # Color palette for models
    MODEL_COLORS = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
    ]

    def plot_model_comparison_overview(
        self,
        experiment_id: str = 'default',
        context: str = 'run_model'
    ) -> Optional[str]:
        """Create comprehensive model comparison overview plot.

        Args:
            experiment_id: Experiment ID for loading results and naming output
            context: Context for the comparison ('run_model' or 'calibrate_model')

        Returns:
            Path to saved plot, or None if creation failed
        """
        try:
            # Collect data - pass context to load from correct location
            results_df, obs_series = self._collect_model_data(experiment_id, context)

            if results_df is None or results_df.empty:
                self.logger.warning("No model data available for comparison overview")
                return None

            # Find model columns (discharge columns)
            model_cols = [c for c in results_df.columns
                         if 'discharge' in c.lower() and 'obs' not in c.lower()]

            if not model_cols:
                self.logger.warning("No model discharge columns found in results")
                return None

            # Calculate metrics for all models
            metrics_dict = self._calculate_all_metrics(results_df, obs_series, model_cols)

            # Setup figure with GridSpec layout
            plt, _ = self._setup_matplotlib()
            import matplotlib.gridspec as gridspec  # type: ignore

            fig = plt.figure(figsize=(18, 14))

            # Create GridSpec layout
            # Row heights: title area, timeseries, FDC/monthly, scatter/residuals
            gs = gridspec.GridSpec(4, 3, height_ratios=[0.05, 1, 1, 1],
                                   width_ratios=[2, 1, 1],
                                   hspace=0.3, wspace=0.3)

            # Title
            context_title = "Post-Calibration" if context == 'calibrate_model' else "Model Run"
            fig.suptitle(f'Model Comparison Overview - {context_title}\n{experiment_id}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Panel 1: Time series (row 1, cols 0-1)
            ax_ts = fig.add_subplot(gs[1, 0:2])
            self._plot_timeseries_panel(ax_ts, results_df, obs_series, model_cols)

            # Panel 2: Metrics table (row 1, col 2)
            ax_metrics = fig.add_subplot(gs[1, 2])
            self._plot_metrics_table(ax_metrics, metrics_dict)

            # Panel 3: Flow Duration Curves (row 2, col 0)
            ax_fdc = fig.add_subplot(gs[2, 0])
            self._plot_fdc_panel(ax_fdc, results_df, obs_series, model_cols)

            # Panel 4: Monthly boxplots (row 2, cols 1-2)
            ax_monthly = fig.add_subplot(gs[2, 1:3])
            self._plot_monthly_boxplots(ax_monthly, results_df, obs_series, model_cols)

            # Panel 5: Scatter plots (row 3, cols 0-1)
            # Create subplot grid for scatter plots
            n_models = len(model_cols)
            if n_models > 0:
                scatter_gs = gridspec.GridSpecFromSubplotSpec(
                    1, min(n_models, 3),
                    subplot_spec=gs[3, 0:2],
                    wspace=0.3
                )
                scatter_axes = [fig.add_subplot(scatter_gs[0, i])
                               for i in range(min(n_models, 3))]
                self._plot_scatter_panels(scatter_axes, results_df, obs_series, model_cols)

            # Panel 6: Residual analysis (row 3, col 2)
            ax_residual = fig.add_subplot(gs[3, 2])
            self._plot_residual_analysis(ax_residual, results_df, obs_series, model_cols)

            # Ensure output directory exists
            output_dir = self._ensure_output_dir('model_comparison')

            # Save plot
            plot_path = output_dir / f"{experiment_id}_comparison_overview.png"
            return self._save_and_close(fig, plot_path)

        except Exception as e:
            self.logger.error(f"Error creating model comparison overview: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _collect_model_data(
        self,
        experiment_id: str,
        context: str = 'run_model'
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load results data and observations.

        If results CSV doesn't exist, auto-detects model outputs in the
        simulations directory and creates the results file.

        For calibration context, prioritizes loading from final_evaluation
        directory where calibrated model outputs are stored.

        Args:
            experiment_id: Experiment ID for locating results file
            context: Context for loading ('run_model' or 'calibrate_model')

        Returns:
            Tuple of (results_df, obs_series) or (None, None) if loading failed
        """
        try:
            # For calibration context, check for calibrated results file first
            if context == 'calibrate_model':
                calibrated_results_file = self.project_dir / "results" / f"{experiment_id}_calibrated_results.csv"
                if calibrated_results_file.exists():
                    results_df = pd.read_csv(calibrated_results_file, index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded calibrated results from: {calibrated_results_file}")
                else:
                    # Auto-detect from final_evaluation directory
                    self.logger.info("Calibrated results file not found, auto-detecting from final_evaluation...")
                    results_df = self._auto_collect_model_outputs(experiment_id, context)
            else:
                results_file = self.project_dir / "results" / f"{experiment_id}_results.csv"
                # Try loading existing results CSV
                if results_file.exists():
                    results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
                else:
                    # Auto-detect and load model outputs
                    self.logger.info("Results file not found, auto-detecting model outputs...")
                    results_df = self._auto_collect_model_outputs(experiment_id, context)

                if results_df is not None and not results_df.empty:
                    # Save for future use
                    results_file.parent.mkdir(parents=True, exist_ok=True)
                    results_df.to_csv(results_file)
                    self.logger.info(f"Auto-generated results saved to: {results_file}")
                else:
                    self.logger.warning("No model outputs found to collect")
                    return None, None

            # For calibration, save calibrated results file
            if context == 'calibrate_model' and results_df is not None and not results_df.empty:
                calibrated_results_file = self.project_dir / "results" / f"{experiment_id}_calibrated_results.csv"
                calibrated_results_file.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(calibrated_results_file)
                self.logger.info(f"Calibrated results saved to: {calibrated_results_file}")

            # Find observation column in results
            obs_series = None
            for col in results_df.columns:
                if 'obs' in col.lower() or 'observed' in col.lower():
                    obs_series = results_df[col]
                    break

            # Fallback: load from observations directory
            if obs_series is None:
                obs_series = self._load_observations(results_df.index)

                # Add to results_df if loaded
                if obs_series is not None:
                    results_df['observed_discharge_cms'] = obs_series

            # Filter out spinup period from all data sources
            spinup_end = self._get_spinup_end_date()
            if spinup_end is not None and results_df is not None:
                original_len = len(results_df)
                results_df = results_df[results_df.index > spinup_end]
                if obs_series is not None:
                    obs_series = obs_series[obs_series.index > spinup_end]
                filtered_count = original_len - len(results_df)
                if filtered_count > 0:
                    self.logger.info(f"Filtered {filtered_count} spinup timesteps (before {spinup_end.strftime('%Y-%m-%d')})")

            return results_df, obs_series

        except Exception as e:
            self.logger.error(f"Error collecting model data: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None, None

    def _auto_collect_model_outputs(
        self,
        experiment_id: str,
        context: str = 'run_model'
    ) -> Optional[pd.DataFrame]:
        """Auto-detect and load model outputs from simulations directory.

        Scans for known model output patterns and loads streamflow data.
        For calibration context, prioritizes final_evaluation directory.

        Args:
            experiment_id: Experiment ID
            context: Context for loading ('run_model' or 'calibrate_model')

        Returns:
            DataFrame with model outputs, or None if no outputs found
        """
        import xarray as xr

        # For calibration context, prioritize final_evaluation directory
        if context == 'calibrate_model':
            final_eval_dir = self.project_dir / "optimization" / "final_evaluation"
            if final_eval_dir.exists():
                self.logger.info(f"Loading calibrated outputs from: {final_eval_dir}")
                results_df = self._load_from_directory(final_eval_dir, experiment_id, label_suffix="_calibrated")
                if results_df is not None:
                    return results_df
                self.logger.info("No outputs in final_evaluation, checking simulations directory...")

        sim_dir = self.project_dir / "simulations" / experiment_id
        if not sim_dir.exists():
            return None

        results_df = None
        basin_area_m2 = self._get_basin_area_m2()

        # Get spinup end date for filtering
        spinup_period = self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default='',
            dict_key='SPINUP_PERIOD'
        )
        spinup_end = None
        if spinup_period and ',' in str(spinup_period):
            try:
                spinup_end = pd.to_datetime(str(spinup_period).split(',')[1].strip())
            except (ValueError, TypeError):
                pass

        # Check for mizuRoute output first (distributed routing)
        # Try multiple possible locations for mizuRoute output
        mizu_dir_candidates = [
            sim_dir / "mizuRoute",           # Direct mizuRoute subdirectory
            sim_dir / "SUMMA" / "mizuRoute", # mizuRoute nested under SUMMA (common for semi-distributed)
        ]

        mizu_files = []
        mizu_dir = None
        for candidate_dir in mizu_dir_candidates:
            if candidate_dir.exists():
                candidate_files = list(candidate_dir.glob("*.nc"))
                if candidate_files:
                    mizu_dir = candidate_dir
                    mizu_files = candidate_files
                    self.logger.debug(f"Found mizuRoute output in: {mizu_dir}")
                    break

        if mizu_files:
            self.logger.info(f"Found {len(mizu_files)} mizuRoute output files")
            try:
                # Load all files using open_mfdataset for multi-year simulations
                if len(mizu_files) > 1:
                    ds = xr.open_mfdataset(
                        sorted(mizu_files),
                        combine='by_coords',
                        data_vars='minimal',
                        compat='override',
                        join='override'
                    )
                    self.logger.info(f"Loaded {len(mizu_files)} files with open_mfdataset")
                else:
                    ds = xr.open_dataset(mizu_files[0])

                # Try multiple mizuRoute output variable names
                # Priority: cumulative/routed flow first, then per-reach delayed runoff last
                routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'sumUpstreamRunoff', 'dlayRunoff', 'averageRoutedRunoff']
                routing_var = None
                for var_name in routing_vars:
                    if var_name in ds:
                        routing_var = var_name
                        break

                if routing_var:
                    self.logger.info(f"Using mizuRoute variable: {routing_var}")
                    # Get outlet reach (last segment or by SIM_REACH_ID)
                    sim_reach_id = self._get_config_value(
                        lambda: self.config.routing.sim_reach_id,
                        default=None,
                        dict_key='SIM_REACH_ID'
                    )

                    var_data = ds[routing_var]

                    # Handle different dimension names (seg vs reachID)
                    if 'seg' in var_data.dims:
                        if sim_reach_id and 'reachID' in ds:
                            segment_mask = ds['reachID'].values == int(sim_reach_id)
                            streamflow = var_data.sel(seg=segment_mask).to_pandas()
                        else:
                            # Use segment with highest mean runoff (outlet)
                            segment_means = var_data.mean(dim='time').values
                            outlet_idx = int(np.argmax(segment_means))
                            streamflow = var_data.isel(seg=outlet_idx).to_pandas()
                    elif 'reachID' in var_data.dims:
                        reach_means = var_data.mean(dim='time').values
                        outlet_idx = int(np.argmax(reach_means))
                        streamflow = var_data.isel(reachID=outlet_idx).to_pandas()
                    else:
                        # No spatial dimension, use directly
                        streamflow = var_data.to_pandas()

                    # Resample hourly to daily
                    streamflow.index = streamflow.index.round('h')
                    streamflow = streamflow.resample('D').mean()

                    results_df = pd.DataFrame(index=streamflow.index)
                    results_df['SUMMA_discharge_cms'] = streamflow
                    self.logger.info(f"Loaded {len(streamflow)} discharge values from mizuRoute")
                else:
                    self.logger.warning(f"No recognized routing variable found in mizuRoute output. Available: {list(ds.data_vars)}")
                ds.close()
            except Exception as e:
                self.logger.warning(f"Error loading mizuRoute output: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # Check for SUMMA output (lumped model without routing)
        if results_df is None:
            summa_dir = sim_dir / "SUMMA"
            summa_files = list(summa_dir.glob("*_timestep.nc")) if summa_dir.exists() else []

            if summa_files:
                self.logger.info("Found SUMMA lumped output")
                try:
                    ds = xr.open_dataset(summa_files[0])
                    if 'averageRoutedRunoff' in ds:
                        streamflow = ds['averageRoutedRunoff'].to_pandas()

                        # Convert from m/s to m³/s using basin area
                        if basin_area_m2:
                            streamflow = streamflow * basin_area_m2

                        results_df = pd.DataFrame(index=streamflow.index)
                        results_df['SUMMA_discharge_cms'] = streamflow
                    ds.close()
                except Exception as e:
                    self.logger.warning(f"Error loading SUMMA output: {e}")

        # Check for FUSE output
        if results_df is None:
            fuse_dir = sim_dir / "FUSE"
            fuse_files = list(fuse_dir.glob("*_runs_best.nc")) if fuse_dir.exists() else []

            if fuse_files:
                self.logger.info("Found FUSE output")
                try:
                    ds = xr.open_dataset(fuse_files[0])
                    if 'q_routed' in ds:
                        streamflow = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0).to_pandas()

                        # FUSE outputs mm/day, convert to cms
                        if basin_area_m2:
                            streamflow = streamflow * (basin_area_m2 / 1e6) / 86400

                        results_df = pd.DataFrame(index=streamflow.index)
                        results_df['FUSE_discharge_cms'] = streamflow
                    ds.close()
                except Exception as e:
                    self.logger.warning(f"Error loading FUSE output: {e}")

        # Check for GR output
        if results_df is None:
            gr_dir = sim_dir / "GR"
            gr_files = list(gr_dir.glob("GR_results.csv")) if gr_dir.exists() else []

            if gr_files:
                self.logger.info("Found GR output")
                try:
                    gr_df = pd.read_csv(gr_files[0], parse_dates=['Date'])
                    gr_df.set_index('Date', inplace=True)

                    # Find discharge column
                    for col in gr_df.columns:
                        if 'sim' in col.lower() or 'qsim' in col.lower():
                            results_df = pd.DataFrame(index=gr_df.index)
                            results_df['GR_discharge_cms'] = gr_df[col]
                            break
                except Exception as e:
                    self.logger.warning(f"Error loading GR output: {e}")

        # Load and add observations
        if results_df is not None:
            obs_series = self._load_observations(results_df.index)
            if obs_series is not None:
                results_df['observed_discharge_cms'] = obs_series

            # Filter by spinup if configured
            if spinup_end is not None:
                results_df = results_df[results_df.index > spinup_end]

        return results_df

    def _load_from_directory(
        self,
        output_dir: Path,
        experiment_id: str,
        label_suffix: str = ""
    ) -> Optional[pd.DataFrame]:
        """Load model outputs from a specific directory.

        Scans for known model output patterns (SUMMA, mizuRoute, etc.)
        and loads streamflow data.

        Args:
            output_dir: Directory to search for outputs
            experiment_id: Experiment ID
            label_suffix: Suffix to add to column names (e.g., "_calibrated")

        Returns:
            DataFrame with model outputs, or None if no outputs found
        """
        import xarray as xr

        if not output_dir.exists():
            return None

        results_df = None
        basin_area_m2 = self._get_basin_area_m2()

        # Get spinup end date for filtering
        spinup_period = self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default='',
            dict_key='SPINUP_PERIOD'
        )
        spinup_end = None
        if spinup_period and ',' in str(spinup_period):
            try:
                spinup_end = pd.to_datetime(str(spinup_period).split(',')[1].strip())
            except (ValueError, TypeError):
                pass

        # Check for SUMMA timestep output
        summa_files = list(output_dir.glob("*_timestep.nc"))
        if summa_files:
            self.logger.info(f"Found SUMMA output in {output_dir}")
            try:
                ds = xr.open_dataset(summa_files[0])
                if 'averageRoutedRunoff' in ds:
                    # Handle different data structures (lumped vs distributed)
                    var_data = ds['averageRoutedRunoff']

                    # Check if there's a hru/gru dimension
                    if 'hru' in var_data.dims:
                        streamflow = var_data.isel(hru=0).to_pandas()
                    elif 'gru' in var_data.dims:
                        streamflow = var_data.isel(gru=0).to_pandas()
                    else:
                        streamflow = var_data.to_pandas()

                    # Convert from m/s to m³/s using basin area
                    if basin_area_m2:
                        streamflow = streamflow * basin_area_m2

                    # Resample to daily if hourly
                    if hasattr(streamflow.index, 'freq') or len(streamflow) > 365 * 4:
                        streamflow.index = pd.to_datetime(streamflow.index)
                        streamflow = streamflow.resample('D').mean()

                    results_df = pd.DataFrame(index=streamflow.index)
                    col_name = f'SUMMA{label_suffix}_discharge_cms'
                    results_df[col_name] = streamflow
                    self.logger.info(f"Loaded {len(streamflow)} discharge values from calibrated SUMMA output")
                ds.close()
            except Exception as e:
                self.logger.warning(f"Error loading SUMMA output from {output_dir}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # Check for mizuRoute output
        if results_df is None:
            mizu_files = list(output_dir.glob("*.nc"))
            # Filter to likely mizuRoute files (contain routing variables)
            for mizu_file in mizu_files:
                try:
                    ds = xr.open_dataset(mizu_file)
                    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'sumUpstreamRunoff', 'dlayRunoff']
                    routing_var = None
                    for var_name in routing_vars:
                        if var_name in ds:
                            routing_var = var_name
                            break

                    if routing_var:
                        self.logger.info(f"Found mizuRoute output: {routing_var}")
                        var_data = ds[routing_var]

                        # Handle spatial dimensions
                        if 'seg' in var_data.dims:
                            segment_means = var_data.mean(dim='time').values
                            outlet_idx = int(np.argmax(segment_means))
                            streamflow = var_data.isel(seg=outlet_idx).to_pandas()
                        elif 'reachID' in var_data.dims:
                            reach_means = var_data.mean(dim='time').values
                            outlet_idx = int(np.argmax(reach_means))
                            streamflow = var_data.isel(reachID=outlet_idx).to_pandas()
                        else:
                            streamflow = var_data.to_pandas()

                        # Resample to daily
                        streamflow.index = pd.to_datetime(streamflow.index)
                        streamflow = streamflow.resample('D').mean()

                        results_df = pd.DataFrame(index=streamflow.index)
                        col_name = f'SUMMA{label_suffix}_discharge_cms'
                        results_df[col_name] = streamflow
                        ds.close()
                        break
                    ds.close()
                except Exception:
                    continue

        # Load and add observations
        if results_df is not None:
            obs_series = self._load_observations(results_df.index)
            if obs_series is not None:
                results_df['observed_discharge_cms'] = obs_series

            # Filter by spinup if configured
            if spinup_end is not None:
                results_df = results_df[results_df.index > spinup_end]

        return results_df

    def _load_observations(
        self,
        target_index: pd.DatetimeIndex
    ) -> Optional[pd.Series]:
        """Load observations from preprocessed streamflow file.

        Args:
            target_index: DatetimeIndex to align observations to

        Returns:
            Observation series or None if not found
        """
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key='DOMAIN_NAME'
        )
        obs_path = (self.project_dir / "observations" / "streamflow" /
                   "preprocessed" / f"{domain_name}_streamflow_processed.csv")

        if not obs_path.exists():
            self.logger.warning(f"Observations file not found: {obs_path}")
            return None

        try:
            obs_df = pd.read_csv(obs_path, parse_dates=['datetime'])
            obs_df.set_index('datetime', inplace=True)

            # Find discharge column
            for col in obs_df.columns:
                if 'discharge' in col.lower() or col.lower() in ['q', 'flow']:
                    return obs_df[col].reindex(target_index)

            return None
        except Exception as e:
            self.logger.warning(f"Error loading observations: {e}")
            return None

    def _get_basin_area_m2(self) -> Optional[float]:
        """Get basin area in m² for unit conversion.

        Returns:
            Basin area in m², or None if not available
        """
        try:
            import geopandas as gpd

            domain_name = self._get_config_value(
                lambda: self.config.domain.name,
                dict_key='DOMAIN_NAME'
            )
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                dict_key='DOMAIN_DEFINITION_METHOD'
            )

            basin_path = (self.project_dir / 'shapefiles' / 'river_basins' /
                         f"{domain_name}_riverBasins_{domain_method}.shp")

            if not basin_path.exists():
                # Try common alternatives
                for alt_method in ['lumped', 'delineate', 'subset']:
                    alt_path = (self.project_dir / 'shapefiles' / 'river_basins' /
                               f"{domain_name}_riverBasins_{alt_method}.shp")
                    if alt_path.exists():
                        basin_path = alt_path
                        break

            if basin_path.exists():
                gdf = gpd.read_file(str(basin_path))
                # Project to UTM for accurate area calculation
                gdf_proj = gdf.to_crs('EPSG:32611')
                return float(gdf_proj.geometry.area.sum())

            return None
        except Exception as e:
            self.logger.warning(f"Could not determine basin area: {e}")
            return None

    def _get_spinup_end_date(self) -> Optional[pd.Timestamp]:
        """Get the end date of the spinup period from config.

        Returns:
            Spinup end date as pandas Timestamp, or None if not configured
        """
        spinup_period = self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default='',
            dict_key='SPINUP_PERIOD'
        )

        if not spinup_period or not isinstance(spinup_period, str):
            return None

        if ',' not in spinup_period:
            return None

        try:
            # SPINUP_PERIOD format: "start_date, end_date"
            spinup_end_str = spinup_period.split(',')[1].strip()
            return pd.to_datetime(spinup_end_str)
        except Exception as e:
            self.logger.debug(f"Could not parse spinup period: {e}")
            return None

    def _calculate_all_metrics(
        self,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for all models.

        Args:
            results_df: DataFrame with model results
            obs_series: Observation series (or None)
            model_cols: List of model column names

        Returns:
            Dict mapping model names to their metrics dicts
        """
        metrics_dict: Dict[str, Dict[str, float]] = {}

        if obs_series is None:
            return metrics_dict

        obs_values = obs_series.values

        for col in model_cols:
            sim_values = results_df[col].values

            # Get aligned, valid data
            valid_mask = ~(np.isnan(obs_values) | np.isnan(sim_values))
            obs_clean = obs_values[valid_mask]
            sim_clean = sim_values[valid_mask]

            if len(obs_clean) < 10:
                continue

            # Calculate metrics using existing utility
            metrics = calculate_metrics(obs_clean, sim_clean)

            # Calculate bias
            mean_obs = np.mean(obs_clean)
            mean_sim = np.mean(sim_clean)
            bias = ((mean_sim - mean_obs) / mean_obs) * 100 if mean_obs != 0 else np.nan
            metrics['Bias%'] = bias

            # Extract model name from column
            model_name = col.replace('_discharge_cms', '').replace('_discharge', '')
            metrics_dict[model_name] = metrics

        return metrics_dict

    def _plot_timeseries_panel(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> None:
        """Plot time series comparison panel.

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with model results
            obs_series: Observation series
            model_cols: List of model column names
        """
        # Plot observations
        if obs_series is not None:
            ax.plot(results_df.index, obs_series,
                   color='black', linewidth=1.5, label='Observed', zorder=10)

        # Plot each model
        for i, col in enumerate(model_cols):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            model_name = col.replace('_discharge_cms', '').replace('_discharge', '')
            ax.plot(results_df.index, results_df[col],
                   color=color, linewidth=1.0, alpha=0.8, label=model_name)

        self._apply_standard_styling(
            ax,
            xlabel='Date',
            ylabel='Discharge (m³/s)',
            title='Time Series Comparison',
            legend=True,
            legend_loc='upper right'
        )
        self._format_date_axis(ax, format_type='full')

    def _plot_fdc_panel(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> None:
        """Plot flow duration curves panel.

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with model results
            obs_series: Observation series
            model_cols: List of model column names
        """
        # Plot observed FDC
        if obs_series is not None:
            exc_obs, flows_obs = calculate_flow_duration_curve(obs_series.values)
            if len(exc_obs) > 0:
                ax.plot(exc_obs * 100, flows_obs, color='black',
                       linewidth=2, label='Observed', zorder=10)

        # Plot model FDCs
        for i, col in enumerate(model_cols):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            exc, flows = calculate_flow_duration_curve(results_df[col].values)
            if len(exc) > 0:
                model_name = col.replace('_discharge_cms', '').replace('_discharge', '')
                ax.plot(exc * 100, flows, color=color, linewidth=1.5,
                       alpha=0.8, label=model_name)

        ax.set_yscale('log')
        ax.set_xlim([0, 100])

        self._apply_standard_styling(
            ax,
            xlabel='Exceedance Probability (%)',
            ylabel='Discharge (m³/s)',
            title='Flow Duration Curves',
            legend=True,
            legend_loc='upper right'
        )

    def _plot_scatter_panels(
        self,
        axes: List[Any],
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> None:
        """Plot scatter plots (obs vs sim) for each model.

        Args:
            axes: List of matplotlib axes
            results_df: DataFrame with model results
            obs_series: Observation series
            model_cols: List of model column names
        """
        if obs_series is None:
            return

        obs_values = obs_series.values

        for i, (ax, col) in enumerate(zip(axes, model_cols[:len(axes)])):
            sim_values = results_df[col].values

            # Get valid data
            valid_mask = ~(np.isnan(obs_values) | np.isnan(sim_values))
            obs_clean = obs_values[valid_mask]
            sim_clean = sim_values[valid_mask]

            if len(obs_clean) < 10:
                ax.text(0.5, 0.5, 'Insufficient data',
                       transform=ax.transAxes, ha='center', va='center')
                continue

            # Scatter plot
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            ax.scatter(obs_clean, sim_clean, c=color, alpha=0.3, s=10, edgecolors='none')

            # 1:1 line
            max_val = max(np.max(obs_clean), np.max(sim_clean))
            min_val = min(np.min(obs_clean), np.min(sim_clean))
            ax.plot([min_val, max_val], [min_val, max_val],
                   'k--', linewidth=1, label='1:1 line')

            # Calculate R²
            correlation = np.corrcoef(obs_clean, sim_clean)[0, 1]
            r_squared = correlation ** 2

            # Add R² annotation
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            model_name = col.replace('_discharge_cms', '').replace('_discharge', '')
            self._apply_standard_styling(
                ax,
                xlabel='Observed (m³/s)',
                ylabel='Simulated (m³/s)',
                title=model_name,
                legend=False
            )

    def _plot_metrics_table(
        self,
        ax: Any,
        metrics_dict: Dict[str, Dict[str, float]]
    ) -> None:
        """Plot performance metrics as a table.

        Args:
            ax: Matplotlib axis
            metrics_dict: Dict mapping model names to metrics dicts
        """
        ax.axis('off')

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
                f"{metrics.get('KGE', np.nan):.3f}",
                f"{metrics.get('NSE', np.nan):.3f}",
                f"{metrics.get('RMSE', np.nan):.2f}",
                f"{metrics.get('Bias%', np.nan):+.1f}%"
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

        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)

    def _plot_monthly_boxplots(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> None:
        """Plot monthly aggregation boxplots.

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with model results
            obs_series: Observation series
            model_cols: List of model column names
        """
        plt, _ = self._setup_matplotlib()

        # Add month column
        months = results_df.index.month
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

        # Prepare data for boxplot
        positions = np.arange(1, 13)

        # Plot observed boxplots
        if obs_series is not None:
            obs_monthly = [obs_series[months == m].dropna().values for m in range(1, 13)]
            bp_obs = ax.boxplot(obs_monthly, positions=positions - 0.2,
                               widths=0.15, patch_artist=True)
            for patch in bp_obs['boxes']:
                patch.set_facecolor('black')
                patch.set_alpha(0.5)

        # Plot model boxplots (first model only to avoid clutter)
        if model_cols:
            col = model_cols[0]
            model_monthly = [results_df[col][months == m].dropna().values
                            for m in range(1, 13)]
            bp_model = ax.boxplot(model_monthly, positions=positions + 0.2,
                                 widths=0.15, patch_artist=True)
            color = self.MODEL_COLORS[0]
            for patch in bp_model['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(month_names)

        # Create legend
        from matplotlib.patches import Patch  # type: ignore
        legend_elements = [Patch(facecolor='black', alpha=0.5, label='Observed')]
        if model_cols:
            model_name = model_cols[0].replace('_discharge_cms', '').replace('_discharge', '')
            legend_elements.append(
                Patch(facecolor=self.MODEL_COLORS[0], alpha=0.5, label=model_name)
            )
        ax.legend(handles=legend_elements, loc='upper right')

        self._apply_standard_styling(
            ax,
            xlabel='Month',
            ylabel='Discharge (m³/s)',
            title='Monthly Distribution',
            legend=False  # Manual legend above
        )

    def _plot_residual_analysis(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str]
    ) -> None:
        """Plot residual analysis (bias by month).

        Args:
            ax: Matplotlib axis
            results_df: DataFrame with model results
            obs_series: Observation series
            model_cols: List of model column names
        """
        if obs_series is None or not model_cols:
            ax.text(0.5, 0.5, 'No data for residual analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            return

        months = results_df.index.month
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

        # Calculate monthly bias for first model
        col = model_cols[0]
        obs_values = obs_series.values
        sim_values = results_df[col].values

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

        # Create bar plot
        positions = np.arange(1, 13)
        colors = [self.MODEL_COLORS[0] if b >= 0 else '#d62728' for b in monthly_bias]
        ax.bar(positions, monthly_bias, color=colors, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(month_names)

        model_name = col.replace('_discharge_cms', '').replace('_discharge', '')
        self._apply_standard_styling(
            ax,
            xlabel='Month',
            ylabel='Bias (%)',
            title=f'Monthly Bias - {model_name}',
            legend=False
        )

    def plot_default_vs_calibrated_comparison(
        self,
        experiment_id: str = 'default'
    ) -> Optional[str]:
        """Create comparison plot showing default vs calibrated model performance.

        Generates a multi-panel visualization comparing:
        - Time series: Observed, Default run, Calibrated run
        - Performance metrics for both runs
        - Flow duration curves comparison
        - Improvement summary

        Args:
            experiment_id: Experiment ID for loading results

        Returns:
            Path to saved plot, or None if creation failed
        """
        try:
            # Load default run results
            default_df, obs_series = self._collect_model_data(experiment_id, context='run_model')

            # Load calibrated run results
            calibrated_df, _ = self._collect_model_data(experiment_id, context='calibrate_model')

            if default_df is None or calibrated_df is None:
                self.logger.warning("Could not load both default and calibrated results for comparison")
                return None

            # Find model columns
            default_cols = [c for c in default_df.columns
                           if 'discharge' in c.lower() and 'obs' not in c.lower()]
            calibrated_cols = [c for c in calibrated_df.columns
                              if 'discharge' in c.lower() and 'obs' not in c.lower()]

            if not default_cols or not calibrated_cols:
                self.logger.warning("No model discharge columns found")
                return None

            # Calculate metrics for both
            default_metrics = self._calculate_all_metrics(default_df, obs_series, default_cols)
            calibrated_metrics = self._calculate_all_metrics(calibrated_df, obs_series, calibrated_cols)

            # Setup figure
            plt, _ = self._setup_matplotlib()
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

            # Title
            fig.suptitle(f'Default vs Calibrated Model Comparison\n{experiment_id}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Panel 1: Time series comparison (full width)
            ax_ts = fig.add_subplot(gs[0, :])
            self._plot_comparison_timeseries(
                ax_ts, default_df, calibrated_df, obs_series,
                default_cols[0], calibrated_cols[0]
            )

            # Panel 2: Flow Duration Curves
            ax_fdc = fig.add_subplot(gs[1, 0])
            self._plot_comparison_fdc(
                ax_fdc, default_df, calibrated_df, obs_series,
                default_cols[0], calibrated_cols[0]
            )

            # Panel 3: Metrics comparison table
            ax_metrics = fig.add_subplot(gs[1, 1])
            self._plot_comparison_metrics_table(
                ax_metrics, default_metrics, calibrated_metrics,
                default_cols[0], calibrated_cols[0]
            )

            # Panel 4: Scatter plot - Default
            ax_scatter_default = fig.add_subplot(gs[2, 0])
            self._plot_single_scatter(
                ax_scatter_default, default_df[default_cols[0]], obs_series,
                'Default Run', self.MODEL_COLORS[0]
            )

            # Panel 5: Scatter plot - Calibrated
            ax_scatter_calib = fig.add_subplot(gs[2, 1])
            self._plot_single_scatter(
                ax_scatter_calib, calibrated_df[calibrated_cols[0]], obs_series,
                'Calibrated Run', self.MODEL_COLORS[1]
            )

            # Save plot
            output_dir = self._ensure_output_dir('model_comparison')
            plot_path = output_dir / f"{experiment_id}_default_vs_calibrated.png"
            return self._save_and_close(fig, plot_path)

        except Exception as e:
            self.logger.error(f"Error creating default vs calibrated comparison: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _plot_comparison_timeseries(
        self,
        ax: Any,
        default_df: pd.DataFrame,
        calibrated_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        default_col: str,
        calibrated_col: str
    ) -> None:
        """Plot time series comparing default and calibrated runs."""
        # Plot observations
        if obs_series is not None:
            ax.plot(obs_series.index, obs_series.values,
                   color='black', linewidth=1.5, label='Observed', zorder=10)

        # Plot default run
        ax.plot(default_df.index, default_df[default_col],
               color=self.MODEL_COLORS[0], linewidth=1.0, alpha=0.8,
               label='Default', linestyle='--')

        # Plot calibrated run
        ax.plot(calibrated_df.index, calibrated_df[calibrated_col],
               color=self.MODEL_COLORS[1], linewidth=1.0, alpha=0.9,
               label='Calibrated')

        self._apply_standard_styling(
            ax,
            xlabel='Date',
            ylabel='Discharge (m³/s)',
            title='Time Series: Default vs Calibrated',
            legend=True,
            legend_loc='upper right'
        )
        self._format_date_axis(ax, format_type='full')

    def _plot_comparison_fdc(
        self,
        ax: Any,
        default_df: pd.DataFrame,
        calibrated_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        default_col: str,
        calibrated_col: str
    ) -> None:
        """Plot FDC comparing default and calibrated runs."""
        # Plot observed FDC
        if obs_series is not None:
            exc_obs, flows_obs = calculate_flow_duration_curve(obs_series.values)
            if len(exc_obs) > 0:
                ax.plot(exc_obs * 100, flows_obs, color='black',
                       linewidth=2, label='Observed', zorder=10)

        # Plot default FDC
        exc_def, flows_def = calculate_flow_duration_curve(default_df[default_col].values)
        if len(exc_def) > 0:
            ax.plot(exc_def * 100, flows_def, color=self.MODEL_COLORS[0],
                   linewidth=1.5, alpha=0.8, label='Default', linestyle='--')

        # Plot calibrated FDC
        exc_cal, flows_cal = calculate_flow_duration_curve(calibrated_df[calibrated_col].values)
        if len(exc_cal) > 0:
            ax.plot(exc_cal * 100, flows_cal, color=self.MODEL_COLORS[1],
                   linewidth=1.5, alpha=0.9, label='Calibrated')

        ax.set_yscale('log')
        ax.set_xlim([0, 100])

        self._apply_standard_styling(
            ax,
            xlabel='Exceedance Probability (%)',
            ylabel='Discharge (m³/s)',
            title='Flow Duration Curves',
            legend=True,
            legend_loc='upper right'
        )

    def _plot_comparison_metrics_table(
        self,
        ax: Any,
        default_metrics: Dict[str, Dict[str, float]],
        calibrated_metrics: Dict[str, Dict[str, float]],
        default_col: str,
        calibrated_col: str
    ) -> None:
        """Plot metrics comparison table."""
        ax.axis('off')

        # Extract metrics for each run
        default_name = default_col.replace('_discharge_cms', '').replace('_discharge', '')
        calibrated_name = calibrated_col.replace('_discharge_cms', '').replace('_discharge', '')

        def_metrics = default_metrics.get(default_name, {})
        cal_metrics = calibrated_metrics.get(calibrated_name, {})

        # Prepare table data with improvement indicators
        headers = ['Metric', 'Default', 'Calibrated', 'Change']
        metrics_to_show = ['KGE', 'NSE', 'RMSE', 'Bias%']
        cell_data = []

        for metric in metrics_to_show:
            def_val = def_metrics.get(metric, np.nan)
            cal_val = cal_metrics.get(metric, np.nan)

            # Calculate change (positive = improvement for KGE/NSE, negative for RMSE/Bias)
            if not np.isnan(def_val) and not np.isnan(cal_val):
                if metric in ['KGE', 'NSE']:
                    change = cal_val - def_val
                    change_str = f"{change:+.3f}" if change != 0 else "0.000"
                elif metric == 'RMSE':
                    change = def_val - cal_val  # Lower is better
                    change_str = f"{change:+.2f}" if change != 0 else "0.00"
                else:  # Bias%
                    change = abs(def_val) - abs(cal_val)  # Closer to 0 is better
                    change_str = f"{change:+.1f}%" if change != 0 else "0.0%"
            else:
                change_str = "N/A"

            if metric == 'Bias%':
                row = [metric, f"{def_val:+.1f}%", f"{cal_val:+.1f}%", change_str]
            elif metric == 'RMSE':
                row = [metric, f"{def_val:.2f}", f"{cal_val:.2f}", change_str]
            else:
                row = [metric, f"{def_val:.3f}", f"{cal_val:.3f}", change_str]

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
        table.set_fontsize(11)
        table.scale(1.3, 1.8)

        ax.set_title('Performance Comparison\n(+ = improvement)', fontsize=12, fontweight='bold', pad=10)

    def _plot_single_scatter(
        self,
        ax: Any,
        sim_series: pd.Series,
        obs_series: Optional[pd.Series],
        label: str,
        color: str
    ) -> None:
        """Plot single scatter plot for one model run."""
        if obs_series is None:
            ax.text(0.5, 0.5, 'No observations',
                   transform=ax.transAxes, ha='center', va='center')
            return

        obs_values = obs_series.values
        sim_values = sim_series.values

        # Get valid data
        valid_mask = ~(np.isnan(obs_values) | np.isnan(sim_values))
        obs_clean = obs_values[valid_mask]
        sim_clean = sim_values[valid_mask]

        if len(obs_clean) < 10:
            ax.text(0.5, 0.5, 'Insufficient data',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Scatter plot
        ax.scatter(obs_clean, sim_clean, c=color, alpha=0.3, s=15, edgecolors='none')

        # 1:1 line
        max_val = max(np.max(obs_clean), np.max(sim_clean))
        min_val = min(np.min(obs_clean), np.min(sim_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 line')

        # Calculate R²
        correlation = np.corrcoef(obs_clean, sim_clean)[0, 1]
        r_squared = correlation ** 2

        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self._apply_standard_styling(
            ax,
            xlabel='Observed (m³/s)',
            ylabel='Simulated (m³/s)',
            title=label,
            legend=False
        )

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Main plot method - delegates to plot_model_comparison_overview.

        Returns:
            Path to saved plot or None
        """
        experiment_id = kwargs.get('experiment_id', 'default')
        context = kwargs.get('context', 'run_model')
        return self.plot_model_comparison_overview(experiment_id, context)
