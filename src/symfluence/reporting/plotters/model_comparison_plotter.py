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
            # Collect data
            results_df, obs_series = self._collect_model_data(experiment_id)

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
        experiment_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load results data and observations.

        Args:
            experiment_id: Experiment ID for locating results file

        Returns:
            Tuple of (results_df, obs_series) or (None, None) if loading failed
        """
        try:
            # Load results CSV
            results_file = self.project_dir / "results" / f"{experiment_id}_results.csv"

            if not results_file.exists():
                self.logger.warning(f"Results file not found: {results_file}")
                return None, None

            results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)

            # Find observation column in results
            obs_series = None
            for col in results_df.columns:
                if 'obs' in col.lower() or 'observed' in col.lower():
                    obs_series = results_df[col]
                    break

            # Fallback: load from observations directory
            if obs_series is None:
                domain_name = self._get_config_value(
                    lambda: self.config.domain.name,
                    dict_key='DOMAIN_NAME'
                )
                obs_path = (self.project_dir / "observations" / "streamflow" /
                           "preprocessed" / f"{domain_name}_streamflow_processed.csv")

                if obs_path.exists():
                    obs_df = pd.read_csv(obs_path, parse_dates=['datetime'])
                    obs_df.set_index('datetime', inplace=True)

                    # Find discharge column
                    for col in obs_df.columns:
                        if 'discharge' in col.lower() or col.lower() in ['q', 'flow']:
                            obs_series = obs_df[col].reindex(results_df.index)
                            break

            return results_df, obs_series

        except Exception as e:
            self.logger.error(f"Error collecting model data: {str(e)}")
            return None, None

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
        metrics_dict = {}

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

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Main plot method - delegates to plot_model_comparison_overview.

        Returns:
            Path to saved plot or None
        """
        experiment_id = kwargs.get('experiment_id', 'default')
        context = kwargs.get('context', 'run_model')
        return self.plot_model_comparison_overview(experiment_id, context)
