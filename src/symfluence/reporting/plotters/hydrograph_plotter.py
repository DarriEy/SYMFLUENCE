"""
Hydrograph and streamflow visualization.

This module provides specialized plotting for hydrograph comparisons,
flow duration curves, and streamflow analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore

from symfluence.reporting.core.base_plotter import BasePlotter


class HydrographPlotter(BasePlotter):
    """
    Specialized plotter for hydrograph visualizations.

    Handles:
    - Observed vs simulated hydrograph comparison
    - Flow duration curves
    - Streamflow comparison with multiple models
    - Hydrograph highlighting for top performers
    """

    def plot_hydrograph(
        self,
        observed: pd.Series,
        simulated: pd.Series,
        title: Optional[str] = None,
        output_file: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot observed vs simulated hydrograph.

        Args:
            observed: Observed streamflow series
            simulated: Simulated streamflow series
            title: Optional plot title
            output_file: Optional output path (uses default if None)

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_metrics

        try:
            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)

            # Plot observed
            ax.plot(
                observed.index, observed,
                label='Observed',
                color=self.plot_config.COLOR_OBSERVED,
                linewidth=self.plot_config.LINE_WIDTH_OBSERVED,
                zorder=5
            )

            # Plot simulated
            ax.plot(
                simulated.index, simulated,
                label='Simulated',
                color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )

            # Calculate and add metrics
            metrics = calculate_metrics(observed.values, simulated.values)
            self._add_metrics_text(ax, metrics)

            self._apply_standard_styling(
                ax,
                xlabel='Date',
                ylabel='Streamflow (m³/s)',
                title=title or 'Hydrograph Comparison',
                legend=True
            )
            self._format_date_axis(ax)

            plt.tight_layout()

            if output_file is None:
                output_file = self._ensure_output_dir('results') / 'hydrograph.png'

            return self._save_and_close(fig, output_file)

        except Exception as e:
            self.logger.error(f"Error in plot_hydrograph: {str(e)}")
            return None

    def plot_flow_duration_curve(
        self,
        data: pd.Series,
        label: str = 'Streamflow',
        output_file: Optional[Path] = None
    ) -> Optional[str]:
        """
        Plot flow duration curve.

        Args:
            data: Streamflow time series
            label: Label for the curve
            output_file: Optional output path

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve

        try:
            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_SMALL)

            exc, flows = calculate_flow_duration_curve(data.values)
            ax.plot(
                exc, flows,
                label=label,
                color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )

            ax.set_xscale('log')
            ax.set_yscale('log')

            self._apply_standard_styling(
                ax,
                xlabel='Exceedance Probability',
                ylabel='Streamflow (m³/s)',
                title='Flow Duration Curve',
                legend=True
            )

            plt.tight_layout()

            if output_file is None:
                output_file = self._ensure_output_dir('results') / 'flow_duration_curve.png'

            return self._save_and_close(fig, output_file)

        except Exception as e:
            self.logger.error(f"Error in plot_flow_duration_curve: {str(e)}")
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
            align_timeseries,
            calculate_flow_duration_curve,
            calculate_metrics,
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

            results_df = pd.read_csv(results_file)

            # Calculate threshold for top 5%
            if metric in ['mae', 'rmse']:
                threshold = results_df[metric].quantile(0.05)
                top_combinations = results_df[results_df[metric] <= threshold]
            else:
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
            fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)

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

            plot_file = output_folder / f'hydrograph_comparison_{metric}.png'
            saved_path = self._save_and_close(fig, plot_file)

            summary_file = output_folder / f'top_combinations_{metric}.csv'
            top_combinations.to_csv(summary_file, index=False)
            self.logger.info(f"Top combinations saved to: {summary_file}")

            return saved_path

        except Exception as e:
            self.logger.error(f"Error creating hydrograph plot: {str(e)}")
            return None

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates to plot_hydrograph() if observed and simulated provided.
        """
        if 'observed' in kwargs and 'simulated' in kwargs:
            return self.plot_hydrograph(
                kwargs['observed'],
                kwargs['simulated'],
                kwargs.get('title'),
                kwargs.get('output_file')
            )
        return None
