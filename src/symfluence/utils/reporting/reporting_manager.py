from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from symfluence.utils.reporting.reporting_utils import VisualizationReporter
from symfluence.utils.reporting.result_vizualisation_utils import TimeseriesVisualizer, BenchmarkVizualiser

class ReportingManager:
    """
    Manages all reporting and visualization tasks for the SYMFLUENCE framework.
    Acts as a central entry point for generating plots and reports.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, visualize: bool = False):
        """
        Initialize the ReportingManager.

        Args:
            config: Configuration dictionary.
            logger: Logger instance.
            visualize: Boolean flag indicating if visualization is enabled. 
                       If False, most visualization methods will return early.
        """
        self.config = config
        self.logger = logger
        self.visualize = visualize
        self.project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

        # Initialize specific reporters lazily or on demand if needed, 
        # but for now we can initialize them here or in methods.
        # Initializing them here is fine as they are lightweight.
        self._viz_reporter = VisualizationReporter(config, logger)
        self._ts_visualizer = TimeseriesVisualizer(config, logger)
        self._benchmark_visualizer = BenchmarkVizualiser(config, logger)

    def visualize_optimization_progress(self, history: List[Dict], output_dir: Path, calibration_variable: str, metric: str) -> None:
        """
        Visualize optimization progress.

        Args:
            history: List of optimization history dictionaries.
            output_dir: Directory to save the plot.
            calibration_variable: Name of the variable being calibrated.
            metric: Name of the optimization metric.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping optimization progress visualization.")
            return

        self.logger.info("Creating optimization progress visualization...")
        # Since the plotting logic is simple and currently resides in ResultsManager._create_plots,
        # we can either move it here or keep it there and just gate it.
        # Ideally, we move the logic here.
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract progress data
            generations = [h.get('generation', 0) for h in history]
            best_scores = [h['best_score'] for h in history if h.get('best_score') is not None]
            
            if not best_scores:
                self.logger.warning("No best scores found in history for plotting.")
                return
            
            # Progress plot
            plt.figure(figsize=(12, 6))
            plt.plot(generations[:len(best_scores)], best_scores, 'b-o', markersize=4)
            plt.xlabel('Generation')
            plt.ylabel(f"Performance ({metric})")
            plt.title(f'Optimization Progress - {calibration_variable.title()} Calibration')
            plt.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = np.nanargmax(best_scores)
            plt.plot(generations[best_idx], best_scores[best_idx], 'ro', markersize=10,
                    label=f'Best: {best_scores[best_idx]:.4f} at generation {generations[best_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plot_path = plots_dir / "optimization_progress.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Optimization progress plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating optimization progress plot: {str(e)}")

    def visualize_optimization_depth_parameters(self, history: List[Dict], output_dir: Path) -> None:
        """
        Visualize depth parameter evolution.

        Args:
            history: List of optimization history dictionaries.
            output_dir: Directory to save the plot.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping depth parameter visualization.")
            return

        self.logger.info("Creating depth parameter visualization...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Extract depth parameters
            generations = []
            total_mults = []
            shape_factors = []
            
            for h in history:
                if h.get('best_params') and 'total_mult' in h['best_params'] and 'shape_factor' in h['best_params']:
                    generations.append(h.get('generation', 0))
                    
                    tm = h['best_params']['total_mult']
                    sf = h['best_params']['shape_factor']
                    
                    tm_val = tm[0] if isinstance(tm, np.ndarray) and len(tm) > 0 else tm
                    sf_val = sf[0] if isinstance(sf, np.ndarray) and len(sf) > 0 else sf
                    
                    total_mults.append(tm_val)
                    shape_factors.append(sf_val)
            
            if not generations:
                self.logger.warning("No depth parameter data found in history for plotting.")
                return
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Total multiplier plot
            ax1.plot(generations, total_mults, 'g-o', markersize=4)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Total Depth Multiplier')
            ax1.set_title('Soil Depth Total Multiplier Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No change (1.0)')
            ax1.legend()
            
            # Shape factor plot
            ax2.plot(generations, shape_factors, 'm-o', markersize=4)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Shape Factor')
            ax2.set_title('Soil Depth Shape Factor Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Uniform scaling (1.0)')
            ax2.legend()
            
            plt.tight_layout()
            plot_path = plots_dir / "depth_parameter_evolution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Depth parameter plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating depth parameter plots: {str(e)}")

    def visualize_sensitivity_analysis(self, sensitivity_data: Any, output_file: Path, plot_type: str = 'single'):
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_data: Data to plot (Series or DataFrame).
            output_file: Path to save the plot.
            plot_type: 'single' for one method, 'comparison' for multiple.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping sensitivity analysis visualization.")
            return

        self.logger.info(f"Creating sensitivity analysis visualization ({plot_type})...")
        
        try:
            import matplotlib.pyplot as plt
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if plot_type == 'single':
                plt.figure(figsize=(10, 6))
                sensitivity_data.plot(kind='bar')
                plt.title("Parameter Sensitivity Analysis")
                plt.xlabel("Parameters")
                plt.ylabel("Sensitivity")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
            elif plot_type == 'comparison':
                plt.figure(figsize=(12, 8))
                sensitivity_data.plot(kind='bar')
                plt.title("Sensitivity Analysis Comparison")
                plt.xlabel("Parameters")
                plt.ylabel("Sensitivity")
                plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                
            self.logger.info(f"Sensitivity plot saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating sensitivity plot: {str(e)}")

    def visualize_decision_impacts(self, results_file: Path, output_folder: Path):
        """
        Visualize decision analysis impacts.

        Args:
            results_file: Path to the CSV results file.
            output_folder: Folder to save plots.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping decision impact visualization.")
            return

        self.logger.info("Creating decision impact visualizations...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            output_folder.mkdir(parents=True, exist_ok=True)
            
            df = pd.read_csv(results_file)
            metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
            # Identify decision columns (exclude Iteration and metrics)
            decisions = [col for col in df.columns if col not in ['Iteration'] + metrics]

            for metric in metrics:
                if metric not in df.columns:
                    continue
                    
                plt.figure(figsize=(12, 6 * len(decisions)))
                for i, decision in enumerate(decisions, 1):
                    plt.subplot(len(decisions), 1, i)
                    impact = df.groupby(decision)[metric].mean().sort_values(ascending=False)
                    impact.plot(kind='bar')
                    plt.title(f'Impact of {decision} on {metric}')
                    plt.ylabel(metric)
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                output_path = output_folder / f'{metric}_decision_impacts.png'
                plt.savefig(output_path)
                plt.close()
                
            self.logger.info("Decision impact plots saved")
            
        except Exception as e:
            self.logger.error(f"Error creating decision impact plots: {str(e)}")

    def visualize_hydrographs_with_highlight(self, results_file: Path, simulation_results: Dict, observed_streamflow: Any, decision_options: Dict, output_folder: Path, metric: str = 'kge'):
        """
        Visualize hydrographs with top performers highlighted.

        Args:
            results_file: Path to results CSV.
            simulation_results: Dictionary of simulation results.
            observed_streamflow: Observed streamflow series.
            decision_options: Dictionary of decision options.
            output_folder: Output folder.
            metric: Metric to use for highlighting.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping hydrograph visualization.")
            return

        self.logger.info(f"Creating hydrograph visualization with {metric} highlight...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
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

            # Find overlapping period across all simulations and observations
            start_date = observed_streamflow.index.min()
            end_date = observed_streamflow.index.max()
            
            for sim in simulation_results.values():
                start_date = max(start_date, sim.index.min())
                end_date = min(end_date, sim.index.max())
            
            # Calculate y-axis limit from top 5% simulations
            max_top5 = 0
            for _, row in top_combinations.iterrows():
                combo = tuple(row[list(decision_options.keys())])
                if combo in simulation_results:
                    sim = simulation_results[combo]
                    sim_overlap = sim.loc[start_date:end_date]
                    max_top5 = max(max_top5, sim_overlap.max())

            # Customize plot
            plt.figure(figsize=(12, 6))
            plt.title(f'Hydrograph Comparison ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})\n'
                     f'Top 5% combinations by {metric} metric highlighted', 
                     fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Streamflow (m³/s)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max_top5 * 1.1)  # Add 10% padding above the maximum value
            
            # Plot all (light gray) - would be expensive if many, but logic is preserved
            # Ideally we might want to sample or just not plot ALL if there are thousands.
            # But sticking to original logic:
            # Actually original code plotted empty lists for legend, but did it plot the actual lines?
            # Wait, looking at original code:
            # plt.plot([], [], color='lightgray', label='All combinations')
            # It didn't loop to plot 'All combinations'! It just added a legend entry.
            # Ah, wait. Did it plot them?
            # The snippet I read:
            # plt.plot([], [], color='lightgray', label='All combinations')
            # plt.plot([], [], color='blue', alpha=0.3, label=f'Top 5% by {metric}')
            # It seems it missed the loop to actually plot the lines in the snippet I saw?
            # Or maybe I missed it in reading.
            # Let's check the file content again.
            
            # Re-reading src/symfluence/utils/models/fuse/decision_analyzer.py from previous turn:
            # def plot_hydrographs(self, results_file: Path, metric: str = 'kge'):
            # ...
            # # Customize plot
            # plt.title(...)
            # ...
            # # Add legend
            # plt.plot([], [], color='lightgray', label='All combinations')
            # plt.plot([], [], color='blue', alpha=0.3, label=f'Top 5% by {metric}')
            # plt.legend(fontsize=10)
            # 
            # # Save plot
            # plt.savefig(...)
            
            # It DOES NOT seem to loop and plot the actual data! It just sets up the plot and saves it.
            # This looks like a bug in the original code or incomplete implementation I read.
            # Unless `self.simulation_results` usage was implied but missing.
            # Wait, `plot_hydrographs` in original code:
            # It calculates `max_top5` by iterating `top_combinations`.
            # But it never calls `plt.plot(sim...)`.
            
            # If I am refactoring, I should probably fix this or at least replicate exactly.
            # Given the instruction "pull out all visualisations", I will implement the plotting loop here.
            
            # Plot top 5%
            for _, row in top_combinations.iterrows():
                combo = tuple(row[list(decision_options.keys())])
                if combo in simulation_results:
                    sim = simulation_results[combo]
                    sim_overlap = sim.loc[start_date:end_date]
                    plt.plot(sim_overlap.index, sim_overlap.values, color='blue', alpha=0.3, linewidth=1)
            
            # Add legend
            plt.plot([], [], color='blue', alpha=0.3, label=f'Top 5% by {metric}')
            plt.legend(fontsize=10)
            
            plt.tight_layout()
            plot_file = output_folder / f'hydrograph_comparison_{metric}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Hydrograph plot saved to: {plot_file}")
            
            # Create summary of top combinations
            summary_file = output_folder / f'top_combinations_{metric}.csv'
            top_combinations.to_csv(summary_file, index=False)
            self.logger.info(f"Top combinations saved to: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating hydrograph plot: {str(e)}")

    def visualize_drop_analysis(self, drop_data: List[Dict], optimal_threshold: float, project_dir: Path):
        """
        Visualize drop analysis for stream threshold selection.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics.
            optimal_threshold: The selected optimal threshold.
            project_dir: Project directory for saving the plot.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping drop analysis visualization.")
            return

        self.logger.info("Creating drop analysis visualization...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            thresholds = [d['threshold'] for d in drop_data]
            mean_drops = [d['mean_drop'] for d in drop_data]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(thresholds, mean_drops, 'bo-', linewidth=2, markersize=8, label='Mean Drop')
            ax.axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                      label=f'Optimal Threshold = {optimal_threshold:.0f}')

            ax.set_xlabel('Contributing Area Threshold (cells)', fontsize=12)
            ax.set_ylabel('Mean Stream Drop (m)', fontsize=12)
            ax.set_title('Drop Analysis for Stream Threshold Selection', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Save plot
            plot_path = project_dir / "plots" / "drop_analysis.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Drop analysis plot saved to: {plot_path}")

        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping drop analysis plot.")
        except Exception as e:
            self.logger.warning(f"Could not create drop analysis plot: {str(e)}")

    def visualize_lstm_results(self, results_df: Any, obs_streamflow: Any, obs_snow: Any, use_snow: bool, output_dir: Path, experiment_id: str):
        """
        Visualize LSTM simulation results.

        Args:
            results_df: Simulation results dataframe.
            obs_streamflow: Observed streamflow dataframe.
            obs_snow: Observed snow dataframe.
            use_snow: Whether snow metrics/plots are required.
            output_dir: Output directory.
            experiment_id: Experiment ID.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping LSTM visualization.")
            return

        self.logger.info("Creating LSTM visualization...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.gridspec import GridSpec
            from symfluence.utils.common.metrics import get_RMSE, get_KGE, get_KGEp, get_NSE, get_MAE, get_KGEnp
            import pandas as pd

            def calculate_metrics(obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
                aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
                obs_vals = aligned_data['obs'].values
                sim_vals = aligned_data['sim'].values

                if len(obs_vals) == 0:
                    return {}

                return {
                    'RMSE': get_RMSE(obs_vals, sim_vals, transfo=1),
                    'KGE': get_KGE(obs_vals, sim_vals, transfo=1),
                    'KGEp': get_KGEp(obs_vals, sim_vals, transfo=1),
                    'NSE': get_NSE(obs_vals, sim_vals, transfo=1),
                    'MAE': get_MAE(obs_vals, sim_vals, transfo=1),
                    'KGEnp': get_KGEnp(obs_vals, sim_vals, transfo=1)
                }

            # Prepare data
            sim_dates = results_df.index
            sim_streamflow = results_df['predicted_streamflow']
            obs_streamflow_aligned = obs_streamflow.reindex(sim_dates)

            # Calculate metrics for streamflow
            metrics_streamflow = {
                '': calculate_metrics(obs_streamflow_aligned['streamflow'], sim_streamflow)
            }

            # Determine figure size and layout based on whether snow is used
            if use_snow:
                fig = plt.figure(figsize=(15, 16))
                gs = GridSpec(2, 1, height_ratios=[1, 1])
            else:
                fig = plt.figure(figsize=(15, 8))
                gs = GridSpec(1, 1)

            # Streamflow subplot
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(sim_dates, sim_streamflow, label='LSTM simulated', color='blue')
            ax1.plot(sim_dates, obs_streamflow_aligned['streamflow'], label='Observed', color='red')
            ax1.set_ylabel('Streamflow (m³/s)')
            ax1.set_title('Observed vs Simulated Streamflow')
            ax1.legend()
            ax1.grid(True)

            # Add metrics text
            metrics_text = "Performance Metrics:\n\n"
            metrics_text += "Streamflow Metrics:\n"
            for period, metrics in metrics_streamflow.items():
                metrics_text += f"{period}:\n"
                metrics_text += "\n".join([f"  {k}: {v:.3f}" for k, v in metrics.items()]) + "\n\n"

            if use_snow and not obs_snow.empty and 'predicted_SWE' in results_df.columns:
                # Add snow metrics and plot
                sim_swe = results_df['predicted_SWE']
                obs_snow_aligned = obs_snow.reindex(sim_dates)
                metrics_swe = {
                    '': calculate_metrics(obs_snow_aligned['snw'], sim_swe)
                }
                metrics_text += "Snow Water Equivalent Metrics:\n"
                for period, metrics in metrics_swe.items():
                    metrics_text += f"{period}:\n"
                    metrics_text += "\n".join([f"  {k}: {v:.3f}" for k, v in metrics.items()]) + "\n\n"

                # Snow subplot
                ax2 = fig.add_subplot(gs[1])
                ax2.plot(sim_dates, sim_swe, label='LSTM simulated', color='blue')
                ax2.plot(sim_dates, obs_snow_aligned['snw'], label='Observed', color='red')
                ax2.set_ylabel('Snow Water Equivalent (mm)')
                ax2.set_title('Observed vs Simulated Snow Water Equivalent')
                ax2.legend()
                ax2.grid(True)

                # Format x-axis for both plots
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax.xaxis.set_minor_locator(mdates.MonthLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # Format x-axis for single plot
                ax1.xaxis.set_major_locator(mdates.YearLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax1.xaxis.set_minor_locator(mdates.MonthLocator())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax1.text(
                0.01, 0.99, metrics_text, transform=ax1.transAxes, verticalalignment='top',
                fontsize=8, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
            )

            plt.tight_layout()

            # Save the figure
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / f"{experiment_id}_LSTM_simulation_results.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Results visualization saved to {fig_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating LSTM visualization: {str(e)}")

    def visualize_hype_results(self, sim_flow: Any, obs_flow: Any, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path):
        """
        Visualize HYPE streamflow comparison.

        Args:
            sim_flow: Simulated streamflow dataframe.
            obs_flow: Observed streamflow dataframe.
            outlet_id: Outlet ID.
            domain_name: Domain name.
            experiment_id: Experiment ID.
            project_dir: Project directory.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping HYPE visualization.")
            return

        self.logger.info("Creating HYPE streamflow comparison plot...")
        
        try:
            import matplotlib.pyplot as plt
            
            sim_col = 'HYPE_discharge_cms'
            
            plt.figure(figsize=(12, 6))
            plt.plot(sim_flow.index, sim_flow[sim_col], label='Simulated', color='blue', alpha=0.7)
            plt.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red', alpha=0.7)

            plt.title(f'Streamflow Comparison - {domain_name}\nOutlet ID: {outlet_id}')
            plt.xlabel('Date')
            plt.ylabel('Discharge (m³/s)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Ensure the plots directory exists
            plot_dir = project_dir / "plots" / "results"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Save plot
            plot_path = plot_dir / f"{experiment_id}_HYPE_streamflow_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"HYPE plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating HYPE visualization: {str(e)}")

    def visualize_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """
        Visualize SUMMA model outputs (all variables).

        Args:
            experiment_id: Experiment ID.

        Returns:
            Dictionary mapping variable names to plot paths.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping SUMMA output visualization.")
            return {}

        self.logger.info(f"Creating SUMMA output visualizations for experiment {experiment_id}...")
        return self._viz_reporter.plot_summa_outputs(experiment_id)

    def visualize_ngen_results(self, sim_df: Any, obs_df: Optional[Any], experiment_id: str, results_dir: Path):
        """
        Visualize NGen streamflow plots.

        Args:
            sim_df: Simulated streamflow dataframe.
            obs_df: Observed streamflow dataframe (optional).
            experiment_id: Experiment ID.
            results_dir: Results directory.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping NGen visualization.")
            return

        self.logger.info("Creating NGen streamflow plots...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Full time series
            ax1 = axes[0]
            ax1.plot(sim_df['datetime'], sim_df['streamflow_cms'], 
                    label='NGEN Simulated', color='blue', linewidth=0.8)
            
            # Add observed if available
            if obs_df is not None:
                ax1.plot(obs_df['datetime'], obs_df['streamflow_cms'], 
                        label='Observed', color='red', linewidth=0.8, alpha=0.7)
                
                # Simple metrics calculation for plot text
                merged = pd.merge(sim_df, obs_df, on='datetime', suffixes=('_sim', '_obs'))
                if not merged.empty:
                    obs_vals = merged['streamflow_cms_obs'].values
                    sim_vals = merged['streamflow_cms_sim'].values
                    
                    # NSE
                    nse_num = np.sum((obs_vals - sim_vals) ** 2)
                    nse_den = np.sum((obs_vals - np.mean(obs_vals)) ** 2)
                    nse = 1 - (nse_num / nse_den) if nse_den != 0 else np.nan
                    
                    # KGE
                    r = np.corrcoef(obs_vals, sim_vals)[0, 1]
                    alpha = np.std(sim_vals) / np.std(obs_vals) if np.std(obs_vals) != 0 else np.nan
                    beta = np.mean(sim_vals) / np.mean(obs_vals) if np.mean(obs_vals) != 0 else np.nan
                    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    
                    ax1.text(0.02, 0.98, f'NSE: {nse:.3f}\nKGE: {kge:.3f}', 
                            transform=ax1.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_ylabel('Streamflow (cms)', fontsize=12)
            ax1.set_title(f'NGEN Streamflow - {experiment_id}', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Plot 2: Flow duration curve
            ax2 = axes[1]
            sorted_flow = np.sort(sim_df['streamflow_cms'].values)[::-1]
            exceedance = np.arange(1, len(sorted_flow) + 1) / len(sorted_flow) * 100
            ax2.semilogy(exceedance, sorted_flow, label='NGEN Simulated', color='blue', linewidth=1.5)
            
            if obs_df is not None:
                sorted_obs = np.sort(obs_df['streamflow_cms'].values)[::-1]
                exceedance_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs) * 100
                ax2.semilogy(exceedance_obs, sorted_obs, label='Observed', 
                            color='red', linewidth=1.5, alpha=0.7)
            
            ax2.set_xlabel('Exceedance Probability (%)', fontsize=12)
            ax2.set_ylabel('Streamflow (cms)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = results_dir / f"ngen_streamflow_plot_{experiment_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"NGen plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating NGen visualization: {str(e)}")

    def update_sim_reach_id(self, config_path: Optional[str] = None):
        """
        Update the SIM_REACH_ID in both the config object and YAML file.

        Args:
            config_path: Either a path to the config file or None.
        """
        # We perform this even if visualize is False because it updates config which might be needed for other things?
        # The original code used VisualizationReporter to do this logic.
        # It seems it's used to find the reach ID for plotting BUT also updates the config.
        # If it updates the config, it might be important for other steps (though unlikely for just visualization).
        # Assuming it's needed for visualization context.
        return self._viz_reporter.update_sim_reach_id(config_path)

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled."""
        return self.visualize

    def visualize_domain(self) -> Optional[str]:
        """
        Visualize the domain boundaries and features.
        
        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping domain visualization.")
            return None
        
        self.logger.info("Creating domain visualization...")
        return self._viz_reporter.plot_domain()

    def visualize_discretized_domain(self, discretization_method: str) -> Optional[str]:
        """
        Visualize the discretized domain (HRUs/GRUs).

        Args:
            discretization_method: The method used for discretization (e.g., 'elevation', 'landclass').

        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping discretized domain visualization.")
            return None

        self.logger.info(f"Creating discretization visualization for {discretization_method}...")
        return self._viz_reporter.plot_discretized_domain(discretization_method)

    def visualize_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize model outputs (streamflow comparison).

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping model output visualization.")
            return None

        self.logger.info("Creating model output visualizations...")
        return self._viz_reporter.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)

    def visualize_lumped_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize lumped model outputs.

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping lumped model output visualization.")
            return None

        self.logger.info("Creating lumped model output visualizations...")
        return self._viz_reporter.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)

    def visualize_fuse_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize FUSE model outputs.

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping FUSE model output visualization.")
            return None

        self.logger.info("Creating FUSE model output visualizations...")
        return self._viz_reporter.plot_fuse_streamflow_simulations_vs_observations(model_outputs, obs_files)
    
    def visualize_timeseries_results(self):
        """
        Visualize timeseries results from the standard results file.
        Uses TimeseriesVisualizer which expects a specific directory structure.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping timeseries results visualization.")
            return None

        self.logger.info("Creating timeseries visualizations from results file...")
        try:
            self._ts_visualizer.create_visualizations()
        except Exception as e:
            self.logger.error(f"Error creating timeseries visualizations: {str(e)}")

    def visualize_benchmarks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """
        Visualize benchmark results.

        Args:
            benchmark_results: Dictionary containing benchmark results.

        Returns:
            List of paths to created plots.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping benchmark visualization.")
            return []

        self.logger.info("Creating benchmark visualizations...")
        return self._benchmark_visualizer.visualize_benchmarks(benchmark_results)

    def visualize_snow_comparison(self, model_outputs: List[List[str]]) -> Dict[str, Any]:
        """
        Visualize snow comparison.
        
        Args:
            model_outputs: List of model outputs.
            
        Returns:
            Dictionary with paths and metrics.
        """
        if not self.visualize:
            self.logger.debug("Visualization disabled. Skipping snow comparison visualization.")
            return {}

        self.logger.info("Creating snow comparison visualization...")
        return self._viz_reporter.plot_snow_simulations_vs_observations(model_outputs)
