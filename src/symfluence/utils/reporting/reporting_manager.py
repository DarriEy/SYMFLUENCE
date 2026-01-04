from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Config
from symfluence.utils.reporting.config.plot_config import PlotConfig, DEFAULT_PLOT_CONFIG

# Processors
from symfluence.utils.reporting.processors.data_processor import DataProcessor
from symfluence.utils.reporting.processors.spatial_processor import SpatialProcessor

# Plotters
from symfluence.utils.reporting.plotters.domain_plotter import DomainPlotter
from symfluence.utils.reporting.plotters.optimization_plotter import OptimizationPlotter
from symfluence.utils.reporting.plotters.analysis_plotter import AnalysisPlotter
from symfluence.utils.reporting.plotters.benchmark_plotter import BenchmarkPlotter
from symfluence.utils.reporting.plotters.snow_plotter import SnowPlotter


class ReportingManager:
    """
    Manages all reporting and visualization tasks for the SYMFLUENCE framework.
    Acts as a central entry point for generating plots and reports, delegating
    to specialized processors and plotters.
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

        # Lazy-loaded components
        self._plot_config = None
        self._data_processor = None
        self._spatial_processor = None
        self._domain_plotter = None
        self._optimization_plotter = None
        self._analysis_plotter = None
        self._benchmark_plotter = None
        self._snow_plotter = None

    # =========================================================================
    # Component Properties (Lazy Initialization)
    # =========================================================================

    @property
    def plot_config(self) -> PlotConfig:
        """Lazy initialization of plot configuration."""
        if self._plot_config is None:
            self._plot_config = DEFAULT_PLOT_CONFIG
        return self._plot_config

    @property
    def data_processor(self) -> DataProcessor:
        """Lazy initialization of data processor."""
        if self._data_processor is None:
            self._data_processor = DataProcessor(self.config, self.logger)
        return self._data_processor

    @property
    def spatial_processor(self) -> SpatialProcessor:
        """Lazy initialization of spatial processor."""
        if self._spatial_processor is None:
            self._spatial_processor = SpatialProcessor(self.config, self.logger)
        return self._spatial_processor

    @property
    def domain_plotter(self) -> DomainPlotter:
        """Lazy initialization of domain plotter."""
        if self._domain_plotter is None:
            self._domain_plotter = DomainPlotter(
                self.config, self.logger, self.plot_config
            )
        return self._domain_plotter

    @property
    def optimization_plotter(self) -> OptimizationPlotter:
        """Lazy initialization of optimization plotter."""
        if self._optimization_plotter is None:
            self._optimization_plotter = OptimizationPlotter(
                self.config, self.logger, self.plot_config
            )
        return self._optimization_plotter

    @property
    def analysis_plotter(self) -> AnalysisPlotter:
        """Lazy initialization of analysis plotter."""
        if self._analysis_plotter is None:
            self._analysis_plotter = AnalysisPlotter(
                self.config, self.logger, self.plot_config
            )
        return self._analysis_plotter

    @property
    def benchmark_plotter(self) -> BenchmarkPlotter:
        """Lazy initialization of benchmark plotter."""
        if self._benchmark_plotter is None:
            self._benchmark_plotter = BenchmarkPlotter(
                self.config, self.logger, self.plot_config
            )
        return self._benchmark_plotter

    @property
    def snow_plotter(self) -> SnowPlotter:
        """Lazy initialization of snow plotter."""
        if self._snow_plotter is None:
            self._snow_plotter = SnowPlotter(
                self.config, self.logger, self.plot_config
            )
        return self._snow_plotter

    # =========================================================================
    # Public Methods
    # =========================================================================

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled."""
        return self.visualize

    def update_sim_reach_id(self, config_path: Optional[str] = None) -> Optional[int]:
        """
        Update the SIM_REACH_ID in both the config object and YAML file.

        Args:
            config_path: Either a path to the config file or None.
            
        Returns:
            The found reach ID, or None if failed.
        """
        return self.spatial_processor.update_sim_reach_id(config_path)

    # --- Domain Visualization ---

    def visualize_domain(self) -> Optional[str]:
        """
        Visualize the domain boundaries and features.
        
        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            return None
        self.logger.info("Creating domain visualization...")
        return self.domain_plotter.plot_domain()

    def visualize_discretized_domain(self, discretization_method: str) -> Optional[str]:
        """
        Visualize the discretized domain (HRUs/GRUs).

        Args:
            discretization_method: The method used for discretization.

        Returns:
            Path to the plot if created, None otherwise.
        """
        if not self.visualize:
            return None
        self.logger.info(f"Creating discretization visualization for {discretization_method}...")
        return self.domain_plotter.plot_discretized_domain(discretization_method)

    # --- Model Output Visualization ---

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
            return None
        self.logger.info("Creating model output visualizations...")
        return self.analysis_plotter.plot_streamflow_comparison(model_outputs, obs_files)

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
            return None
        self.logger.info("Creating lumped model output visualizations...")
        return self.analysis_plotter.plot_streamflow_comparison(model_outputs, obs_files, lumped=True)

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
            return None
        self.logger.info("Creating FUSE model output visualizations...")
        return self.analysis_plotter.plot_fuse_streamflow(model_outputs, obs_files)

    def visualize_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """
        Visualize SUMMA model outputs (all variables).

        Args:
            experiment_id: Experiment ID.

        Returns:
            Dictionary mapping variable names to plot paths.
        """
        if not self.visualize:
            return {}
        self.logger.info(f"Creating SUMMA output visualizations for experiment {experiment_id}...")
        return self.analysis_plotter.plot_summa_outputs(experiment_id)

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
            return
        self.logger.info("Creating NGen streamflow plots...")
        self.analysis_plotter.plot_ngen_results(sim_df, obs_df, experiment_id, results_dir)

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
            return
        self.logger.info("Creating LSTM visualization...")
        self.analysis_plotter.plot_lstm_results(
            results_df, obs_streamflow, obs_snow, use_snow, output_dir, experiment_id
        )

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
            return
        self.logger.info("Creating HYPE streamflow comparison plot...")
        self.analysis_plotter.plot_hype_results(
            sim_flow, obs_flow, outlet_id, domain_name, experiment_id, project_dir
        )

    # --- Analysis Visualization ---

    def visualize_timeseries_results(self):
        """
        Visualize timeseries results from the standard results file.
        Reads results using DataProcessor and plots using AnalysisPlotter.
        """
        if not self.visualize:
            return None

        self.logger.info("Creating timeseries visualizations from results file...")
        
        try:
            # Use new DataProcessor to read results
            df = self.data_processor.read_results_file()
            
            exp_id = self.config.get('EXPERIMENT_ID', 'default')
            domain_name = self.config.get('DOMAIN_NAME', 'unknown')
            
            self.analysis_plotter.plot_timeseries_results(df, exp_id, domain_name)
            self.analysis_plotter.plot_diagnostics(df, exp_id, domain_name)
                
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
            return []
        self.logger.info("Creating benchmark visualizations...")
        return self.benchmark_plotter.plot_benchmarks(benchmark_results)

    def visualize_snow_comparison(self, model_outputs: List[List[str]]) -> Dict[str, Any]:
        """
        Visualize snow comparison.
        
        Args:
            model_outputs: List of model outputs (list of [name, path]).
            
        Returns:
            Dictionary with paths and metrics.
        """
        if not self.visualize:
            return {}
        self.logger.info("Creating snow comparison visualization...")
        # Convert List[List[str]] to List[Tuple[str, str]] for consistency if needed
        formatted_outputs = [tuple(item) for item in model_outputs]
        return self.snow_plotter.plot_snow_comparison(formatted_outputs)

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
            return
        self.logger.info("Creating optimization progress visualization...")
        self.optimization_plotter.plot_optimization_progress(
            history, output_dir, calibration_variable, metric
        )

    def visualize_optimization_depth_parameters(self, history: List[Dict], output_dir: Path) -> None:
        """
        Visualize depth parameter evolution.

        Args:
            history: List of optimization history dictionaries.
            output_dir: Directory to save the plot.
        """
        if not self.visualize:
            return
        self.logger.info("Creating depth parameter visualization...")
        self.optimization_plotter.plot_depth_parameters(history, output_dir)

    def visualize_sensitivity_analysis(self, sensitivity_data: Any, output_file: Path, plot_type: str = 'single'):
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_data: Data to plot (Series or DataFrame).
            output_file: Path to save the plot.
            plot_type: 'single' for one method, 'comparison' for multiple.
        """
        if not self.visualize:
            return
        self.logger.info(f"Creating sensitivity analysis visualization ({plot_type})...")
        self.analysis_plotter.plot_sensitivity_analysis(
            sensitivity_data, output_file, plot_type
        )

    def visualize_decision_impacts(self, results_file: Path, output_folder: Path):
        """
        Visualize decision analysis impacts.

        Args:
            results_file: Path to the CSV results file.
            output_folder: Folder to save plots.
        """
        if not self.visualize:
            return
        self.logger.info("Creating decision impact visualizations...")
        self.analysis_plotter.plot_decision_impacts(results_file, output_folder)

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
            return
        self.logger.info(f"Creating hydrograph visualization with {metric} highlight...")
        self.analysis_plotter.plot_hydrographs_with_highlight(
            results_file, simulation_results, observed_streamflow,
            decision_options, output_folder, metric
        )

    def visualize_drop_analysis(self, drop_data: List[Dict], optimal_threshold: float, project_dir: Path):
        """
        Visualize drop analysis for stream threshold selection.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics.
            optimal_threshold: The selected optimal threshold.
            project_dir: Project directory for saving the plot.
        """
        if not self.visualize:
            return
        self.logger.info("Creating drop analysis visualization...")
        self.analysis_plotter.plot_drop_analysis(
            drop_data, optimal_threshold, project_dir
        )
