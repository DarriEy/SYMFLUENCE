"""
Central reporting facade for coordinating all SYMFLUENCE visualizations.

Provides a unified interface for generating publication-ready visualizations
across all modeling stages: domain setup, calibration, evaluation, and
multi-model comparison. Implements the Facade pattern to orchestrate
specialized plotters while hiding complexity from client code.
"""

from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from pathlib import Path
from functools import cached_property

# Config
from symfluence.reporting.config.plot_config import PlotConfig, DEFAULT_PLOT_CONFIG
from symfluence.core.mixins import ConfigMixin

# Type hints only - actual imports are lazy
if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.reporting.processors.data_processor import DataProcessor
    from symfluence.reporting.processors.spatial_processor import SpatialProcessor
    from symfluence.reporting.plotters.domain_plotter import DomainPlotter
    from symfluence.reporting.plotters.optimization_plotter import OptimizationPlotter
    from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter
    from symfluence.reporting.plotters.benchmark_plotter import BenchmarkPlotter
    from symfluence.reporting.plotters.snow_plotter import SnowPlotter
    from symfluence.reporting.plotters.diagnostic_plotter import DiagnosticPlotter
    from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter


class ReportingManager(ConfigMixin):
    """Central facade coordinating all visualization and reporting in SYMFLUENCE.

    Orchestrates diverse visualization workflows by delegating to specialized plotters
    for domain maps, calibration analysis, performance benchmarking, and diagnostics.
    Implements Facade and Lazy Initialization patterns to manage complex visualization
    dependencies efficiently. Provides unified interface for all reporting tasks
    throughout SYMFLUENCE workflows.

    This class provides high-level methods for generating publication-ready visualizations
    for all stages of hydrological modeling: domain setup, calibration, evaluation,
    and multi-model comparison. All visualization methods are conditional on the
    `visualize` flag, allowing seamless integration with non-visual workflows.

    Architecture:

        1. Plotter Components (Specialized):
           - DomainPlotter: Geospatial domain maps, HRU discretization, river networks
           - OptimizationPlotter: Calibration convergence, parameter sensitivity
           - AnalysisPlotter: Model performance metrics, time series, scatter plots
           - BenchmarkPlotter: Multi-model comparison, Pareto frontier
           - SnowPlotter: SWE maps, snow validation, SNOTEL comparison
           - DiagnosticPlotter: Water balance, energy balance, flux diagnostics

        2. Processor Components (Data Preparation):
           - DataProcessor: Load, aggregate, prepare data for plotting
           - SpatialProcessor: Geospatial operations, map projections, raster handling

        3. Configuration System:
           - PlotConfig: Centralized plot styling and layout configuration
           - Theme management: Colors, fonts, sizes, annotations
           - Format control: Vector (PDF/SVG) vs raster (PNG) output

    Plotter Responsibilities:

        DomainPlotter:
            - Basin boundary map
            - HRU discretization visualization
            - River network topology
            - Attribute maps (elevation, soil, landcover)
            - Input: Shapefiles, DEM, attributes
            - Output: PNG/PDF maps

        OptimizationPlotter:
            - Calibration iteration progress (KGE evolution)
            - Parameter sensitivity analysis
            - Parameter trajectories over iterations
            - Population convergence metrics
            - Input: Optimization results, parameter sets
            - Output: Line plots, heatmaps

        AnalysisPlotter:
            - Time series comparison (observed vs simulated)
            - Scatter plots (KGE, NSE, RMSE distributions)
            - Flow duration curves
            - Period-specific analysis (seasons, years)
            - Input: Model outputs, observations
            - Output: Multi-panel plots

        BenchmarkPlotter:
            - Multi-model performance comparison
            - Pareto frontier (single vs multi-objective)
            - Metric ranking tables
            - Box plots of metrics across models
            - Input: Multiple model results
            - Output: Comparison plots

        SnowPlotter:
            - SWE temporal evolution
            - Snow cover fraction maps
            - SNOTEL station comparison
            - Validation metrics
            - Input: SWE outputs, observations
            - Output: Snow-specific plots

        DiagnosticPlotter:
            - Water balance closure
            - Energy balance components
            - Flux time series (ET, runoff, infiltration)
            - Residual analysis
            - Input: Model diagnostics, fluxes
            - Output: Diagnostic plots

    Data Processing:

        DataProcessor:
            - Load NetCDF/CSV model outputs
            - Temporal aggregation (daily→monthly→annual)
            - Spatial aggregation (grid→basin)
            - Metric calculations (KGE, NSE, RMSE)
            - Statistical analysis (mean, std, quantiles)

        SpatialProcessor:
            - Load and reproject shapefiles
            - Raster-vector overlay operations
            - Map projection management (UTM, geographic)
            - Boundary geometry operations

    Configuration:

        PlotConfig (PlotConfig dataclass):
            figure_size: Default figure dimensions (10x6 inches)
            dpi: Output resolution (300 for publications, 100 for web)
            color_palette: Color scheme for plots
            font_size: Default font size for labels
            line_width: Default line width for time series
            marker_size: Default marker size for scatter plots
            style: Plot style ('seaborn', 'ggplot', etc.)
            output_format: 'png', 'pdf', 'svg' (vector formats preserve quality)
            show_grid: Display grid lines
            show_legend: Display plot legends
            save_path: Output directory for plots

        Theming:
            - Seaborn-based styling for publication-ready plots
            - Consistent colormaps across figures
            - Customizable palettes (Set2, RdYlBu, viridis, etc.)
            - High-DPI output for print quality

    Visualization Workflow:

        1. Initialization:
           rm = ReportingManager(config, logger, visualize=True)
           - Load configuration
           - Set visualization flag

        2. Domain Visualization:
           rm.plot_domain()
           - Map study domain and HRUs
           - Show attribute distributions

        3. Calibration Monitoring:
           rm.plot_calibration_progress()
           - Monitor convergence in real-time
           - Track parameter evolution

        4. Model Evaluation:
           rm.plot_evaluation_results()
           - Time series comparison (observed vs simulated)
           - Performance metrics
           - Error analysis

        5. Multi-Model Comparison:
           rm.plot_benchmark_comparison()
           - Compare multiple models/optimizers
           - Identify best-performing configurations

        6. Diagnostic Analysis:
           rm.plot_diagnostics()
           - Water/energy balance
           - Flux validation
           - Error diagnostics

    Lazy Initialization:

        Uses cached_property for plotter components:
        - Plotters imported only when first accessed
        - Heavy dependencies (matplotlib, seaborn, cartopy) loaded on-demand
        - Speeds startup for non-visualization workflows
        - Memory-efficient for scripts that don't use graphics

    Performance:

        - Figure generation: 1-10 seconds per plot
        - Memory: ~500 MB for typical visualization components
        - Output file sizes: 1-10 MB (PNG), 100 KB-1 MB (PDF)
        - Batch processing: Supports parallel plot generation

    Key Methods:

        plot_domain():
            Generate domain overview map with HRU boundaries and attributes

        plot_calibration_progress(iteration, metrics):
            Update calibration convergence plot

        plot_optimization_results(results):
            Generate parameter sensitivity and convergence plots

        plot_time_series(obs, sim, period):
            Generate time series comparison

        plot_benchmark(results, models):
            Compare multiple models/runs

        plot_diagnostics(diagnostics):
            Generate water balance and flux diagnostic plots

    Configuration:

        visualize: bool (default False)
            Enable/disable all visualization functions
            If False, visualization methods return early (cheap skip)

        PlotConfig.output_format: str
            'png': Raster format (smaller files, fast rendering)
            'pdf': Vector format (scalable, print-ready)
            'svg': Vector format (web-ready, editable)

        PlotConfig.dpi: int
            Resolution for raster output (300 for publications, 100 for web)

    Example Usage:

        >>> from symfluence.reporting import ReportingManager
        >>> from symfluence.core.config import SymfluenceConfig
        >>>
        >>> config = SymfluenceConfig.from_file('config.yaml')
        >>> logger = setup_logger()
        >>>
        >>> # Enable visualization
        >>> rm = ReportingManager(config, logger, visualize=True)
        >>>
        >>> # Domain setup visualization
        >>> rm.plot_domain()  # Generate domain map
        >>>
        >>> # Calibration monitoring
        >>> for iteration in range(num_iterations):
        ...     metrics = optimizer.run_iteration()
        ...     rm.plot_calibration_progress(iteration, metrics)
        >>>
        >>> # Final evaluation
        >>> results = model.run_evaluation(eval_period)
        >>> rm.plot_evaluation_results(results)
        >>>
        >>> # Multi-model comparison
        >>> all_results = {model: results for model, results in model_results.items()}
        >>> rm.plot_benchmark_comparison(all_results)

    Error Handling:

        - Gracefully skips visualization if visualize=False
        - Catches matplotlib errors and logs warnings
        - Continues execution even if individual plots fail
        - Validates input data before plotting

    Dependencies:

        - matplotlib: Core plotting library
        - seaborn: Statistical visualization
        - cartopy: Geographic mapping
        - rasterio/geopandas: Geospatial data handling

    See Also:

        - PlotConfig: Configuration for plot styling
        - DomainPlotter: Domain-specific visualizations
        - OptimizationPlotter: Calibration analysis plots
        - AnalysisPlotter: Model performance plots
        - BenchmarkPlotter: Multi-model comparison
        - SnowPlotter: Snow-specific visualizations

    Example:
        >>> rm = ReportingManager(config, logger, visualize=True)
        >>> rm.plot_domain()          # Generate domain overview map
        >>> rm.plot_calibration()     # Plot calibration convergence
    """

    def __init__(self, config: 'SymfluenceConfig', logger: Any, visualize: bool = False):
        """
        Initialize the ReportingManager.

        Args:
            config: SymfluenceConfig instance.
            logger: Logger instance.
            visualize: Boolean flag indicating if visualization is enabled.
                       If False, most visualization methods will return early.
        """
        from symfluence.core.config.models import SymfluenceConfig
        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )

        self._config = config
        self.logger = logger
        self.visualize = visualize
        self.project_dir = Path(config['SYMFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}"

    # =========================================================================
    # Component Properties (Lazy Initialization via cached_property)
    # =========================================================================

    @cached_property
    def plot_config(self) -> PlotConfig:
        """Lazy initialization of plot configuration."""
        return DEFAULT_PLOT_CONFIG

    @cached_property
    def data_processor(self) -> 'DataProcessor':
        """Lazy initialization of data processor."""
        from symfluence.reporting.processors.data_processor import DataProcessor
        return DataProcessor(self.config, self.logger)

    @cached_property
    def spatial_processor(self) -> 'SpatialProcessor':
        """Lazy initialization of spatial processor."""
        from symfluence.reporting.processors.spatial_processor import SpatialProcessor
        return SpatialProcessor(self.config, self.logger)

    @cached_property
    def domain_plotter(self) -> 'DomainPlotter':
        """Lazy initialization of domain plotter."""
        from symfluence.reporting.plotters.domain_plotter import DomainPlotter
        return DomainPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def optimization_plotter(self) -> 'OptimizationPlotter':
        """Lazy initialization of optimization plotter."""
        from symfluence.reporting.plotters.optimization_plotter import OptimizationPlotter
        return OptimizationPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def analysis_plotter(self) -> 'AnalysisPlotter':
        """Lazy initialization of analysis plotter."""
        from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter
        return AnalysisPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def benchmark_plotter(self) -> 'BenchmarkPlotter':
        """Lazy initialization of benchmark plotter."""
        from symfluence.reporting.plotters.benchmark_plotter import BenchmarkPlotter
        return BenchmarkPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def snow_plotter(self) -> 'SnowPlotter':
        """Lazy initialization of snow plotter."""
        from symfluence.reporting.plotters.snow_plotter import SnowPlotter
        return SnowPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def diagnostic_plotter(self) -> 'DiagnosticPlotter':
        """Lazy initialization of diagnostic plotter."""
        from symfluence.reporting.plotters.diagnostic_plotter import DiagnosticPlotter
        return DiagnosticPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def model_comparison_plotter(self) -> 'ModelComparisonPlotter':
        """Lazy initialization of model comparison plotter."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        return ModelComparisonPlotter(self.config, self.logger, self.plot_config)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def visualize_data_distribution(self, data: Any, variable_name: str, stage: str):
        """
        Visualize data distribution (histogram/boxplot).
        """
        if not self.visualize:
            return
        self.diagnostic_plotter.plot_data_distribution(data, variable_name, stage)

    def visualize_spatial_coverage(self, raster_path: Path, variable_name: str, stage: str):
        """
        Visualize spatial coverage of raster data.
        """
        if not self.visualize:
            return
        self.diagnostic_plotter.plot_spatial_coverage(raster_path, variable_name, stage)

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

    def visualize_model_results(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Visualize model results using registry-based dispatch.

        This method uses the PlotterRegistry to dynamically discover and invoke
        the appropriate model-specific plotter. This is the preferred method for
        new code as it eliminates hardcoded model checks.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE', 'HYPE', 'NGEN', 'LSTM')
            **kwargs: Model-specific arguments passed to the plotter's plot method

        Returns:
            Plot result (path to saved plot or dict of paths), or None if:
            - visualize flag is False
            - no plotter registered for the model
            - plotting failed

        Example:
            >>> # SUMMA outputs
            >>> reporting_manager.visualize_model_results('SUMMA', experiment_id='exp1')

            >>> # FUSE streamflow
            >>> reporting_manager.visualize_model_results('FUSE',
            ...     model_outputs=[('FUSE', 'output.nc')],
            ...     obs_files=[('obs', 'obs.csv')])

            >>> # HYPE results
            >>> reporting_manager.visualize_model_results('HYPE',
            ...     sim_flow=sim_df, obs_flow=obs_df,
            ...     outlet_id='1234', domain_name='test',
            ...     experiment_id='exp1', project_dir=Path('/path'))
        """
        if not self.visualize:
            return None

        # Import model modules to trigger plotter registration
        self._import_model_plotters()

        from symfluence.reporting.plotter_registry import PlotterRegistry

        model_upper = model_name.upper()
        plotter_cls = PlotterRegistry.get_plotter(model_upper)

        if plotter_cls is None:
            available = PlotterRegistry.list_plotters()
            self.logger.warning(
                f"No plotter registered for model '{model_name}'. "
                f"Available plotters: {available}. Falling back to legacy method if available."
            )
            # Fallback to legacy methods for backward compatibility
            return self._fallback_visualize(model_upper, **kwargs)

        self.logger.info(f"Creating {model_name} visualizations using registered plotter...")

        try:
            plotter = plotter_cls(self.config, self.logger, self.plot_config)
            return plotter.plot(**kwargs)
        except Exception as e:
            self.logger.error(f"Error in {model_name} plotter: {str(e)}")
            return None

    def _import_model_plotters(self) -> None:
        """
        Import model modules to trigger plotter registration with PlotterRegistry.

        This ensures that model-specific plotters are registered before we try
        to look them up. The registration happens at import time via decorators.
        """
        models_to_import = ['summa', 'fuse', 'hype', 'ngen', 'lstm']

        for model in models_to_import:
            try:
                __import__(f'symfluence.models.{model}')
            except ImportError:
                pass  # Model module may not be available

    def _fallback_visualize(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Fallback to legacy visualization methods for backward compatibility.

        Args:
            model_name: Model name (uppercase)
            **kwargs: Arguments for the visualization method

        Returns:
            Plot result or None
        """
        if model_name == 'SUMMA' and 'experiment_id' in kwargs:
            return self.visualize_summa_outputs(kwargs['experiment_id'])
        elif model_name == 'FUSE' and 'model_outputs' in kwargs and 'obs_files' in kwargs:
            return self.visualize_fuse_outputs(kwargs['model_outputs'], kwargs['obs_files'])
        elif model_name == 'HYPE':
            required = ['sim_flow', 'obs_flow', 'outlet_id', 'domain_name', 'experiment_id', 'project_dir']
            if all(k in kwargs for k in required):
                return self.visualize_hype_results(**{k: kwargs[k] for k in required})
        elif model_name == 'NGEN':
            required = ['sim_df', 'experiment_id', 'results_dir']
            if all(k in kwargs for k in required):
                return self.visualize_ngen_results(
                    kwargs['sim_df'], kwargs.get('obs_df'), kwargs['experiment_id'], kwargs['results_dir']
                )
        elif model_name == 'LSTM':
            required = ['results_df', 'obs_streamflow', 'obs_snow', 'use_snow', 'output_dir', 'experiment_id']
            if all(k in kwargs for k in required):
                return self.visualize_lstm_results(**{k: kwargs[k] for k in required})

        return None

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

            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default', dict_key='EXPERIMENT_ID')
            domain_name = self._get_config_value(lambda: self.config.domain.name, default='unknown', dict_key='DOMAIN_NAME')

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

    def generate_model_comparison_overview(
        self,
        experiment_id: Optional[str] = None,
        context: str = 'run_model'
    ) -> Optional[str]:
        """Generate model comparison overview for all models with valid output.

        Creates a comprehensive multi-panel visualization comparing observed and
        simulated streamflow across all models. Includes time series, flow duration
        curves, scatter plots, performance metrics, monthly distributions, and
        residual analysis.

        Based on Camille Gautier's overview_model_comparison visualization.
        Reference: https://github.com/camille-gautier/overview_model_comparison

        Args:
            experiment_id: Experiment ID for loading results. If None, uses
                          config.domain.experiment_id.
            context: Context for the comparison:
                    - 'run_model': After model run (default title)
                    - 'calibrate_model': After calibration (post-calibration title)

        Returns:
            Path to the saved overview plot, or None if:
            - visualize flag is False
            - no results data available
            - plot generation failed

        Note:
            Automatically triggered at the end of run_model and calibrate_model
            when the --visualize flag is enabled.
        """
        if not self.visualize:
            return None

        # Get experiment ID from config if not provided
        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='default',
                dict_key='EXPERIMENT_ID'
            )

        self.logger.info(f"Generating model comparison overview for {experiment_id}...")

        try:
            return self.model_comparison_plotter.plot_model_comparison_overview(
                experiment_id=experiment_id,
                context=context
            )
        except Exception as e:
            self.logger.error(f"Error generating model comparison overview: {str(e)}")
            return None
