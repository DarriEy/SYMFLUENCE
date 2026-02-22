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
from symfluence.core.exceptions import ReportingError, symfluence_error_handler
from symfluence.core.mixins import ConfigMixin
from symfluence.reporting.core.decorators import skip_if_not_visualizing, skip_if_not_diagnostic
from symfluence.core.constants import ConfigKeys

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
    from symfluence.reporting.plotters.forcing_comparison_plotter import ForcingComparisonPlotter
    from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter


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

    def __init__(self, config: 'SymfluenceConfig', logger: Any, visualize: bool = False, diagnostic: bool = False):
        """
        Initialize the ReportingManager.

        Args:
            config: SymfluenceConfig instance.
            logger: Logger instance.
            visualize: Boolean flag indicating if visualization is enabled.
                       If False, most visualization methods will return early.
            diagnostic: Boolean flag indicating if diagnostic mode is enabled.
                       If True, generates validation plots at workflow step completion.
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
        self.diagnostic = diagnostic
        self.project_dir = Path(config.system.data_dir) / f"domain_{config.domain.name}"

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

    @cached_property
    def forcing_comparison_plotter(self) -> 'ForcingComparisonPlotter':
        """Lazy initialization of forcing comparison plotter."""
        from symfluence.reporting.plotters.forcing_comparison_plotter import ForcingComparisonPlotter
        return ForcingComparisonPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def workflow_diagnostic_plotter(self) -> 'WorkflowDiagnosticPlotter':
        """Lazy initialization of workflow diagnostic plotter."""
        from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter
        return WorkflowDiagnosticPlotter(self.config, self.logger, self.plot_config)

    # =========================================================================
    # Public Methods
    # =========================================================================

    @skip_if_not_visualizing()
    def visualize_data_distribution(self, data: Any, variable_name: str, stage: str) -> None:
        """
        Visualize data distribution (histogram/boxplot).
        """
        self.diagnostic_plotter.plot_data_distribution(data, variable_name, stage)

    @skip_if_not_visualizing()
    def visualize_spatial_coverage(self, raster_path: Path, variable_name: str, stage: str) -> None:
        """
        Visualize spatial coverage of raster data.
        """
        self.diagnostic_plotter.plot_spatial_coverage(raster_path, variable_name, stage)

    @skip_if_not_visualizing()
    def visualize_forcing_comparison(
        self,
        raw_forcing_file: Path,
        remapped_forcing_file: Path,
        forcing_grid_shp: Path,
        hru_shp: Path,
        variable: str = 'pptrate',
        time_index: int = 0
    ) -> Optional[str]:
        """
        Visualize raw vs. remapped forcing data comparison.

        Creates a side-by-side map visualization comparing raw gridded forcing
        data with HRU-remapped forcing data.

        Args:
            raw_forcing_file: Path to raw NetCDF forcing file
            remapped_forcing_file: Path to remapped NetCDF forcing file
            forcing_grid_shp: Path to forcing grid shapefile
            hru_shp: Path to HRU/catchment shapefile
            variable: Variable to visualize (default: 'pptrate')
            time_index: Time index to visualize (default: 0)

        Returns:
            Path to saved plot, or None if visualization is disabled or failed
        """
        self.logger.info("Creating raw vs. remapped forcing comparison visualization...")
        return self.forcing_comparison_plotter.plot_raw_vs_remapped(
            raw_forcing_file=raw_forcing_file,
            remapped_forcing_file=remapped_forcing_file,
            forcing_grid_shp=forcing_grid_shp,
            hru_shp=hru_shp,
            variable=variable,
            time_index=time_index
        )

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

    @skip_if_not_visualizing()
    def visualize_domain(self) -> Optional[str]:
        """
        Visualize the domain boundaries and features.

        Returns:
            Path to the plot if created, None otherwise.
        """
        self.logger.info("Creating domain visualization...")
        return self.domain_plotter.plot_domain()

    @skip_if_not_visualizing()
    def visualize_discretized_domain(self, discretization_method: str) -> Optional[str]:
        """
        Visualize the discretized domain (HRUs/GRUs).

        Args:
            discretization_method: The method used for discretization.

        Returns:
            Path to the plot if created, None otherwise.
        """
        self.logger.info(f"Creating discretization visualization for {discretization_method}...")
        return self.domain_plotter.plot_discretized_domain(discretization_method)

    # --- Model Output Visualization ---

    @skip_if_not_visualizing()
    def visualize_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize model outputs (streamflow comparison).

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        self.logger.info("Creating model output visualizations...")
        return self.analysis_plotter.plot_streamflow_comparison(model_outputs, obs_files)

    @skip_if_not_visualizing()
    def visualize_lumped_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize lumped model outputs.

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        self.logger.info("Creating lumped model output visualizations...")
        return self.analysis_plotter.plot_streamflow_comparison(model_outputs, obs_files, lumped=True)

    @skip_if_not_visualizing()
    def visualize_fuse_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """
        Visualize FUSE model outputs.

        Args:
            model_outputs: List of tuples (model_name, file_path).
            obs_files: List of tuples (obs_name, file_path).

        Returns:
            Path to the plot if created, None otherwise.
        """
        self.logger.info("Creating FUSE model output visualizations...")
        return self.analysis_plotter.plot_fuse_streamflow(model_outputs, obs_files)

    @skip_if_not_visualizing(default={})
    def visualize_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """
        Visualize SUMMA model outputs (all variables).

        Args:
            experiment_id: Experiment ID.

        Returns:
            Dictionary mapping variable names to plot paths.
        """
        self.logger.info(f"Creating SUMMA output visualizations for experiment {experiment_id}...")
        return self.analysis_plotter.plot_summa_outputs(experiment_id)

    @skip_if_not_visualizing()
    def visualize_ngen_results(self, sim_df: Any, obs_df: Optional[Any], experiment_id: str, results_dir: Path) -> None:
        """
        Visualize NGen streamflow plots.

        Args:
            sim_df: Simulated streamflow dataframe.
            obs_df: Observed streamflow dataframe (optional).
            experiment_id: Experiment ID.
            results_dir: Results directory.
        """
        self.logger.info("Creating NGen streamflow plots...")
        self.analysis_plotter.plot_ngen_results(sim_df, obs_df, experiment_id, results_dir)

    @skip_if_not_visualizing()
    def visualize_lstm_results(self, results_df: Any, obs_streamflow: Any, obs_snow: Any, use_snow: bool, output_dir: Path, experiment_id: str) -> None:
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
        self.logger.info("Creating LSTM visualization...")
        self.analysis_plotter.plot_lstm_results(
            results_df, obs_streamflow, obs_snow, use_snow, output_dir, experiment_id
        )

    @skip_if_not_visualizing()
    def visualize_hype_results(self, sim_flow: Any, obs_flow: Any, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path) -> None:
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
        self.logger.info("Creating HYPE streamflow comparison plot...")
        self.analysis_plotter.plot_hype_results(
            sim_flow, obs_flow, outlet_id, domain_name, experiment_id, project_dir
        )

    @skip_if_not_visualizing()
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

        with symfluence_error_handler(
            f"{model_name} plotter",
            self.logger,
            reraise=False,
            error_type=ReportingError
        ):
            plotter = plotter_cls(self.config, self.logger, self.plot_config)
            return plotter.plot(**kwargs)

        return None

    def _import_model_plotters(self) -> None:
        """
        Import model modules to trigger plotter registration with PlotterRegistry.

        This ensures that model-specific plotters are registered before we try
        to look them up. The registration happens at import time via decorators.
        """
        from symfluence.core.constants import SupportedModels

        for model in SupportedModels.WITH_PLOTTERS:
            try:
                __import__(f'symfluence.models.{model}')
            except ImportError:
                self.logger.debug(f"Model plotter module '{model}' not available")

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

    @skip_if_not_visualizing()
    def visualize_timeseries_results(self) -> None:
        """
        Visualize timeseries results from the standard results file.
        Reads results using DataProcessor and plots using AnalysisPlotter.
        """
        self.logger.info("Creating timeseries visualizations from results file...")

        with symfluence_error_handler(
            "creating timeseries visualizations",
            self.logger,
            reraise=False,
            error_type=ReportingError
        ):
            # Use new DataProcessor to read results
            df = self.data_processor.read_results_file()

            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default', dict_key=ConfigKeys.EXPERIMENT_ID)
            domain_name = self._get_config_value(lambda: self.config.domain.name, default='unknown', dict_key=ConfigKeys.DOMAIN_NAME)

            self.analysis_plotter.plot_timeseries_results(df, exp_id, domain_name)
            self.analysis_plotter.plot_diagnostics(df, exp_id, domain_name)

    @skip_if_not_visualizing(default=[])
    def visualize_benchmarks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """
        Visualize benchmark results.

        Args:
            benchmark_results: Dictionary containing benchmark results.

        Returns:
            List of paths to created plots.
        """
        self.logger.info("Creating benchmark visualizations...")
        return self.benchmark_plotter.plot_benchmarks(benchmark_results)

    @skip_if_not_visualizing(default={})
    def visualize_snow_comparison(self, model_outputs: List[List[str]]) -> Dict[str, Any]:
        """
        Visualize snow comparison.

        Args:
            model_outputs: List of model outputs (list of [name, path]).

        Returns:
            Dictionary with paths and metrics.
        """
        self.logger.info("Creating snow comparison visualization...")
        # Convert List[List[str]] to List[Tuple[str, str]] for consistency if needed
        formatted_outputs = [tuple(item) for item in model_outputs]
        return self.snow_plotter.plot_snow_comparison(formatted_outputs)

    @skip_if_not_visualizing()
    def visualize_optimization_progress(self, history: List[Dict], output_dir: Path, calibration_variable: str, metric: str) -> None:
        """
        Visualize optimization progress.

        Args:
            history: List of optimization history dictionaries.
            output_dir: Directory to save the plot.
            calibration_variable: Name of the variable being calibrated.
            metric: Name of the optimization metric.
        """
        self.logger.info("Creating optimization progress visualization...")
        self.optimization_plotter.plot_optimization_progress(
            history, output_dir, calibration_variable, metric
        )

    @skip_if_not_visualizing()
    def visualize_optimization_depth_parameters(self, history: List[Dict], output_dir: Path) -> None:
        """
        Visualize depth parameter evolution.

        Args:
            history: List of optimization history dictionaries.
            output_dir: Directory to save the plot.
        """
        self.logger.info("Creating depth parameter visualization...")
        self.optimization_plotter.plot_depth_parameters(history, output_dir)

    @skip_if_not_visualizing()
    def visualize_sensitivity_analysis(self, sensitivity_data: Any, output_file: Path, plot_type: str = 'single') -> None:
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_data: Data to plot (Series or DataFrame).
            output_file: Path to save the plot.
            plot_type: 'single' for one method, 'comparison' for multiple.
        """
        self.logger.info(f"Creating sensitivity analysis visualization ({plot_type})...")
        self.analysis_plotter.plot_sensitivity_analysis(
            sensitivity_data, output_file, plot_type
        )

    @skip_if_not_visualizing()
    def visualize_decision_impacts(self, results_file: Path, output_folder: Path) -> None:
        """
        Visualize decision analysis impacts.

        Args:
            results_file: Path to the CSV results file.
            output_folder: Folder to save plots.
        """
        self.logger.info("Creating decision impact visualizations...")
        self.analysis_plotter.plot_decision_impacts(results_file, output_folder)

    @skip_if_not_visualizing()
    def visualize_hydrographs_with_highlight(self, results_file: Path, simulation_results: Dict, observed_streamflow: Any, decision_options: Dict, output_folder: Path, metric: str = 'kge') -> None:
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
        self.logger.info(f"Creating hydrograph visualization with {metric} highlight...")
        self.analysis_plotter.plot_hydrographs_with_highlight(
            results_file, simulation_results, observed_streamflow,
            decision_options, output_folder, metric
        )

    @skip_if_not_visualizing()
    def visualize_drop_analysis(self, drop_data: List[Dict], optimal_threshold: float, project_dir: Path) -> None:
        """
        Visualize drop analysis for stream threshold selection.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics.
            optimal_threshold: The selected optimal threshold.
            project_dir: Project directory for saving the plot.
        """
        self.logger.info("Creating drop analysis visualization...")
        self.analysis_plotter.plot_drop_analysis(
            drop_data, optimal_threshold, project_dir
        )

    @skip_if_not_visualizing()
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
        # Get experiment ID from config if not provided
        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='default',
                dict_key=ConfigKeys.EXPERIMENT_ID
            )

        self.logger.info(f"Generating model comparison overview for {experiment_id}...")

        with symfluence_error_handler(
            "generating model comparison overview",
            self.logger,
            reraise=False,
            error_type=ReportingError
        ):
            return self.model_comparison_plotter.plot_model_comparison_overview(
                experiment_id=experiment_id,
                context=context
            )

        return None

    @skip_if_not_visualizing(default={})
    def visualize_calibration_results(
        self,
        experiment_id: Optional[str] = None,
        calibration_target: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive post-calibration visualizations.

        Creates visualizations appropriate for the calibration target:
        - Optimization progress/convergence plot
        - Model performance comparison (obs vs sim with metrics)

        Automatically detects the calibration target from config if not specified:
        - 'streamflow'/'q' -> Streamflow comparison (Camille's model comparison)
        - 'swe'/'snow' -> SWE comparison (scalarSWE plot with observations)
        - 'et'/'evapotranspiration' -> Energy flux comparison (LE/H plots)

        Args:
            experiment_id: Experiment ID. If None, uses config value.
            calibration_target: Target variable being calibrated. If None,
                               auto-detected from config.optimization.target.

        Returns:
            Dictionary mapping plot names to file paths:
            - 'optimization_progress': Convergence plot (if history exists)
            - 'model_comparison': Main comparison plot
            - 'scalarSWE': SWE plot (for snow calibration)
            - 'scalarLatHeatTotal': LE plot (for ET calibration)
            - 'scalarSenHeatTotal': H plot (for ET calibration)

        Example:
            >>> # After calibration
            >>> plot_paths = reporting_manager.visualize_calibration_results(
            ...     experiment_id='run_1'
            ... )
            >>> for name, path in plot_paths.items():
            ...     print(f"{name}: {path}")
        """
        import json
        plot_paths: Dict[str, str] = {}

        # Get experiment ID from config if not provided
        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='default',
                dict_key=ConfigKeys.EXPERIMENT_ID
            )

        # Get calibration target from config if not provided
        if calibration_target is None:
            calibration_target = self._get_config_value(
                lambda: self.config.optimization.target,
                default='streamflow',
                dict_key=ConfigKeys.OPTIMIZATION_TARGET
            )
        calibration_target = str(calibration_target).lower()

        self.logger.info(f"Generating post-calibration visualizations for {experiment_id} (target: {calibration_target})")

        # 1. Generate optimization progress plot (if history exists)
        with symfluence_error_handler(
            "generating optimization progress plot",
            self.logger,
            reraise=False,
            error_type=ReportingError
        ):
            opt_dir = self.project_dir / "optimization"
            history_files = list(opt_dir.glob("*history*.json")) + list(opt_dir.glob("*history*.csv"))

            if history_files:
                # Try to load history from JSON first
                history = []
                for hf in history_files:
                    if hf.suffix == '.json':
                        try:
                            with open(hf, encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    history = data
                                    break
                                elif isinstance(data, dict) and 'history' in data:
                                    history = data['history']
                                    break
                        except (json.JSONDecodeError, OSError, KeyError):
                            continue

                if history:
                    metric = self._get_config_value(
                        lambda: self.config.optimization.metric,
                        default='KGE',
                        dict_key=ConfigKeys.OPTIMIZATION_METRIC
                    )
                    progress_plot = self.optimization_plotter.plot_optimization_progress(
                        history, opt_dir, calibration_target, metric
                    )
                    if progress_plot:
                        plot_paths['optimization_progress'] = progress_plot

        # 2. Generate appropriate model comparison based on calibration target
        with symfluence_error_handler(
            "generating calibration comparison plots",
            self.logger,
            reraise=False,
            error_type=ReportingError
        ):
            if calibration_target in ('streamflow', 'q', 'discharge', 'runoff'):
                # Streamflow calibration -> Camille's model comparison overview
                comparison_plot = self.generate_model_comparison_overview(
                    experiment_id=experiment_id,
                    context='calibrate_model'
                )
                if comparison_plot:
                    plot_paths['model_comparison'] = comparison_plot

                # Also generate default vs calibrated comparison
                with symfluence_error_handler(
                    "generating default vs calibrated comparison",
                    self.logger,
                    reraise=False,
                    error_type=ReportingError
                ):
                    default_vs_calibrated_plot = self.model_comparison_plotter.plot_default_vs_calibrated_comparison(
                        experiment_id=experiment_id
                    )
                    if default_vs_calibrated_plot:
                        plot_paths['default_vs_calibrated'] = default_vs_calibrated_plot

            elif calibration_target in ('swe', 'snow', 'snow_water_equivalent'):
                # SWE calibration -> SUMMA outputs with SWE observations
                summa_plots = self.visualize_summa_outputs(experiment_id)
                if 'scalarSWE' in summa_plots:
                    plot_paths['scalarSWE'] = summa_plots['scalarSWE']
                    plot_paths['model_comparison'] = summa_plots['scalarSWE']
                # Include other snow-related variables if present
                for var in ['scalarSnowDepth', 'scalarSnowfall']:
                    if var in summa_plots:
                        plot_paths[var] = summa_plots[var]

            elif calibration_target in ('et', 'evapotranspiration', 'latent_heat', 'le'):
                # ET/energy flux calibration -> SUMMA outputs with energy observations
                summa_plots = self.visualize_summa_outputs(experiment_id)
                if 'scalarLatHeatTotal' in summa_plots:
                    plot_paths['scalarLatHeatTotal'] = summa_plots['scalarLatHeatTotal']
                    plot_paths['model_comparison'] = summa_plots['scalarLatHeatTotal']
                if 'scalarSenHeatTotal' in summa_plots:
                    plot_paths['scalarSenHeatTotal'] = summa_plots['scalarSenHeatTotal']
                # Include ET-related variables if present
                for var in ['scalarCanopyEvaporation', 'scalarGroundEvaporation', 'scalarTotalET']:
                    if var in summa_plots:
                        plot_paths[var] = summa_plots[var]

            else:
                # Unknown target - try both streamflow and SUMMA outputs
                self.logger.info(f"Unknown calibration target '{calibration_target}', generating all available plots")
                comparison_plot = self.generate_model_comparison_overview(
                    experiment_id=experiment_id,
                    context='calibrate_model'
                )
                if comparison_plot:
                    plot_paths['model_comparison'] = comparison_plot

                summa_plots = self.visualize_summa_outputs(experiment_id)
                plot_paths.update(summa_plots)

        self.logger.info(f"Generated {len(plot_paths)} calibration visualization(s)")
        return plot_paths

    # =========================================================================
    # Workflow Diagnostic Methods
    # =========================================================================

    @skip_if_not_diagnostic()
    def diagnostic_domain_definition(self, basin_gdf: Any, dem_path: Optional[Path] = None) -> Optional[str]:
        """
        Generate diagnostic plots for domain definition step.

        Creates validation plots including:
        - DEM coverage vs basin boundary
        - NoData percentage analysis
        - Elevation histogram

        Args:
            basin_gdf: GeoDataFrame of basin boundaries
            dem_path: Path to DEM raster file

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating domain definition diagnostics...")
        return self.workflow_diagnostic_plotter.plot_domain_definition_diagnostic(
            basin_gdf=basin_gdf,
            dem_path=dem_path
        )

    @skip_if_not_diagnostic()
    def diagnostic_discretization(self, hru_gdf: Any, method: str) -> Optional[str]:
        """
        Generate diagnostic plots for discretization step.

        Creates validation plots including:
        - HRU area distribution
        - Tiny/huge HRU warnings
        - Count by elevation band

        Args:
            hru_gdf: GeoDataFrame of HRU polygons
            method: Discretization method used

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating discretization diagnostics...")
        return self.workflow_diagnostic_plotter.plot_discretization_diagnostic(
            hru_gdf=hru_gdf,
            method=method
        )

    @skip_if_not_diagnostic()
    def diagnostic_observations(self, obs_df: Any, obs_type: str) -> Optional[str]:
        """
        Generate diagnostic plots for observation processing step.

        Creates validation plots including:
        - Gap analysis timeline
        - Outlier detection
        - Value range histogram

        Args:
            obs_df: DataFrame of observations
            obs_type: Type of observations (e.g., 'streamflow', 'swe')

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating observation diagnostics for {obs_type}...")
        return self.workflow_diagnostic_plotter.plot_observations_diagnostic(
            obs_df=obs_df,
            obs_type=obs_type
        )

    @skip_if_not_diagnostic()
    def diagnostic_forcing_raw(self, forcing_nc: Path, domain_shp: Optional[Path] = None) -> Optional[str]:
        """
        Generate diagnostic plots for raw forcing acquisition step.

        Creates validation plots including:
        - Spatial coverage map
        - Variable completeness
        - Temporal extent

        Args:
            forcing_nc: Path to raw forcing NetCDF file
            domain_shp: Optional path to domain shapefile for overlay

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating raw forcing diagnostics...")
        return self.workflow_diagnostic_plotter.plot_forcing_raw_diagnostic(
            forcing_nc=forcing_nc,
            domain_shp=domain_shp
        )

    @skip_if_not_diagnostic()
    def diagnostic_forcing_remapped(
        self,
        raw_nc: Path,
        remapped_nc: Path,
        hru_shp: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for forcing remapping step.

        Creates validation plots including:
        - Raw vs remapped comparison
        - Conservation check
        - Per-HRU coverage

        Args:
            raw_nc: Path to raw forcing NetCDF file
            remapped_nc: Path to remapped forcing NetCDF file
            hru_shp: Optional path to HRU shapefile

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating forcing remapping diagnostics...")
        return self.workflow_diagnostic_plotter.plot_forcing_remapped_diagnostic(
            raw_nc=raw_nc,
            remapped_nc=remapped_nc,
            hru_shp=hru_shp
        )

    @skip_if_not_diagnostic()
    def diagnostic_model_preprocessing(self, input_dir: Path, model_name: str) -> Optional[str]:
        """
        Generate diagnostic plots for model preprocessing step.

        Creates validation plots including:
        - Required vs present files
        - Variable inventory
        - Config validation

        Args:
            input_dir: Path to model input directory
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating model preprocessing diagnostics for {model_name}...")
        return self.workflow_diagnostic_plotter.plot_model_preprocessing_diagnostic(
            input_dir=input_dir,
            model_name=model_name
        )

    @skip_if_not_diagnostic()
    def diagnostic_model_output(self, output_nc: Path, model_name: str) -> Optional[str]:
        """
        Generate diagnostic plots for model output step.

        Creates validation plots including:
        - Output variable ranges
        - NaN heatmap
        - Mass/energy balance check

        Args:
            output_nc: Path to model output NetCDF file
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating model output diagnostics for {model_name}...")
        return self.workflow_diagnostic_plotter.plot_model_output_diagnostic(
            output_nc=output_nc,
            model_name=model_name
        )

    @skip_if_not_diagnostic()
    def diagnostic_attributes(
        self,
        dem_path: Optional[Path] = None,
        soil_path: Optional[Path] = None,
        land_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for attribute acquisition step.

        Creates validation plots including:
        - DEM coverage and statistics
        - Soil class distribution
        - Land class distribution

        Args:
            dem_path: Path to DEM raster file
            soil_path: Path to soil class raster file
            land_path: Path to land class raster file

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating attribute acquisition diagnostics...")
        return self.workflow_diagnostic_plotter.plot_attributes_diagnostic(
            dem_path=dem_path,
            soil_path=soil_path,
            land_path=land_path
        )

    @skip_if_not_diagnostic()
    def diagnostic_calibration(
        self,
        history: Optional[List[Dict]] = None,
        best_params: Optional[Dict[str, float]] = None,
        obs_vs_sim: Optional[Dict[str, Any]] = None,
        model_name: str = 'Unknown'
    ) -> Optional[str]:
        """
        Generate diagnostic plots for calibration step.

        Creates validation plots including:
        - Optimization convergence
        - Parameter evolution/values
        - Observed vs simulated comparison

        Args:
            history: List of optimization history dictionaries
            best_params: Dictionary of best parameter values
            obs_vs_sim: Dictionary with 'observed' and 'simulated' arrays
            model_name: Name of the model being calibrated

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating calibration diagnostics for {model_name}...")
        return self.workflow_diagnostic_plotter.plot_calibration_diagnostic(
            history=history,
            best_params=best_params,
            obs_vs_sim=obs_vs_sim,
            model_name=model_name
        )

    @skip_if_not_diagnostic()
    def diagnostic_coupling_conservation(
        self, graph: Any, output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """Generate conservation diagnostic for a coupled model run.

        Extracts the conservation log from a dCoupler CouplingGraph,
        logs a summary table, and optionally generates a bar-chart plot.

        Args:
            graph: A dCoupler CouplingGraph instance with conservation enabled.
            output_dir: Directory for saving the plot. If None, uses
                        project_dir/diagnostics/.

        Returns:
            Path to saved diagnostic plot, or None if conservation is disabled.
        """
        try:
            from symfluence.coupling.diagnostics import CouplingDiagnostics
        except ImportError:
            self.logger.debug("coupling.diagnostics not available")
            return None

        report = CouplingDiagnostics.extract_conservation_report(graph)
        if report["mode"] == "disabled":
            self.logger.debug("Conservation checking disabled; skipping diagnostic")
            return None

        # Log the summary table
        table = CouplingDiagnostics.format_conservation_table(report)
        self.logger.info(table)

        # Generate plot
        if output_dir is None:
            output_dir = self.project_dir / "diagnostics"
        plot_path = CouplingDiagnostics.plot_conservation_errors(
            report, Path(output_dir) / "conservation.png"
        )
        return str(plot_path) if plot_path else None
