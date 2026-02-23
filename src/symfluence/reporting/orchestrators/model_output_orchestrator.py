"""Orchestrator for model output visualizations.

Extracted from ``ReportingManager`` — handles registry-based model plotter
dispatch (``visualize_model_results``) and the individual
``visualize_*_outputs`` methods.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from symfluence.core.exceptions import ReportingError, symfluence_error_handler
from symfluence.reporting.core.decorators import skip_if_not_visualizing

if TYPE_CHECKING:
    from symfluence.reporting.config.plot_config import PlotConfig
    from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter


class ModelOutputOrchestrator:
    """Orchestrates model output visualizations via registry dispatch."""

    def __init__(
        self,
        config: Any,
        logger: Any,
        visualize: bool,
        plot_config: 'PlotConfig',
        analysis_plotter: 'AnalysisPlotter',
    ) -> None:
        self._config = config
        self.config = config
        self.logger = logger
        self.visualize = visualize
        self.plot_config = plot_config
        self.analysis_plotter = analysis_plotter

    # --- Specific model output methods ---

    @skip_if_not_visualizing()
    def visualize_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """Visualize model outputs (streamflow comparison).

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
        """Visualize lumped model outputs.

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
        """Visualize FUSE model outputs.

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
        """Visualize SUMMA model outputs (all variables).

        Args:
            experiment_id: Experiment ID.

        Returns:
            Dictionary mapping variable names to plot paths.
        """
        self.logger.info(f"Creating SUMMA output visualizations for experiment {experiment_id}...")
        return self.analysis_plotter.plot_summa_outputs(experiment_id)

    @skip_if_not_visualizing()
    def visualize_ngen_results(self, sim_df: Any, obs_df: Optional[Any], experiment_id: str, results_dir: Path) -> None:
        """Visualize NGen streamflow plots.

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
        """Visualize LSTM simulation results.

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
            results_df, obs_streamflow, obs_snow, use_snow, output_dir, experiment_id,
        )

    @skip_if_not_visualizing()
    def visualize_hype_results(self, sim_flow: Any, obs_flow: Any, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path) -> None:
        """Visualize HYPE streamflow comparison.

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
            sim_flow, obs_flow, outlet_id, domain_name, experiment_id, project_dir,
        )

    # --- Registry-based dispatch ---

    @skip_if_not_visualizing()
    def visualize_model_results(self, model_name: str, **kwargs) -> Optional[Any]:
        """Visualize model results using registry-based dispatch.

        Uses the PlotterRegistry to dynamically discover and invoke the
        appropriate model-specific plotter.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE', 'HYPE', 'NGEN', 'LSTM')
            **kwargs: Model-specific arguments passed to the plotter's plot method

        Returns:
            Plot result (path to saved plot or dict of paths), or None.
        """
        # Import model modules to trigger plotter registration
        from symfluence.core.constants import SupportedModels

        for model in SupportedModels.WITH_PLOTTERS:
            try:
                __import__(f'symfluence.models.{model}')
            except ImportError:
                self.logger.debug(f"Model plotter module '{model}' not available")

        from symfluence.reporting.plotter_registry import PlotterRegistry

        model_upper = model_name.upper()
        plotter_cls = PlotterRegistry.get_plotter(model_upper)

        if plotter_cls is None:
            available = PlotterRegistry.list_plotters()
            self.logger.warning(
                f"No plotter registered for model '{model_name}'. "
                f"Available plotters: {available}. Use a direct visualize_*_outputs method instead."
            )
            return None

        self.logger.info(f"Creating {model_name} visualizations using registered plotter...")

        with symfluence_error_handler(
            f"{model_name} plotter",
            self.logger,
            reraise=False,
            error_type=ReportingError,
        ):
            plotter = plotter_cls(self.config, self.logger, self.plot_config)
            return plotter.plot(**kwargs)

        return None
