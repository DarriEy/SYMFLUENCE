"""Orchestrator for workflow-step diagnostic visualizations.

Extracted from ``ReportingManager`` — each ``diagnostic_*`` method generates
validation plots at the completion of a workflow step (domain definition,
discretization, forcing, calibration, etc.).  All methods use the
``@skip_if_not_diagnostic()`` decorator so they are no-ops when diagnostic
mode is disabled.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from symfluence.reporting.core.decorators import skip_if_not_diagnostic

if TYPE_CHECKING:
    from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter


class DiagnosticsOrchestrator:
    """Orchestrates per-step diagnostic plots for SYMFLUENCE workflows."""

    def __init__(
        self,
        config: Any,
        logger: Any,
        diagnostic: bool,
        project_dir: Path,
        workflow_diagnostic_plotter: 'WorkflowDiagnosticPlotter',
    ) -> None:
        self._config = config
        self.logger = logger
        self.diagnostic = diagnostic
        self.project_dir = project_dir
        self.workflow_diagnostic_plotter = workflow_diagnostic_plotter

    @skip_if_not_diagnostic()
    def diagnostic_domain_definition(self, basin_gdf: Any, dem_path: Optional[Path] = None) -> Optional[str]:
        """Generate diagnostic plots for domain definition step.

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
            dem_path=dem_path,
        )

    @skip_if_not_diagnostic()
    def diagnostic_discretization(self, hru_gdf: Any, method: str) -> Optional[str]:
        """Generate diagnostic plots for discretization step.

        Args:
            hru_gdf: GeoDataFrame of HRU polygons
            method: Discretization method used

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating discretization diagnostics...")
        return self.workflow_diagnostic_plotter.plot_discretization_diagnostic(
            hru_gdf=hru_gdf,
            method=method,
        )

    @skip_if_not_diagnostic()
    def diagnostic_observations(self, obs_df: Any, obs_type: str) -> Optional[str]:
        """Generate diagnostic plots for observation processing step.

        Args:
            obs_df: DataFrame of observations
            obs_type: Type of observations (e.g., 'streamflow', 'swe')

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating observation diagnostics for {obs_type}...")
        return self.workflow_diagnostic_plotter.plot_observations_diagnostic(
            obs_df=obs_df,
            obs_type=obs_type,
        )

    @skip_if_not_diagnostic()
    def diagnostic_forcing_raw(self, forcing_nc: Path, domain_shp: Optional[Path] = None) -> Optional[str]:
        """Generate diagnostic plots for raw forcing acquisition step.

        Args:
            forcing_nc: Path to raw forcing NetCDF file
            domain_shp: Optional path to domain shapefile for overlay

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info("Generating raw forcing diagnostics...")
        return self.workflow_diagnostic_plotter.plot_forcing_raw_diagnostic(
            forcing_nc=forcing_nc,
            domain_shp=domain_shp,
        )

    @skip_if_not_diagnostic()
    def diagnostic_forcing_remapped(
        self,
        raw_nc: Path,
        remapped_nc: Path,
        hru_shp: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate diagnostic plots for forcing remapping step.

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
            hru_shp=hru_shp,
        )

    @skip_if_not_diagnostic()
    def diagnostic_model_preprocessing(self, input_dir: Path, model_name: str) -> Optional[str]:
        """Generate diagnostic plots for model preprocessing step.

        Args:
            input_dir: Path to model input directory
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating model preprocessing diagnostics for {model_name}...")
        return self.workflow_diagnostic_plotter.plot_model_preprocessing_diagnostic(
            input_dir=input_dir,
            model_name=model_name,
        )

    @skip_if_not_diagnostic()
    def diagnostic_model_output(self, output_nc: Path, model_name: str) -> Optional[str]:
        """Generate diagnostic plots for model output step.

        Args:
            output_nc: Path to model output NetCDF file
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        self.logger.info(f"Generating model output diagnostics for {model_name}...")
        return self.workflow_diagnostic_plotter.plot_model_output_diagnostic(
            output_nc=output_nc,
            model_name=model_name,
        )

    @skip_if_not_diagnostic()
    def diagnostic_attributes(
        self,
        dem_path: Optional[Path] = None,
        soil_path: Optional[Path] = None,
        land_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate diagnostic plots for attribute acquisition step.

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
            land_path=land_path,
        )

    @skip_if_not_diagnostic()
    def diagnostic_calibration(
        self,
        history: Optional[List[Dict]] = None,
        best_params: Optional[Dict[str, float]] = None,
        obs_vs_sim: Optional[Dict[str, Any]] = None,
        model_name: str = 'Unknown',
    ) -> Optional[str]:
        """Generate diagnostic plots for calibration step.

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
            model_name=model_name,
        )

    @skip_if_not_diagnostic()
    def diagnostic_coupling_conservation(
        self, graph: Any, output_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate conservation diagnostic for a coupled model run.

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
            report, Path(output_dir) / "conservation.png",
        )
        return str(plot_path) if plot_path else None
