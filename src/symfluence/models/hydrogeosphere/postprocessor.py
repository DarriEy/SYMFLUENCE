"""
HydroGeoSphere Model Postprocessor

Extracts hydrograph discharge from HGS output and optionally
combines with SUMMA surface runoff for total streamflow.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_postprocessor("HYDROGEOSPHERE")
class HGSPostProcessor(StandardModelPostprocessor):
    """
    Postprocesses HGS output.

    Extracts hydrograph discharge and optionally combines with
    land surface model (SUMMA) surface runoff for total streamflow.
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='unknown',
            dict_key='DOMAIN_NAME',
        )

    def _get_model_name(self) -> str:
        return "HYDROGEOSPHERE"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract streamflow from HGS output, optionally combined with SUMMA."""
        if output_dir is None:
            output_dir = (
                self.project_dir / "simulations"
                / self.config.domain.experiment_id / "HYDROGEOSPHERE"
            )

        try:
            from .extractor import HGSResultExtractor
            extractor = HGSResultExtractor()

            start_date = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START',
            )

            hydrograph = extractor.extract_variable(
                output_dir, 'hydrograph',
                start_date=str(start_date),
            )
        except Exception as e:
            logger.error(f"Failed to extract HGS hydrograph: {e}")
            return None

        if hydrograph.empty:
            logger.warning("No hydrograph data extracted from HGS")
            return None

        coupling_source = self._get_config_value(
            lambda: self.config.model.hydrogeosphere.coupling_source,
            default=None,
            dict_key='HGS_COUPLING_SOURCE',
        )

        if coupling_source and str(coupling_source).upper() == 'SUMMA':
            return self._extract_coupled_streamflow(hydrograph, output_dir)
        else:
            return self._extract_standalone_baseflow(hydrograph)

    def _extract_standalone_baseflow(
        self, hydrograph: pd.Series
    ) -> Tuple[pd.Series, Dict]:
        """Extract standalone streamflow from HGS hydrograph."""
        hydrograph.name = 'streamflow_m3s'

        metadata = {
            'units': 'm3/s',
            'source': 'HGS hydrograph outlet',
            'mode': 'standalone',
        }

        logger.info(
            f"Extracted HGS hydrograph: {len(hydrograph)} timesteps, "
            f"mean={hydrograph.mean():.4f} m3/s"
        )

        self.save_streamflow_to_results(hydrograph)
        self._try_generate_plot()

        return hydrograph, metadata

    def _extract_coupled_streamflow(
        self,
        hydrograph: pd.Series,
        hgs_output_dir: Path,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract combined SUMMA surface runoff + HGS baseflow."""
        try:
            from .coupling import SUMMAToHGSCoupler

            coupler = SUMMAToHGSCoupler(self.config_dict, self.logger)

            summa_output_dir = (
                self.project_dir / "simulations"
                / self.config.domain.experiment_id / "SUMMA"
            )

            if not summa_output_dir.exists():
                logger.warning(
                    f"SUMMA output directory not found: {summa_output_dir}. "
                    "Falling back to standalone."
                )
                return self._extract_standalone_baseflow(hydrograph)

            surface_runoff = coupler.extract_surface_runoff(summa_output_dir)
            area_m2 = self._get_catchment_area() * 1e6

            total_streamflow = coupler.combine_flows(
                surface_runoff, hydrograph, area_m2
            )

            metadata = {
                'units': 'm3/s',
                'source': 'SUMMA surface runoff + HGS hydrograph',
                'mode': 'coupled_SUMMA_HGS',
                'area_km2': self._get_catchment_area(),
            }

            logger.info(
                f"Extracted coupled streamflow: {len(total_streamflow)} timesteps, "
                f"mean={total_streamflow.mean():.4f} m3/s"
            )

            self.save_streamflow_to_results(total_streamflow)
            self._try_generate_plot()

            return total_streamflow, metadata

        except Exception as e:
            logger.error(f"Coupled extraction failed: {e}. Falling back to standalone.")
            return self._extract_standalone_baseflow(hydrograph)

    def _try_generate_plot(self) -> None:
        """Attempt to generate coupling diagnostics plot."""
        try:
            from .plotter import HGSPlotter
            plotter = HGSPlotter(self.config_dict, self.logger)
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            )
            result = plotter.plot_coupling_results(experiment_id)
            if result:
                logger.debug(f"HGS coupling plot saved: {result}")
        except Exception as e:
            logger.debug(f"Could not generate HGS coupling plot: {e}")

    def _get_catchment_area(self) -> float:
        """Get catchment area in km2."""
        area = self._get_config_value(
            lambda: self.config.domain.catchment_area,
            default=None,
            dict_key='CATCHMENT_AREA',
        )
        if area:
            return float(area)

        try:
            from symfluence.evaluation.utilities.streamflow_metrics import StreamflowMetrics
            metrics = StreamflowMetrics()
            return metrics.get_catchment_area(
                self.config_dict, self.project_dir, self.domain_name,
                source='shapefile'
            )
        except Exception:
            logger.warning("Could not determine catchment area, using default 2210 km2")
            return 2210.0
