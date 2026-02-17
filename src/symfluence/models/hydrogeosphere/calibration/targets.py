"""
HydroGeoSphere Streamflow Calibration Target

Extracts simulated streamflow from HGS hydrograph output and
compares against observed streamflow for metric calculation
during calibration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator
from symfluence.optimization.registry import OptimizerRegistry

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_calibration_target('HYDROGEOSPHERE', 'streamflow')
class HGSStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for HydroGeoSphere."""

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
        settings_dir: Optional[str] = None,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'HYDROGEOSPHERE'
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return HGS output files as simulation file list."""
        hydrograph_files = sorted(sim_dir.glob('*hydrograph*'))
        if hydrograph_files:
            return hydrograph_files
        return sorted(sim_dir.glob('*head*'))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from HGS output."""
        if not sim_files:
            return pd.Series(dtype=float)

        output_dir = sim_files[0].parent
        streamflow = self._extract_hgs_streamflow(output_dir)
        if streamflow is None:
            return pd.Series(dtype=float)
        return streamflow

    def _extract_hgs_streamflow(
        self,
        output_dir: Path,
    ) -> Optional[pd.Series]:
        """Extract streamflow from HGS output directory."""
        try:
            from symfluence.models.hydrogeosphere.extractor import HGSResultExtractor

            extractor = HGSResultExtractor()
            start_date = str(self.config_dict.get('EXPERIMENT_TIME_START', '2000-01-01'))

            streamflow = extractor.extract_variable(
                output_dir, 'hydrograph', start_date=start_date,
            )

            if streamflow.empty:
                self.logger.warning("No HGS hydrograph data extracted")
                return None

            streamflow.name = 'streamflow_m3s'

            self.logger.debug(
                f"HGS streamflow: {len(streamflow)} daily values, "
                f"mean={streamflow.mean():.4f} m3/s"
            )

            return streamflow

        except Exception as e:
            self.logger.error(f"Failed to extract HGS streamflow: {e}")
            return None
