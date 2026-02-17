"""
PIHM Streamflow Calibration Target

Extracts simulated streamflow from PIHM .rivflx output (river flux)
and compares against observed streamflow for metric calculation
during calibration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator
from symfluence.optimization.registry import OptimizerRegistry

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_calibration_target('PIHM', 'streamflow')
class PIHMStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for PIHM integrated hydrologic model."""

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
        settings_dir: Optional[str] = None,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'PIHM'
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return PIHM output files as simulation file list.

        MM-PIHM output is in: sim_dir/output/pihm_lumped/*.river.flx1.txt
        Search both the direct directory and the MM-PIHM subdirectory.
        """
        # MM-PIHM directory structure: output/pihm_lumped/
        pihm_output = sim_dir / "output" / "pihm_lumped"
        if pihm_output.exists():
            files = sorted(pihm_output.glob('*.river.flx*.txt'))
            if files:
                return files
            files = sorted(pihm_output.glob('*.gw.txt'))
            if files:
                return files

        # Fallback: search recursively
        files = sorted(sim_dir.rglob('*.river.flx*.txt'))
        if files:
            return files

        # Legacy patterns
        rivflx_files = sorted(sim_dir.glob('*.rivflx*'))
        if rivflx_files:
            return rivflx_files
        return sorted(sim_dir.glob('*.gwhead*'))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from PIHM output."""
        if not sim_files:
            return pd.Series(dtype=float)

        output_dir = sim_files[0].parent
        streamflow = self._extract_pihm_streamflow(output_dir)
        if streamflow is None:
            return pd.Series(dtype=float)
        return streamflow

    def _extract_pihm_streamflow(
        self,
        output_dir: Path,
    ) -> Optional[pd.Series]:
        """Extract streamflow from PIHM output directory."""
        try:
            from symfluence.models.pihm.extractor import PIHMResultExtractor

            extractor = PIHMResultExtractor()
            start_date = str(self.config_dict.get('EXPERIMENT_TIME_START', '2000-01-01'))

            streamflow = extractor.extract_variable(
                output_dir, 'river_flux', start_date=start_date,
            )

            if streamflow.empty:
                self.logger.warning("No PIHM river flux data extracted")
                return None

            streamflow.name = 'streamflow_m3s'

            self.logger.debug(
                f"PIHM streamflow: {len(streamflow)} daily values, "
                f"mean={streamflow.mean():.4f} m3/s"
            )

            return streamflow

        except Exception as e:
            self.logger.error(f"Failed to extract PIHM streamflow: {e}")
            return None
