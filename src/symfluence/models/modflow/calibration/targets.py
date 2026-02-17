"""
Coupled Groundwater Calibration Target

Extracts combined streamflow from any land surface model's surface runoff
+ MODFLOW drain discharge for comparison against observed streamflow.

The land surface model is determined by LAND_SURFACE_MODEL config key.
Uses SUMMAToMODFLOWCoupler for recharge extraction and flow combination,
and MODFLOWResultExtractor for drain discharge from .bud files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator
from symfluence.optimization.registry import OptimizerRegistry

logger = logging.getLogger(__name__)

# Map land surface models to their total routed runoff variable names.
# Used to compute fast flow = total_runoff - soil_drainage (recharge).
TOTAL_RUNOFF_VARIABLES = {
    'SUMMA': 'averageRoutedRunoff',
    'CLM': 'QOVER',
    'MESH': 'RUNOFF',
}

DRAINAGE_VARIABLES = {
    'SUMMA': 'scalarSoilDrainage',
    'CLM': 'QCHARGE',
    'MESH': 'DRAINAGE',
}


@OptimizerRegistry.register_calibration_target('COUPLED_GW', 'streamflow')
class CoupledGWStreamflowTarget(StreamflowEvaluator):
    """Streamflow target for coupled land-surface + MODFLOW calibration.

    Combines any land surface model's surface runoff with MODFLOW
    groundwater baseflow (drain discharge) to produce total simulated
    streamflow.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'COUPLED_GW'
        self.land_model_name = config.get('LAND_SURFACE_MODEL', 'SUMMA').upper()

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return output files from both land surface and MODFLOW sub-directories."""
        files = []

        # Land surface model output
        land_dir = sim_dir / self.land_model_name
        if land_dir.exists():
            files.extend(sorted(land_dir.glob('*.nc')))

        # MODFLOW output
        modflow_dir = sim_dir / 'MODFLOW'
        if modflow_dir.exists():
            files.extend(sorted(modflow_dir.glob('*.bud')))
            files.extend(sorted(modflow_dir.glob('*.hds')))

        return files

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract combined streamflow from land-surface + MODFLOW output."""
        if not sim_files:
            return pd.Series(dtype=float)

        # Determine directories from file paths
        land_dir = None
        modflow_dir = None
        for f in sim_files:
            if f.parent.name == self.land_model_name:
                land_dir = f.parent
            elif f.parent.name == 'MODFLOW':
                modflow_dir = f.parent

        if land_dir is None or modflow_dir is None:
            self.logger.error(
                f"Could not find {self.land_model_name} and/or MODFLOW output directories"
            )
            return pd.Series(dtype=float)

        try:
            return self._extract_combined_streamflow(land_dir, modflow_dir)
        except Exception as e:
            self.logger.error(f"Failed to extract combined streamflow: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return pd.Series(dtype=float)

    def _extract_combined_streamflow(
        self,
        land_dir: Path,
        modflow_dir: Path,
    ) -> pd.Series:
        """Extract and combine surface runoff + baseflow.

        Returns:
            Total streamflow in m3/s with daily datetime index.
        """
        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler
        from symfluence.models.modflow.extractor import MODFLOWResultExtractor

        coupler = SUMMAToMODFLOWCoupler(self.config_dict, self.logger)
        extractor = MODFLOWResultExtractor()

        # Extract fast runoff = total routed runoff minus soil drainage.
        # scalarSurfaceRunoff only captures Hortonian overland flow (near-zero
        # in most catchments).  The correct fast-flow component subtracts the
        # recharge that feeds MODFLOW from the total SUMMA-routed runoff.
        total_var = TOTAL_RUNOFF_VARIABLES.get(
            self.land_model_name, 'averageRoutedRunoff',
        )
        drainage_var = DRAINAGE_VARIABLES.get(
            self.land_model_name, 'scalarSoilDrainage',
        )
        surface_runoff = coupler.extract_fast_runoff(
            land_dir, total_var=total_var, drainage_var=drainage_var,
        )

        # Extract MODFLOW drain discharge (m3/d)
        # Use date-only (no time component) so dates align with SUMMA daily output
        start_str = str(self.config_dict.get('EXPERIMENT_TIME_START', '2000-01-01'))
        start_date = pd.Timestamp(start_str).strftime('%Y-%m-%d')
        drain_discharge = extractor.extract_variable(
            modflow_dir,
            'drain_discharge',
            start_date=start_date,
            stress_period_length=1.0,
        )

        # Get catchment area for unit conversion
        catchment_area_m2 = self._get_catchment_area()

        # Combine flows
        total_streamflow = coupler.combine_flows(
            surface_runoff, drain_discharge, catchment_area_m2,
        )

        return total_streamflow
