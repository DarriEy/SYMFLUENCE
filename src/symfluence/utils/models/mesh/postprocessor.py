"""
MESH model postprocessor.

Handles extraction and processing of MESH model simulation results.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..registry import ModelRegistry
from ..base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('MESH')
class MESHPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for the MESH model.

    Handles extraction and processing of MESH model simulation results.
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "MESH"

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.gr_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_gr_path = self.project_dir / 'forcing' / 'MESH_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from MESH outputs.

        Note: MESH streamflow extraction not yet implemented.
        """
        self.logger.warning("MESH streamflow extraction not yet implemented")
        return None
