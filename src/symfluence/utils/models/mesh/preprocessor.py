"""
MESH model preprocessor.

Handles data preparation using meshflow library for MESH model setup.
"""

import os
from typing import Dict, Any
from pathlib import Path
import meshflow  # type: ignore  # version >= v0.1.0.dev5

from ..base import BaseModelPreProcessor
from ..mixins import ObservationLoaderMixin
from symfluence.utils.exceptions import ConfigurationError, ModelExecutionError, symfluence_error_handler


class MESHPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Preprocessor for the MESH model.

    Handles data preparation using meshflow library for MESH model setup.
    Inherits common functionality from BaseModelPreProcessor and observation loading from ObservationLoaderMixin.

    Attributes:
        config: Configuration settings for MESH
        logger: Logger object for recording processing information
        project_dir: Directory for the current project
        setup_dir: Directory for MESH setup files (inherited)
        domain_name: Name of the domain being processed (inherited)
    """

    def _get_model_name(self) -> str:
        """Return model name for MESH."""
        return "MESH"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles common paths)
        super().__init__(config, logger)

        # MESH-specific catchment path (uses river basins instead of catchment)
        self.catchment_path = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')

        # Phase 3: Use typed config when available
        if self.typed_config:
            self.catchment_name = self.typed_config.paths.river_basins_name
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.typed_config.domain.definition_method}.shp"
        else:
            self.catchment_name = self.config.get('RIVER_BASINS_NAME')
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"

        # River network paths
        self.rivers_path = self.get_river_network_path().parent
        self.rivers_name = self.get_river_network_path().name

    def run_preprocessing(self):
        """
        Run the complete MESH preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info("Starting MESH preprocessing")
        return self.run_preprocessing_template()

    def _pre_setup(self) -> None:
        """MESH-specific pre-setup: create meshflow config (template hook)."""
        self._meshflow_config = self.create_json()

    def _prepare_forcing(self) -> None:
        """MESH-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data(self._meshflow_config)

    def create_json(self):
        """Create configuration dictionary for meshflow."""

        def _get_config_value(key: str, default_value):
            value = self.config.get(key)
            if value is None or value == 'default':
                return default_value
            return value

        default_forcing_vars = {
            "RDRS_v2.1_P_P0_SFC": "air_pressure",
            "RDRS_v2.1_P_HU_09944": "specific_humidity",
            "RDRS_v2.1_P_TT_09944": "air_temperature",
            "RDRS_v2.1_P_UVC_09944": "wind_speed",
            "RDRS_v2.1_A_PR0_SFC": "precipitation",
            "RDRS_v2.1_P_FB_SFC": "shortwave_radiation",
            "RDRS_v2.1_P_FI_SFC": "longwave_radiation",
        }

        default_forcing_units = {
            "RDRS_v2.1_P_P0_SFC": 'millibar',
            "RDRS_v2.1_P_HU_09944": 'kg/kg',
            "RDRS_v2.1_P_TT_09944": 'celsius',
            "RDRS_v2.1_P_UVC_09944": 'knot',
            "RDRS_v2.1_A_PR0_SFC": 'm/hr',
            "RDRS_v2.1_P_FB_SFC": 'W/m^2',
            "RDRS_v2.1_P_FI_SFC": 'W/m^2',
        }

        default_forcing_to_units = {
            "RDRS_v2.1_P_P0_SFC": 'pascal',
            "RDRS_v2.1_P_HU_09944": 'kg/kg',
            "RDRS_v2.1_P_TT_09944": 'kelvin',
            "RDRS_v2.1_P_UVC_09944": 'm/s',
            "RDRS_v2.1_A_PR0_SFC": 'mm/s',
            "RDRS_v2.1_P_FB_SFC": 'W/m^2',
            "RDRS_v2.1_P_FI_SFC": 'W/m^2',
        }

        default_landcover_classes = {
            1: 'Temperate or sub-polar needleleaf forest',
            2: 'Sub-polar taiga needleleaf forest',
            3: 'Tropical or sub-tropical broadleaf evergreen forest',
            4: 'Tropical or sub-tropical broadleaf deciduous forest',
            5: 'Temperate or sub-polar broadleaf deciduous forest',
            6: 'Mixed forest',
            7: 'Tropical or sub-tropical shrubland',
            8: 'Temperate or sub-polar shrubland',
            9: 'Tropical or sub-tropical grassland',
            10: 'Temperate or sub-polar grassland',
            11: 'Sub-polar or polar shrubland-lichen-moss',
            12: 'Sub-polar or polar grassland-lichen-moss',
            13: 'Sub-polar or polar barren-lichen-moss',
            14: 'Wetland',
            15: 'Cropland',
            16: 'Barren lands',
            17: 'Urban',
            18: 'Water',
            19: 'Snow and Ice',
        }

        default_ddb_vars = {
            'Slope': 'river_slope',
            'Length': 'river_length',
            'Rank': 'rank',
            'Next': 'next',
            'landcover': 'landclass',
            'GRU_area': 'subbasin_area',
        }

        default_ddb_units = {
            'river_slope': 'm/m',
            'river_length': 'm',
            'rank': 'dimensionless',
            'next': 'dimensionless',
            'landclass': 'dimensionless',
            'subbasin_area': 'm^2',
        }

        default_ddb_to_units = default_ddb_units.copy()

        default_ddb_min_values = {
            'river_slope': 1e-10,
            'river_length': 1e-3,
            'subbasin_area': 1e-3,
        }

        forcing_vars = _get_config_value('MESH_FORCING_VARS', default_forcing_vars)
        forcing_units = default_forcing_units.copy()
        forcing_units.update(_get_config_value('MESH_FORCING_UNITS', {}))
        forcing_to_units = default_forcing_to_units.copy()
        forcing_to_units.update(_get_config_value('MESH_FORCING_TO_UNITS', {}))

        missing_units = [var for var in forcing_vars if var not in forcing_units]
        missing_to_units = [var for var in forcing_vars if var not in forcing_to_units]
        if missing_units or missing_to_units:
            raise ConfigurationError(
                "MESH forcing units are incomplete. Missing units for: "
                f"{', '.join(sorted(set(missing_units + missing_to_units)))}"
            )

        landcover_stats_path = _get_config_value('MESH_LANDCOVER_STATS_PATH', None)
        if landcover_stats_path:
            landcover_path = Path(landcover_stats_path)
        else:
            landcover_file = _get_config_value(
                'MESH_LANDCOVER_STATS_FILE',
                'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv',
            )
            landcover_dir = Path(
                _get_config_value(
                    'MESH_LANDCOVER_STATS_DIR',
                    self.project_dir / 'attributes' / 'gistool-outputs',
                )
            )
            landcover_path = landcover_dir / landcover_file

        forcing_files_path = Path(
            _get_config_value(
                'MESH_FORCING_PATH',
                self.project_dir / 'forcing' / 'easymore-outputs',
            )
        )

        # using meshflow >= v0.1.0.dev5
        # modify the following to match your settings
        config = {
            'riv': os.path.join(str(self.rivers_path / self.rivers_name)),
            'cat': os.path.join(str(self.catchment_path / self.catchment_name)),
            'landcover': os.path.join(str(landcover_path)),
            'forcing_files': os.path.join(str(forcing_files_path)),
            'forcing_vars': forcing_vars,
            'forcing_units': forcing_units,
            'forcing_to_units': forcing_to_units,
            'main_id': _get_config_value('MESH_MAIN_ID', 'GRU_ID'),
            'ds_main_id': _get_config_value('MESH_DS_MAIN_ID', 'DSLINKNO'),
            'landcover_classes': _get_config_value('MESH_LANDCOVER_CLASSES', default_landcover_classes),
            'ddb_vars': _get_config_value('MESH_DDB_VARS', default_ddb_vars),
            'ddb_units': _get_config_value('MESH_DDB_UNITS', default_ddb_units),
            'ddb_to_units': _get_config_value('MESH_DDB_TO_UNITS', default_ddb_to_units),
            'ddb_min_values': _get_config_value('MESH_DDB_MIN_VALUES', default_ddb_min_values),
            'gru_dim': _get_config_value('MESH_GRU_DIM', 'NGRU'),
            'hru_dim': _get_config_value('MESH_HRU_DIM', 'subbasin'),
            'outlet_value': _get_config_value('MESH_OUTLET_VALUE', 0),
        }
        return config

    def prepare_forcing_data(self, config):
        """Prepare forcing data using meshflow."""
        exp = meshflow.MESHWorkflow(**config)
        exp.run(save_path=self.forcing_dir)

        # Save drainage database and forcing files
        # (forcing_dir already created by base class create_directories())
        exp.save(self.forcing_dir)
