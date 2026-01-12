"""
MESH Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the Modélisation Environmentale Communautaire - Surface and Hydrology (MESH) model.

This module registers MESH-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import MESHConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# MESH Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('MESH')
class MESHDefaults:
    """Default configuration values for MESH model."""

    # Core settings
    MESH_INSTALL_PATH = 'default'
    MESH_EXE = 'sa_mesh'
    MESH_SPATIAL_MODE = 'distributed'
    SETTINGS_MESH_PATH = 'default'
    EXPERIMENT_OUTPUT_MESH = 'default'

    # Calibration parameters
    MESH_PARAMS_TO_CALIBRATE = 'ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR'

    # Spinup
    MESH_SPINUP_DAYS = 365

    # GRU/HRU configuration
    MESH_GRU_MIN_TOTAL = 1


# ============================================================================
# MESH Field Transformers (Flat to Nested Mapping)
# ============================================================================

MESH_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'MESH_INSTALL_PATH': ('model', 'mesh', 'install_path'),
    'MESH_EXE': ('model', 'mesh', 'exe'),
    'MESH_SPATIAL_MODE': ('model', 'mesh', 'spatial_mode'),
    'SETTINGS_MESH_PATH': ('model', 'mesh', 'settings_path'),
    'EXPERIMENT_OUTPUT_MESH': ('model', 'mesh', 'experiment_output'),
    'MESH_FORCING_PATH': ('model', 'mesh', 'forcing_path'),
    'MESH_FORCING_VARS': ('model', 'mesh', 'forcing_vars'),
    'MESH_FORCING_UNITS': ('model', 'mesh', 'forcing_units'),
    'MESH_FORCING_TO_UNITS': ('model', 'mesh', 'forcing_to_units'),
    'MESH_LANDCOVER_STATS_PATH': ('model', 'mesh', 'landcover_stats_path'),
    'MESH_LANDCOVER_STATS_DIR': ('model', 'mesh', 'landcover_stats_dir'),
    'MESH_LANDCOVER_STATS_FILE': ('model', 'mesh', 'landcover_stats_file'),
    'MESH_MAIN_ID': ('model', 'mesh', 'main_id'),
    'MESH_DS_MAIN_ID': ('model', 'mesh', 'ds_main_id'),
    'MESH_LANDCOVER_CLASSES': ('model', 'mesh', 'landcover_classes'),
    'MESH_DDB_VARS': ('model', 'mesh', 'ddb_vars'),
    'MESH_DDB_UNITS': ('model', 'mesh', 'ddb_units'),
    'MESH_DDB_TO_UNITS': ('model', 'mesh', 'ddb_to_units'),
    'MESH_DDB_MIN_VALUES': ('model', 'mesh', 'ddb_min_values'),
    'MESH_GRU_DIM': ('model', 'mesh', 'gru_dim'),
    'MESH_HRU_DIM': ('model', 'mesh', 'hru_dim'),
    'MESH_OUTLET_VALUE': ('model', 'mesh', 'outlet_value'),
    'SETTINGS_MESH_INPUT': ('model', 'mesh', 'input_file'),
    'MESH_PARAMS_TO_CALIBRATE': ('model', 'mesh', 'params_to_calibrate'),
    'MESH_SPINUP_DAYS': ('model', 'mesh', 'spinup_days'),
    'MESH_GRU_MIN_TOTAL': ('model', 'mesh', 'gru_min_total'),
}


# ============================================================================
# MESH Config Adapter
# ============================================================================

class MESHConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for MESH model."""

    def __init__(self, model_name: str = 'MESH'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for MESH configuration."""
        return MESHConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for MESH."""
        return {
            k: v for k, v in vars(MESHDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for MESH."""
        return MESH_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate MESH-specific configuration."""
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"MESH configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for MESH."""
        return [
            'MESH_EXE',
            'SETTINGS_MESH_PATH',
        ]
