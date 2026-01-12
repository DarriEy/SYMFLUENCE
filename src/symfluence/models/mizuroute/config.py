"""
mizuRoute Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the mizuRoute routing model.

This module registers mizuRoute-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import MizuRouteConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# mizuRoute Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('MIZUROUTE')
class MizuRouteDefaults:
    """Default configuration values for mizuRoute model."""

    # Core settings
    INSTALL_PATH_MIZUROUTE = 'default'
    EXE_NAME_MIZUROUTE = 'mizuroute.exe'
    SETTINGS_MIZU_PATH = 'default'
    EXPERIMENT_OUTPUT_MIZUROUTE = 'default'
    EXPERIMENT_LOG_MIZUROUTE = 'default'

    # Calibration
    MIZUROUTE_PARAMS_TO_CALIBRATE = 'velo,diff'
    CALIBRATE_MIZUROUTE = False

    # Timeout
    MIZUROUTE_TIMEOUT = 3600


# ============================================================================
# mizuRoute Field Transformers (Flat to Nested Mapping)
# ============================================================================

MIZUROUTE_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'INSTALL_PATH_MIZUROUTE': ('model', 'mizuroute', 'install_path'),
    'EXE_NAME_MIZUROUTE': ('model', 'mizuroute', 'exe'),
    'SETTINGS_MIZU_PATH': ('model', 'mizuroute', 'settings_path'),
    'SETTINGS_MIZU_WITHIN_BASIN': ('model', 'mizuroute', 'within_basin'),
    'SETTINGS_MIZU_ROUTING_DT': ('model', 'mizuroute', 'routing_dt'),
    'SETTINGS_MIZU_ROUTING_UNITS': ('model', 'mizuroute', 'routing_units'),
    'SETTINGS_MIZU_ROUTING_VAR': ('model', 'mizuroute', 'routing_var'),
    'SETTINGS_MIZU_OUTPUT_FREQ': ('model', 'mizuroute', 'output_freq'),
    'SETTINGS_MIZU_OUTPUT_VARS': ('model', 'mizuroute', 'output_vars'),
    'SETTINGS_MIZU_MAKE_OUTLET': ('model', 'mizuroute', 'make_outlet'),
    'SETTINGS_MIZU_NEEDS_REMAP': ('model', 'mizuroute', 'needs_remap'),
    'SETTINGS_MIZU_TOPOLOGY': ('model', 'mizuroute', 'topology'),
    'SETTINGS_MIZU_PARAMETERS': ('model', 'mizuroute', 'parameters'),
    'SETTINGS_MIZU_CONTROL_FILE': ('model', 'mizuroute', 'control_file'),
    'SETTINGS_MIZU_REMAP': ('model', 'mizuroute', 'remap'),
    'MIZU_FROM_MODEL': ('model', 'mizuroute', 'from_model'),
    'EXPERIMENT_LOG_MIZUROUTE': ('model', 'mizuroute', 'experiment_log'),
    'EXPERIMENT_OUTPUT_MIZUROUTE': ('model', 'mizuroute', 'experiment_output'),
    'SETTINGS_MIZU_OUTPUT_VAR': ('model', 'mizuroute', 'output_var'),
    'SETTINGS_MIZU_PARAMETER_FILE': ('model', 'mizuroute', 'parameter_file'),
    'SETTINGS_MIZU_REMAP_FILE': ('model', 'mizuroute', 'remap_file'),
    'SETTINGS_MIZU_TOPOLOGY_FILE': ('model', 'mizuroute', 'topology_file'),
    'MIZUROUTE_PARAMS_TO_CALIBRATE': ('model', 'mizuroute', 'params_to_calibrate'),
    'CALIBRATE_MIZUROUTE': ('model', 'mizuroute', 'calibrate'),
    'MIZUROUTE_TIMEOUT': ('model', 'mizuroute', 'timeout'),
}


# ============================================================================
# mizuRoute Config Adapter
# ============================================================================

class MizuRouteConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for mizuRoute model."""

    def __init__(self, model_name: str = 'MIZUROUTE'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for mizuRoute configuration."""
        return MizuRouteConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for mizuRoute."""
        return {
            k: v for k, v in vars(MizuRouteDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for mizuRoute."""
        return MIZUROUTE_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate mizuRoute-specific configuration."""
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"mizuRoute configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for mizuRoute."""
        return [
            'EXE_NAME_MIZUROUTE',
            'INSTALL_PATH_MIZUROUTE',
        ]
