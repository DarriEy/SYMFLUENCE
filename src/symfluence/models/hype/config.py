"""
HYPE Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the Hydrological Predictions for the Environment (HYPE) model.

This module registers HYPE-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import HYPEConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# HYPE Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('HYPE')
class HYPEDefaults:
    """Default configuration values for HYPE model."""

    # Core settings
    SETTINGS_HYPE_PATH = 'default'
    HYPE_INSTALL_PATH = 'default'
    HYPE_EXE = 'hype'
    SETTINGS_HYPE_CONTROL_FILE = 'info.txt'

    # Calibration parameters
    HYPE_PARAMS_TO_CALIBRATE = 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp'

    # Spinup period
    HYPE_SPINUP_DAYS = 365


# ============================================================================
# HYPE Field Transformers (Flat to Nested Mapping)
# ============================================================================

HYPE_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'HYPE_INSTALL_PATH': ('model', 'hype', 'install_path'),
    'HYPE_EXE': ('model', 'hype', 'exe'),
    'SETTINGS_HYPE_PATH': ('model', 'hype', 'settings_path'),
    'SETTINGS_HYPE_INFO': ('model', 'hype', 'info_file'),
    'HYPE_PARAMS_TO_CALIBRATE': ('model', 'hype', 'params_to_calibrate'),
    'HYPE_SPINUP_DAYS': ('model', 'hype', 'spinup_days'),
}


# ============================================================================
# HYPE Config Adapter
# ============================================================================

class HYPEConfigAdapter(ModelConfigAdapter):
    """
    Configuration adapter for HYPE model.

    Provides schema, defaults, transformers, and validation for HYPE-specific
    configuration. Registered with ModelRegistry to enable core config system
    to be model-agnostic.
    """

    def __init__(self, model_name: str = 'HYPE'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for HYPE configuration."""
        return HYPEConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for HYPE."""
        return {
            k: v for k, v in vars(HYPEDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for HYPE."""
        return HYPE_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate HYPE-specific configuration."""
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"HYPE configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for HYPE."""
        return [
            'SETTINGS_HYPE_PATH',
        ]
