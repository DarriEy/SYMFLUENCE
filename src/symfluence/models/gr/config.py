"""
GR Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the GR4J/GR5J/GR6J family of hydrological models.

This module registers GR-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import GRConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# GR Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('GR')
class GRDefaults:
    """Default configuration values for GR models."""

    # Core settings
    GR_MODEL_TYPE = 'GR4J'
    GR_SPATIAL_MODE = 'lumped'
    GR_INSTALL_PATH = 'default'
    GR_EXE = 'GR.r'
    SETTINGS_GR_PATH = 'default'
    SETTINGS_GR_CONTROL = 'default'

    # Routing
    GR_ROUTING_INTEGRATION = 'none'

    # Calibration parameters
    GR_PARAMS_TO_CALIBRATE = 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff'


# ============================================================================
# GR Field Transformers (Flat to Nested Mapping)
# ============================================================================

GR_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'GR_INSTALL_PATH': ('model', 'gr', 'install_path'),
    'GR_EXE': ('model', 'gr', 'exe'),
    'GR_SPATIAL_MODE': ('model', 'gr', 'spatial_mode'),
    'GR_ROUTING_INTEGRATION': ('model', 'gr', 'routing_integration'),
    'SETTINGS_GR_PATH': ('model', 'gr', 'settings_path'),
    'SETTINGS_GR_CONTROL': ('model', 'gr', 'control'),
    'GR_PARAMS_TO_CALIBRATE': ('model', 'gr', 'params_to_calibrate'),
}


# ============================================================================
# GR Config Adapter
# ============================================================================

class GRConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for GR models."""

    def __init__(self, model_name: str = 'GR'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for GR configuration."""
        return GRConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for GR."""
        return {
            k: v for k, v in vars(GRDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for GR."""
        return GR_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate GR-specific configuration."""
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"GR configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

        # Validate GR model type
        model_type = config.get('GR_MODEL_TYPE', 'GR4J')
        valid_types = ['GR4J', 'GR5J', 'GR6J']
        if model_type not in valid_types:
            raise ConfigValidationError(
                f"Invalid GR_MODEL_TYPE '{model_type}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )

        # Validate spatial mode
        spatial_mode = config.get('GR_SPATIAL_MODE', 'lumped')
        valid_modes = ['lumped', 'semi_distributed', 'distributed']
        if spatial_mode not in valid_modes:
            raise ConfigValidationError(
                f"Invalid GR_SPATIAL_MODE '{spatial_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for GR."""
        return [
            'GR_EXE',
            'SETTINGS_GR_PATH',
        ]
