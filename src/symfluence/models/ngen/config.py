"""
NGEN Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the NOAA Next Generation (NextGen) Water Resources Modeling Framework.

This module registers NGEN-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import NGENConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# NGEN Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('NGEN')
class NGENDefaults:
    """Default configuration values for NGEN model."""

    # Core settings
    NGEN_INSTALL_PATH = 'default'
    NGEN_EXE = 'ngen'

    # Calibration settings
    NGEN_MODULES_TO_CALIBRATE = 'CFE'
    NGEN_CFE_PARAMS_TO_CALIBRATE = 'maxsmc,satdk,bb,slop'
    NGEN_NOAH_PARAMS_TO_CALIBRATE = 'refkdt,slope,smcmax,dksat'
    NGEN_PET_PARAMS_TO_CALIBRATE = 'wind_speed_measurement_height_m'

    # Active catchment
    NGEN_ACTIVE_CATCHMENT_ID = None


# ============================================================================
# NGEN Field Transformers (Flat to Nested Mapping)
# ============================================================================

NGEN_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'NGEN_INSTALL_PATH': ('model', 'ngen', 'install_path'),
    'NGEN_EXE': ('model', 'ngen', 'exe'),
    'NGEN_MODULES_TO_CALIBRATE': ('model', 'ngen', 'modules_to_calibrate'),
    'NGEN_CFE_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'cfe_params_to_calibrate'),
    'NGEN_NOAH_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'noah_params_to_calibrate'),
    'NGEN_PET_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'pet_params_to_calibrate'),
    'NGEN_ACTIVE_CATCHMENT_ID': ('model', 'ngen', 'active_catchment_id'),
}


# ============================================================================
# NGEN Config Adapter
# ============================================================================

class NGENConfigAdapter(ModelConfigAdapter):
    """
    Configuration adapter for NGEN model.

    Provides schema, defaults, transformers, and validation for NGEN-specific
    configuration. Registered with ModelRegistry to enable core config system
    to be model-agnostic.

    Example:
        >>> from symfluence.models.registry import ModelRegistry
        >>> adapter = ModelRegistry.get_config_adapter('NGEN')
        >>> defaults = adapter.get_defaults()
        >>> schema = adapter.get_config_schema()
    """

    def __init__(self, model_name: str = 'NGEN'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for NGEN configuration."""
        return NGENConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for NGEN."""
        return {
            k: v for k, v in vars(NGENDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for NGEN."""
        return NGEN_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate NGEN-specific configuration.

        Args:
            config: Configuration dictionary (flat format with uppercase keys)

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Check required fields
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"NGEN configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

        # Validate modules to calibrate
        modules = config.get('NGEN_MODULES_TO_CALIBRATE', 'CFE')
        if modules:
            valid_modules = ['CFE', 'NOAH', 'PET', 'TOPMODEL', 'LASAM']
            module_list = [m.strip().upper() for m in modules.split(',')]
            invalid_modules = [m for m in module_list if m not in valid_modules]
            if invalid_modules:
                raise ConfigValidationError(
                    f"Invalid NGEN modules in NGEN_MODULES_TO_CALIBRATE: {', '.join(invalid_modules)}. "
                    f"Valid modules: {', '.join(valid_modules)}"
                )

            # Check that appropriate params are specified for selected modules
            for module in module_list:
                param_key = f'NGEN_{module}_PARAMS_TO_CALIBRATE'
                params = config.get(param_key)
                if not params or params == '':
                    raise ConfigValidationError(
                        f"Module '{module}' selected for calibration but {param_key} is not specified"
                    )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for NGEN."""
        return [
            'NGEN_EXE',
            'NGEN_INSTALL_PATH',
        ]

    def get_conditional_requirements(self) -> Dict[str, Dict[str, list]]:
        """Get conditional requirements based on other config values."""
        return {
            'NGEN_MODULES_TO_CALIBRATE': {
                # If a module is selected, its params must be specified
                # This is a simplification - actual validation is in validate()
            },
        }
