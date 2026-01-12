"""
FUSE Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the Framework for Understanding Structural Errors (FUSE).

This module registers FUSE-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import FUSEConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# FUSE Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('FUSE')
class FUSEDefaults:
    """Default configuration values for FUSE model."""

    # Core settings
    FUSE_SPATIAL_MODE = 'lumped'
    ROUTING_MODEL = 'none'
    FUSE_INSTALL_PATH = 'default'
    SETTINGS_FUSE_PATH = 'default'
    SETTINGS_FUSE_FILEMANAGER = 'fm_catch.txt'
    FUSE_EXE = 'fuse.exe'
    EXPERIMENT_OUTPUT_FUSE = 'default'

    # Calibration parameters
    SETTINGS_FUSE_PARAMS_TO_CALIBRATE = 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE'

    # Routing integration
    FUSE_ROUTING_INTEGRATION = 'default'

    # Subcatchment dimension
    FUSE_SUBCATCHMENT_DIM = 'longitude'

    # File ID (optional)
    FUSE_FILE_ID = None

    # Elevation bands
    FUSE_N_ELEVATION_BANDS = 1

    # Execution timeout
    FUSE_TIMEOUT = 3600  # seconds


# ============================================================================
# FUSE Field Transformers (Flat to Nested Mapping)
# ============================================================================

FUSE_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'FUSE_INSTALL_PATH': ('model', 'fuse', 'install_path'),
    'FUSE_EXE': ('model', 'fuse', 'exe'),
    'FUSE_ROUTING_INTEGRATION': ('model', 'fuse', 'routing_integration'),
    'SETTINGS_FUSE_PATH': ('model', 'fuse', 'settings_path'),
    'SETTINGS_FUSE_FILEMANAGER': ('model', 'fuse', 'filemanager'),
    'FUSE_SPATIAL_MODE': ('model', 'fuse', 'spatial_mode'),
    'FUSE_SUBCATCHMENT_DIM': ('model', 'fuse', 'subcatchment_dim'),
    'EXPERIMENT_OUTPUT_FUSE': ('model', 'fuse', 'experiment_output'),
    'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': ('model', 'fuse', 'params_to_calibrate'),
    'FUSE_DECISION_OPTIONS': ('model', 'fuse', 'decision_options'),
    'FUSE_FILE_ID': ('model', 'fuse', 'file_id'),
    'FUSE_N_ELEVATION_BANDS': ('model', 'fuse', 'n_elevation_bands'),
    'FUSE_TIMEOUT': ('model', 'fuse', 'timeout'),
}


# ============================================================================
# FUSE Config Adapter
# ============================================================================

class FUSEConfigAdapter(ModelConfigAdapter):
    """
    Configuration adapter for FUSE model.

    Provides schema, defaults, transformers, and validation for FUSE-specific
    configuration. Registered with ModelRegistry to enable core config system
    to be model-agnostic.

    Example:
        >>> from symfluence.models.registry import ModelRegistry
        >>> adapter = ModelRegistry.get_config_adapter('FUSE')
        >>> defaults = adapter.get_defaults()
        >>> schema = adapter.get_config_schema()
    """

    def __init__(self, model_name: str = 'FUSE'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for FUSE configuration."""
        return FUSEConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for FUSE."""
        return {
            k: v for k, v in vars(FUSEDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for FUSE."""
        return FUSE_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate FUSE-specific configuration.

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
                f"FUSE configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

        # Validate spatial mode
        spatial_mode = config.get('FUSE_SPATIAL_MODE', 'lumped')
        valid_modes = ['lumped', 'semi_distributed', 'distributed']
        if spatial_mode not in valid_modes:
            raise ConfigValidationError(
                f"Invalid FUSE_SPATIAL_MODE '{spatial_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

        # Validate routing requirements for distributed mode
        if spatial_mode in ['semi_distributed', 'distributed']:
            routing_model = config.get('ROUTING_MODEL', 'none')
            if routing_model == 'none':
                raise ConfigValidationError(
                    f"FUSE_SPATIAL_MODE='{spatial_mode}' requires ROUTING_MODEL "
                    f"to be set (e.g., 'mizuRoute')"
                )

        # Validate mizuRoute requirements if routing is enabled
        routing_model = config.get('ROUTING_MODEL', '').upper()
        if routing_model == 'MIZUROUTE':
            mizu_required = ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE']
            mizu_missing = [
                f for f in mizu_required
                if not config.get(f) or config.get(f) in ['', 'None']
            ]
            if mizu_missing:
                raise ConfigValidationError(
                    f"mizuRoute routing enabled but configuration incomplete:\n"
                    + "\n".join(f"  • {field}" for field in mizu_missing)
                )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for FUSE."""
        return [
            'FUSE_EXE',
            'SETTINGS_FUSE_PATH',
        ]

    def get_conditional_requirements(self) -> Dict[str, Dict[str, list]]:
        """Get conditional requirements based on other config values."""
        return {
            'ROUTING_MODEL': {
                'mizuRoute': ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE'],
                'MIZUROUTE': ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE'],
                'none': [],
            },
            'FUSE_SPATIAL_MODE': {
                'distributed': ['ROUTING_MODEL'],
                'semi_distributed': ['ROUTING_MODEL'],
                'lumped': [],
            },
        }
