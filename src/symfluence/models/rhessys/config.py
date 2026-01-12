"""
Configuration schema for the RHESSys model.
"""
from typing import Dict, Any, Tuple
from symfluence.models.config.model_config_schema import (
    ModelConfigSchema,
    InstallationConfig,
    ExecutionConfig,
    InputConfig,
    OutputConfig,
    ConfigKey,
    ConfigKeyType,
)
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError


def create_rhessys_schema() -> ModelConfigSchema:
    """Create configuration schema for the RHESSys model."""
    return ModelConfigSchema(
        model_name='RHESSys',
        description='Regional Hydro-Ecologic Simulation System',
        installation=InstallationConfig(
            install_path_key='RHESSYS_INSTALL_PATH',
            default_install_subpath='installs/rhessys/bin',
            exe_name_key='RHESSYS_EXE',
            default_exe_name='rhessys'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=True,
            default_timeout=14400,
            default_memory='8G'
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_RHESSYS_PATH',
            default_forcing_subpath='forcing/RHESSYS_input',
            forcing_file_pattern='{domain}_forcing.nc',
            required_variables=['time', 'pr', 't_mean', 't_max', 't_min']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_RHESSYS',
            default_output_subpath='simulations/{experiment_id}/RHESSys',
            output_file_pattern='{experiment_id}_basin.daily',
            primary_output_var='streamflow'
        ),
        config_keys=[
            ConfigKey('SETTINGS_RHESSYS_PATH', ConfigKeyType.PATH, True,
                      description='Path to RHESSys settings (templates, worldfiles)'),
            ConfigKey('RHESSYS_WORLD_TEMPLATE', ConfigKeyType.STRING, True,
                      default='world.template',
                      description='Template for the RHESSys worldfile'),
            ConfigKey('RHESSYS_FLOW_TEMPLATE', ConfigKeyType.STRING, True,
                      default='flow.template',
                      description='Template for the RHESSys flow table'),
            ConfigKey('RHESSYS_SKIP_CALIBRATION', ConfigKeyType.BOOLEAN, False,
                      default=True, description='Skip calibration for RHESSys'),
            # WMFire Integration (Wildfire spread module)
            ConfigKey('RHESSYS_USE_WMFIRE', ConfigKeyType.BOOLEAN, False,
                      default=False, description='Enable WMFire fire spread support'),
            ConfigKey('WMFIRE_INSTALL_PATH', ConfigKeyType.PATH, False,
                      default='installs/wmfire/lib',
                      description='Path to WMFire library installation'),
            ConfigKey('WMFIRE_LIB', ConfigKeyType.STRING, False,
                      default='libwmfire.so',
                      description='WMFire shared library name'),
            # Legacy VMFire aliases (for backwards compatibility)
            ConfigKey('RHESSYS_USE_VMFIRE', ConfigKeyType.BOOLEAN, False,
                      default=False, description='Alias for RHESSYS_USE_WMFIRE'),
            ConfigKey('VMFIRE_INSTALL_PATH', ConfigKeyType.PATH, False,
                      default='installs/wmfire/lib',
                      description='Alias for WMFIRE_INSTALL_PATH'),
        ]
    )


# Field Transformers (Flat to Nested Mapping)
RHESSYS_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'RHESSYS_INSTALL_PATH': ('model', 'rhessys', 'install_path'),
    'RHESSYS_EXE': ('model', 'rhessys', 'exe'),
    'SETTINGS_RHESSYS_PATH': ('model', 'rhessys', 'settings_path'),
    'RHESSYS_WORLD_TEMPLATE': ('model', 'rhessys', 'world_template'),
    'RHESSYS_FLOW_TEMPLATE': ('model', 'rhessys', 'flow_template'),
    'RHESSYS_SKIP_CALIBRATION': ('model', 'rhessys', 'skip_calibration'),
    'RHESSYS_USE_WMFIRE': ('model', 'rhessys', 'use_wmfire'),
    'WMFIRE_INSTALL_PATH': ('model', 'rhessys', 'wmfire_install_path'),
    'WMFIRE_LIB': ('model', 'rhessys', 'wmfire_lib'),
    'RHESSYS_USE_VMFIRE': ('model', 'rhessys', 'use_vmfire'),
    'VMFIRE_INSTALL_PATH': ('model', 'rhessys', 'vmfire_install_path'),
}


class RHESSysConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for RHESSys model."""

    def __init__(self, model_name: str = 'RHESSYS'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for RHESSys configuration."""
        # RHESSys doesn't have a Pydantic schema yet
        return None

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for RHESSys."""
        from .defaults import RHESSysDefaults
        return {
            k: v for k, v in vars(RHESSysDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for RHESSys."""
        return RHESSYS_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate RHESSys-specific configuration."""
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"RHESSys configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for RHESSys."""
        return [
            'RHESSYS_EXE',
            'SETTINGS_RHESSYS_PATH',
        ]
