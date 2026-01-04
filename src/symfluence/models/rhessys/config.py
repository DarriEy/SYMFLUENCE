"""
Configuration schema for the RHESSys model.
"""
from symfluence.models.config.model_config_schema import (
    ModelConfigSchema,
    InstallationConfig,
    ExecutionConfig,
    InputConfig,
    OutputConfig,
    ConfigKey,
    ConfigKeyType,
)


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
            # VMFire Integration
            ConfigKey('RHESSYS_USE_VMFIRE', ConfigKeyType.BOOLEAN, False,
                      default=False, description='Enable VMFire pre-processing'),
            ConfigKey('VMFIRE_INSTALL_PATH', ConfigKeyType.PATH, False,
                      default='installs/vmfire/bin',
                      description='Path to VMFire installation'),
            ConfigKey('VMFIRE_EXE', ConfigKeyType.STRING, False,
                      default='vmfire',
                      description='VMFire executable name'),
        ]
    )
