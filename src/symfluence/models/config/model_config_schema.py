# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-model ConfigKey schema definitions (scaffolding metadata).

The machinery (ConfigKey, ModelConfigSchema, the registry and lookup
functions) lives in ``symfluence.core.modeling.config_schema``; this module
holds the in-tree models' schema definitions and registers them on import —
the same path an external model package uses (``register_model_schema``).
"""
from __future__ import annotations

from symfluence.core.modeling.config_schema import (  # noqa: F401 — re-exported for compat
    REGISTERED_SCHEMAS,
    ConfigKey,
    ConfigKeyType,
    ExecutionConfig,
    InputConfig,
    InstallationConfig,
    ModelConfigSchema,
    OutputConfig,
    get_model_schema,
    register_model_schema,
    validate_model_config,
)


def _create_summa_schema() -> ModelConfigSchema:
    """Create configuration schema for SUMMA model."""
    return ModelConfigSchema(
        model_name='SUMMA',
        description='Structure for Unifying Multiple Modeling Alternatives',
        installation=InstallationConfig(
            install_path_key='SUMMA_INSTALL_PATH',
            default_install_subpath='installs/summa/bin',
            exe_name_key='SUMMA_EXE',
            default_exe_name='summa_sundials.exe'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=True,
            parallel_key='SETTINGS_SUMMA_USE_PARALLEL_SUMMA',
            default_timeout=14400,  # 4 hours
            default_memory='4G'
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_SUMMA_PATH',
            default_forcing_subpath='forcing/SUMMA_input',
            forcing_file_pattern='{domain}_forcing.nc',
            required_variables=['time', 'precipitation_flux', 'air_temperature', 'specific_humidity', 'wind_speed', 'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux', 'surface_air_pressure']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_SUMMA',
            default_output_subpath='simulations/{experiment_id}/SUMMA',
            output_file_pattern='{experiment_id}_timestep.nc',
            primary_output_var='averageRoutedRunoff'
        ),
        spatial_mode_key='DOMAIN_DEFINITION_METHOD',
        routing_key='ROUTING_DELINEATION',
        config_keys=[
            ConfigKey('SETTINGS_SUMMA_PATH', ConfigKeyType.PATH, True,
                      description='Path to SUMMA settings directory'),
            ConfigKey('SETTINGS_SUMMA_FILEMANAGER', ConfigKeyType.STRING, True,
                      default='fileManager.txt',
                      description='Name of SUMMA file manager'),
            ConfigKey('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', ConfigKeyType.BOOLEAN, False,
                      default=False,
                      description='Enable parallel SUMMA execution with specified backend'),
            ConfigKey('SETTINGS_SUMMA_PARALLEL_BACKEND', ConfigKeyType.ENUM, False,
                      default='slurm', valid_values=['slurm', 'local'],
                      description='SUMMA parallel backend'),
            ConfigKey('EXPERIMENT_LOG_SUMMA', ConfigKeyType.PATH, False,
                      description='Path for SUMMA log files'),
            ConfigKey('EXPERIMENT_BACKUP_SETTINGS', ConfigKeyType.ENUM, False,
                      default='no', valid_values=['yes', 'no'],
                      description='Backup settings to output directory'),
            ConfigKey('MONITOR_SLURM_JOB', ConfigKeyType.BOOLEAN, False,
                      default=True,
                      description='Monitor SLURM job until completion'),
        ]
    )


def _create_fuse_schema() -> ModelConfigSchema:
    """Create configuration schema for FUSE model."""
    return ModelConfigSchema(
        model_name='FUSE',
        description='Framework for Understanding Structural Errors',
        installation=InstallationConfig(
            install_path_key='FUSE_INSTALL_PATH',
            default_install_subpath='installs/fuse/bin',
            exe_name_key='FUSE_EXE',
            default_exe_name='fuse.exe'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_FUSE_PATH',
            default_forcing_subpath='forcing/FUSE_input',
            forcing_file_pattern='{domain}_input.nc',
            required_variables=['time', 'pr', 'temp', 'pet']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_FUSE',
            default_output_subpath='simulations/{experiment_id}/FUSE',
            output_file_pattern='{domain}_{experiment_id}_runs_def.nc',
            primary_output_var='q_routed'
        ),
        spatial_mode_key='FUSE_SPATIAL_MODE',
        routing_key='FUSE_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_FUSE_FILEMANAGER', ConfigKeyType.STRING, False,
                      default='fm_catch.txt',
                      description='Name of FUSE file manager'),
            ConfigKey('FUSE_SPATIAL_MODE', ConfigKeyType.ENUM, False,
                      default='lumped',
                      valid_values=['lumped', 'semi_distributed', 'distributed'],
                      description='Spatial discretization mode'),
            ConfigKey('FUSE_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
            ConfigKey('FUSE_FILE_ID', ConfigKeyType.STRING, False,
                      description='File identifier for FUSE outputs'),
        ]
    )


def _create_gr_schema() -> ModelConfigSchema:
    """Create configuration schema for GR model."""
    return ModelConfigSchema(
        model_name='GR',
        description='GR4J/GR6J Rainfall-Runoff Model',
        installation=InstallationConfig(
            install_path_key='GR_INSTALL_PATH',
            default_install_subpath='installs/airGR',
            exe_name_key=None,
            default_exe_name=None  # R-based, no executable
        ),
        execution=ExecutionConfig(
            method='subprocess',  # Runs via Rscript
            supports_parallel=False,
            default_timeout=1800
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_GR_PATH',
            default_forcing_subpath='forcing/GR_input',
            forcing_file_pattern='{domain}_input.nc',
            required_variables=['time', 'pr', 'pet']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_GR',
            default_output_subpath='simulations/{experiment_id}/GR',
            output_file_pattern='{experiment_id}_output.nc',
            primary_output_var='Qsim'
        ),
        spatial_mode_key='GR_SPATIAL_MODE',
        routing_key='GR_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('GR_MODEL_TYPE', ConfigKeyType.ENUM, False,
                      default='GR4J',
                      valid_values=['GR4J', 'GR5J', 'GR6J'],
                      description='GR model variant'),
            ConfigKey('GR_SPATIAL_MODE', ConfigKeyType.ENUM, False,
                      default='lumped',
                      valid_values=['lumped', 'semi_distributed', 'distributed'],
                      description='Spatial discretization mode'),
            ConfigKey('GR_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
        ]
    )


def _create_ngen_schema() -> ModelConfigSchema:
    """Create configuration schema for NextGen model."""
    return ModelConfigSchema(
        model_name='NGEN',
        description='NextGen Water Resources Modeling Framework',
        installation=InstallationConfig(
            install_path_key='NGEN_INSTALL_PATH',
            default_install_subpath='installs/ngen',
            exe_name_key='NGEN_EXE',
            default_exe_name='ngen'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=True,
            default_timeout=7200,
            default_memory='8G'
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_NGEN_PATH',
            default_forcing_subpath='forcing/NGEN_input',
            forcing_file_pattern='{domain}_forcing.csv',
            required_variables=['time', 'APCP_surface', 'TMP_2maboveground']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_NGEN',
            default_output_subpath='simulations/{experiment_id}/NGEN',
            output_file_pattern='nex-*_output.csv',
            primary_output_var='q_out'
        ),
        spatial_mode_key='NGEN_SPATIAL_MODE',
        config_keys=[
            ConfigKey('NGEN_REALIZATION_FILE', ConfigKeyType.STRING, True,
                      description='Path to realization configuration'),
            ConfigKey('NGEN_CATCHMENT_FILE', ConfigKeyType.STRING, True,
                      description='Path to catchment GeoJSON'),
            ConfigKey('NGEN_NEXUS_FILE', ConfigKeyType.STRING, True,
                      description='Path to nexus GeoJSON'),
        ]
    )


def _create_hype_schema() -> ModelConfigSchema:
    """Create configuration schema for HYPE model."""
    return ModelConfigSchema(
        model_name='HYPE',
        description='Hydrological Predictions for the Environment',
        installation=InstallationConfig(
            install_path_key='HYPE_INSTALL_PATH',
            default_install_subpath='installs/hype',
            exe_name_key='HYPE_EXE',
            default_exe_name='hype'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_HYPE_PATH',
            default_forcing_subpath='forcing/HYPE_input',
            forcing_file_pattern='Pobs.txt',
            required_variables=['DATE', 'precip', 'temp']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_HYPE',
            default_output_subpath='simulations/{experiment_id}/HYPE',
            output_file_pattern='timeOUT.txt',
            primary_output_var='cout'
        ),
        config_keys=[
            ConfigKey('SETTINGS_HYPE_PATH', ConfigKeyType.PATH, True,
                      description='Path to HYPE settings directory'),
            ConfigKey('HYPE_INFO_FILE', ConfigKeyType.STRING, False,
                      default='info.txt',
                      description='HYPE info configuration file'),
        ]
    )


def _create_mesh_schema() -> ModelConfigSchema:
    """Create configuration schema for MESH model."""
    return ModelConfigSchema(
        model_name='MESH',
        description='Modélisation Environmentale Surface et Hydrologie',
        installation=InstallationConfig(
            install_path_key='MESH_INSTALL_PATH',
            default_install_subpath='installs/mesh/bin',
            exe_name_key='MESH_EXE',
            default_exe_name='mesh.exe'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=True,
            default_timeout=7200,
            default_memory='8G'
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_MESH_PATH',
            default_forcing_subpath='forcing/MESH_input',
            forcing_file_pattern='basin_forcing.nc',
            required_variables=['time', 'RDRS_v2.1_A_PR0_SFC', 'RDRS_v2.1_P_TT_1.5m']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_MESH',
            default_output_subpath='simulations/{experiment_id}/MESH',
            output_file_pattern='Basin_average_water_balance.csv',
            primary_output_var='QOMEAS'
        ),
        config_keys=[
            ConfigKey('SETTINGS_MESH_PATH', ConfigKeyType.PATH, True,
                      description='Path to MESH settings directory'),
            ConfigKey('MESH_DRAINAGE_DB', ConfigKeyType.STRING, False,
                      default='MESH_drainage_database.nc',
                      description='MESH drainage database file'),
        ]
    )


def _create_gnn_schema() -> ModelConfigSchema:
    """Create configuration schema for GNN model."""
    return ModelConfigSchema(
        model_name='GNN',
        description='Spatio-Temporal Graph Neural Network for Hydrology',
        installation=InstallationConfig(
            install_path_key='GNN_INSTALL_PATH', # Not really used, but required by schema
            default_install_subpath='models',
            exe_name_key=None,
            default_exe_name=None
        ),
        execution=ExecutionConfig(
            method='python',
            supports_parallel=True, # GPU support
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_GNN_PATH',
            default_forcing_subpath='forcing/basin_averaged_data',
            forcing_file_pattern='*.nc',
            required_variables=['time', 'precipitation_flux', 'air_temperature']
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_GNN',
            default_output_subpath='simulations/{experiment_id}/GNN',
            output_file_pattern='gnn_output.csv',
            primary_output_var='streamflow'
        ),
        config_keys=[
            ConfigKey('GNN_HIDDEN_SIZE', ConfigKeyType.INTEGER, False, default=64),
            ConfigKey('GNN_OUTPUT_SIZE', ConfigKeyType.INTEGER, False, default=32),
            ConfigKey('GNN_EPOCHS', ConfigKeyType.INTEGER, False, default=100),
            ConfigKey('GNN_BATCH_SIZE', ConfigKeyType.INTEGER, False, default=16),
            ConfigKey('GNN_LEARNING_RATE', ConfigKeyType.FLOAT, False, default=0.005),
            ConfigKey('GNN_DROPOUT', ConfigKeyType.FLOAT, False, default=0.2),
            ConfigKey('GNN_USE_SNOW', ConfigKeyType.BOOLEAN, False, default=False),
            ConfigKey('GNN_LOAD', ConfigKeyType.BOOLEAN, False, default=False),
        ]
    )


from symfluence.models.rhessys.config import create_rhessys_schema  # noqa: E402


def _register_schemas():
    """Register the in-tree model schemas with the core machinery."""
    for name, schema in {
        'SUMMA': _create_summa_schema(),
        'FUSE': _create_fuse_schema(),
        'GR': _create_gr_schema(),
        'NGEN': _create_ngen_schema(),
        'HYPE': _create_hype_schema(),
        'MESH': _create_mesh_schema(),
        'RHESSYS': create_rhessys_schema(),
        'GNN': _create_gnn_schema(),
    }.items():
        register_model_schema(name, schema)


# Initialize on module load
_register_schemas()
