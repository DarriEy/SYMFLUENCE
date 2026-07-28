# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-model ConfigKey schema definitions (scaffolding metadata).

The machinery (ConfigKey, ModelConfigSchema, the registry and lookup
functions) lives in ``symfluence.core.modeling.config_schema``; this module
holds the in-tree models' schema definitions and registers them on import —
the same path an external model package uses (``register_model_schema``).

A schema is also where a model declares the metadata core used to hardcode
per model: ``spatial_mode_key`` and ``routing_integration_key`` (read by
``RoutingDecider``), ``runoff`` (read by ``runoff_loader``) and
``parallel_calibration`` (read by the parallel-calibration
``ConfigurationUpdater``). Registering a schema is the only step needed for
those to take effect.

``runoff`` and ``parallel_calibration`` describe *different files* for the same
model and must not be conflated -- see ``ParallelCalibrationConfig``'s docstring
and the per-model notes below.
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
    ParallelCalibrationConfig,
    RunoffConfig,
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
        runoff=RunoffConfig(
            output_dir_key='EXPERIMENT_OUTPUT_SUMMA',
            output_dir_name='SUMMA',
            default_var='averageRoutedRunoff',
            default_units='m/s',
            default_dt='3600',
            output_file_pattern='{experiment_id}_timestep.nc',
            hru_dim='hru',
            hru_var='hruId',
            comment_name='SUMMA',
        ),
        # Parallel calibration writes the same content under a different name:
        # ConfigurationUpdater sets ``outFilePrefix 'proc_{NN}_{experiment_id}'``
        # and SUMMA appends '_timestep.nc'.
        # runoff_var_from_config=False: SETTINGS_MIZU_ROUTING_VAR is not read on
        # SUMMA's branch (unlike FUSE/GR); the name is fixed by SUMMA's output.
        # KNOWN WRONG, preserved value-for-value from core's table and reported
        # rather than fixed here: hru_dim/hru_var say gru/gruId while ``runoff``
        # above says hru/hruId. The non-parallel writer starts from hru/hruId and
        # switches to gru/gruId only when topology detects n_hrus > n_grus
        # (``summa_uses_gru_runoff``), so for a 1-HRU-per-GRU domain the two
        # paths disagree.
        parallel_calibration=ParallelCalibrationConfig(
            fname_pattern='proc_{proc_id:02d}_{experiment_id}_timestep.nc',
            runoff_var='averageRoutedRunoff',
            runoff_var_from_config=False,
            dt_qsim=None,  # SETTINGS_MIZU_ROUTING_DT decides (default '3600')
            sim_start_time='00:00',
            sim_end_time='00:00',
            hru_dim='gru',
            hru_var='gruId',
        ),
        # SUMMA has no '<MODEL>_SPATIAL_MODE' key anywhere -- its spatial mode
        # *is* the domain definition. Declared explicitly so no convention can
        # derive 'SUMMA_SPATIAL_MODE' and quietly change the decision for the
        # most-used model.
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
            # Same file as the runoff declaration below. '{domain_name}' is the
            # token resolve_runoff_file() substitutes; the '{domain}' spelling
            # this carried was a drift that would have raised KeyError had this
            # declaration ever been used for formatting.
            output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
            primary_output_var='q_routed'
        ),
        runoff=RunoffConfig(
            output_dir_key='EXPERIMENT_OUTPUT_FUSE',
            output_dir_name='FUSE',
            default_var='q_routed',
            default_units='m/s',
            default_dt='86400',
            output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
            hru_dim='gru',
            hru_var='gruId',
            comment_name='FUSE',
        ),
        # Genuinely a DIFFERENT file from the ``runoff`` declaration above, not
        # a drifted copy of it. Verified: FUSE's calibration worker converts its
        # own output before mizuRoute runs --
        # ``FuseToMizurouteConverter.convert`` (models/fuse/utilities/
        # mizuroute_converter.py) reads '{domain_name}_{fuse_id}_runs_def.nc'
        # and writes a new file named, verbatim,
        # f"proc_{proc_id:02d}_{experiment_id}_timestep.nc" into the process
        # sim dir, which is the same directory ConfigurationUpdater points
        # <input_dir> at. The serial path (models/fuse/runner.py) instead
        # overwrites runs_def in place -- that is what ``runoff`` describes.
        #
        # The converted dataset is (time, gru) with a 'gruId' variable, and its
        # routing variable name comes from SETTINGS_MIZU_ROUTING_VAR with the
        # same 'q_routed' default and the same 'default'/empty sentinel
        # handling, so both sides read one key. dt_qsim is pinned to '86400'
        # because FUSE writes daily output regardless of forcing cadence.
        parallel_calibration=ParallelCalibrationConfig(
            fname_pattern='proc_{proc_id:02d}_{experiment_id}_timestep.nc',
            runoff_var='q_routed',
            runoff_var_from_config=True,
            dt_qsim='86400',
            sim_start_time='00:00',
            sim_end_time='00:00',
            hru_dim='gru',
            hru_var='gruId',
        ),
        # FUSE's typed default is the concrete 'lumped', not a deferral, so it
        # does not qualify for the automatic opt-in RoutingDecider applies to
        # 'auto'-defaulting models. Declared explicitly because the lumped veto
        # is live behaviour the repro campaign depends on: a lumped FUSE run
        # must keep suppressing the template-wide ROUTING_MODEL: mizuRoute.
        spatial_mode_key='FUSE_SPATIAL_MODE',
        routing_key='FUSE_ROUTING_INTEGRATION',
        routing_integration_key='FUSE_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_FUSE_FILEMANAGER', ConfigKeyType.STRING, False,
                      default='fm_catch.txt',
                      description='Name of FUSE file manager'),
            ConfigKey('FUSE_SPATIAL_MODE', ConfigKeyType.ENUM, False,
                      default='lumped',
                      valid_values=['auto', 'lumped', 'semi_distributed', 'distributed'],
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
            # In lumped mode GR writes GR_results.csv (columns 'datetime',
            # 'q_sim'), which GRResultExtractor._extract_from_csv and the lumped
            # postprocessor consume; the old '{experiment_id}_output.nc'/'Qsim'
            # declaration named a file the GR adapter never writes or reads.
            # The distributed artifact is the ``runoff`` declaration below.
            output_file_pattern='GR_results.csv',
            primary_output_var='q_sim'
        ),
        # Distributed mode: GRRunner._save_distributed_results_for_routing()
        # writes '{domain_name}_{experiment_id}_runs_def.nc' with (time, gru),
        # 'gruId' and 'q_routed' in m/s -- the mizuRoute-shaped file routing
        # consumes. FUSE-shaped by construction, not by copy-paste.
        runoff=RunoffConfig(
            output_dir_key='EXPERIMENT_OUTPUT_GR',
            output_dir_name='GR',
            default_var='q_routed',
            default_units='m/s',
            default_dt='86400',
            output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
            hru_dim='gru',
            hru_var='gruId',
            comment_name='GR4J',
            aliases=('GR4J', 'GR5J', 'GR6J'),
        ),
        # GR is the one model whose parallel-calibration file keeps the same
        # name as its ``runoff`` declaration, with no 'proc_' prefix -- correct,
        # not an oversight: GRRunner._save_distributed_results_for_routing()
        # writes to runner.output_path, which the calibration worker overrides
        # to the per-process sim_dir, and ConfigurationUpdater points
        # <input_dir> at that same directory, so the path already disambiguates.
        #
        # KNOWN WRONG, preserved value-for-value from core's table and reported
        # rather than fixed here. All three disagree with GR's actual daily,
        # gru-dimensioned output, and the non-parallel writer
        # (models/mizuroute/control_writer.py) gets each of them right:
        #   * hru_dim/hru_var say hru/hruId; GR writes (time, gru) with 'gruId'.
        #   * dt_qsim=None -> SETTINGS_MIZU_ROUTING_DT ('3600'); GR is daily
        #     ('86400' in ``runoff`` above).
        #   * sim times 01:00/23:00; GR's daily data needs midnight alignment.
        parallel_calibration=ParallelCalibrationConfig(
            fname_pattern='{domain_name}_{experiment_id}_runs_def.nc',
            runoff_var='q_routed',
            runoff_var_from_config=True,
            dt_qsim=None,
            sim_start_time='01:00',
            sim_end_time='23:00',
            hru_dim='hru',
            hru_var='hruId',
        ),
        spatial_mode_key='GR_SPATIAL_MODE',
        routing_key='GR_ROUTING_INTEGRATION',
        # Declaring this puts GR into RoutingDecider's routing-integration
        # check, which used to read FUSE's key alone -- not by design, but
        # because GR/CRHM/MHM/SWAT/VIC had no registered schema for the table to
        # see. All five now declare theirs (same rationale, not repeated per
        # model); each defaults to 'none' and no config in the tree sets one, so
        # widening the table changed no run.
        routing_integration_key='GR_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('GR_MODEL_TYPE', ConfigKeyType.ENUM, False,
                      default='GR4J',
                      valid_values=['GR4J', 'GR5J', 'GR6J'],
                      description='GR model variant'),
            # 'auto' matches GRConfig.spatial_mode's typed default (and
            # SpatialModeType's value set). This used to declare 'lumped',
            # which no GR run ever saw: the typed config is what reaches
            # config_dict, so apply_defaults() was seeding a value the model
            # itself never uses and validate() would have rejected 'auto'.
            ConfigKey('GR_SPATIAL_MODE', ConfigKeyType.ENUM, False,
                      default='auto',
                      valid_values=['auto', 'lumped', 'semi_distributed', 'distributed'],
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
        # Routing reads a NetCDF aggregate, not NGEN's per-nexus CSVs.
        # Confirmed: ngen writes nex-*_output.csv (NGENPostProcessor
        # .output_file_glob), and NGENPostProcessor derives
        # '{experiment_id}_runoff.nc' with a 'runoff' (time, hru) variable in
        # m/s from them for routing. Two genuinely different files.
        runoff=RunoffConfig(
            output_dir_key='EXPERIMENT_OUTPUT_NGEN',
            output_dir_name='NGEN',
            default_var='runoff',
            default_units='m/s',
            default_dt='3600',
            output_file_pattern='{experiment_id}_runoff.nc',
            hru_dim='hru',
            hru_var='hruId',
            comment_name='NGEN',
        ),
        # No spatial_mode_key: 'NGEN_SPATIAL_MODE' was a dead declaration.
        # NGENConfig declares no spatial_mode field and no config, template or
        # test in the tree ever sets the key, so the entry could only ever have
        # matched a hand-built raw dict. NGEN's discretization comes from its
        # realization/catchment GeoJSON, not from a config enum.
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
            # info.txt requests 'timeoutput variable COUT EVAP SNOW'
            # (config_manager.py), so HYPE writes timeCOUT.txt, which all five
            # consumers read. The 'timeOUT.txt' spelling this used to declare is
            # a file HYPE never writes.
            output_file_pattern='timeCOUT.txt',
            primary_output_var='cout'
        ),
        # No runoff declaration: HYPE is NOT a routable source. Its 'cout' is
        # already routed discharge at subbasin outlets, so feeding it to
        # mizuRoute would route it a second time -- and the
        # '{experiment_id}_timestep.nc' the old declaration named is a file the
        # HYPE adapter never writes. Routing HYPE now fails in
        # get_model_config() with an explicit "a model without one cannot feed a
        # routing model", instead of a missing-file error one stage later.
        #
        # No spatial_mode_key either: 'HYPE_SPATIAL_MODE' was dead -- HYPEConfig
        # declares no spatial_mode field and nothing in the tree sets the key.
        #
        # parallel_calibration below is half live:
        #  * LIVE, and HYPE's alone: ``settings_values_quoted=False`` (info.txt
        #    is bare tab-separated key/value, so quoting would pass every line
        #    through untouched) and ``output_dir_directive='resultdir'`` (HYPE
        #    has no 'outputPath'). _setup_parallel_dirs calls
        #    update_file_managers('HYPE', ..., info.txt) unconditionally, so
        #    dropping either breaks parallel HYPE calibration.
        #  * The mizuRoute fields are reachable but non-functional: they run
        #    only when settings/mizuRoute/ exists, which nothing generates for
        #    HYPE (requires_routing=False for every spatial mode; the mizuRoute
        #    preprocessor dispatches only FUSE/GR/NGEN), and they name a
        #    '*_timestep.nc' nothing writes. The get_model_config raise does NOT
        #    catch this -- the parallel updater reads this declaration, never
        #    runoff_loader. Preserved value-for-value so a hand-placed config
        #    does not silently change mid-campaign; failing early here is a
        #    behaviour change for its own reviewed PR.
        parallel_calibration=ParallelCalibrationConfig(
            fname_pattern='proc_{proc_id:02d}_{experiment_id}_timestep.nc',
            runoff_var='q_routed',
            runoff_var_from_config=True,
            dt_qsim='86400',
            sim_start_time='00:00',
            sim_end_time='00:00',
            hru_dim='hru',
            hru_var='hruId',
            settings_values_quoted=False,
            output_dir_directive='resultdir',
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
        spatial_mode_key='MESH_SPATIAL_MODE',
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


# ---------------------------------------------------------------------------
# Models whose only core-visible declaration is their routing-integration key.
#
# CRHM/MHM/SWAT/VIC each define a ``<MODEL>_ROUTING_INTEGRATION`` key that
# their calibration optimizer reads and that ``spatial_orchestrator`` derives
# by the same convention, but they had no registered schema — so
# ``RoutingDecider`` never consulted the key and the routing *decision*
# disagreed with the orchestrator that acted on it. Registering the schema is
# what makes the key visible to core.
#
# None of the four declares ``spatial_mode_key``: their spatial mode already
# resolves by lowercase-section convention, and putting them in
# ``RoutingDecider.SPATIAL_MODE_KEYS`` would change a second decision path.
# None declares ``runoff``: none of them writes a runoff artifact a routing
# model consumes today (CRHM and mHM route internally), so they stay
# unroutable-as-a-source, which ``get_model_config`` now reports explicitly.
# ---------------------------------------------------------------------------


def _create_crhm_schema() -> ModelConfigSchema:
    """Create configuration schema for CRHM model."""
    return ModelConfigSchema(
        model_name='CRHM',
        description='Cold Regions Hydrological Model',
        installation=InstallationConfig(
            install_path_key='CRHM_INSTALL_PATH',
            default_install_subpath='installs/crhm',
            exe_name_key='CRHM_EXE',
            default_exe_name='crhm'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_CRHM_PATH',
            default_forcing_subpath='forcing/CRHM_input',
            forcing_file_pattern='forcing.obs',
            required_variables=[]
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_CRHM',
            default_output_subpath='simulations/{experiment_id}/CRHM',
            output_file_pattern='crhm_output.txt',
            primary_output_var='basinflow'
        ),
        routing_integration_key='CRHM_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_CRHM_PATH', ConfigKeyType.PATH, False,
                      description='Path to CRHM settings directory'),
            ConfigKey('CRHM_PROJECT_FILE', ConfigKeyType.STRING, False,
                      default='model.prj',
                      description='CRHM project (.prj) file name'),
            ConfigKey('CRHM_OBSERVATION_FILE', ConfigKeyType.STRING, False,
                      default='forcing.obs',
                      description='CRHM observation (.obs) file name'),
            ConfigKey('CRHM_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
        ]
    )


def _create_mhm_schema() -> ModelConfigSchema:
    """Create configuration schema for mHM model."""
    return ModelConfigSchema(
        model_name='MHM',
        description='mesoscale Hydrological Model',
        installation=InstallationConfig(
            install_path_key='MHM_INSTALL_PATH',
            default_install_subpath='installs/mhm',
            exe_name_key='MHM_EXE',
            default_exe_name='mhm'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_MHM_PATH',
            default_forcing_subpath='forcing/MHM_input',
            forcing_file_pattern='*.nc',
            required_variables=[]
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_MHM',
            default_output_subpath='simulations/{experiment_id}/MHM',
            output_file_pattern='discharge_*.nc',
            primary_output_var='Qsim'
        ),
        routing_integration_key='MHM_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_MHM_PATH', ConfigKeyType.PATH, False,
                      description='Path to mHM settings directory'),
            ConfigKey('MHM_NAMELIST_FILE', ConfigKeyType.STRING, False,
                      default='mhm.nml',
                      description='mHM namelist file name'),
            ConfigKey('MHM_ROUTING_NAMELIST', ConfigKeyType.STRING, False,
                      default='mrm.nml',
                      description='mRM routing namelist file name'),
            ConfigKey('MHM_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
        ]
    )


def _create_swat_schema() -> ModelConfigSchema:
    """Create configuration schema for SWAT model."""
    return ModelConfigSchema(
        model_name='SWAT',
        description='Soil and Water Assessment Tool',
        installation=InstallationConfig(
            install_path_key='SWAT_INSTALL_PATH',
            default_install_subpath='installs/swat',
            exe_name_key='SWAT_EXE',
            default_exe_name='swat_rel.exe'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=3600
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_SWAT_PATH',
            default_forcing_subpath='forcing/SWAT_input',
            forcing_file_pattern='*.txt',
            required_variables=[]
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_SWAT',
            default_output_subpath='simulations/{experiment_id}/SWAT',
            output_file_pattern='output.rch',
            primary_output_var='FLOW_OUTcms'
        ),
        routing_integration_key='SWAT_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_SWAT_PATH', ConfigKeyType.PATH, False,
                      description='Path to SWAT settings directory'),
            ConfigKey('SWAT_TXTINOUT_DIR', ConfigKeyType.STRING, False,
                      default='TxtInOut',
                      description='SWAT TxtInOut directory name'),
            ConfigKey('SWAT_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
        ]
    )


def _create_vic_schema() -> ModelConfigSchema:
    """Create configuration schema for VIC model."""
    return ModelConfigSchema(
        model_name='VIC',
        description='Variable Infiltration Capacity model',
        installation=InstallationConfig(
            install_path_key='VIC_INSTALL_PATH',
            default_install_subpath='installs/vic',
            exe_name_key='VIC_EXE',
            default_exe_name='vic_image.exe'
        ),
        execution=ExecutionConfig(
            method='subprocess',
            supports_parallel=False,
            default_timeout=7200
        ),
        input=InputConfig(
            forcing_dir_key='FORCING_VIC_PATH',
            default_forcing_subpath='forcing/VIC_input',
            forcing_file_pattern='*.nc',
            required_variables=[]
        ),
        output=OutputConfig(
            output_dir_key='EXPERIMENT_OUTPUT_VIC',
            default_output_subpath='simulations/{experiment_id}/VIC',
            output_file_pattern='vic_output*.nc',
            primary_output_var='OUT_RUNOFF'
        ),
        routing_integration_key='VIC_ROUTING_INTEGRATION',
        config_keys=[
            ConfigKey('SETTINGS_VIC_PATH', ConfigKeyType.PATH, False,
                      description='Path to VIC settings directory'),
            ConfigKey('VIC_GLOBAL_PARAM_FILE', ConfigKeyType.STRING, False,
                      default='vic_global.txt',
                      description='VIC global parameter file name'),
            ConfigKey('VIC_ROUTING_INTEGRATION', ConfigKeyType.ENUM, False,
                      default='none',
                      valid_values=['none', 'mizuRoute'],
                      description='Routing model integration'),
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
        'CRHM': _create_crhm_schema(),
        'MHM': _create_mhm_schema(),
        'SWAT': _create_swat_schema(),
        'VIC': _create_vic_schema(),
    }.items():
        register_model_schema(name, schema)


# Initialize on module load
_register_schemas()
