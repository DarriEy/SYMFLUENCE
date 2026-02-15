"""
Hydrological model configuration models.

Contains configuration classes for all supported hydrological models:
SUMMAConfig, FUSEConfig, GRConfig, HYPEConfig, NGENConfig, MESHConfig,
MizuRouteConfig, LSTMConfig, and the parent ModelConfig.
"""

from typing import List, Literal, Optional, Dict, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from .base import FROZEN_CONFIG

# Spatial mode types for hydrological models
SpatialModeType = Literal['lumped', 'semi_distributed', 'distributed', 'auto']


class SUMMAConfig(BaseModel):
    """SUMMA hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='SUMMA_INSTALL_PATH')
    exe: str = Field(default='summa_sundials.exe', alias='SUMMA_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_SUMMA_PATH')
    filemanager: str = Field(default='fileManager.txt', alias='SETTINGS_SUMMA_FILEMANAGER')
    forcing_list: str = Field(default='forcingFileList.txt', alias='SETTINGS_SUMMA_FORCING_LIST')
    coldstate: str = Field(default='coldState.nc', alias='SETTINGS_SUMMA_COLDSTATE')
    trialparams: str = Field(default='trialParams.nc', alias='SETTINGS_SUMMA_TRIALPARAMS')
    attributes: str = Field(default='attributes.nc', alias='SETTINGS_SUMMA_ATTRIBUTES')
    output: str = Field(default='outputControl.txt', alias='SETTINGS_SUMMA_OUTPUT')
    basin_params_file: str = Field(default='basinParamInfo.txt', alias='SETTINGS_SUMMA_BASIN_PARAMS_FILE')
    local_params_file: str = Field(default='localParamInfo.txt', alias='SETTINGS_SUMMA_LOCAL_PARAMS_FILE')
    connect_hrus: bool = Field(default=True, alias='SETTINGS_SUMMA_CONNECT_HRUS')
    trialparam_n: int = Field(default=0, alias='SETTINGS_SUMMA_TRIALPARAM_N')
    trialparam_1: Optional[str] = Field(default=None, alias='SETTINGS_SUMMA_TRIALPARAM_1')
    use_parallel: bool = Field(default=False, alias='SETTINGS_SUMMA_USE_PARALLEL_SUMMA')
    cpus_per_task: int = Field(default=32, alias='SETTINGS_SUMMA_CPUS_PER_TASK', ge=1, le=256)
    time_limit: str = Field(default='01:00:00', alias='SETTINGS_SUMMA_TIME_LIMIT')
    mem: Union[int, str] = Field(default='5G', alias='SETTINGS_SUMMA_MEM')  # SLURM-style memory spec like "12G"
    gru_count: int = Field(default=85, alias='SETTINGS_SUMMA_GRU_COUNT')
    gru_per_job: int = Field(default=5, alias='SETTINGS_SUMMA_GRU_PER_JOB')
    parallel_path: str = Field(default='default', alias='SETTINGS_SUMMA_PARALLEL_PATH')
    parallel_exe: str = Field(default='summa_actors.exe', alias='SETTINGS_SUMMA_PARALLEL_EXE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SUMMA')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_SUMMA')
    params_to_calibrate: str = Field(
        default='albedo_max,albedo_min,canopy_capacity,slow_drainage',
        alias='PARAMS_TO_CALIBRATE'
    )
    basin_params_to_calibrate: str = Field(
        default='routingGammaShape,routingGammaScale',
        alias='BASIN_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='SUMMA_DECISION_OPTIONS')
    calibrate_depth: bool = Field(default=False, alias='CALIBRATE_DEPTH')
    depth_total_mult_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_TOTAL_MULT_BOUNDS')
    depth_shape_factor_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_SHAPE_FACTOR_BOUNDS')
    # Glacier-related settings
    glacier_mode: bool = Field(default=False, alias='SETTINGS_SUMMA_GLACIER_MODE')
    glacier_attributes: str = Field(default='attributes_glac.nc', alias='SETTINGS_SUMMA_GLACIER_ATTRIBUTES')
    glacier_coldstate: str = Field(default='coldState_glac.nc', alias='SETTINGS_SUMMA_GLACIER_COLDSTATE')
    # Execution settings
    timeout: int = Field(default=7200, alias='SUMMA_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)


class FUSEConfig(BaseModel):
    """FUSE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='FUSE_INSTALL_PATH')
    exe: str = Field(default='fuse.exe', alias='FUSE_EXE')
    routing_integration: str = Field(default='default', alias='FUSE_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_FUSE_PATH')
    filemanager: str = Field(default='default', alias='SETTINGS_FUSE_FILEMANAGER')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='FUSE_SPATIAL_MODE')
    subcatchment_dim: str = Field(default='longitude', alias='FUSE_SUBCATCHMENT_DIM')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_FUSE')
    params_to_calibrate: str = Field(
        default='MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',
        alias='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='FUSE_DECISION_OPTIONS')
    # Additional FUSE settings
    file_id: Optional[str] = Field(default=None, alias='FUSE_FILE_ID')
    n_elevation_bands: int = Field(default=1, alias='FUSE_N_ELEVATION_BANDS', ge=1)
    timeout: int = Field(default=3600, alias='FUSE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    run_internal_calibration: bool = Field(default=True, alias='FUSE_RUN_INTERNAL_CALIBRATION')


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')
    params_to_calibrate: str = Field(
        default='X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
        alias='GR_PARAMS_TO_CALIBRATE'
    )
    # Fallback behavior control - default to False to prevent silent data corruption
    allow_dummy_observations: bool = Field(
        default=False,
        alias='GR_ALLOW_DUMMY_OBSERVATIONS',
        description='If True, use zero-filled dummy observations when no streamflow data found'
    )
    allow_default_area: bool = Field(
        default=False,
        alias='GR_ALLOW_DEFAULT_AREA',
        description='If True, use 1.0 km² default area when basin shapefile not found'
    )


class HBVConfig(BaseModel):
    """HBV-96 hydrological model configuration"""
    model_config = FROZEN_CONFIG

    spatial_mode: SpatialModeType = Field(default='auto', alias='HBV_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='HBV_ROUTING_INTEGRATION')
    backend: Literal['jax', 'numpy'] = Field(default='jax', alias='HBV_BACKEND')
    use_gpu: bool = Field(default=False, alias='HBV_USE_GPU')
    jit_compile: bool = Field(default=True, alias='HBV_JIT_COMPILE')
    warmup_days: int = Field(default=365, alias='HBV_WARMUP_DAYS', ge=0)
    timestep_hours: int = Field(default=24, alias='HBV_TIMESTEP_HOURS', ge=1, le=24)
    params_to_calibrate: str = Field(
        default='default',  # 'default' triggers use of all available HBV parameters
        alias='HBV_PARAMS_TO_CALIBRATE',
        description="Parameters to calibrate. Use 'default' for all parameters, or specify comma-separated list."
    )
    use_gradient_calibration: bool = Field(default=True, alias='HBV_USE_GRADIENT_CALIBRATION')
    calibration_metric: Literal['KGE', 'NSE'] = Field(default='KGE', alias='HBV_CALIBRATION_METRIC')
    # Initial state values
    initial_snow: float = Field(default=0.0, alias='HBV_INITIAL_SNOW', ge=0.0)
    initial_sm: float = Field(default=150.0, alias='HBV_INITIAL_SM', ge=0.0)
    initial_suz: float = Field(default=10.0, alias='HBV_INITIAL_SUZ', ge=0.0)
    initial_slz: float = Field(default=10.0, alias='HBV_INITIAL_SLZ', ge=0.0)
    # PET configuration
    pet_method: Literal['input', 'hamon', 'thornthwaite'] = Field(default='input', alias='HBV_PET_METHOD')
    latitude: Optional[float] = Field(default=None, alias='HBV_LATITUDE', ge=-90.0, le=90.0)
    allow_unit_heuristics: bool = Field(
        default=False,
        alias='HBV_ALLOW_UNIT_HEURISTICS',
        description='Allow magnitude-based unit detection for precip/PET when units are missing or ambiguous'
    )
    # Output configuration
    save_states: bool = Field(default=False, alias='HBV_SAVE_STATES')
    output_frequency: Literal['daily', 'timestep'] = Field(default='daily', alias='HBV_OUTPUT_FREQUENCY')
    # Default parameter values
    default_tt: float = Field(default=0.0, alias='HBV_DEFAULT_TT')
    default_cfmax: float = Field(default=3.5, alias='HBV_DEFAULT_CFMAX')
    default_sfcf: float = Field(default=0.9, alias='HBV_DEFAULT_SFCF')
    default_cfr: float = Field(default=0.05, alias='HBV_DEFAULT_CFR')
    default_cwh: float = Field(default=0.1, alias='HBV_DEFAULT_CWH')
    default_fc: float = Field(default=250.0, alias='HBV_DEFAULT_FC')
    default_lp: float = Field(default=0.7, alias='HBV_DEFAULT_LP')
    default_beta: float = Field(default=2.5, alias='HBV_DEFAULT_BETA')
    default_k0: float = Field(default=0.3, alias='HBV_DEFAULT_K0')
    default_k1: float = Field(default=0.1, alias='HBV_DEFAULT_K1')
    default_k2: float = Field(default=0.01, alias='HBV_DEFAULT_K2')
    default_uzl: float = Field(default=30.0, alias='HBV_DEFAULT_UZL')
    default_perc: float = Field(default=2.5, alias='HBV_DEFAULT_PERC')
    default_maxbas: float = Field(default=2.5, alias='HBV_DEFAULT_MAXBAS')


class HYPEConfig(BaseModel):
    """HYPE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='HYPE_INSTALL_PATH')
    exe: str = Field(default='hype', alias='HYPE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_HYPE_PATH')
    info_file: str = Field(default='info.txt', alias='SETTINGS_HYPE_INFO')
    params_to_calibrate: str = Field(
        default='ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp',
        alias='HYPE_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='HYPE_SPINUP_DAYS')


class NGENConfig(BaseModel):
    """NGEN (Next Generation Water Resources Modeling Framework) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='NGEN_INSTALL_PATH')
    exe: str = Field(default='ngen', alias='NGEN_EXE')
    modules_to_calibrate: str = Field(default='CFE', alias='NGEN_MODULES_TO_CALIBRATE')
    cfe_params_to_calibrate: str = Field(
        default='maxsmc,satdk,bb,slop',
        alias='NGEN_CFE_PARAMS_TO_CALIBRATE'
    )
    noah_params_to_calibrate: str = Field(
        default='refkdt,slope,smcmax,dksat',
        alias='NGEN_NOAH_PARAMS_TO_CALIBRATE'
    )
    pet_params_to_calibrate: str = Field(
        default='wind_speed_measurement_height_m',
        alias='NGEN_PET_PARAMS_TO_CALIBRATE'
    )
    active_catchment_id: Optional[str] = Field(default=None, alias='NGEN_ACTIVE_CATCHMENT_ID')


class MESHConfig(BaseModel):
    """MESH (Modélisation Environnementale-Surface Hydrology) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MESH_INSTALL_PATH')
    exe: str = Field(default='mesh.exe', alias='MESH_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='MESH_SPATIAL_MODE')
    settings_path: str = Field(default='default', alias='SETTINGS_MESH_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MESH')
    forcing_path: str = Field(default='default', alias='MESH_FORCING_PATH')
    forcing_vars: str = Field(default='default', alias='MESH_FORCING_VARS')
    forcing_units: str = Field(default='default', alias='MESH_FORCING_UNITS')
    forcing_to_units: str = Field(default='default', alias='MESH_FORCING_TO_UNITS')
    landcover_stats_path: str = Field(default='default', alias='MESH_LANDCOVER_STATS_PATH')
    landcover_stats_dir: str = Field(default='default', alias='MESH_LANDCOVER_STATS_DIR')
    landcover_stats_file: str = Field(default='default', alias='MESH_LANDCOVER_STATS_FILE')
    main_id: str = Field(default='default', alias='MESH_MAIN_ID')
    ds_main_id: str = Field(default='default', alias='MESH_DS_MAIN_ID')
    landcover_classes: str = Field(default='default', alias='MESH_LANDCOVER_CLASSES')
    ddb_vars: str = Field(default='default', alias='MESH_DDB_VARS')
    ddb_units: str = Field(default='default', alias='MESH_DDB_UNITS')
    ddb_to_units: str = Field(default='default', alias='MESH_DDB_TO_UNITS')
    ddb_min_values: str = Field(default='default', alias='MESH_DDB_MIN_VALUES')
    gru_dim: str = Field(default='default', alias='MESH_GRU_DIM')
    hru_dim: str = Field(default='default', alias='MESH_HRU_DIM')
    outlet_value: str = Field(default='default', alias='MESH_OUTLET_VALUE')
    # Additional MESH settings
    input_file: str = Field(default='default', alias='SETTINGS_MESH_INPUT')
    params_to_calibrate: str = Field(
        default='ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR',
        alias='MESH_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='MESH_SPINUP_DAYS')
    gru_min_total: float = Field(default=0.0, alias='MESH_GRU_MIN_TOTAL')
    # Lumped mode enforcement settings
    force_single_gru: bool = Field(default=True, alias='MESH_FORCE_SINGLE_GRU')
    apply_params_all_grus: bool = Field(default=True, alias='MESH_APPLY_PARAMS_ALL_GRUS')
    use_landcover_multipliers: bool = Field(default=True, alias='MESH_USE_LANDCOVER_MULTIPLIERS')
    enable_frozen_soil: bool = Field(default=True, alias='MESH_ENABLE_FROZEN_SOIL')


class MizuRouteConfig(BaseModel):
    """mizuRoute routing model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='INSTALL_PATH_MIZUROUTE')
    exe: str = Field(default='mizuRoute.exe', alias='EXE_NAME_MIZUROUTE')
    settings_path: str = Field(default='default', alias='SETTINGS_MIZU_PATH')
    within_basin: int = Field(default=0, alias='SETTINGS_MIZU_WITHIN_BASIN')
    routing_dt: int = Field(default=3600, alias='SETTINGS_MIZU_ROUTING_DT')
    routing_units: str = Field(default='m/s', alias='SETTINGS_MIZU_ROUTING_UNITS')
    routing_var: str = Field(default='averageRoutedRunoff', alias='SETTINGS_MIZU_ROUTING_VAR')
    output_freq: str = Field(default='single', alias='SETTINGS_MIZU_OUTPUT_FREQ')
    output_vars: str = Field(default='1', alias='SETTINGS_MIZU_OUTPUT_VARS')
    make_outlet: str = Field(default='n/a', alias='SETTINGS_MIZU_MAKE_OUTLET')
    needs_remap: bool = Field(default=False, alias='SETTINGS_MIZU_NEEDS_REMAP')
    topology: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY')
    parameters: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETERS')
    control_file: str = Field(default='mizuroute.control', alias='SETTINGS_MIZU_CONTROL_FILE')
    remap: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP')
    from_model: str = Field(default='default', alias='MIZU_FROM_MODEL')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_MIZUROUTE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MIZUROUTE')
    # Additional mizuRoute settings
    output_var: str = Field(default='IRFroutedRunoff', alias='SETTINGS_MIZU_OUTPUT_VAR')
    parameter_file: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETER_FILE')
    remap_file: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP_FILE')
    topology_file: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY_FILE')
    params_to_calibrate: str = Field(
        default='velo,diff',
        alias='MIZUROUTE_PARAMS_TO_CALIBRATE'
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_MIZUROUTE')
    timeout: int = Field(default=3600, alias='MIZUROUTE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    time_rounding_freq: str = Field(
        default='h',
        alias='MIZUROUTE_TIME_ROUNDING_FREQ',
        description='Frequency for rounding time values (e.g., "h" for hour, "min" for minute, "none" to disable)'
    )

    @field_validator('output_vars', mode='before')
    @classmethod
    def normalize_output_vars(cls, v):
        """Convert list or other types to string for output_vars"""
        if isinstance(v, list):
            return ' '.join(str(item).strip() for item in v)
        return str(v)


class DRouteConfig(BaseModel):
    """dRoute routing model configuration (EXPERIMENTAL)

    dRoute is a C++ river routing library with Python bindings that offers:
    - Multiple routing methods (Muskingum-Cunge, IRF, Lag, Diffusive Wave, KWT)
    - Native automatic differentiation for gradient-based calibration
    - mizuRoute-compatible network topology format
    """
    model_config = FROZEN_CONFIG

    # Execution settings
    execution_mode: Literal['python', 'subprocess'] = Field(
        default='python',
        alias='DROUTE_EXECUTION_MODE',
        description='Execution mode: python API (preferred) or subprocess fallback'
    )
    install_path: str = Field(default='default', alias='DROUTE_INSTALL_PATH')
    exe: str = Field(default='droute', alias='DROUTE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_DROUTE_PATH')

    # Routing configuration
    routing_method: Literal['muskingum_cunge', 'irf', 'lag', 'diffusive_wave', 'kwt'] = Field(
        default='muskingum_cunge',
        alias='DROUTE_ROUTING_METHOD',
        description='Routing scheme to use'
    )
    routing_dt: int = Field(
        default=3600,
        alias='DROUTE_ROUTING_DT',
        ge=60,
        le=86400,
        description='Routing timestep in seconds'
    )

    # Gradient/AD settings
    enable_gradients: bool = Field(
        default=False,
        alias='DROUTE_ENABLE_GRADIENTS',
        description='Enable automatic differentiation for gradient-based calibration'
    )
    ad_backend: Literal['codipack', 'enzyme'] = Field(
        default='codipack',
        alias='DROUTE_AD_BACKEND',
        description='AD backend (requires dRoute compiled with AD support)'
    )

    # Topology configuration
    topology_file: str = Field(default='topology.nc', alias='DROUTE_TOPOLOGY_FILE')
    topology_format: Literal['netcdf', 'geojson', 'csv'] = Field(
        default='netcdf',
        alias='DROUTE_TOPOLOGY_FORMAT'
    )
    config_file: str = Field(default='droute_config.yaml', alias='DROUTE_CONFIG_FILE')

    # Integration settings
    from_model: str = Field(
        default='default',
        alias='DROUTE_FROM_MODEL',
        description='Source model for runoff input (SUMMA, FUSE, GR, etc.)'
    )

    # Output settings
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_DROUTE')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_DROUTE')

    # Calibration settings
    params_to_calibrate: str = Field(
        default='velocity,diffusivity',
        alias='DROUTE_PARAMS_TO_CALIBRATE'
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_DROUTE')
    timeout: int = Field(default=3600, alias='DROUTE_TIMEOUT', ge=60, le=86400)


class TRouteConfig(BaseModel):
    """T-Route (NOAA OWP) channel routing configuration.

    T-Route is NOAA's Office of Water Prediction channel routing model
    supporting Muskingum-Cunge and diffusive wave routing methods for
    large-scale river network simulations.
    """
    model_config = FROZEN_CONFIG

    # Installation and paths
    install_path: str = Field(default='default', alias='TROUTE_INSTALL_PATH')
    pkg_path: str = Field(
        default='troute/network/__init__.py',
        alias='TROUTE_PKG_PATH',
    )
    settings_path: str = Field(default='default', alias='SETTINGS_TROUTE_PATH')

    # Topology and config files
    topology_file: str = Field(
        default='troute_topology.nc',
        alias='SETTINGS_TROUTE_TOPOLOGY',
    )
    config_file: str = Field(
        default='troute_config.yml',
        alias='SETTINGS_TROUTE_CONFIG_FILE',
    )

    # Routing configuration
    dt_seconds: int = Field(
        default=3600,
        alias='SETTINGS_TROUTE_DT_SECONDS',
        ge=60,
        le=86400,
        description='Routing timestep in seconds',
    )
    routing_method: Literal['muskingum_cunge', 'diffusive_wave'] = Field(
        default='muskingum_cunge',
        alias='TROUTE_ROUTING_METHOD',
        description='Routing scheme: muskingum_cunge or diffusive_wave',
    )

    # Integration settings
    from_model: str = Field(
        default='SUMMA',
        alias='TROUTE_FROM_MODEL',
        description='Source model for runoff input (SUMMA, FUSE, etc.)',
    )
    mannings_n: float = Field(
        default=0.035,
        alias='TROUTE_MANNINGS_N',
        gt=0,
        description="Manning's roughness coefficient",
    )

    # Output settings
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_TROUTE')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_TROUTE')

    # Hydraulic geometry for channel width estimation (W = a * A^b)
    hg_width_coeff: float = Field(
        default=2.71,
        alias='TROUTE_HG_WIDTH_COEFF',
        gt=0,
        description='Hydraulic geometry width coefficient (a in W=a*A^b)',
    )
    hg_width_exp: float = Field(
        default=0.557,
        alias='TROUTE_HG_WIDTH_EXP',
        gt=0,
        le=1.0,
        description='Hydraulic geometry width exponent (b in W=a*A^b)',
    )

    # Sub-timestep for Courant stability
    qts_subdivisions: int = Field(
        default=0,
        alias='TROUTE_QTS_SUBDIVISIONS',
        ge=0,
        le=20,
        description='Sub-timestep divisions (0=auto from Courant)',
    )

    # Calibration settings
    params_to_calibrate: str = Field(
        default='mannings_n',
        alias='TROUTE_PARAMS_TO_CALIBRATE',
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_TROUTE')
    timeout: int = Field(default=3600, alias='TROUTE_TIMEOUT', ge=60, le=86400)


class LSTMConfig(BaseModel):
    """LSTM neural network emulator configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='LSTM_LOAD')
    hidden_size: int = Field(default=128, alias='LSTM_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='LSTM_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='LSTM_EPOCHS', ge=1, le=10000)
    batch_size: int = Field(default=64, alias='LSTM_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='LSTM_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='LSTM_LEARNING_PATIENCE', ge=1)
    lookback: int = Field(default=700, alias='LSTM_LOOKBACK', ge=1)
    dropout: float = Field(default=0.2, alias='LSTM_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='LSTM_L2_REGULARIZATION', ge=0)
    use_attention: bool = Field(default=True, alias='LSTM_USE_ATTENTION')
    use_snow: bool = Field(default=False, alias='LSTM_USE_SNOW')
    train_through_routing: bool = Field(default=False, alias='LSTM_TRAIN_THROUGH_ROUTING')


class WMFireConfig(BaseModel):
    """WMFire wildfire spread module configuration for RHESSys.

    WMFire is a fire spread model that couples with RHESSys to simulate
    wildfire spread based on fuel loads, moisture, wind, and topography.

    Reference:
        Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. 2017.
        Balancing uncertainty and complexity to incorporate fire spread in
        an eco-hydrological model. International Journal of Wildland Fire.
    """
    model_config = FROZEN_CONFIG

    # Grid resolution and timestep
    grid_resolution: int = Field(
        default=30,
        alias='WMFIRE_GRID_RESOLUTION',
        ge=10,
        le=200,
        description='Fire grid cell resolution in meters (30, 60, or 90 recommended)'
    )
    timestep_hours: int = Field(
        default=24,
        alias='WMFIRE_TIMESTEP_HOURS',
        ge=1,
        le=24,
        description='Fire spread timestep in hours (1-24)'
    )

    # Fuel and moisture configuration
    ndays_average: float = Field(
        default=30.0,
        alias='WMFIRE_NDAYS_AVERAGE',
        ge=1.0,
        le=365.0,
        description='Fuel moisture averaging window in days'
    )
    fuel_source: Literal['static', 'rhessys_litter'] = Field(
        default='static',
        alias='WMFIRE_FUEL_SOURCE',
        description='Source of fuel load data: static values or RHESSys litter pools'
    )
    moisture_source: Literal['static', 'rhessys_soil'] = Field(
        default='static',
        alias='WMFIRE_MOISTURE_SOURCE',
        description='Source of moisture data: static values or RHESSys soil moisture'
    )
    carbon_to_fuel_ratio: float = Field(
        default=2.0,
        alias='WMFIRE_CARBON_TO_FUEL_RATIO',
        ge=1.0,
        le=5.0,
        description='Conversion factor from kg carbon to kg fuel'
    )

    # Ignition configuration
    ignition_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_SHAPEFILE',
        description='Path to ignition point shapefile (overrides ignition_point if set)'
    )
    ignition_point: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_POINT',
        description='Ignition point as "lat/lon" (e.g., "51.2096/-115.7539")'
    )
    ignition_date: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_DATE',
        description='Ignition date as "YYYY-MM-DD" for fire simulation start'
    )
    ignition_name: Optional[str] = Field(
        default='ignition',
        alias='WMFIRE_IGNITION_NAME',
        description='Name for the ignition point (used in output shapefile)'
    )

    # Fire perimeter validation
    perimeter_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_SHAPEFILE',
        description='Path to observed fire perimeter shapefile for validation'
    )
    perimeter_dir: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_DIR',
        description='Directory containing fire perimeter shapefiles for comparison'
    )

    # Output options
    write_geotiff: bool = Field(
        default=True,
        alias='WMFIRE_WRITE_GEOTIFF',
        description='Write georeferenced GeoTIFF outputs for visualization'
    )

    # Optional coefficient overrides (None = use defaults from fire.def)
    load_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K1',
        description='Fuel load coefficient k1 (default 3.9)'
    )
    load_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K2',
        description='Fuel load coefficient k2 (default 0.07)'
    )
    moisture_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K1',
        description='Moisture coefficient k1 (default 3.8)'
    )
    moisture_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K2',
        description='Moisture coefficient k2 (default 0.27)'
    )
    ign_def_mod: Optional[float] = Field(
        default=None,
        alias='WMFIRE_IGN_DEF_MOD',
        description='Ignition probability modifier (default 1.0, increase for more fires)'
    )
    mean_ign: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MEAN_IGN',
        description='Mean ignition events per timestep (default 1.0)'
    )
    windmax: Optional[float] = Field(
        default=None,
        alias='WMFIRE_WINDMAX',
        description='Maximum wind speed multiplier (default 1.0)'
    )
    slope_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_SLOPE_K1',
        description='Slope effect coefficient k1 (default 0.91)'
    )

    @field_validator('grid_resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate grid resolution is reasonable."""
        recommended = [30, 60, 90]
        if v not in recommended:
            import warnings
            warnings.warn(
                f"Grid resolution {v}m is not standard. "
                f"Recommended values: {recommended}"
            )
        return v

    @field_validator('ignition_point')
    @classmethod
    def validate_ignition_point(cls, v):
        """Validate ignition point format."""
        if v is not None:
            parts = v.split('/')
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid ignition_point format: {v}. "
                    f"Expected 'lat/lon' (e.g., '51.2096/-115.7539')"
                )
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude {lat} out of range [-90, 90]")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude {lon} out of range [-180, 180]")
            except ValueError as e:
                raise ValueError(f"Invalid ignition_point coordinates: {e}")
        return v


class RHESSysConfig(BaseModel):
    """RHESSys (Regional Hydro-Ecologic Simulation System) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='RHESSYS_INSTALL_PATH')
    exe: str = Field(default='rhessys', alias='RHESSYS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_RHESSYS_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_RHESSYS')
    forcing_path: str = Field(default='default', alias='FORCING_RHESSYS_PATH')
    world_template: str = Field(default='world.template', alias='RHESSYS_WORLD_TEMPLATE')
    flow_template: str = Field(default='flow.template', alias='RHESSYS_FLOW_TEMPLATE')
    # LNA/TWI controls (optional; None disables caps)
    lna_area_cap_m2: Optional[float] = Field(default=None, alias='RHESSYS_LNA_AREA_CAP_M2')
    lna_min: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MIN')
    lna_max: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MAX')
    params_to_calibrate: str = Field(
        default=(
            'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,porosity_decay,'
            'soil_depth,snow_melt_Tcoef,max_snow_temp,min_rain_temp,theta_mean_std_p1'
        ),
        alias='RHESSYS_PARAMS_TO_CALIBRATE'
    )
    skip_calibration: bool = Field(default=True, alias='RHESSYS_SKIP_CALIBRATION')
    # WMFire integration (wildfire spread module)
    use_wmfire: bool = Field(default=False, alias='RHESSYS_USE_WMFIRE')
    wmfire_install_path: str = Field(default='installs/wmfire/lib', alias='WMFIRE_INSTALL_PATH')
    wmfire_lib: str = Field(default='libwmfire.so', alias='WMFIRE_LIB')
    wmfire: Optional[WMFireConfig] = Field(default=None, description='Enhanced WMFire configuration')
    # Legacy VMFire aliases
    use_vmfire: bool = Field(default=False, alias='RHESSYS_USE_VMFIRE')
    vmfire_install_path: str = Field(default='installs/wmfire/lib', alias='VMFIRE_INSTALL_PATH')
    # Execution settings
    timeout: int = Field(default=7200, alias='RHESSYS_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    # Grow mode for Farquhar photosynthesis and transpiration (default True)
    use_grow_mode: bool = Field(default=True, alias='RHESSYS_USE_GROW_MODE')


class VICConfig(BaseModel):
    """VIC (Variable Infiltration Capacity) model configuration.

    VIC is a large-scale, semi-distributed hydrological model that solves
    full water and energy balances. It uses a grid-based structure and is
    typically applied to large river basins.

    Reference:
        Liang, X., D. P. Lettenmaier, E. F. Wood, and S. J. Burges, 1994:
        A simple hydrologically based model of land surface water and energy
        fluxes for general circulation models. J. Geophys. Res., 99(D7), 14415-14428.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='VIC_INSTALL_PATH')
    exe: str = Field(default='vic_image.exe', alias='VIC_EXE')
    driver: Literal['image', 'classic'] = Field(default='image', alias='VIC_DRIVER')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_VIC_PATH')
    global_param_file: str = Field(default='vic_global.txt', alias='VIC_GLOBAL_PARAM_FILE')
    domain_file: str = Field(default='vic_domain.nc', alias='VIC_DOMAIN_FILE')
    params_file: str = Field(default='vic_params.nc', alias='VIC_PARAMS_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='auto', alias='VIC_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_VIC')
    output_prefix: str = Field(default='vic_output', alias='VIC_OUTPUT_PREFIX')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default=None,
        alias='VIC_PARAMS_TO_CALIBRATE'
    )

    # Model options
    full_energy: bool = Field(default=True, alias='VIC_FULL_ENERGY')
    frozen_soil: bool = Field(default=True, alias='VIC_FROZEN_SOIL')
    snow_band: bool = Field(default=False, alias='VIC_SNOW_BAND')
    n_snow_bands: int = Field(default=10, alias='VIC_N_SNOW_BANDS', ge=1, le=25)
    pfactor_per_km: float = Field(default=0.0005, alias='VIC_PFACTOR_PER_KM', ge=0.0, le=0.01)

    # Timing
    model_steps_per_day: int = Field(default=24, alias='VIC_STEPS_PER_DAY', ge=1, le=48)

    # Execution
    timeout: int = Field(default=7200, alias='VIC_TIMEOUT', ge=60, le=86400)


class CLMConfig(BaseModel):
    """CLM (Community Land Model / CTSM 5.x) configuration.

    CLM5 is the land component of CESM, providing comprehensive
    biogeophysics, biogeochemistry, hydrology, snow, and vegetation
    dynamics. It is the most physics-heavy LSM in the ensemble.

    Reference:
        Lawrence, D. M., et al. (2019): The Community Land Model version 5.
        JAMES, 11, 4245-4287.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='CLM_INSTALL_PATH')
    exe: str = Field(default='cesm.exe', alias='CLM_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_CLM_PATH')
    compset: str = Field(default='I2000Clm50SpGs', alias='CLM_COMPSET')
    params_file: str = Field(default='clm5_params.nc', alias='CLM_PARAMS_FILE')
    surfdata_file: str = Field(default='surfdata_clm.nc', alias='CLM_SURFDATA_FILE')
    domain_file: str = Field(default='domain.nc', alias='CLM_DOMAIN_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CLM_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CLM')
    hist_nhtfrq: int = Field(default=-24, alias='CLM_HIST_NHTFRQ')
    hist_mfilt: int = Field(default=365, alias='CLM_HIST_MFILT')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default=None,
        alias='CLM_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='CLM_TIMEOUT', ge=60, le=86400)
    warmup_days: int = Field(default=365, alias='CLM_WARMUP_DAYS', ge=0, le=3650)


class MODFLOWConfig(BaseModel):
    """MODFLOW 6 (USGS modular groundwater flow model) configuration.

    MODFLOW 6 simulates three-dimensional groundwater flow using the
    finite-difference method. In SYMFLUENCE it is used as a lumped
    single-cell groundwater model coupled with land surface models
    (e.g., SUMMA) to separate baseflow from surface runoff.

    Reference:
        Langevin, C.D., et al. (2017): Documentation for the MODFLOW 6
        Groundwater Flow Model. USGS Techniques and Methods 6-A55.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='MODFLOW_INSTALL_PATH')
    exe: str = Field(default='mf6', alias='MODFLOW_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_MODFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MODFLOW_SPATIAL_MODE')

    # Grid discretization
    grid_type: str = Field(default='dis', alias='MODFLOW_GRID_TYPE')
    nlay: int = Field(default=1, alias='MODFLOW_NLAY', ge=1, le=100)
    nrow: int = Field(default=1, alias='MODFLOW_NROW', ge=1, le=10000)
    ncol: int = Field(default=1, alias='MODFLOW_NCOL', ge=1, le=10000)
    cell_size: float = Field(default=1000.0, alias='MODFLOW_CELL_SIZE', gt=0)

    # Aquifer properties
    k: float = Field(default=5.0, alias='MODFLOW_K', gt=0)
    sy: float = Field(default=0.15, alias='MODFLOW_SY', gt=0, le=0.5)
    ss: float = Field(default=1e-5, alias='MODFLOW_SS', gt=0, le=0.1)
    strt: Optional[float] = Field(default=None, alias='MODFLOW_STRT')
    top: float = Field(default=1500.0, alias='MODFLOW_TOP')
    bot: float = Field(default=1400.0, alias='MODFLOW_BOT')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='MODFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='MODFLOW_RECHARGE_VARIABLE')

    # Drain package
    drain_elevation: Optional[float] = Field(default=None, alias='MODFLOW_DRAIN_ELEVATION')
    drain_conductance: float = Field(default=50.0, alias='MODFLOW_DRAIN_CONDUCTANCE', gt=0)

    # Stress period
    stress_period_length: float = Field(default=1.0, alias='MODFLOW_STRESS_PERIOD_LENGTH', gt=0)
    nstp: int = Field(default=1, alias='MODFLOW_NSTP', ge=1, le=1000)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MODFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K,SY,DRAIN_CONDUCTANCE',
        alias='MODFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='MODFLOW_TIMEOUT', ge=60, le=86400)


class ParFlowConfig(BaseModel):
    """ParFlow integrated hydrologic model configuration.

    ParFlow solves variably-saturated flow (Richards equation) and
    overland flow. In SYMFLUENCE it is used as an alternative to MODFLOW
    for coupled land surface + groundwater simulations with full vadose
    zone support.

    Reference:
        Kollet, S.J. & Maxwell, R.M. (2006): Integrated surface-groundwater
        flow modeling. Advances in Water Resources 29(7).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='PARFLOW_INSTALL_PATH')
    exe: str = Field(default='parflow', alias='PARFLOW_EXE')
    parflow_dir: str = Field(default='default', alias='PARFLOW_DIR')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_PARFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='PARFLOW_SPATIAL_MODE')

    # Grid discretization
    nx: int = Field(default=1, alias='PARFLOW_NX', ge=1, le=10000)
    ny: int = Field(default=1, alias='PARFLOW_NY', ge=1, le=10000)
    nz: int = Field(default=1, alias='PARFLOW_NZ', ge=1, le=100)
    dx: float = Field(default=1000.0, alias='PARFLOW_DX', gt=0)
    dy: float = Field(default=1000.0, alias='PARFLOW_DY', gt=0)
    dz: float = Field(default=100.0, alias='PARFLOW_DZ', gt=0)

    # Domain geometry
    top: float = Field(default=1500.0, alias='PARFLOW_TOP')
    bot: float = Field(default=1400.0, alias='PARFLOW_BOT')

    # Subsurface properties
    k_sat: float = Field(default=5.0, alias='PARFLOW_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='PARFLOW_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='PARFLOW_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='PARFLOW_VG_N', gt=1.0)
    s_res: float = Field(default=0.1, alias='PARFLOW_S_RES', ge=0, lt=1.0)
    s_sat: float = Field(default=1.0, alias='PARFLOW_S_SAT', gt=0, le=1.0)
    specific_storage: float = Field(default=1e-5, alias='PARFLOW_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='PARFLOW_MANNINGS_N', gt=0)

    # Initial conditions
    initial_pressure: Optional[float] = Field(default=None, alias='PARFLOW_INITIAL_PRESSURE')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='PARFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='PARFLOW_RECHARGE_VARIABLE')

    # Solver
    solver: str = Field(default='Richards', alias='PARFLOW_SOLVER')
    timestep_hours: float = Field(default=1.0, alias='PARFLOW_TIMESTEP_HOURS', gt=0)

    # Parallel execution
    num_procs: int = Field(default=1, alias='PARFLOW_NUM_PROCS', ge=1, le=1024)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PARFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,MANNINGS_N',
        alias='PARFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='PARFLOW_TIMEOUT', ge=60, le=86400)


class GNNConfig(BaseModel):
    """GNN (Graph Neural Network) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='GNN_LOAD')
    hidden_size: int = Field(default=128, alias='GNN_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='GNN_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='GNN_EPOCHS', ge=1, le=10000)
    batch_size: int = Field(default=64, alias='GNN_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='GNN_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='GNN_LEARNING_PATIENCE', ge=1)
    dropout: float = Field(default=0.2, alias='GNN_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='GNN_L2_REGULARIZATION', ge=0)
    params_to_calibrate: str = Field(
        default='precip_mult,temp_offset,routing_velocity',
        alias='GNN_PARAMS_TO_CALIBRATE'
    )
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(default=None, alias='GNN_PARAMETER_BOUNDS')


class IGNACIOConfig(BaseModel):
    """IGNACIO fire spread model configuration.

    IGNACIO implements the Canadian Forest Fire Behavior Prediction (FBP) System
    with Richards' elliptical wave propagation for fire spread modeling.

    This is a standalone fire model that can be run independently or compared
    with WMFire results for validation.

    Reference:
        IGNACIO: https://github.com/KatherineHopeReece/Fire-Engine-Framework
    """
    model_config = FROZEN_CONFIG

    # Project settings
    project_name: str = Field(
        default='ignacio_run',
        alias='IGNACIO_PROJECT_NAME',
        description='Name for the IGNACIO simulation project'
    )
    output_dir: str = Field(
        default='default',
        alias='IGNACIO_OUTPUT_DIR',
        description='Output directory for IGNACIO results'
    )

    # Terrain inputs
    dem_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_DEM_PATH',
        description='Path to DEM raster file'
    )

    # Fuel inputs
    fuel_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_FUEL_PATH',
        description='Path to fuel type raster'
    )
    default_fuel_type: str = Field(
        default='C-2',
        alias='IGNACIO_DEFAULT_FUEL',
        description='Default FBP fuel type code'
    )

    # Ignition configuration
    ignition_shapefile: Optional[str] = Field(
        default=None,
        alias='IGNACIO_IGNITION_SHAPEFILE',
        description='Path to ignition point shapefile'
    )
    ignition_date: Optional[str] = Field(
        default=None,
        alias='IGNACIO_IGNITION_DATE',
        description='Ignition date/time as YYYY-MM-DD HH:MM:SS'
    )

    # Weather configuration
    station_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_STATION_PATH',
        description='Path to weather station CSV file'
    )
    calculate_fwi: bool = Field(
        default=True,
        alias='IGNACIO_CALCULATE_FWI',
        description='Calculate FWI from weather data'
    )

    # Simulation parameters
    dt: float = Field(
        default=1.0,
        alias='IGNACIO_DT',
        ge=0.1,
        le=60.0,
        description='Simulation timestep in minutes'
    )
    max_duration: int = Field(
        default=480,
        alias='IGNACIO_MAX_DURATION',
        ge=1,
        le=43200,
        description='Maximum simulation duration in minutes'
    )

    # Output configuration
    save_perimeters: bool = Field(
        default=True,
        alias='IGNACIO_SAVE_PERIMETERS',
        description='Save fire perimeter shapefiles'
    )

    # Comparison with WMFire
    compare_with_wmfire: bool = Field(
        default=False,
        alias='IGNACIO_COMPARE_WMFIRE',
        description='Compare fire perimeters with WMFire results'
    )

    @field_validator('ignition_date')
    @classmethod
    def validate_ignition_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate ignition date format."""
        if v is not None:
            from datetime import datetime
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    datetime.strptime(v, fmt)
                    return v
                except ValueError:
                    continue
            raise ValueError(
                f"Invalid ignition date format: {v}. "
                f"Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
            )
        return v


class MIKESHEConfig(BaseModel):
    """MIKE-SHE (DHI) integrated physics-based model configuration.

    MIKE-SHE is a fully integrated, physically-based modelling system
    that simulates overland flow, unsaturated flow, groundwater flow,
    channel flow, and evapotranspiration.

    Reference:
        Graham, D.N. & Butts, M.B. (2005): Flexible, integrated watershed
        modelling with MIKE SHE. Watershed Models, 245-272.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MIKESHE_INSTALL_PATH')
    exe: str = Field(default='MikeSheEngine.exe', alias='MIKESHE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MIKESHE_PATH')
    setup_file: str = Field(default='model.she', alias='MIKESHE_SETUP_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MIKESHE_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MIKESHE')
    use_wine: bool = Field(default=False, alias='MIKESHE_USE_WINE')
    params_to_calibrate: str = Field(
        default='manning_m,detention_storage,Ks_uz,theta_sat,theta_fc,theta_wp,Ks_sz_h,specific_yield,ddf,snow_threshold,max_canopy_storage',
        alias='MIKESHE_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=7200, alias='MIKESHE_TIMEOUT', ge=60, le=86400)


class SWATConfig(BaseModel):
    """SWAT (Soil and Water Assessment Tool) model configuration.

    SWAT is a river basin scale model developed by USDA-ARS to predict
    the impact of land management on water, sediment, and agricultural
    chemical yields.

    Reference:
        Arnold, J.G., et al. (1998): Large area hydrologic modeling and
        assessment Part I: Model development. JAWRA, 34(1), 73-89.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='SWAT_INSTALL_PATH')
    exe: str = Field(default='swat_rel.exe', alias='SWAT_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_SWAT_PATH')
    txtinout_dir: str = Field(default='TxtInOut', alias='SWAT_TXTINOUT_DIR')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='SWAT_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SWAT')
    params_to_calibrate: str = Field(
        default='CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP',
        alias='SWAT_PARAMS_TO_CALIBRATE'
    )
    warmup_years: int = Field(default=2, alias='SWAT_WARMUP_YEARS', ge=0, le=10)
    timeout: int = Field(default=3600, alias='SWAT_TIMEOUT', ge=60, le=86400)


class MHMConfig(BaseModel):
    """mHM (mesoscale Hydrological Model) configuration.

    mHM is a spatially distributed hydrological model developed at the
    Helmholtz Centre for Environmental Research (UFZ). It uses multiscale
    parameter regionalization (MPR) for parameter transfer.

    Reference:
        Samaniego, L., et al. (2010): Multiscale parameter regionalization
        of a grid-based hydrologic model at the mesoscale. Water Resources
        Research, 46, W05523.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MHM_INSTALL_PATH')
    exe: str = Field(default='mhm', alias='MHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MHM_PATH')
    namelist_file: str = Field(default='mhm.nml', alias='MHM_NAMELIST_FILE')
    routing_namelist: str = Field(default='mrm.nml', alias='MHM_ROUTING_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MHM')
    params_to_calibrate: str = Field(
        default='canopyInterceptionFactor,snowThresholdTemperature,degreeDayFactor_forest,degreeDayFactor_pervious,PTF_Ks,interflowRecession_slope,rechargeCoefficient,baseflowRecession,saturatedHydraulicConductivity',
        alias='MHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='MHM_TIMEOUT', ge=60, le=86400)


class CRHMConfig(BaseModel):
    """CRHM (Cold Regions Hydrological Model) configuration.

    CRHM is a physically-based, object-oriented hydrological model
    designed specifically for cold-region processes including blowing
    snow, energy-balance snowmelt, and frozen soil infiltration.

    Reference:
        Pomeroy, J.W., et al. (2007): The Cold Regions Hydrological Model:
        a platform for basing process representation and model structure on
        physical evidence. Hydrological Processes, 21(19), 2650-2667.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='CRHM_INSTALL_PATH')
    exe: str = Field(default='crhm', alias='CRHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_CRHM_PATH')
    project_file: str = Field(default='model.prj', alias='CRHM_PROJECT_FILE')
    observation_file: str = Field(default='forcing.obs', alias='CRHM_OBSERVATION_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CRHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CRHM')
    params_to_calibrate: str = Field(
        default='basin_area,Ht,Asnow,inhibit_evap,Ksat,soil_rechr_max,soil_moist_max,soil_gw_K,Sdmax,fetch',
        alias='CRHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='CRHM_TIMEOUT', ge=60, le=86400)


class ModelConfig(BaseModel):
    """Hydrological model configuration"""
    model_config = FROZEN_CONFIG

    # Required model selection
    hydrological_model: Union[str, List[str]] = Field(alias='HYDROLOGICAL_MODEL')
    routing_model: Optional[str] = Field(default=None, alias='ROUTING_MODEL')
    fire_model: Optional[str] = Field(default=None, alias='FIRE_MODEL')

    # Model-specific configurations (optional, validated only if model is selected)
    summa: Optional[SUMMAConfig] = Field(default=None)
    fuse: Optional[FUSEConfig] = Field(default=None)
    gr: Optional[GRConfig] = Field(default=None)
    hbv: Optional[HBVConfig] = Field(default=None)
    hype: Optional[HYPEConfig] = Field(default=None)
    ngen: Optional[NGENConfig] = Field(default=None)
    mesh: Optional[MESHConfig] = Field(default=None)
    mizuroute: Optional[MizuRouteConfig] = Field(default=None)
    droute: Optional[DRouteConfig] = Field(default=None)
    troute: Optional[TRouteConfig] = Field(default=None)
    lstm: Optional[LSTMConfig] = Field(default=None, alias='lstm')
    rhessys: Optional[RHESSysConfig] = Field(default=None)
    gnn: Optional[GNNConfig] = Field(default=None)
    ignacio: Optional[IGNACIOConfig] = Field(default=None)
    vic: Optional[VICConfig] = Field(default=None)
    clm: Optional[CLMConfig] = Field(default=None)
    modflow: Optional[MODFLOWConfig] = Field(default=None)
    parflow: Optional[ParFlowConfig] = Field(default=None)
    mikeshe: Optional[MIKESHEConfig] = Field(default=None)
    swat: Optional[SWATConfig] = Field(default=None)
    mhm: Optional[MHMConfig] = Field(default=None)
    crhm: Optional[CRHMConfig] = Field(default=None)

    @field_validator('hydrological_model')
    @classmethod
    def validate_hydrological_model(cls, v):
        """Normalize model list to comma-separated string"""
        if isinstance(v, list):
            return ",".join(str(i).strip() for i in v)
        return v

    @model_validator(mode='before')
    @classmethod
    def auto_populate_model_configs(cls, values):
        """Auto-populate model-specific configs when model is selected."""
        if not isinstance(values, dict):
            return values

        # Get hydrological_model from values (check both alias and field name)
        hydrological_model = values.get('HYDROLOGICAL_MODEL') or values.get('hydrological_model')
        if not hydrological_model:
            return values

        # Parse models from hydrological_model string
        if isinstance(hydrological_model, list):
            models = [str(m).strip().upper() for m in hydrological_model]
        else:
            models = [m.strip().upper() for m in str(hydrological_model).split(',')]

        # Auto-create model configs if not already set
        if 'SUMMA' in models and values.get('summa') is None:
            values['summa'] = SUMMAConfig()
        if 'FUSE' in models and values.get('fuse') is None:
            values['fuse'] = FUSEConfig()
        if 'GR' in models and values.get('gr') is None:
            values['gr'] = GRConfig()
        if 'HBV' in models and values.get('hbv') is None:
            values['hbv'] = HBVConfig()
        if 'HYPE' in models and values.get('hype') is None:
            values['hype'] = HYPEConfig()
        if 'NGEN' in models and values.get('ngen') is None:
            values['ngen'] = NGENConfig()
        if 'MESH' in models and values.get('mesh') is None:
            values['mesh'] = MESHConfig()
        if 'LSTM' in models and values.get('lstm') is None:
            values['lstm'] = LSTMConfig()
        if 'RHESSYS' in models and values.get('rhessys') is None:
            values['rhessys'] = RHESSysConfig()
        if 'GNN' in models and values.get('gnn') is None:
            values['gnn'] = GNNConfig()
        if 'VIC' in models and values.get('vic') is None:
            values['vic'] = VICConfig()
        if 'CLM' in models and values.get('clm') is None:
            values['clm'] = CLMConfig()
        if 'MODFLOW' in models and values.get('modflow') is None:
            values['modflow'] = MODFLOWConfig()
        if 'PARFLOW' in models and values.get('parflow') is None:
            values['parflow'] = ParFlowConfig()
        if 'MIKESHE' in models and values.get('mikeshe') is None:
            values['mikeshe'] = MIKESHEConfig()
        if 'SWAT' in models and values.get('swat') is None:
            values['swat'] = SWATConfig()
        if 'MHM' in models and values.get('mhm') is None:
            values['mhm'] = MHMConfig()
        if 'CRHM' in models and values.get('crhm') is None:
            values['crhm'] = CRHMConfig()

        # Auto-create routing model config if needed
        routing_model = values.get('ROUTING_MODEL') or values.get('routing_model')
        if routing_model:
            routing_upper = str(routing_model).upper()
            if routing_upper == 'MIZUROUTE' and values.get('mizuroute') is None:
                values['mizuroute'] = MizuRouteConfig()
            elif routing_upper == 'DROUTE' and values.get('droute') is None:
                values['droute'] = DRouteConfig()

        # Auto-create fire model config if needed
        fire_model = values.get('FIRE_MODEL') or values.get('fire_model')
        if fire_model:
            fire_upper = str(fire_model).upper()
            if fire_upper == 'IGNACIO' and values.get('ignacio') is None:
                values['ignacio'] = IGNACIOConfig()

        return values
