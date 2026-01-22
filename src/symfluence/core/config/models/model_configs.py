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


class HBVConfig(BaseModel):
    """HBV-96 hydrological model configuration"""
    model_config = FROZEN_CONFIG

    spatial_mode: SpatialModeType = Field(default='auto', alias='HBV_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='HBV_ROUTING_INTEGRATION')
    backend: Literal['jax', 'numpy'] = Field(default='jax', alias='HBV_BACKEND')
    use_gpu: bool = Field(default=False, alias='HBV_USE_GPU')
    jit_compile: bool = Field(default=True, alias='HBV_JIT_COMPILE')
    warmup_days: int = Field(default=365, alias='HBV_WARMUP_DAYS', ge=0)
    params_to_calibrate: str = Field(
        default='tt,cfmax,fc,lp,beta,k0,k1,k2,uzl,perc,maxbas',
        alias='HBV_PARAMS_TO_CALIBRATE'
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
    params_to_calibrate: str = Field(
        default='sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef',
        alias='RHESSYS_PARAMS_TO_CALIBRATE'
    )
    skip_calibration: bool = Field(default=True, alias='RHESSYS_SKIP_CALIBRATION')
    # WMFire integration (wildfire spread module)
    use_wmfire: bool = Field(default=False, alias='RHESSYS_USE_WMFIRE')
    wmfire_install_path: str = Field(default='installs/wmfire/lib', alias='WMFIRE_INSTALL_PATH')
    wmfire_lib: str = Field(default='libwmfire.so', alias='WMFIRE_LIB')
    # Legacy VMFire aliases
    use_vmfire: bool = Field(default=False, alias='RHESSYS_USE_VMFIRE')
    vmfire_install_path: str = Field(default='installs/wmfire/lib', alias='VMFIRE_INSTALL_PATH')
    # Execution settings
    timeout: int = Field(default=7200, alias='RHESSYS_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)


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


class ModelConfig(BaseModel):
    """Hydrological model configuration"""
    model_config = FROZEN_CONFIG

    # Required model selection
    hydrological_model: Union[str, List[str]] = Field(alias='HYDROLOGICAL_MODEL')
    routing_model: Optional[str] = Field(default=None, alias='ROUTING_MODEL')

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
    lstm: Optional[LSTMConfig] = Field(default=None, alias='lstm')
    rhessys: Optional[RHESSysConfig] = Field(default=None)
    gnn: Optional[GNNConfig] = Field(default=None)

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

        # Auto-create routing model config if needed
        routing_model = values.get('ROUTING_MODEL') or values.get('routing_model')
        if routing_model:
            routing_upper = str(routing_model).upper()
            if routing_upper == 'MIZUROUTE' and values.get('mizuroute') is None:
                values['mizuroute'] = MizuRouteConfig()
            elif routing_upper == 'DROUTE' and values.get('droute') is None:
                values['droute'] = DRouteConfig()

        return values
