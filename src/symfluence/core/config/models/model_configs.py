"""
Hydrological model configuration models.

Contains configuration classes for all supported hydrological models:
SUMMAConfig, FUSEConfig, GRConfig, HYPEConfig, NGENConfig, MESHConfig,
MizuRouteConfig, LSTMConfig, and the parent ModelConfig.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .base import FROZEN_CONFIG


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
    cpus_per_task: int = Field(default=32, alias='SETTINGS_SUMMA_CPUS_PER_TASK')
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
    timeout: int = Field(default=7200, alias='SUMMA_TIMEOUT')  # seconds


class FUSEConfig(BaseModel):
    """FUSE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='FUSE_INSTALL_PATH')
    exe: str = Field(default='fuse.exe', alias='FUSE_EXE')
    routing_integration: str = Field(default='default', alias='FUSE_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_FUSE_PATH')
    filemanager: str = Field(default='default', alias='SETTINGS_FUSE_FILEMANAGER')
    spatial_mode: str = Field(default='lumped', alias='FUSE_SPATIAL_MODE')
    subcatchment_dim: str = Field(default='longitude', alias='FUSE_SUBCATCHMENT_DIM')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_FUSE')
    params_to_calibrate: str = Field(
        default='MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',
        alias='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='FUSE_DECISION_OPTIONS')
    # Additional FUSE settings
    file_id: Optional[str] = Field(default=None, alias='FUSE_FILE_ID')
    n_elevation_bands: int = Field(default=1, alias='FUSE_N_ELEVATION_BANDS')
    timeout: int = Field(default=3600, alias='FUSE_TIMEOUT')  # seconds


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: str = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')
    params_to_calibrate: str = Field(
        default='X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
        alias='GR_PARAMS_TO_CALIBRATE'
    )


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
    spatial_mode: str = Field(default='auto', alias='MESH_SPATIAL_MODE')  # 'auto', 'lumped', or 'distributed'
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
    routing_dt: int = Field(default=86400, alias='SETTINGS_MIZU_ROUTING_DT')
    routing_units: str = Field(default='m/s', alias='SETTINGS_MIZU_ROUTING_UNITS')
    routing_var: str = Field(default='q_routed', alias='SETTINGS_MIZU_ROUTING_VAR')
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
    timeout: int = Field(default=3600, alias='MIZUROUTE_TIMEOUT')  # seconds

    @field_validator('output_vars', mode='before')
    @classmethod
    def normalize_output_vars(cls, v):
        """Convert list or other types to string for output_vars"""
        if isinstance(v, list):
            return ' '.join(str(item).strip() for item in v)
        return str(v)


class LSTMConfig(BaseModel):
    """LSTM neural network emulator configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='LSTM_LOAD')
    hidden_size: int = Field(default=128, alias='LSTM_HIDDEN_SIZE')
    num_layers: int = Field(default=3, alias='LSTM_NUM_LAYERS')
    epochs: int = Field(default=300, alias='LSTM_EPOCHS')
    batch_size: int = Field(default=64, alias='LSTM_BATCH_SIZE')
    learning_rate: float = Field(default=0.001, alias='LSTM_LEARNING_RATE')
    learning_patience: int = Field(default=30, alias='LSTM_LEARNING_PATIENCE')
    lookback: int = Field(default=700, alias='LSTM_LOOKBACK')
    dropout: float = Field(default=0.2, alias='LSTM_DROPOUT')
    l2_regularization: float = Field(default=1e-6, alias='LSTM_L2_REGULARIZATION')
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
    timeout: int = Field(default=7200, alias='RHESSYS_TIMEOUT')  # seconds


class GNNConfig(BaseModel):
    """GNN (Graph Neural Network) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='GNN_LOAD')
    hidden_size: int = Field(default=128, alias='GNN_HIDDEN_SIZE')
    num_layers: int = Field(default=3, alias='GNN_NUM_LAYERS')
    epochs: int = Field(default=300, alias='GNN_EPOCHS')
    batch_size: int = Field(default=64, alias='GNN_BATCH_SIZE')
    learning_rate: float = Field(default=0.001, alias='GNN_LEARNING_RATE')
    learning_patience: int = Field(default=30, alias='GNN_LEARNING_PATIENCE')
    dropout: float = Field(default=0.2, alias='GNN_DROPOUT')
    l2_regularization: float = Field(default=1e-6, alias='GNN_L2_REGULARIZATION')
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
    hype: Optional[HYPEConfig] = Field(default=None)
    ngen: Optional[NGENConfig] = Field(default=None)
    mesh: Optional[MESHConfig] = Field(default=None)
    mizuroute: Optional[MizuRouteConfig] = Field(default=None)
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

    @model_validator(mode='after')
    def auto_populate_model_configs(self):
        """Auto-populate model-specific configs when model is selected."""
        # Parse models from hydrological_model string
        if isinstance(self.hydrological_model, str):
            models = [m.strip().upper() for m in self.hydrological_model.split(',')]
        else:
            models = [str(self.hydrological_model).upper()]

        # Auto-create model configs if not already set
        # We use model_copy to create a new instance with updated fields (required for frozen models)
        updates = {}

        if 'SUMMA' in models and self.summa is None:
            updates['summa'] = SUMMAConfig()
        if 'FUSE' in models and self.fuse is None:
            updates['fuse'] = FUSEConfig()
        if 'GR' in models and self.gr is None:
            updates['gr'] = GRConfig()
        if 'HYPE' in models and self.hype is None:
            updates['hype'] = HYPEConfig()
        if 'NGEN' in models and self.ngen is None:
            updates['ngen'] = NGENConfig()
        if 'MESH' in models and self.mesh is None:
            updates['mesh'] = MESHConfig()
        if 'LSTM' in models and self.lstm is None:
            updates['lstm'] = LSTMConfig()
        if 'RHESSYS' in models and self.rhessys is None:
            updates['rhessys'] = RHESSysConfig()
        if 'GNN' in models and self.gnn is None:
            updates['gnn'] = GNNConfig()

        # Auto-create routing model config if needed
        if self.routing_model and self.routing_model.upper() == 'MIZUROUTE' and self.mizuroute is None:
            updates['mizuroute'] = MizuRouteConfig()

        # Apply updates if any
        if updates:
            return self.model_copy(update=updates)
        return self
