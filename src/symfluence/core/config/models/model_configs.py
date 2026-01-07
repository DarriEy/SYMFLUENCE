"""
Hydrological model configuration models.

Contains configuration classes for all supported hydrological models:
SUMMAConfig, FUSEConfig, GRConfig, HYPEConfig, NGENConfig, MESHConfig,
MizuRouteConfig, LSTMConfig, and the parent ModelConfig.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

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
    mem: int = Field(default=5, alias='SETTINGS_SUMMA_MEM')
    gru_count: int = Field(default=85, alias='SETTINGS_SUMMA_GRU_COUNT')
    gru_per_job: int = Field(default=5, alias='SETTINGS_SUMMA_GRU_PER_JOB')
    parallel_path: str = Field(default='default', alias='SETTINGS_SUMMA_PARALLEL_PATH')
    parallel_exe: str = Field(default='summa_actors.exe', alias='SETTINGS_SUMMA_PARALLEL_EXE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SUMMA')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_SUMMA')
    params_to_calibrate: Optional[str] = Field(default=None, alias='PARAMS_TO_CALIBRATE')
    basin_params_to_calibrate: Optional[str] = Field(default=None, alias='BASIN_PARAMS_TO_CALIBRATE')
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='SUMMA_DECISION_OPTIONS')
    calibrate_depth: bool = Field(default=False, alias='CALIBRATE_DEPTH')
    depth_total_mult_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_TOTAL_MULT_BOUNDS')
    depth_shape_factor_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_SHAPE_FACTOR_BOUNDS')


class FUSEConfig(BaseModel):
    """FUSE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='FUSE_INSTALL_PATH')
    exe: str = Field(default='fuse.exe', alias='FUSE_EXE')
    routing_integration: str = Field(default='default', alias='FUSE_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_FUSE_PATH')
    filemanager: str = Field(default='default', alias='SETTINGS_FUSE_FILEMANAGER')
    spatial_mode: str = Field(default='lumped', alias='FUSE_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_FUSE')
    params_to_calibrate: Optional[str] = Field(default=None, alias='SETTINGS_FUSE_PARAMS_TO_CALIBRATE')
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='FUSE_DECISION_OPTIONS')


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: str = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')


class HYPEConfig(BaseModel):
    """HYPE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='HYPE_INSTALL_PATH')
    exe: str = Field(default='hype', alias='HYPE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_HYPE_PATH')


class NGENConfig(BaseModel):
    """NGEN (Next Generation Water Resources Modeling Framework) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='NGEN_INSTALL_PATH')
    exe: str = Field(default='ngen', alias='NGEN_EXE')
    modules_to_calibrate: Optional[str] = Field(default=None, alias='NGEN_MODULES_TO_CALIBRATE')
    cfe_params_to_calibrate: Optional[str] = Field(default=None, alias='NGEN_CFE_PARAMS_TO_CALIBRATE')
    noah_params_to_calibrate: Optional[str] = Field(default=None, alias='NGEN_NOAH_PARAMS_TO_CALIBRATE')
    pet_params_to_calibrate: Optional[str] = Field(default=None, alias='NGEN_PET_PARAMS_TO_CALIBRATE')
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

    @field_validator('hydrological_model')
    @classmethod
    def validate_hydrological_model(cls, v):
        """Normalize model list to comma-separated string"""
        if isinstance(v, list):
            return ",".join(str(i).strip() for i in v)
        return v
