"""
Evaluation configuration models.

Contains configuration classes for observation data sources:
StreamflowConfig, SNOTELConfig, FluxNetConfig, USGSGWConfig, SMAPConfig,
GRACEConfig, MODISSnowConfig, AttributesConfig, and the parent EvaluationConfig.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import FROZEN_CONFIG


class StreamflowConfig(BaseModel):
    """Streamflow observation data settings"""
    model_config = FROZEN_CONFIG

    data_provider: Optional[str] = Field(default=None, alias='STREAMFLOW_DATA_PROVIDER')
    download_usgs: bool = Field(default=False, alias='DOWNLOAD_USGS_DATA')
    download_wsc: bool = Field(default=False, alias='DOWNLOAD_WSC_DATA')
    station_id: Optional[str] = Field(default=None, alias='STATION_ID')
    raw_path: str = Field(default='default', alias='STREAMFLOW_RAW_PATH')
    raw_name: str = Field(default='default', alias='STREAMFLOW_RAW_NAME')
    processed_path: str = Field(default='default', alias='STREAMFLOW_PROCESSED_PATH')
    hydat_path: str = Field(default='default', alias='HYDAT_PATH')


class SNOTELConfig(BaseModel):
    """SNOTEL observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_SNOTEL')
    station: Optional[str] = Field(default=None, alias='SNOTEL_STATION')
    path: Optional[str] = Field(default=None, alias='SNOTEL_PATH')


class FluxNetConfig(BaseModel):
    """FluxNet observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_FLUXNET')
    station: Optional[str] = Field(default=None, alias='FLUXNET_STATION')
    path: Optional[str] = Field(default=None, alias='FLUXNET_PATH')


class USGSGWConfig(BaseModel):
    """USGS groundwater observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_USGS_GW')
    station: Optional[str] = Field(default=None, alias='USGS_STATION')


class SMAPConfig(BaseModel):
    """SMAP soil moisture observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_SMAP')
    product: str = Field(default='SPL4SMGP', alias='SMAP_PRODUCT')
    path: str = Field(default='default', alias='SMAP_PATH')


class GRACEConfig(BaseModel):
    """GRACE terrestrial water storage observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_GRACE')
    product: str = Field(default='RL06', alias='GRACE_PRODUCT')
    path: str = Field(default='default', alias='GRACE_PATH')


class MODISSnowConfig(BaseModel):
    """MODIS snow cover observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_MODIS_SNOW')
    product: str = Field(default='MOD10A1.006', alias='MODIS_SNOW_PRODUCT')
    path: str = Field(default='default', alias='MODIS_SNOW_PATH')


class AttributesConfig(BaseModel):
    """Catchment attributes data settings"""
    model_config = FROZEN_CONFIG

    data_dir: str = Field(default='default', alias='ATTRIBUTES_DATA_DIR')
    soilgrids_path: str = Field(default='default', alias='ATTRIBUTES_SOILGRIDS_PATH')
    pelletier_path: str = Field(default='default', alias='ATTRIBUTES_PELLETIER_PATH')
    merit_path: str = Field(default='default', alias='ATTRIBUTES_MERIT_PATH')
    modis_path: str = Field(default='default', alias='ATTRIBUTES_MODIS_PATH')
    glclu_path: str = Field(default='default', alias='ATTRIBUTES_GLCLU_PATH')
    forest_height_path: str = Field(default='default', alias='ATTRIBUTES_FOREST_HEIGHT_PATH')
    worldclim_path: str = Field(default='default', alias='ATTRIBUTES_WORLDCLIM_PATH')
    glim_path: str = Field(default='default', alias='ATTRIBUTES_GLIM_PATH')
    groundwater_path: str = Field(default='default', alias='ATTRIBUTES_GROUNDWATER_PATH')
    streamflow_path: str = Field(default='default', alias='ATTRIBUTES_STREAMFLOW_PATH')
    glwd_path: str = Field(default='default', alias='ATTRIBUTES_GLWD_PATH')
    hydrolakes_path: str = Field(default='default', alias='ATTRIBUTES_HYDROLAKES_PATH')
    output_dir: str = Field(default='default', alias='ATTRIBUTES_OUTPUT_DIR')


class EvaluationConfig(BaseModel):
    """Evaluation data and analysis configuration"""
    model_config = FROZEN_CONFIG

    evaluation_data: Optional[List[str]] = Field(default=None, alias='EVALUATION_DATA')
    analyses: Optional[List[str]] = Field(default=None, alias='ANALYSES')
    sim_reach_id: Optional[int] = Field(default=None, alias='SIM_REACH_ID')

    # Observation data sources
    streamflow: Optional[StreamflowConfig] = Field(default_factory=StreamflowConfig)
    snotel: Optional[SNOTELConfig] = Field(default_factory=SNOTELConfig)
    fluxnet: Optional[FluxNetConfig] = Field(default_factory=FluxNetConfig)
    usgs_gw: Optional[USGSGWConfig] = Field(default_factory=USGSGWConfig)
    smap: Optional[SMAPConfig] = Field(default_factory=SMAPConfig)
    grace: Optional[GRACEConfig] = Field(default_factory=GRACEConfig)
    modis_snow: Optional[MODISSnowConfig] = Field(default_factory=MODISSnowConfig)
    attributes: Optional[AttributesConfig] = Field(default_factory=AttributesConfig)
    hru_gauge_mapping: Optional[Dict[str, Any]] = Field(default_factory=dict, alias='HRU_GAUGE_MAPPING')

    @field_validator('evaluation_data', 'analyses', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
