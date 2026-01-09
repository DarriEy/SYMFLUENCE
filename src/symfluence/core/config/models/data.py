"""
Data management configuration models.

Contains DataConfig for high-level data acquisition and processing settings.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class DataConfig(BaseModel):
    """Configuration for data acquisition and processing"""
    model_config = FROZEN_CONFIG

    # High-level acquisition flags
    additional_observations: Optional[List[str]] = Field(default=None, alias='ADDITIONAL_OBSERVATIONS')
    supplement_forcing: bool = Field(default=False, alias='SUPPLEMENT_FORCING')
    force_download: bool = Field(default=False, alias='FORCE_DOWNLOAD')
    
    # Streamflow provider
    streamflow_data_provider: Optional[str] = Field(default=None, alias='STREAMFLOW_DATA_PROVIDER')
    
    # Acquisition flags for specific datasets
    download_usgs_gw: bool = Field(default=False, alias='DOWNLOAD_USGS_GW')
    download_modis_snow: bool = Field(default=False, alias='DOWNLOAD_MODIS_SNOW')
    download_snotel: bool = Field(default=False, alias='DOWNLOAD_SNOTEL')
    download_smhi_data: bool = Field(default=False, alias='DOWNLOAD_SMHI_DATA')
    download_lamah_ice_data: bool = Field(default=False, alias='DOWNLOAD_LAMAH_ICE_DATA')
    download_ismn: bool = Field(default=False, alias='DOWNLOAD_ISMN')
    
    # Dataset-specific paths
    lamah_ice_path: Optional[str] = Field(default=None, alias='LAMAH_ICE_PATH')

    @field_validator('additional_observations', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
