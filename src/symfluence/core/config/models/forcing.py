"""
Forcing configuration models.

Contains NexConfig, EMEarthConfig, and ForcingConfig for meteorological forcing data.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import FROZEN_CONFIG


class NexConfig(BaseModel):
    """NASA NEX-GDDP climate projection settings"""
    model_config = FROZEN_CONFIG

    models: Optional[List[str]] = Field(default=None, alias='NEX_MODELS')
    scenarios: Optional[List[str]] = Field(default=None, alias='NEX_SCENARIOS')
    ensembles: Optional[List[str]] = Field(default=None, alias='NEX_ENSEMBLES')
    variables: Optional[List[str]] = Field(default=None, alias='NEX_VARIABLES')

    @field_validator('models', 'scenarios', 'ensembles', 'variables', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class EMEarthConfig(BaseModel):
    """EM-Earth ensemble meteorological forcing settings"""
    model_config = FROZEN_CONFIG

    prcp_dir: Optional[str] = Field(default=None, alias='EM_EARTH_PRCP_DIR')
    tmean_dir: Optional[str] = Field(default=None, alias='EM_EARTH_TMEAN_DIR')
    min_bbox_size: float = Field(default=0.1, alias='EM_EARTH_MIN_BBOX_SIZE')
    max_expansion: float = Field(default=0.2, alias='EM_EARTH_MAX_EXPANSION')
    prcp_var: str = Field(default='prcp', alias='EM_PRCP')
    data_type: str = Field(default='deterministic', alias='EM_EARTH_DATA_TYPE')


class ForcingConfig(BaseModel):
    """Meteorological forcing configuration"""
    model_config = FROZEN_CONFIG

    # Required dataset
    dataset: str = Field(alias='FORCING_DATASET')

    # Forcing settings
    time_step_size: int = Field(default=3600, alias='FORCING_TIME_STEP_SIZE')
    variables: str = Field(default='default', alias='FORCING_VARIABLES')
    measurement_height: int = Field(default=2, alias='FORCING_MEASUREMENT_HEIGHT')
    apply_lapse_rate: bool = Field(default=True, alias='APPLY_LAPSE_RATE')
    lapse_rate: float = Field(default=0.0065, alias='LAPSE_RATE')
    shape_lat_name: str = Field(default='lat', alias='FORCING_SHAPE_LAT_NAME')
    shape_lon_name: str = Field(default='lon', alias='FORCING_SHAPE_LON_NAME')
    pet_method: str = Field(default='oudin', alias='PET_METHOD')
    supplement: bool = Field(default=False, alias='SUPPLEMENT_FORCING')

    # ERA5-specific settings
    era5_use_cds: Optional[bool] = Field(default=None, alias='ERA5_USE_CDS')

    # Dataset-specific settings
    nex: Optional[NexConfig] = Field(default=None)
    em_earth: Optional[EMEarthConfig] = Field(default=None)
