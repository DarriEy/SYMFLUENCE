"""
Observation data handlers for various data sources.

This module provides handlers for acquiring and processing observation data
from multiple sources including satellite products, in-situ networks, and
reanalysis datasets.
"""

from .fluxcom import FLUXCOMETHandler
from .fluxnet import FLUXNETObservationHandler
from .ggmn import GGMNHandler
from .gleam import GLEAMETHandler
from .grace import GRACEHandler
from .lamah_ice import LamahIceStreamflowHandler
from .modis_et import MODISETHandler
from .modis_snow import MODISSnowHandler, MODISSCAHandler
from .modis_utils import (
    MODIS_FILL_VALUES,
    CLOUD_VALUE,
    VALID_SNOW_RANGE,
    MODIS_ET_COLUMN_MAP,
    convert_cftime_to_datetime,
    standardize_et_columns,
    interpolate_8day_to_daily,
    apply_modis_quality_filter,
    extract_spatial_average,
    find_variable_in_dataset,
)
from .smhi import SMHIStreamflowHandler
from .snotel import SNOTELHandler
from .soil_moisture import SMAPHandler, ISMNHandler, ESACCISMHandler
from .usgs import USGSStreamflowHandler, USGSGroundwaterHandler
from .wsc import WSCStreamflowHandler

__all__ = [
    # FLUXCOM
    "FLUXCOMETHandler",
    # FLUXNET
    "FLUXNETObservationHandler",
    # GGMN
    "GGMNHandler",
    # GLEAM
    "GLEAMETHandler",
    # GRACE
    "GRACEHandler",
    # LamaH-Ice
    "LamahIceStreamflowHandler",
    # MODIS ET
    "MODISETHandler",
    # MODIS Snow
    "MODISSnowHandler",
    "MODISSCAHandler",
    # MODIS utilities
    "MODIS_FILL_VALUES",
    "CLOUD_VALUE",
    "VALID_SNOW_RANGE",
    "MODIS_ET_COLUMN_MAP",
    "convert_cftime_to_datetime",
    "standardize_et_columns",
    "interpolate_8day_to_daily",
    "apply_modis_quality_filter",
    "extract_spatial_average",
    "find_variable_in_dataset",
    # SMHI
    "SMHIStreamflowHandler",
    # SNOTEL
    "SNOTELHandler",
    # Soil moisture
    "SMAPHandler",
    "ISMNHandler",
    "ESACCISMHandler",
    # USGS
    "USGSStreamflowHandler",
    "USGSGroundwaterHandler",
    # WSC
    "WSCStreamflowHandler",
]
