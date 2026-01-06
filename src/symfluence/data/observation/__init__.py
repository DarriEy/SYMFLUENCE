from .registry import ObservationRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import grace, modis_snow, modis_et, usgs, wsc, snotel, soil_moisture, fluxcom, smhi, lamah_ice
except ImportError:
    pass

__all__ = ["ObservationRegistry"]
