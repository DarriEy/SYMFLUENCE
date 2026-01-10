from .registry import ObservationRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import grace, modis_snow, modis_et, usgs, wsc, snotel, soil_moisture, fluxcom, gleam, smhi, lamah_ice, ggmn, fluxnet
except ImportError:
    pass

__all__ = ["ObservationRegistry"]
