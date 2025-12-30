from .registry import ObservationRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import grace, modis_snow, modis_et
except ImportError:
    pass

__all__ = ["ObservationRegistry"]
