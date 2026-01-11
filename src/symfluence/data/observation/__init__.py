"""Observation data acquisition and processing module.

This module provides handlers for acquiring observational data from various sources
(GRACE, MODIS, USGS, etc.) and processing them into SYMFLUENCE-standard formats.

The module uses a plugin-style registry pattern: observation handlers are defined
in separate modules and self-register using the @ObservationRegistry.register() decorator.
Importing the handlers package triggers their registration, making them available
for dynamic instantiation by type string.

Key Classes:
    ObservationRegistry: Central registry for managing handler instances.

Usage:
    >>> from symfluence.data.observation import ObservationRegistry
    >>> handler = ObservationRegistry.get_handler('GRACE', config, logger)
    >>> raw_data = handler.acquire()
    >>> processed = handler.process(raw_data)
"""

from .registry import ObservationRegistry
from . import handlers

# Trigger registration of all observation handler plugins when this module is imported.
# This allows handlers to self-register without hardcoding imports in the registry.
# We catch ImportError for optional dependencies (e.g., if certain data sources
# require external packages that may not be installed).
try:
    from .handlers import grace, modis_snow, modis_et, usgs, wsc, snotel, soil_moisture, fluxcom, gleam, smhi, lamah_ice, ggmn, fluxnet
except ImportError:
    pass

__all__ = ["ObservationRegistry"]
