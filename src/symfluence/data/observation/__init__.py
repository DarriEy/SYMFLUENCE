# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Observation data acquisition and processing module.

This module provides handlers for acquiring observational data from various sources
(GRACE, MODIS, USGS, etc.) and processing them into SYMFLUENCE-standard formats.

The module uses a plugin-style registry pattern: observation handlers are defined
in separate modules and self-register using the @ObservationRegistry.register() decorator.
Importing the handlers package triggers their registration, making them available
for dynamic instantiation by type string.

Key Classes:
    ObservationRegistry: Central registry for managing handler instances.
    BaseObservationHandler: Abstract base class for all handlers.
    ObservationMetadata: Dataclass for standardized output metadata.

Exceptions:
    ObservationError: Base exception for observation handling errors.
    ObservationAcquisitionError: Error during data acquisition.
    ObservationProcessingError: Error during data processing.
    ObservationValidationError: Error during data validation.

Usage:
    >>> from symfluence.data.observation import ObservationRegistry
    >>> handler = ObservationRegistry.get_handler('grace', config, logger)
    >>> raw_data = handler.acquire()
    >>> processed = handler.process(raw_data)
"""

from . import handlers
from .base import (
    STANDARD_COLUMNS,
    BaseObservationHandler,
    ObservationAcquisitionError,
    ObservationError,
    ObservationMetadata,
    ObservationProcessingError,
    ObservationValidationError,
)
from .registry import ObservationRegistry

# Trigger registration of all observation handler plugins when this module is imported.
# This allows handlers to self-register without hardcoding imports in the registry.
# We catch ImportError for optional dependencies (e.g., if certain data sources
# require external packages that may not be installed).
try:
    from .handlers import (
        fluxcom,
        fluxnet,
        ggmn,
        gleam,
        grace,
        lamah_ice,
        modis_et,
        modis_snow,
        smhi,
        snotel,
        soil_moisture,
        usgs,
        wsc,
    )
except ImportError:
    pass

__all__ = [
    # Registry
    "ObservationRegistry",
    # Base class
    "BaseObservationHandler",
    # Data contract
    "ObservationMetadata",
    "STANDARD_COLUMNS",
    # Exceptions
    "ObservationError",
    "ObservationAcquisitionError",
    "ObservationProcessingError",
    "ObservationValidationError",
]
