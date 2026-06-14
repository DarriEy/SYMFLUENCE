# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Observation Registry for SYMFLUENCE

Provides a central registry for observational data handlers (GRACE, MODIS, etc.).

This module implements a plugin-style registry pattern that allows observation handlers
to self-register and be dynamically instantiated by type string. This decouples handler
implementations from the core acquisition system and enables easy addition of new
data sources without modifying the registry code.

Phase 4 delegation shim: all state lives in ``R.observation_handlers``.

Example:
    Register a custom handler:

    >>> @R.observation_handlers.add('custom_sensor')
    ... class CustomHandler(BaseObservationHandler):
    ...     def acquire(self): ...
    ...     def process(self, input_path): ...

    Get a handler instance:

    >>> handler = ObservationRegistry.get_handler('custom_sensor', config, logger)
    >>> raw_data = handler.acquire()
    >>> processed = handler.process(raw_data)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from symfluence.data.base_registry import HandlerRegistry

if TYPE_CHECKING:
    from symfluence.data.observation.base import BaseObservationHandler  # noqa: F401


class ObservationRegistry(HandlerRegistry["BaseObservationHandler"]):
    """Plugin registry for observation data handlers.

    Inherits from HandlerRegistry which provides:
    - get_handler(name, config, logger)
    - is_registered(name)
    - list_handlers()
    - clear() for testing

    All keys are automatically normalized to lowercase for consistency.
    Lookups are case-insensitive (e.g., 'GRACE' and 'grace' both work).
    Registration goes through ``R.observation_handlers.add()``.
    """

    _r_registry_name = "observation_handlers"

    @classmethod
    def list_observations(cls) -> list:
        """Get sorted list of all registered observation types.

        This is an alias for list_handlers() for backward compatibility.

        Returns:
            list: Registered observation type strings, sorted alphabetically.

        Example:
            >>> ObservationRegistry.list_observations()
            ['gleam_et', 'grace', 'modis_et', 'modis_snow', 'usgs_streamflow']
        """
        return cls.list_handlers()
