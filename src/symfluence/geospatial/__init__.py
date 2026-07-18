# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geospatial processing utilities.

This module provides domain delineation, raster processing, and coordinate
utilities for SYMFLUENCE watershed modeling workflows.

Key Components:
    - DomainDelineator: Orchestrator for domain delineation
    - DelineationRegistry: Registry for delineation strategies
    - DelineationArtifacts: Tracking of delineation outputs
    - Exceptions: Geospatial-specific error types
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562): the delineation machinery pulls the full raster
# stack (~0.9 s), which must not load at `import symfluence`. Delineation
# strategies still self-register reliably — the registry bootstrap installs a
# seeder on R.delineation_strategies that imports this machinery on first
# strategy lookup.
_LAZY_IMPORTS = {
    'DelineationArtifacts': ('symfluence.geospatial.delineation', 'DelineationArtifacts'),
    'DomainDelineator': ('symfluence.geospatial.delineation', 'DomainDelineator'),
    'create_point_domain_shapefile': ('symfluence.geospatial.delineation', 'create_point_domain_shapefile'),
    'DelineationResult': ('symfluence.geospatial.delineation_protocol', 'DelineationResult'),
    'DelineationStrategy': ('symfluence.geospatial.delineation_protocol', 'DelineationStrategy'),
    'DelineationRegistry': ('symfluence.geospatial.delineation_registry', 'DelineationRegistry'),
    'DelineationError': ('symfluence.geospatial.exceptions', 'DelineationError'),
    'GeospatialError': ('symfluence.geospatial.exceptions', 'GeospatialError'),
    'GridCreationError': ('symfluence.geospatial.exceptions', 'GridCreationError'),
    'RasterError': ('symfluence.geospatial.exceptions', 'RasterError'),
    'ShapefileError': ('symfluence.geospatial.exceptions', 'ShapefileError'),
    'SubsettingError': ('symfluence.geospatial.exceptions', 'SubsettingError'),
    'TauDEMError': ('symfluence.geospatial.exceptions', 'TauDEMError'),
    'TopologyError': ('symfluence.geospatial.exceptions', 'TopologyError'),
    'geospatial_error_handler': ('symfluence.geospatial.exceptions', 'geospatial_error_handler'),
}


def __getattr__(name: str):
    """Lazy import handler for geospatial re-exports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        value = getattr(import_module(module_path), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY_IMPORTS.keys())


if TYPE_CHECKING:
    from symfluence.geospatial.delineation import (
        DelineationArtifacts,
        DomainDelineator,
        create_point_domain_shapefile,
    )
    from symfluence.geospatial.delineation_protocol import (
        DelineationResult,
        DelineationStrategy,
    )
    from symfluence.geospatial.delineation_registry import DelineationRegistry
    from symfluence.geospatial.exceptions import (
        DelineationError,
        GeospatialError,
        GridCreationError,
        RasterError,
        ShapefileError,
        SubsettingError,
        TauDEMError,
        TopologyError,
        geospatial_error_handler,
    )

__all__ = [
    # Main orchestrator
    'DomainDelineator',
    'DelineationArtifacts',
    'create_point_domain_shapefile',
    # Registry and protocol
    'DelineationRegistry',
    'DelineationResult',
    'DelineationStrategy',
    # Exceptions
    'GeospatialError',
    'DelineationError',
    'TauDEMError',
    'GridCreationError',
    'SubsettingError',
    'ShapefileError',
    'RasterError',
    'TopologyError',
    'geospatial_error_handler',
]
