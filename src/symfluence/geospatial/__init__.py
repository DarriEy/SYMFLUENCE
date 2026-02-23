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
