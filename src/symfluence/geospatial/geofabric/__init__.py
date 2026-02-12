"""
Geofabric delineation module.

This module provides utilities for geofabric delineation and processing in SYMFLUENCE.
It includes classes for distributed delineation, coastal watersheds, subsetting, and
lumped watershed delineation.

The module has been refactored from a single large file (geofabric_utils.py, 2,246 lines)
into a modular package structure for better maintainability and organization.

Components:
    - GeofabricDelineator: Main distributed delineation (distributed_delineator.py)
    - GeofabricSubsetter: Subset existing geofabric data (subsetter.py)
    - LumpedWatershedDelineator: Lumped watershed delineation (lumped_delineator.py)

For backward compatibility, all classes are re-exported at the package level,
so existing code using `from symfluence.geospatial.geofabric_utils import ...`
can be updated to `from symfluence.geospatial.geofabric import ...`.

Refactored: 2026-01-01
"""

# Import all classes from their respective modules
from .delineators.distributed_delineator import GeofabricDelineator
from .delineators.subsetter import GeofabricSubsetter
from .delineators.lumped_delineator import LumpedWatershedDelineator
from .delineators.point_delineator import PointDelineator
from .delineators.grid_delineator import GridDelineator

__all__ = [
    'GeofabricDelineator',
    'GeofabricSubsetter',
    'LumpedWatershedDelineator',
    'PointDelineator',
    'GridDelineator',
]
