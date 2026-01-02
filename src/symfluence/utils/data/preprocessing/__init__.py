"""
Data Preprocessing utilities for SYMFLUENCE.

This module provides:
- ForcingResampler: Orchestrates forcing data remapping
- ShapefileManager: CRS alignment and HRU ID management
- ElevationCalculator: DEM-based elevation statistics
- RemappingWeightGenerator/Applier: EASYMORE weight operations
- GeospatialStatistics: Catchment attribute computation
"""

from .forcing_resampler import ForcingResampler
from .geospatial_statistics import GeospatialStatistics
from .shapefile_manager import ShapefileManager
from .elevation_calculator import ElevationCalculator, create_elevation_calculator
from .remapping_weights import (
    RemappingWeightGenerator,
    RemappingWeightApplier,
    BatchProcessor,
)

__all__ = [
    'ForcingResampler',
    'GeospatialStatistics',
    'ShapefileManager',
    'ElevationCalculator',
    'create_elevation_calculator',
    'RemappingWeightGenerator',
    'RemappingWeightApplier',
    'BatchProcessor',
]
