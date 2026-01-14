"""
Data Preprocessing utilities for SYMFLUENCE.

This module provides:
- ForcingResampler: Orchestrates forcing data remapping
- ShapefileManager: CRS alignment and HRU ID management
- ElevationCalculator: DEM-based elevation statistics
- RemappingWeightGenerator/Applier: EASYMORE weight operations
- GeospatialStatistics: Catchment attribute computation
"""

import sys as _sys
from typing import Any

# Fail-safe imports to allow partial loading
try:
    from .forcing_resampler import ForcingResampler
except ImportError as _e:
    ForcingResampler: Any = None  # type: ignore
    print(f"WARNING: Failed to import ForcingResampler: {_e}", file=_sys.stderr)

try:
    from .geospatial_statistics import GeospatialStatistics
except ImportError as _e:
    GeospatialStatistics: Any = None  # type: ignore
    print(f"WARNING: Failed to import GeospatialStatistics: {_e}", file=_sys.stderr)

try:
    from .shapefile_manager import ShapefileManager
except ImportError as _e:
    ShapefileManager: Any = None  # type: ignore
    print(f"WARNING: Failed to import ShapefileManager: {_e}", file=_sys.stderr)

try:
    from .elevation_calculator import ElevationCalculator, create_elevation_calculator
except ImportError as _e:
    ElevationCalculator: Any = None  # type: ignore
    create_elevation_calculator: Any = None  # type: ignore
    print(f"WARNING: Failed to import elevation_calculator: {_e}", file=_sys.stderr)

try:
    from .remapping_weights import (
        RemappingWeightGenerator,
        RemappingWeightApplier,
        BatchProcessor,
    )
except ImportError as _e:
    RemappingWeightGenerator: Any = None  # type: ignore
    RemappingWeightApplier: Any = None  # type: ignore
    BatchProcessor: Any = None  # type: ignore
    print(f"WARNING: Failed to import remapping_weights: {_e}", file=_sys.stderr)

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
