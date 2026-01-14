"""
Data utilities for SYMFLUENCE.

Provides common utilities for data processing:
- Spatial operations (cropping, subsetting, masking)
- Variable standardization and unit conversion
- Archive utilities
"""

import sys as _sys
from typing import Any

# Fail-safe imports to prevent package loading failures in CI
# If an import fails, we set None and log the error

# Spatial utilities
try:
    from .spatial_utils import (
        crop_raster_to_bbox,
        read_raster_window,
        read_raster_multiband_window,
        create_spatial_mask,
        subset_xarray_to_bbox,
        normalize_longitude,
        validate_bbox,
        SpatialSubsetMixin,
        BBox,
    )
except ImportError as e:
    print(f"WARNING: Failed to import spatial_utils: {e}", file=_sys.stderr)
    crop_raster_to_bbox: Any = None  # type: ignore
    read_raster_window: Any = None  # type: ignore
    read_raster_multiband_window: Any = None  # type: ignore
    create_spatial_mask: Any = None  # type: ignore
    subset_xarray_to_bbox: Any = None  # type: ignore
    normalize_longitude: Any = None  # type: ignore
    validate_bbox: Any = None  # type: ignore
    SpatialSubsetMixin: Any = None  # type: ignore
    BBox: Any = None  # type: ignore

# Variable utilities
try:
    from .variable_utils import VariableHandler, VariableStandardizer
except ImportError as e:
    print(f"WARNING: Failed to import variable_utils: {e}", file=_sys.stderr)
    VariableHandler: Any = None  # type: ignore
    VariableStandardizer: Any = None  # type: ignore

__all__ = [
    # Spatial utilities
    'crop_raster_to_bbox',
    'read_raster_window',
    'read_raster_multiband_window',
    'create_spatial_mask',
    'subset_xarray_to_bbox',
    'normalize_longitude',
    'validate_bbox',
    'SpatialSubsetMixin',
    'BBox',
    # Variable utilities
    'VariableHandler',
    'VariableStandardizer',
]
