"""
Data utilities for SYMFLUENCE.

Provides common utilities for data processing:
- Spatial operations (cropping, subsetting, masking)
- Variable standardization and unit conversion
- Archive utilities
"""

import sys as _sys

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
    crop_raster_to_bbox = None
    read_raster_window = None
    read_raster_multiband_window = None
    create_spatial_mask = None
    subset_xarray_to_bbox = None
    normalize_longitude = None
    validate_bbox = None
    SpatialSubsetMixin = None
    BBox = None

# Variable utilities
try:
    from .variable_utils import VariableHandler, VariableStandardizer
except ImportError as e:
    print(f"WARNING: Failed to import variable_utils: {e}", file=_sys.stderr)
    VariableHandler = None
    VariableStandardizer = None

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
