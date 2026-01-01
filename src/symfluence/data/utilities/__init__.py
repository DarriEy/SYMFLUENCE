"""
Data utilities for SYMFLUENCE.

Provides common utilities for data processing:
- Spatial operations (cropping, subsetting, masking)
- Variable standardization and unit conversion
- Archive utilities
"""

from .spatial_utils import (
    # Functions
    crop_raster_to_bbox,
    read_raster_window,
    read_raster_multiband_window,
    create_spatial_mask,
    subset_xarray_to_bbox,
    normalize_longitude,
    validate_bbox,
    # Mixin
    SpatialSubsetMixin,
    # Type
    BBox,
)

from .variable_utils import VariableHandler, VariableStandardizer

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
