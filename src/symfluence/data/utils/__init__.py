"""
Data utilities for SYMFLUENCE.

Provides common utilities for data processing:
- Spatial operations (cropping, subsetting, masking)
- Variable standardization and unit conversion
- Archive utilities
"""

import logging as _logging
from typing import Any

_logger = _logging.getLogger(__name__)

# Fail-safe imports to prevent package loading failures in CI
# If an import fails, we set None and log the error

# Spatial utilities
try:
    from .spatial_utils import (
        crop_raster_to_bbox,
        read_raster_window,
        read_raster_multiband_window,
        create_spatial_mask,
        find_nearest_grid_point,
        get_bbox_center,
        subset_xarray_to_bbox,
        normalize_longitude,
        validate_bbox,
        SpatialSubsetMixin,
        BBox,
    )
except ImportError as e:
    _logger.warning("Failed to import spatial_utils: %s", e)
    crop_raster_to_bbox: Any = None  # type: ignore
    read_raster_window: Any = None  # type: ignore
    read_raster_multiband_window: Any = None  # type: ignore
    create_spatial_mask: Any = None  # type: ignore
    find_nearest_grid_point: Any = None  # type: ignore
    get_bbox_center: Any = None  # type: ignore
    subset_xarray_to_bbox: Any = None  # type: ignore
    normalize_longitude: Any = None  # type: ignore
    validate_bbox: Any = None  # type: ignore
    SpatialSubsetMixin: Any = None  # type: ignore
    BBox: Any = None  # type: ignore

# Variable utilities
try:
    from .variable_utils import VariableHandler, VariableStandardizer
except ImportError as e:
    _logger.warning("Failed to import variable_utils: %s", e)
    VariableHandler: Any = None  # type: ignore
    VariableStandardizer: Any = None  # type: ignore

__all__ = [
    # Spatial utilities
    'crop_raster_to_bbox',
    'read_raster_window',
    'read_raster_multiband_window',
    'create_spatial_mask',
    'find_nearest_grid_point',
    'get_bbox_center',
    'subset_xarray_to_bbox',
    'normalize_longitude',
    'validate_bbox',
    'SpatialSubsetMixin',
    'BBox',
    # Variable utilities
    'VariableHandler',
    'VariableStandardizer',
]
