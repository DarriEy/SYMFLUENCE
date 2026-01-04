"""Common utilities shared across SYMFLUENCE modules."""

from .path_resolver import resolve_path, resolve_file_path, PathResolverMixin
from .file_utils import ensure_dir, copy_file, copy_tree, safe_delete
from .validation import validate_config_keys, validate_file_exists, validate_directory_exists
from .geospatial_utils import (
    calculate_catchment_centroid,
    calculate_catchment_area_km2,
    calculate_feature_centroids,
    validate_and_fix_crs,
    GeospatialUtilsMixin
)
from .constants import UnitConversion, PhysicalConstants, ModelDefaults
from . import metrics
from . import coordinate_utils
from .coordinate_utils import BoundingBox, parse_bbox, normalize_longitude, CoordinateUtilsMixin
from .mixins import (
    LoggingMixin, 
    ConfigMixin, 
    ProjectContextMixin, 
    ConfigurableMixin, 
    FileUtilsMixin,
    ValidationMixin
)

__all__ = [
    'resolve_path',
    'resolve_file_path',
    'PathResolverMixin',
    'ensure_dir',
    'copy_file',
    'copy_tree',
    'safe_delete',
    'validate_config_keys',
    'validate_file_exists',
    'validate_directory_exists',
    'calculate_catchment_centroid',
    'calculate_catchment_area_km2',
    'calculate_feature_centroids',
    'validate_and_fix_crs',
    'GeospatialUtilsMixin',
    'CoordinateUtilsMixin',
    'FileUtilsMixin',
    'ValidationMixin',
    'LoggingMixin',
    'ConfigMixin',
    'ProjectContextMixin',
    'ConfigurableMixin',
    'UnitConversion',
    'PhysicalConstants',
    'ModelDefaults',
    'metrics',
    'coordinate_utils',
    'BoundingBox',
    'parse_bbox',
    'normalize_longitude',
]
