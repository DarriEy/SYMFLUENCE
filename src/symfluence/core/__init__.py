"""Common utilities and core system components."""

from .base_manager import BaseManager
from .path_resolver import resolve_path, resolve_file_path, PathResolverMixin
from .file_utils import ensure_dir, copy_file, copy_tree, safe_delete
from .validation import validate_config_keys, validate_file_exists, validate_directory_exists
from .constants import UnitConversion, PhysicalConstants, ModelDefaults
from .mixins import (
    LoggingMixin,
    ConfigMixin,
    ProjectContextMixin,
    ConfigurableMixin,
    FileUtilsMixin,
    ValidationMixin,
    ShapefileAccessMixin
)
# CoordinateUtilsMixin is in geospatial module but re-exported here for convenience
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

from .system import SYMFLUENCE

__all__ = [
    'SYMFLUENCE',
    'BaseManager',
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
    'FileUtilsMixin',
    'ValidationMixin',
    'LoggingMixin',
    'ConfigMixin',
    'ProjectContextMixin',
    'ConfigurableMixin',
    'ShapefileAccessMixin',
    'CoordinateUtilsMixin',
    'UnitConversion',
    'PhysicalConstants',
    'ModelDefaults',
]