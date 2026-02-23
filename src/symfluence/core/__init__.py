"""Common utilities and core system components."""

# CoordinateUtilsMixin is in geospatial module but re-exported here for convenience
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

from .base_manager import BaseManager
from .constants import ModelDefaults, PhysicalConstants, UnitConversion
from .file_utils import copy_file, copy_tree, ensure_dir, safe_delete
from .mixins import (
    ConfigMixin,
    ConfigurableMixin,
    FileUtilsMixin,
    LoggingMixin,
    ProjectContextMixin,
    ShapefileAccessMixin,
    ValidationMixin,
)
from .path_resolver import PathResolverMixin, resolve_file_path, resolve_path

# Profiling module for IOPS diagnostics
from .profiling import IOProfiler, ProfilerContext, get_profiler, profiling_enabled

# Unified component registry
from .registries import R, Registries
from .registry import Registry, model_manifest
from .system import SYMFLUENCE
from .validation import validate_config_keys, validate_directory_exists, validate_file_exists

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
    # Unified registry
    'Registry',
    'Registries',
    'R',
    'model_manifest',
    # Profiling
    'IOProfiler',
    'ProfilerContext',
    'get_profiler',
    'profiling_enabled',
]
