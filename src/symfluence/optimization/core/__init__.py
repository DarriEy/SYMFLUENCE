"""
Core optimization utilities.

This module provides shared infrastructure for model calibration:
- BaseParameterManager: Abstract base for model-specific parameter managers
- ParameterBoundsRegistry: Centralized parameter bounds definitions
- ParameterManager: DEPRECATED - use SUMMAParameterManager from parameter_managers instead
- TransformationManager: Manages parameter transformations
- DirectoryConventionRegistry: Model-specific directory structure conventions
- ModelDirectoryConvention: Dataclass defining directory layout for a model

Note:
    The SUMMA-specific ParameterManager has been moved to:
    symfluence.optimization.parameter_managers.summa_parameter_manager

    For new code, use:
    >>> from symfluence.optimization.parameter_managers import SUMMAParameterManager
"""

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import (
    ParameterBoundsRegistry,
    get_registry,
    get_fuse_bounds,
    get_ngen_bounds,
    get_ngen_cfe_bounds,
    get_ngen_noah_bounds,
    get_ngen_pet_bounds,
    get_mizuroute_bounds,
    get_depth_bounds,
)
from symfluence.optimization.core.parameter_manager import ParameterManager
from symfluence.optimization.core.transformers import TransformationManager
from symfluence.optimization.core.directory_conventions import (
    ModelDirectoryConvention,
    DirectoryConventionRegistry,
    get_model_directories,
)

__all__ = [
    'BaseParameterManager',
    'ParameterBoundsRegistry',
    'ParameterManager',
    'TransformationManager',
    'ModelDirectoryConvention',
    'DirectoryConventionRegistry',
    'get_model_directories',
    'get_registry',
    'get_fuse_bounds',
    'get_ngen_bounds',
    'get_ngen_cfe_bounds',
    'get_ngen_noah_bounds',
    'get_ngen_pet_bounds',
    'get_mizuroute_bounds',
    'get_depth_bounds',
]
