"""
Core optimization utilities.

This module provides shared infrastructure for model calibration:
- BaseParameterManager: Abstract base for model-specific parameter managers
- ParameterBoundsRegistry: Centralized parameter bounds definitions
- ParameterManager: SUMMA-specific parameter manager
"""

from symfluence.utils.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.utils.optimization.core.parameter_bounds_registry import (
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
from symfluence.utils.optimization.core.parameter_manager import ParameterManager

__all__ = [
    'BaseParameterManager',
    'ParameterBoundsRegistry',
    'ParameterManager',
    'get_registry',
    'get_fuse_bounds',
    'get_ngen_bounds',
    'get_ngen_cfe_bounds',
    'get_ngen_noah_bounds',
    'get_ngen_pet_bounds',
    'get_mizuroute_bounds',
    'get_depth_bounds',
]
