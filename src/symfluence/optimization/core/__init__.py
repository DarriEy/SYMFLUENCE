# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Core optimization utilities.

This module provides shared infrastructure for model calibration:
- BaseParameterManager: Abstract base for model-specific parameter managers
- ParameterBoundsRegistry: Centralized parameter bounds definitions
- TransformationManager: Manages parameter transformations
- DirectoryConventionRegistry: Model-specific directory structure conventions
- ModelDirectoryConvention: Dataclass defining directory layout for a model

Note:
    Model-specific parameter managers live under
    symfluence.optimization.parameter_managers, e.g.:
    >>> from symfluence.optimization.parameter_managers import SUMMAParameterManager
"""
from __future__ import annotations

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.directory_conventions import (
    DirectoryConventionRegistry,
    ModelDirectoryConvention,
    get_model_directories,
)
from symfluence.optimization.core.parameter_bounds_registry import (
    ParameterBoundsRegistry,
    get_depth_bounds,
    get_fuse_bounds,
    get_mizuroute_bounds,
    get_ngen_bounds,
    get_ngen_cfe_bounds,
    get_ngen_noah_bounds,
    get_ngen_pet_bounds,
    get_registry,
)
from symfluence.optimization.core.transformers import TransformationManager

__all__ = [
    'BaseParameterManager',
    'ParameterBoundsRegistry',
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
