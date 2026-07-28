# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Parameter-management contract: base manager and bounds registry."""
from __future__ import annotations

from .base_parameter_manager import BaseParameterManager
from .parameter_bounds_registry import (
    ParameterBoundsRegistry,
    ParameterInfo,
    bounds_registration_conflicts,
    get_depth_bounds,
    get_fuse_bounds,
    get_mizuroute_bounds,
    get_model_bounds,
    get_ngen_bounds,
    get_ngen_cfe_bounds,
    get_ngen_noah_bounds,
    get_ngen_pet_bounds,
    get_registry,
    register_model_bounds,
    registered_bound_models,
)

__all__ = [
    'BaseParameterManager',
    'ParameterBoundsRegistry',
    'ParameterInfo',
    'get_registry',
    'get_model_bounds',
    'register_model_bounds',
    'registered_bound_models',
    'bounds_registration_conflicts',
    'get_depth_bounds',
    'get_fuse_bounds',
    'get_mizuroute_bounds',
    'get_ngen_bounds',
    'get_ngen_cfe_bounds',
    'get_ngen_noah_bounds',
    'get_ngen_pet_bounds',
]
