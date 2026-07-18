# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WMFire model module for SYMFLUENCE.

WMFire is a wildfire spread model designed to be coupled with RHESSys.
It simulates fire spread based on:
- Litter load
- Relative moisture deficit
- Wind direction
- Topographic slope

This module provides:
- FireGrid, FireGridManager: Georeferenced grid management
- FuelCalculator, FuelMoistureModel: Fuel load and moisture calculations
- FireDefGenerator: Dynamic fire.def parameter generation

Reference:
Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. 2017.
Balancing uncertainty and complexity to incorporate fire spread in
an eco-hydrological model. International Journal of Wildland Fire. 26(8): 706-718.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — the fire grid/fuel/postprocessing classes pull the
# geospatial stack and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'FireDefGenerator': ('.fire_def_generator', 'FireDefGenerator'),
    'FireDefParameters': ('.fire_def_generator', 'FireDefParameters'),
    'validate_fire_def': ('.fire_def_generator', 'validate_fire_def'),
    'FireGrid': ('.fire_grid', 'FireGrid'),
    'FireGridManager': ('.fire_grid', 'FireGridManager'),
    'FuelCalculator': ('.fuel_calculator', 'FuelCalculator'),
    'FuelMoistureModel': ('.fuel_calculator', 'FuelMoistureModel'),
    'FuelStats': ('.fuel_calculator', 'FuelStats'),
    'estimate_initial_moisture': ('.fuel_calculator', 'estimate_initial_moisture'),
    'FirePerimeterValidator': ('.ignition', 'FirePerimeterValidator'),
    'IgnitionManager': ('.ignition', 'IgnitionManager'),
    'IgnitionPoint': ('.ignition', 'IgnitionPoint'),
    'WMFirePostProcessor': ('.postprocessor', 'WMFirePostProcessor'),
}


def __getattr__(name: str):
    """Lazy import handler for WMFire module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY_IMPORTS.keys())


def register() -> None:
    """Register WMFire components with the unified registry.

    The postprocessor is registered lazily — imported on first registry
    access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    R.postprocessors.add_lazy(
        'WMFire', 'symfluence.models.wmfire.postprocessor.WMFirePostProcessor'
    )
    # Import build instructions to register with BuildInstructionsRegistry
    try:
        from . import build_instructions  # noqa: F401
    except ImportError:
        pass


if TYPE_CHECKING:
    from .fire_def_generator import (
        FireDefGenerator,
        FireDefParameters,
        validate_fire_def,
    )
    from .fire_grid import FireGrid, FireGridManager
    from .fuel_calculator import (
        FuelCalculator,
        FuelMoistureModel,
        FuelStats,
        estimate_initial_moisture,
    )
    from .ignition import (
        FirePerimeterValidator,
        IgnitionManager,
        IgnitionPoint,
    )
    from .postprocessor import WMFirePostProcessor

__all__ = [
    # Grid management
    'FireGrid',
    'FireGridManager',
    # Fuel calculations
    'FuelCalculator',
    'FuelMoistureModel',
    'FuelStats',
    'estimate_initial_moisture',
    # Fire definition
    'FireDefGenerator',
    'FireDefParameters',
    'validate_fire_def',
    # Ignition and perimeter
    'IgnitionPoint',
    'IgnitionManager',
    'FirePerimeterValidator',
    # Postprocessor
    'WMFirePostProcessor',
]
