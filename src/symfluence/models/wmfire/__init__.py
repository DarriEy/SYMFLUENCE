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

# Import core classes
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

# Import build instructions to register with BuildInstructionsRegistry
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass

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
