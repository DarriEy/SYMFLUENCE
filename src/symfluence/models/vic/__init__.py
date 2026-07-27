# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""VIC (Variable Infiltration Capacity) Hydrological Model.

This module implements VIC 5.x support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install vic`
- Preprocessing (domain, parameters, forcing)
- Model execution (image driver)
- Result extraction
- Calibration support

VIC is a large-scale, semi-distributed hydrological model that solves
full water and energy balances. It is typically applied to large river
basins using gridded forcing data.

Model Architecture:
    VIC uses a grid-based structure with:

    1. **Domain File**: NetCDF file defining the model grid
       - Grid mask, cell area, fractional coverage
       - Latitude/longitude coordinates

    2. **Parameter File**: NetCDF file with soil and vegetation parameters
       - Soil parameters (infilt, Ds, Dsmax, Ws, soil depth)
       - Vegetation parameters (from MODIS or similar)

    3. **Forcing Files**: NetCDF files with meteorological forcing
       - Precipitation, temperature, wind, humidity, etc.

    4. **Global Parameter File**: Text file with model settings
       - File paths, simulation dates, output options

Design Rationale:
    VIC is well-suited for:
    - Large-scale water balance studies
    - Land surface-atmosphere interactions
    - Grid-based distributed modeling
    - Studies requiring full energy balance

Key Components:
    VICPreProcessor: Domain, parameter, and forcing file generation
    VICRunner: Model execution with image driver
    VICResultExtractor: Output extraction and analysis

Configuration Parameters:
    VIC_INSTALL_PATH: Path to VIC installation
    VIC_EXE: Executable name (default: vic_image.exe)
    VIC_DRIVER: Driver type ('image' or 'classic')
    VIC_SPATIAL_MODE: 'lumped' or 'distributed'
    VIC_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'infilt,Ds,Dsmax,Ws,depth1,depth2,depth3')

Typical Workflow:
    1. Define domain grid (from catchment shapefile or DEM)
    2. Generate parameter file with soil/veg properties
    3. Prepare forcing data in VIC NetCDF format
    4. Create global parameter file
    5. Run VIC image driver
    6. Extract and analyze outputs

References:
    Liang, X., D. P. Lettenmaier, E. F. Wood, and S. J. Burges, 1994:
    A simple hydrologically based model of land surface water and energy
    fluxes for general circulation models. J. Geophys. Res., 99(D7), 14415-14428.

    https://github.com/UW-Hydro/VIC
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'VICPreProcessor': ('.preprocessor', 'VICPreProcessor'),
    'VICRunner': ('.runner', 'VICRunner'),
    'VICResultExtractor': ('.extractor', 'VICResultExtractor'),
    'VICPostProcessor': ('.postprocessor', 'VICPostProcessor'),
    'VICModelOptimizer': ('.calibration', 'VICModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for VIC module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['VICConfigAdapter'])


__all__ = [
    "VICPreProcessor",
    "VICRunner",
    "VICResultExtractor",
    "VICPostProcessor",
    "VICConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import VICConfigAdapter


def register() -> None:
    """Register VIC components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "VIC",
        config_adapter=VICConfigAdapter,
        build_instructions_module="symfluence.models.vic.build_instructions",
    )
    base = 'symfluence.models.vic'
    R.preprocessors.add_lazy("VIC", f"{base}.preprocessor.VICPreProcessor")
    R.runners.add_lazy("VIC", f"{base}.runner.VICRunner")
    R.postprocessors.add_lazy("VIC", f"{base}.postprocessor.VICPostProcessor")
    R.result_extractors.add_lazy("VIC", f"{base}.extractor.VICResultExtractor")
    R.optimizers.add_lazy("VIC", f"{base}.calibration.optimizer.VICModelOptimizer")
    R.workers.add_lazy("VIC", f"{base}.calibration.worker.VICWorker")
    R.parameter_managers.add_lazy("VIC", f"{base}.calibration.parameter_manager.VICParameterManager")
    # Calibration bounds are owned by this package (service-decomposition
    # item 2): registering here means plugin discovery is what makes them
    # servable, so a bound change never needs a core release.
    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .calibration import VICModelOptimizer
    from .extractor import VICResultExtractor
    from .postprocessor import VICPostProcessor
    from .preprocessor import VICPreProcessor
    from .runner import VICRunner
