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
from .config import VICConfigAdapter
from .extractor import VICResultExtractor
from .postprocessor import VICPostProcessor
from .preprocessor import VICPreProcessor
from .runner import VICRunner

__all__ = [
    "VICPreProcessor",
    "VICRunner",
    "VICResultExtractor",
    "VICPostProcessor",
    "VICConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

# Register preprocessor
ModelRegistry.register_preprocessor('VIC')(VICPreProcessor)

# Register runner
ModelRegistry.register_runner('VIC')(VICRunner)

# Register result extractor
ModelRegistry.register_result_extractor('VIC')(VICResultExtractor)

# Register config adapter
ModelRegistry.register_config_adapter('VIC')(VICConfigAdapter)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import VICModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
