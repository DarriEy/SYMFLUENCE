"""mHM (mesoscale Hydrological Model).

This module implements mHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install mhm`
- Preprocessing (forcing, morphology, namelists)
- Model execution (Fortran binary)
- Result extraction
- Calibration support

mHM is a spatially distributed hydrological model developed at the
Helmholtz Centre for Environmental Research (UFZ). It uses Multiscale
Parameter Regionalization (MPR) for parameter transfer across scales.

Model Architecture:
    mHM uses a grid-based structure with:

    1. **Forcing Files**: NetCDF grids with meteorological data
       - Precipitation (pre), temperature (tavg), PET (pet)

    2. **Morphological Inputs**: DEM, soil, land cover grids
       - Used by MPR for parameter regionalization

    3. **Namelist Files**: Fortran namelists controlling simulation
       - mhm.nml: Main model configuration
       - mrm.nml: Routing (mRM) configuration

    4. **Output Files**: NetCDF files with results
       - discharge_*.nc: Simulated discharge [m3/s]
       - mHM_Fluxes_States_*.nc: Fluxes and states

Design Rationale:
    mHM is well-suited for:
    - Mesoscale hydrological modeling
    - Parameter regionalization studies
    - Multi-basin hydrological assessment
    - Spatially distributed process modeling

Key Components:
    MHMPreProcessor: Forcing, morphology, and namelist generation
    MHMRunner: Model execution with Fortran binary
    MHMResultExtractor: Output extraction and analysis

Configuration Parameters:
    MHM_INSTALL_PATH: Path to mHM installation
    MHM_EXE: Executable name (default: mhm)
    MHM_NAMELIST_FILE: Main namelist (default: mhm.nml)
    MHM_ROUTING_NAMELIST: Routing namelist (default: mrm.nml)
    MHM_SPATIAL_MODE: 'lumped' or 'distributed'
    MHM_PARAMS_TO_CALIBRATE: Calibration parameters

Typical Workflow:
    1. Prepare forcing data (precipitation, temperature, PET)
    2. Generate morphological inputs (DEM, soil, land cover)
    3. Create Fortran namelists (mhm.nml, mrm.nml)
    4. Run mHM binary from settings directory
    5. Extract and analyze discharge and fluxes/states

References:
    Samaniego, L., et al. (2010): Multiscale parameter regionalization
    of a grid-based hydrologic model at the mesoscale. Water Resources
    Research, 46, W05523.

    Kumar, R., et al. (2013): Toward computationally efficient large-scale
    hydrologic predictions with a multiscale regionalization scheme. Water
    Resources Research, 49, 5700-5714.

    https://git.ufz.de/mhm/mhm
"""
from .preprocessor import MHMPreProcessor
from .runner import MHMRunner
from .extractor import MHMResultExtractor
from .postprocessor import MHMPostProcessor
from .config import MHMConfigAdapter

__all__ = [
    "MHMPreProcessor",
    "MHMRunner",
    "MHMResultExtractor",
    "MHMPostProcessor",
    "MHMConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

# Register preprocessor
ModelRegistry.register_preprocessor('MHM')(MHMPreProcessor)

# Register runner
ModelRegistry.register_runner('MHM')(MHMRunner)

# Register result extractor
ModelRegistry.register_result_extractor('MHM')(MHMResultExtractor)

# Register config adapter
ModelRegistry.register_config_adapter('MHM')(MHMConfigAdapter)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import MHMModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
