"""WRF-Hydro (NCAR) Coupled Atmosphere-Hydrology Model.

This module implements WRF-Hydro support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install wrfhydro`
- Preprocessing (HRLDAS namelist, hydro namelist, geogrid, routing files)
- Model execution (wrf_hydro.exe with NoahMP LSM)
- Result extraction (CHRTOUT streamflow, LDASOUT land surface variables)
- Calibration support (8 parameters: REFKDT, SLOPE, OVROUGHRTFAC, etc.)

WRF-Hydro is NCAR's community hydrological modeling system and forms the
backbone of the US National Water Model. It couples the Noah-MP land surface
model with terrain-following routing for distributed hydrological simulation.

Model Architecture:
    WRF-Hydro uses a coupled LSM + routing structure with:

    1. **HRLDAS Namelist**: Controls the Noah-MP land surface model
       - Simulation timing, output frequency, restart options
       - Physics options for radiation, PBL, microphysics

    2. **Hydro Namelist**: Controls the hydrological routing
       - Channel routing (Muskingum-Cunge or diffusive wave)
       - Overland flow routing (gridded or reach-based)
       - Subsurface routing options

    3. **Geogrid/WRFinput Files**: Static domain definition
       - Terrain, land use, soil type grids
       - Channel network (Fulldom_hires.nc)

    4. **Forcing Files**: LDASIN meteorological forcing
       - Precipitation, temperature, radiation, humidity, wind

    5. **Output Files**: NetCDF time series
       - CHRTOUT: Channel discharge at reach points
       - LDASOUT: Land surface fluxes (ET, soil moisture, SWE)

Design Rationale:
    WRF-Hydro is well-suited for:
    - National/continental-scale water prediction
    - Coupled atmosphere-hydrology studies
    - Flood forecasting with distributed routing
    - Multi-physics land surface modeling

Key Components:
    WRFHydroPreProcessor: HRLDAS/hydro namelist and forcing generation
    WRFHydroRunner: Model execution with wrf_hydro.exe
    WRFHydroResultExtractor: CHRTOUT/LDASOUT extraction
    WRFHydroPostProcessor: Streamflow extraction and unit handling
    WRFHydroConfigAdapter: Configuration schema and validation

Configuration Parameters:
    WRFHYDRO_INSTALL_PATH: Path to WRF-Hydro installation
    WRFHYDRO_EXE: Executable name (default: wrf_hydro.exe)
    WRFHYDRO_SPATIAL_MODE: 'distributed' (default)
    WRFHYDRO_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'REFKDT,SLOPE,OVROUGHRTFAC,RETDEPRTFAC,LKSATFAC,BEXP,DKSAT,SMCMAX')

Typical Workflow:
    1. Define domain from geogrid and Fulldom routing files
    2. Generate HRLDAS and hydro namelists
    3. Prepare LDASIN forcing files from ERA5 data
    4. Run wrf_hydro.exe (or wrf_hydro_NoahMP.exe)
    5. Extract streamflow from CHRTOUT and fluxes from LDASOUT

References:
    Gochis, D.J., et al. (2020): The WRF-Hydro modeling system technical
    description, (Version 5.1.1). NCAR Technical Note.

    https://github.com/NCAR/wrf_hydro_nwm_public
"""
from .preprocessor import WRFHydroPreProcessor
from .runner import WRFHydroRunner
from .extractor import WRFHydroResultExtractor
from .postprocessor import WRFHydroPostProcessor
from .config import WRFHydroConfigAdapter

__all__ = [
    "WRFHydroPreProcessor",
    "WRFHydroRunner",
    "WRFHydroResultExtractor",
    "WRFHydroPostProcessor",
    "WRFHydroConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

# Register preprocessor
ModelRegistry.register_preprocessor('WRFHYDRO')(WRFHydroPreProcessor)

# Register runner
ModelRegistry.register_runner('WRFHYDRO')(WRFHydroRunner)

# Register result extractor
ModelRegistry.register_result_extractor('WRFHYDRO')(WRFHydroResultExtractor)

# Register config adapter
ModelRegistry.register_config_adapter('WRFHYDRO')(WRFHydroConfigAdapter)

# Register calibration components
try:
    from .calibration import WRFHydroModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
