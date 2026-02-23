"""CRHM (Cold Regions Hydrological Model).

This module implements CRHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install crhm`
- Preprocessing (project file, observation/forcing file)
- Model execution (crhm binary)
- Result extraction
- Calibration support

CRHM is a physically-based, object-oriented hydrological model designed
specifically for cold-region processes. It includes modules for blowing
snow (PBSM), energy-balance snowmelt (EBSM), frozen soil infiltration,
and prairie/alpine hydrology.

Model Architecture:
    CRHM uses a module-based structure with:

    1. **Project File** (.prj): Text file defining model structure
       - Module selection and connectivity
       - Parameter values (key-value format)
       - Basin and HRU definitions

    2. **Observation File** (.obs): Text file with forcing data
       - Header with variable names and metadata
       - Space-separated meteorological data
       - Variables: temperature, precipitation, humidity, wind, radiation

    3. **Output**: CSV files with simulated variables
       - Date, flow, SWE, soil moisture, evapotranspiration

Design Rationale:
    CRHM is well-suited for:
    - Cold-region hydrology (Arctic, subarctic, alpine)
    - Blowing snow redistribution and sublimation
    - Energy-balance snowmelt processes
    - Frozen soil infiltration dynamics
    - Prairie and wetland hydrology

Key Components:
    CRHMPreProcessor: Observation file generation from ERA5 forcing
    CRHMRunner: Model execution with crhm binary
    CRHMResultExtractor: Output extraction from CSV results

Configuration Parameters:
    CRHM_INSTALL_PATH: Path to CRHM installation
    CRHM_EXE: Executable name (default: crhm)
    CRHM_PROJECT_FILE: Project file name (default: model.prj)
    CRHM_OBSERVATION_FILE: Observation file name (default: forcing.obs)
    CRHM_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'basin_area,Ht,Asnow,inhibit_evap,Ksat,soil_rechr_max,
                   soil_moist_max,soil_gw_K,Sdmax,fetch')

Typical Workflow:
    1. Prepare .obs observation/forcing file from ERA5 data
    2. Configure .prj project file with modules and parameters
    3. Run CRHM binary
    4. Extract and analyze CSV outputs

References:
    Pomeroy, J.W., et al. (2007): The Cold Regions Hydrological Model:
    a platform for basing process representation and model structure on
    physical evidence. Hydrological Processes, 21(19), 2650-2667.

    https://github.com/CentreForHydrology/CRHM
"""
from .config import CRHMConfigAdapter
from .extractor import CRHMResultExtractor
from .postprocessor import CRHMPostProcessor
from .preprocessor import CRHMPreProcessor
from .runner import CRHMRunner

__all__ = [
    "CRHMPreProcessor",
    "CRHMRunner",
    "CRHMResultExtractor",
    "CRHMPostProcessor",
    "CRHMConfigAdapter",
]

# Register all CRHM components via unified registry
from symfluence.core.registry import model_manifest

model_manifest(
    "CRHM",
    preprocessor=CRHMPreProcessor,
    runner=CRHMRunner,
    result_extractor=CRHMResultExtractor,
    config_adapter=CRHMConfigAdapter,
    build_instructions_module="symfluence.models.crhm.build_instructions",
)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import CRHMModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
