"""MIKE-SHE (DHI) Integrated Physics-Based Hydrological Model.

This module implements MIKE-SHE support for SYMFLUENCE, including:
- Preprocessing (ERA5 forcing conversion, .she XML setup generation)
- Model execution (MikeSheEngine.exe, optionally via WINE on Unix)
- Result extraction (CSV / dfs0 output parsing)
- Calibration support (XML-based parameter updates)

MIKE-SHE is a fully integrated, physically-based modelling system
developed by DHI (Danish Hydraulic Institute) that simulates:
- Overland flow (2D diffusive wave with Manning's equation)
- Unsaturated zone flow (Richards' equation or simplified)
- Saturated zone flow (3D finite difference groundwater model)
- Channel flow (MIKE 11 coupling)
- Evapotranspiration (Kristensen & Jensen or Penman-Monteith)
- Snow accumulation and melt (degree-day method)

Model Architecture:
    MIKE-SHE uses a grid-based structure with:

    1. **.she Setup File**: XML file defining all model components
       - Simulation period, spatial discretization
       - Component configuration and parameters

    2. **Forcing Data**: Time series of meteorological inputs
       - Precipitation, temperature, potential ET

    3. **Spatial Data**: Grids for topography, soil, land use
       - Typically in DHI's .dfs2 grid format

    4. **Output**: Time series in .dfs0 format or CSV export
       - Discharge, groundwater levels, soil moisture, snow

Design Rationale:
    MIKE-SHE is well-suited for:
    - Integrated surface water / groundwater studies
    - Physically-based distributed modelling
    - Detailed process representation
    - Catchments where groundwater-surface water interactions matter

Key Components:
    MIKESHEPreProcessor: Forcing conversion and .she file generation
    MIKESHERunner: Model execution with optional WINE wrapper
    MIKESHEResultExtractor: Output extraction and analysis
    MIKESHEPostProcessor: Streamflow extraction from CSV/dfs0

Configuration Parameters:
    MIKESHE_INSTALL_PATH: Path to MIKE-SHE installation
    MIKESHE_EXE: Executable name (default: MikeSheEngine.exe)
    MIKESHE_SETUP_FILE: Setup file name (default: model.she)
    MIKESHE_USE_WINE: Use WINE on Unix (default: False)
    MIKESHE_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'manning_m,detention_storage,Ks_uz,theta_sat,
         theta_fc,theta_wp,Ks_sz_h,specific_yield,ddf,
         snow_threshold,max_canopy_storage')

Typical Workflow:
    1. Prepare forcing data (ERA5 -> CSV)
    2. Generate .she XML setup file with default parameters
    3. Run MikeSheEngine.exe (or via WINE on Linux/macOS)
    4. Extract results from CSV or dfs0 output
    5. Calibrate by modifying parameters in .she XML

Note:
    MIKE-SHE is proprietary software from DHI. A valid license is
    required. It cannot be built from source.

References:
    Graham, D.N. & Butts, M.B. (2005): Flexible, integrated watershed
    modelling with MIKE SHE. Watershed Models, 245-272.

    DHI (2017): MIKE SHE User Manual. Danish Hydraulic Institute.
"""
from .preprocessor import MIKESHEPreProcessor
from .runner import MIKESHERunner
from .extractor import MIKESHEResultExtractor
from .postprocessor import MIKESHEPostProcessor
from .config import MIKESHEConfigAdapter

__all__ = [
    "MIKESHEPreProcessor",
    "MIKESHERunner",
    "MIKESHEResultExtractor",
    "MIKESHEPostProcessor",
    "MIKESHEConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

# Register preprocessor
ModelRegistry.register_preprocessor('MIKESHE')(MIKESHEPreProcessor)

# Register runner
ModelRegistry.register_runner('MIKESHE')(MIKESHERunner)

# Register result extractor
ModelRegistry.register_result_extractor('MIKESHE')(MIKESHEResultExtractor)

# Register config adapter
ModelRegistry.register_config_adapter('MIKESHE')(MIKESHEConfigAdapter)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import MIKESHEModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
