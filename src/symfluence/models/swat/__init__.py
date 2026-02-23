"""SWAT (Soil and Water Assessment Tool) Hydrological Model.

This module implements SWAT support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install swat`
- Preprocessing (TxtInOut directory with forcing, soil, management files)
- Model execution (swat_rel.exe from TxtInOut directory)
- Result extraction (output.rch fixed-width text)
- Calibration support (14 parameters across .bsn, .gw, .hru, .sol, .mgt files)

SWAT is a river basin scale model developed by the USDA Agricultural
Research Service (ARS) to predict the impact of land management practices
on water, sediment, and agricultural chemical yields in large complex
watersheds with varying soils, land use, and management conditions.

Model Architecture:
    SWAT uses a semi-distributed, HRU-based structure with:

    1. **TxtInOut Directory**: All input and output files in one directory
       - file.cio: Master control file
       - .pcp/.tmp: Precipitation and temperature forcing
       - .sub/.hru/.gw/.mgt/.sol: Sub-basin and HRU parameters
       - .bsn: Basin-level parameters

    2. **Output Files**: Fixed-width text format
       - output.rch: Reach-level results (FLOW_OUTcms, etc.)
       - output.sub: Sub-basin results
       - output.hru: HRU-level results

    3. **Calibration**: Parameters modified via text file editing
       - r__ (relative): new = original * (1 + change)
       - v__ (value replacement): new = change
       - a__ (absolute): new = original + change

Design Rationale:
    SWAT is well-suited for:
    - Agricultural watershed management
    - Water quality and sediment yield assessment
    - Land use change impact studies
    - Long-term continuous simulation

Key Components:
    SWATPreProcessor: TxtInOut directory and forcing file generation
    SWATRunner: Model execution with swat_rel.exe
    SWATResultExtractor: Output extraction from output.rch
    SWATPostProcessor: Streamflow extraction and unit handling
    SWATConfigAdapter: Configuration schema and validation

Configuration Parameters:
    SWAT_INSTALL_PATH: Path to SWAT installation
    SWAT_EXE: Executable name (default: swat_rel.exe)
    SWAT_TXTINOUT_DIR: TxtInOut directory name (default: TxtInOut)
    SWAT_SPATIAL_MODE: 'lumped' or 'semi_distributed'
    SWAT_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP')

Typical Workflow:
    1. Prepare TxtInOut directory with catchment data
    2. Generate forcing files (.pcp, .tmp) from ERA5 data
    3. Configure basin, sub-basin, HRU, groundwater, management, and soil files
    4. Create file.cio master control file
    5. Run swat_rel.exe from within TxtInOut
    6. Extract streamflow from output.rch

References:
    Arnold, J.G., Srinivasan, R., Muttiah, R.S., and Williams, J.R. (1998):
    Large area hydrologic modeling and assessment Part I: Model development.
    Journal of the American Water Resources Association, 34(1), 73-89.

    Neitsch, S.L., Arnold, J.G., Kiniry, J.R., and Williams, J.R. (2011):
    Soil and Water Assessment Tool Theoretical Documentation Version 2009.
    Texas Water Resources Institute Technical Report No. 406.

    https://github.com/WatershedModels/SWAT
"""
from .config import SWATConfigAdapter
from .extractor import SWATResultExtractor
from .postprocessor import SWATPostProcessor
from .preprocessor import SWATPreProcessor
from .runner import SWATRunner

__all__ = [
    "SWATPreProcessor",
    "SWATRunner",
    "SWATResultExtractor",
    "SWATPostProcessor",
    "SWATConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

# Register preprocessor
ModelRegistry.register_preprocessor('SWAT')(SWATPreProcessor)

# Register runner
ModelRegistry.register_runner('SWAT')(SWATRunner)

# Register result extractor
ModelRegistry.register_result_extractor('SWAT')(SWATResultExtractor)

# Register config adapter
ModelRegistry.register_config_adapter('SWAT')(SWATConfigAdapter)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import SWATModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
