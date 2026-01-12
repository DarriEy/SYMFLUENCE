"""HYPE (HYdrological Predictions for the Environment) Model.

This module implements the HYPE semi-distributed process-based hydrological model
developed by SMHI (Swedish Meteorological and Hydrological Institute). HYPE is
designed for large-scale operational hydrological prediction and has been applied
from catchment to continental scales (e.g., E-HYPE covering all of Europe).

Model Architecture:
    1. **Spatial Discretization**: Subbasins containing Soil-Land Classes (SLCs)
       that combine soil type and land use for parameter regionalization
    2. **Snow Processes**: Degree-day snowmelt with liquid water refreezing
    3. **Soil Moisture**: Multi-layer soil model with infiltration and percolation
    4. **Evapotranspiration**: Penman-Monteith or simpler temperature-based methods
    5. **Groundwater**: Upper and lower groundwater boxes with regional flow
    6. **Routing**: Internal subbasin routing with river delay and dampening

Design Rationale:
    HYPE addresses large-scale operational prediction needs:
    - SLC-based parameterization enables parameter transfer to ungauged basins
    - Process-based structure supports scenario analysis (land use, climate)
    - Proven operational use in national flood forecasting services
    - Supports multiple output types (water balance, nutrients, loads)

Spatial Structure:
    - Subbasins: Hydrological response units for routing
    - SLCs: Soil-land class combinations within each subbasin
    - Outlets: Defined pour points for streamflow comparison

Key Components:
    HYPEPreProcessor: Orchestrates preprocessing pipeline
    HYPERunner: Model execution and simulation management
    HYPEPostProcessor: Output extraction and analysis
    HYPEForcingProcessor: Forcing data conversion (hourly to daily aggregation)
    HYPEConfigManager: Configuration file generation (info.txt, par.txt, filedir.txt)
    HYPEGeoDataManager: Geographic data files (GeoData.txt, GeoClass.txt, ForcKey.txt)

Configuration Parameters:
    HYPE_SPINUP_DAYS: Model spinup period in days (default: 365)
    SETTINGS_HYPE_INFO: Info file name (default: 'info.txt')
    HYPE_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp')
        ttmp: Temperature threshold for snow/rain
        cmlt: Degree-day snowmelt factor
        cevp: Evapotranspiration coefficient
        lp: Soil moisture threshold for ET reduction
        rrcs1/rrcs2: Recession coefficients for upper/lower response
        rcgrw: Regional groundwater flow coefficient
        rivvel: River routing velocity
        damp: River routing dampening

Typical Workflow:
    1. Initialize HYPEPreProcessor with configuration
    2. Process forcing data via HYPEForcingProcessor (temporal aggregation)
    3. Generate geographic data files via HYPEGeoDataManager
    4. Create configuration files via HYPEConfigManager
    5. Execute HYPE via HYPERunner
    6. Extract results via HYPEPostProcessor

Limitations and Considerations:
    - Requires HYPE executable (compiled from source or from SMHI)
    - SLC delineation requires soil and land use spatial data
    - Daily timestep is standard; sub-daily requires special configuration
    - Spinup period needed to initialize soil moisture and groundwater states
"""

from .preprocessor import HYPEPreProcessor
from .runner import HYPERunner
from .postprocessor import HYPEPostProcessor
from .visualizer import visualize_hype
from .forcing_processor import HYPEForcingProcessor
from .config_manager import HYPEConfigManager
from .geodata_manager import HYPEGeoDataManager

__all__ = [
    'HYPEPreProcessor',
    'HYPERunner',
    'HYPEPostProcessor',
    'visualize_hype',
    'HYPEForcingProcessor',
    'HYPEConfigManager',
    'HYPEGeoDataManager',
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
from .config import HYPEConfigAdapter
ModelRegistry.register_config_adapter('HYPE')(HYPEConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import HYPEResultExtractor
ModelRegistry.register_result_extractor('HYPE')(HYPEResultExtractor)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import HYPEPlotter  # noqa: F401
