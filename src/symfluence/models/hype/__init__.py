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

from .config_manager import HYPEConfigManager
from .forcing_processor import HYPEForcingProcessor
from .geodata_manager import HYPEGeoDataManager
from .postprocessor import HYPEPostProcessor
from .preprocessor import HYPEPreProcessor
from .runner import HYPERunner
from .visualizer import visualize_hype

__all__ = [
    'HYPEPreProcessor',
    'HYPERunner',
    'HYPEPostProcessor',
    'visualize_hype',
    'HYPEForcingProcessor',
    'HYPEConfigManager',
    'HYPEGeoDataManager',
]

# Register all HYPE components via unified registry
from symfluence.core.registry import model_manifest

from .config import HYPEConfigAdapter
from .extractor import HYPEResultExtractor
from .plotter import HYPEPlotter

model_manifest(
    "HYPE",
    config_adapter=HYPEConfigAdapter,
    result_extractor=HYPEResultExtractor,
    plotter=HYPEPlotter,
    build_instructions_module="symfluence.models.hype.build_instructions",
)
