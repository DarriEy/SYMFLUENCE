"""RHESSys (Regional Hydro-Ecologic Simulation System) Ecohydrological Model.

This module implements RHESSys, a spatially distributed, process-based model
that couples hydrological, carbon, and nitrogen cycling. RHESSys is designed
for watershed-scale analysis of eco-hydrological interactions, particularly
in forested and mountainous environments.

Model Architecture:
    RHESSys uses a hierarchical spatial structure:

    1. **Spatial Hierarchy**:
       - Basin: Top-level container for the watershed
       - Hillslope: Drainage area to a stream segment
       - Zone: Climate zone (elevation band)
       - Patch: Fundamental hydrological unit (soil column)
       - Stratum: Vegetation layer within a patch

    2. **Hydrological Processes**:
       - Snow accumulation and melt (energy balance or degree-day)
       - Canopy interception and throughfall
       - Infiltration and vertical drainage
       - Lateral subsurface flow (topographic index based)
       - Groundwater recharge and baseflow

    3. **Biogeochemical Processes**:
       - Photosynthesis and plant respiration (Farquhar model)
       - Soil decomposition and nitrogen mineralization
       - Nitrogen uptake, nitrification, denitrification
       - Litterfall and carbon allocation

Design Rationale:
    RHESSys addresses coupled eco-hydrological questions:
    - How do forests affect watershed hydrology and water yield?
    - What are the impacts of climate change on vegetation and streamflow?
    - How does fire affect watershed carbon and water cycling?
    - What are the nitrogen export dynamics from forested watersheds?

Key Components:
    RHESSysPreProcessor: World file and flow table generation from GIS
    RHESSysRunner: Model execution with optional WMFire coupling
    RHESSysPostProcessor: Output extraction and analysis

Configuration Parameters:
    RHESSYS_WORLD_TEMPLATE: Template for world file generation
    RHESSYS_FLOW_TEMPLATE: Template for flow table generation
    RHESSYS_USE_WMFIRE: Enable WMFire wildfire spread module (default: False)
    RHESSYS_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'sat_to_gw_coeff,gw_loss_coeff,gw_loss_fast_coeff,gw_loss_fast_threshold,m,Ksat_0,...')
        sat_to_gw_coeff: Saturation excess to groundwater coefficient
        gw_loss_coeff: Groundwater loss coefficient (slow baseflow)
        gw_loss_fast_coeff: Fast groundwater loss coefficient (for storage > threshold)
        gw_loss_fast_threshold: Storage threshold (m) above which fast flow activates
        m: TOPMODEL decay parameter
        Ksat_0: Surface saturated hydraulic conductivity
        porosity_0: Surface porosity
        soil_depth: Total soil depth
        snow_melt_Tcoef: Degree-day snowmelt coefficient

Typical Workflow:
    1. Prepare GIS data (DEM, soils, vegetation, streams)
    2. Generate world file and flow table via RHESSysPreProcessor
    3. Create forcing data (climate time series)
    4. Run spinup to initialize carbon and nitrogen pools
    5. Execute main simulation via RHESSysRunner
    6. Analyze outputs via RHESSysPostProcessor

Limitations and Considerations:
    - Requires extensive GIS preprocessing (world file, flow table)
    - Spinup can take hundreds of years for carbon equilibrium
    - WMFire requires additional library compilation
    - Computationally intensive for large watersheds
    - Patch-level output can generate very large files
"""
from .preprocessor import RHESSysPreProcessor
from .runner import RHESSysRunner
from .postprocessor import RHESSysPostProcessor

__all__ = ["RHESSysPreProcessor", "RHESSysRunner", "RHESSysPostProcessor"]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
from .config import RHESSysConfigAdapter
ModelRegistry.register_config_adapter('RHESSYS')(RHESSysConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import RHESSysResultExtractor
ModelRegistry.register_result_extractor('RHESSYS')(RHESSysResultExtractor)

# Register preprocessor with ModelRegistry
ModelRegistry.register_preprocessor('RHESSYS')(RHESSysPreProcessor)

# Register runner with ModelRegistry
ModelRegistry.register_runner('RHESSYS')(RHESSysRunner)

# Register postprocessor with ModelRegistry
ModelRegistry.register_postprocessor('RHESSYS')(RHESSysPostProcessor)
