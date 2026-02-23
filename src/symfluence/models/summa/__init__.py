"""SUMMA (Structure for Unifying Multiple Modeling Alternatives) Hydrological Model.

This module implements SUMMA, a unified framework for process-based hydrological
modeling that enables systematic exploration of different model representations.
SUMMA allows users to select from multiple physically-based options for each
hydrological process, generating 200+ unique model configurations.

Model Architecture:
    SUMMA uses a layered approach with configurable process representations:

    1. **Canopy Processes**: Interception, throughfall, canopy snow
       - Options: Big-leaf, two-stream radiation, CLM-style

    2. **Snow Processes**: Accumulation, metamorphism, melt
       - Options: Temperature index, energy balance, layered snow

    3. **Soil Processes**: Infiltration, percolation, drainage
       - Options: Richards equation, simplified bucket, Green-Ampt

    4. **Groundwater**: Baseflow generation, aquifer dynamics
       - Options: TOPMODEL, linear reservoir, power-law

    5. **Runoff Generation**: Surface and subsurface routing
       - Options: Saturation excess, infiltration excess, variable area

Design Rationale:
    SUMMA addresses model structural uncertainty systematically:
    - Most models hard-code process representations
    - SUMMA exposes alternatives as runtime decisions
    - Enables hypothesis testing across process formulations
    - Reduces need for multiple model codebases
    - Supports ensemble modeling with structural uncertainty

Spatial Structure:
    - GRU (Grouped Response Unit): Routing unit containing multiple HRUs
    - HRU (Hydrologic Response Unit): Fundamental computational unit
    - Layers: Vertical discretization for snow and soil

Key Components:
    SummaPreProcessor: Forcing preparation, attributes, trial parameters
    SummaRunner: Model execution with parallel support (summa_actors)
    SUMMAPostprocessor: Output extraction and NetCDF processing
    SummaStructureAnalyzer: Decision ensemble analysis
    SummaForcingProcessor: Forcing file preparation
    SummaConfigManager: Configuration file generation
    SummaAttributesManager: HRU attribute management

Configuration Parameters:
    SETTINGS_SUMMA_CONNECT_HRUS: Enable lateral HRU connectivity (default: True)
    SUMMA_DECISION_OPTIONS: Dictionary of decision choices for ensemble runs
    SETTINGS_SUMMA_GLACIER_MODE: Enable glacier dynamics (default: False)
    SETTINGS_SUMMA_USE_PARALLEL_SUMMA: Use parallel execution (default: False)
    PARAMS_TO_CALIBRATE: Local parameters
        (default: 'albedo_max,albedo_min,canopy_capacity,slow_drainage')
    BASIN_PARAMS_TO_CALIBRATE: Basin-scale routing parameters
        (default: 'routingGammaShape,routingGammaScale')

Typical Workflow:
    1. Initialize SummaPreProcessor with configuration
    2. Process forcing data via SummaForcingProcessor
    3. Generate attributes and trial parameters via managers
    4. Create file manager and decision files
    5. Execute SUMMA (serial or parallel) via SummaRunner
    6. Extract results and analyze decisions via SUMMAPostprocessor

Limitations and Considerations:
    - Requires SUMMA executable (compiled with Sundials solver recommended)
    - Decision ensemble runs multiply computational cost
    - Glacier mode requires additional attribute preparation
    - Large domains benefit from parallel execution (summa_actors)
    - Some decision combinations may be incompatible or unstable
"""

from .attributes_manager import SummaAttributesManager
from .config_manager import SummaConfigManager
from .forcing_processor import SummaForcingProcessor
from .postprocessor import SUMMAPostprocessor
from .preprocessor import SummaPreProcessor
from .runner import SummaRunner
from .structure_analyzer import SummaStructureAnalyzer
from .visualizer import visualize_summa

__all__ = [
    'SummaPreProcessor',
    'SummaRunner',
    'SUMMAPostprocessor',
    'SummaStructureAnalyzer',
    'SummaForcingProcessor',
    'SummaConfigManager',
    'SummaAttributesManager'
]

# Register all SUMMA components via unified registry
from symfluence.core.registry import model_manifest

from .config import SUMMAConfigAdapter
from .extractor import SUMMAResultExtractor
from .plotter import SUMMAPlotter

model_manifest(
    "SUMMA",
    config_adapter=SUMMAConfigAdapter,
    result_extractor=SUMMAResultExtractor,
    decision_analyzer=SummaStructureAnalyzer,
    plotter=SUMMAPlotter,
    build_instructions_module="symfluence.models.summa.build_instructions",
)
