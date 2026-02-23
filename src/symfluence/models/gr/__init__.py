"""GR (Génie Rural) Family of Conceptual Hydrological Models.

This module implements the GR family of rainfall-runoff models developed by INRAE
(formerly Cemagref/IRSTEA). The GR models are parsimonious conceptual models known
for excellent performance relative to their simplicity.

Model Variants:
    GR4J: 4-parameter daily model (production store, routing store, unit hydrographs)
    GR5J: 5-parameter version with inter-catchment groundwater exchange
    GR6J: 6-parameter version with additional exponential store

    All variants can be coupled with CemaNeige, a 2-parameter snow module using
    degree-day snowmelt with cold content accounting.

Model Architecture:
    1. **Interception**: Neutralization of P and PE via threshold
    2. **Production Store**: Soil moisture accounting (parameter X1 = capacity)
    3. **Percolation**: Leakage from production store to routing
    4. **Unit Hydrographs**: Two UHs split flow (90%/10%) with X4 time base
    5. **Routing Store**: Nonlinear reservoir (X3 = capacity) with exchange (X2)
    6. **CemaNeige** (optional): Degree-day snow with CTG and Kf parameters

Design Rationale:
    The GR approach emphasizes parsimony and robustness:
    - Few parameters reduce equifinality and overfitting risk
    - Proven performance across diverse climates and catchment sizes
    - R-based implementation via airGR package for reliability
    - CemaNeige provides simple but effective snow representation

Spatial Modes:
    lumped: Single GRU for entire catchment
    distributed: Multiple subcatchments with optional mizuRoute routing
    auto: Automatically detect based on domain configuration

Key Components:
    GRPreProcessor: Forcing preparation, catchment setup, R script generation
    GRRunner: Model execution via Rscript with airGR/airGRdatassim packages
    GRPostprocessor: Output parsing, result formatting, metric calculation

Configuration Parameters:
    GR_SPATIAL_MODE: Spatial setup ('auto', 'lumped', 'distributed')
    GR_ROUTING_INTEGRATION: Routing model ('none', 'mizuroute')
    GR_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff')
        X1: Production store capacity (mm)
        X2: Groundwater exchange coefficient (mm/day)
        X3: Routing store capacity (mm)
        X4: Unit hydrograph time base (days)
        CTG: CemaNeige degree-day factor
        Kf: CemaNeige cold content factor

Typical Workflow:
    1. Initialize GRPreProcessor with configuration
    2. Process forcing data (precipitation, temperature, PET)
    3. Generate R control script and input CSV files
    4. Execute GR via GRRunner (calls Rscript)
    5. Parse outputs and format results via GRPostprocessor

Limitations and Considerations:
    - Requires R with airGR/airGRdatassim packages installed
    - CemaNeige is daily timestep only (no sub-daily snow)
    - Parameter bounds should respect physical constraints (X1, X3 > 0)
    - For distributed mode, subcatchment areas needed for routing
"""

from .postprocessor import GRPostprocessor
from .preprocessor import GRPreProcessor
from .runner import GRRunner
from .visualizer import visualize_gr

__all__ = [
    'GRPreProcessor',
    'GRRunner',
    'GRPostprocessor',
    'visualize_gr'
]


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry

from .config import GRConfigAdapter

ModelRegistry.register_config_adapter('GR')(GRConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import GRResultExtractor

ModelRegistry.register_result_extractor('GR')(GRResultExtractor)
