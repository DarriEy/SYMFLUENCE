"""FUSE (Framework for Understanding Structural Errors) Hydrological Model.

This module implements the FUSE modular modeling framework, which enables systematic
exploration of model structural uncertainty by combining different representations
of hydrological processes. FUSE can generate up to 1,248 unique model structures
by mixing and matching components for upper/lower soil zones, percolation, routing,
evaporation, and baseflow.

Model Architecture:
    1. **Upper Zone**: Tension storage, free storage with configurable overflow/drainage
    2. **Lower Zone**: Single or dual baseflow reservoirs with linear/power-law release
    3. **Percolation**: Saturation excess, field capacity, or demand-based drainage
    4. **Surface Runoff**: Infiltration excess (Horton) or saturation excess (Dunne)
    5. **Snow Module**: Temperature-index snowmelt with optional elevation bands

Design Rationale:
    FUSE addresses the challenge of model structural uncertainty:
    - Most calibration focuses only on parameter uncertainty
    - Different process representations can yield equally good fits but different predictions
    - FUSE enables ensemble runs across multiple structures for robust uncertainty estimates
    - Structure selection can be automated via structure ensemble calibration

Spatial Modes:
    lumped: Single spatial unit representing entire catchment
    semi-distributed: Multiple HRUs with elevation bands for snow processes
    distributed: Grid-based or subcatchment-based with optional mizuRoute routing

Key Components:
    FUSEPreProcessor: Forcing preparation, spatial setup, file manager generation
    FUSERunner: Model execution with structure selection and parameter mapping
    FUSEPostprocessor: Output extraction and result formatting
    FuseStructureAnalyzer: Ensemble analysis comparing different model structures

Configuration Parameters:
    FUSE_SPATIAL_MODE: Spatial discretization (default: 'lumped')
    FUSE_N_ELEVATION_BANDS: Number of elevation bands for snow (default: 1)
    FUSE_ROUTING_INTEGRATION: Routing model (default: 'default', options: 'none', 'mizuroute')
    FUSE_DECISION_OPTIONS: Structure decision dictionary for ensemble runs
    SETTINGS_FUSE_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE')

Typical Workflow:
    1. Initialize FUSEPreProcessor with configuration
    2. Process forcing data (precipitation, temperature, PET)
    3. Generate elevation bands if semi-distributed mode
    4. Create file manager and control files
    5. Execute FUSE via FUSERunner for one or more structures
    6. Postprocess outputs, optionally analyze structure ensemble

Limitations and Considerations:
    - Elevation band mode requires DEM and careful band delineation
    - Structure ensemble runs increase computational cost significantly
    - Some structure combinations may be physically inconsistent
    - Requires FUSE executable built from source (Fortran)
"""

# Import main classes
from .elevation_band_manager import FuseElevationBandManager

# Import manager classes (for advanced usage)
from .forcing_processor import FuseForcingProcessor
from .postprocessor import FUSEPostprocessor
from .preprocessor import FUSEPreProcessor
from .runner import FUSERunner
from .structure_analyzer import FuseStructureAnalyzer
from .synthetic_data_generator import FuseSyntheticDataGenerator
from .visualizer import visualize_fuse

__all__ = [
    # Main classes (public API)
    'FUSEPreProcessor',
    'FUSERunner',
    'FUSEPostprocessor',
    'FuseStructureAnalyzer',
    # Manager classes (advanced usage)
    'FuseForcingProcessor',
    'FuseElevationBandManager',
    'FuseSyntheticDataGenerator',
]

# Register all FUSE components via unified registry
from symfluence.core.registry import model_manifest

from .config import FUSEConfigAdapter
from .extractor import FUSEResultExtractor
from .plotter import FUSEPlotter

model_manifest(
    "FUSE",
    config_adapter=FUSEConfigAdapter,
    result_extractor=FUSEResultExtractor,
    decision_analyzer=FuseStructureAnalyzer,
    plotter=FUSEPlotter,
    build_instructions_module="symfluence.models.fuse.build_instructions",
)
