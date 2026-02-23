"""MESH (Modélisation Environmentale - Surface and Hydrology) Model.

This module implements MESH, a coupled land surface-hydrology model developed
by Environment and Climate Change Canada. MESH combines the Canadian Land
Surface Scheme (CLASS) or SVS with WATFLOOD routing for operational
hydrological prediction, particularly in cold regions.

Model Architecture:
    MESH couples two main components:

    1. **Land Surface Scheme** (CLASS or SVS):
       - Energy balance: radiation, sensible/latent heat fluxes
       - Snow processes: multi-layer snow, metamorphism, melt
       - Soil processes: heat conduction, moisture dynamics, freeze-thaw
       - Vegetation: phenology, transpiration, interception

    2. **Routing Component** (WATFLOOD-derived):
       - Overland flow routing
       - Channel routing with Manning's equation
       - Lake and wetland storage
       - Gridded or GRU-based spatial structure

Design Rationale:
    MESH addresses Canadian operational needs:
    - Process-based for scenario analysis (climate, land use)
    - Cold region processes (permafrost, snow redistribution, ice)
    - Operational use in Canadian flood forecasting
    - Energy balance critical for snowmelt timing

Spatial Structure:
    - GRUs (Grouped Response Units): Tiles with similar hydrological response
    - Tiles: Land cover types within each GRU
    - Grid: Optional regular grid for spatially distributed simulations

Key Components:
    MESHPreProcessor: DDB preparation, forcing setup, parameter files
    MESHRunner: Model execution and simulation management
    MESHPostProcessor: Output extraction and result formatting

Configuration Parameters:
    MESH_SPATIAL_MODE: Spatial setup ('auto', 'lumped', 'distributed')
    MESH_FORCING_PATH: Path to forcing data files
    MESH_FORCING_VARS: Forcing variable names
    MESH_FORCING_UNITS: Forcing variable units

Typical Workflow:
    1. Prepare drainage database (DDB) with GRU/tile definitions
    2. Process forcing data (hourly or sub-hourly for energy balance)
    3. Set up CLASS/SVS parameters and initial conditions
    4. Configure WATFLOOD routing parameters
    5. Execute MESH via MESHRunner
    6. Extract results (streamflow, SWE, soil moisture, energy fluxes)

Limitations and Considerations:
    - Requires MESH executable (compiled from source)
    - CLASS/SVS have different parameter requirements
    - Energy balance requires radiation and wind data (not just P/T)
    - Canadian datasets (CaPA, RDRS) well-supported
    - DDB preparation can be complex for new domains
    - Primarily tested for Canadian applications
"""

from .postprocessor import MESHPostProcessor
from .preprocessor import MESHPreProcessor
from .runner import MESHRunner
from .visualizer import visualize_mesh

__all__ = [
    'MESHPreProcessor',
    'MESHRunner',
    'MESHPostProcessor',
    'visualize_mesh'
]

# Register all MESH components via unified registry
from symfluence.core.registry import model_manifest

from .config import MESHConfigAdapter
from .extractor import MESHResultExtractor

model_manifest(
    "MESH",
    config_adapter=MESHConfigAdapter,
    result_extractor=MESHResultExtractor,
    build_instructions_module="symfluence.models.mesh.build_instructions",
)
