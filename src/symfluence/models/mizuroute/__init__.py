"""MizuRoute River Routing Model.

This module implements integration with mizuRoute, a large-scale river routing
model developed at NCAR (National Center for Atmospheric Research). MizuRoute
routes runoff from hydrological models through river networks to produce
streamflow at any location in the network.

Routing Schemes:
    1. **Impulse Response Function (IRF)**: Unit hydrograph approach that
       convolves upstream runoff with a gamma-shaped transfer function.
       Fast and suitable for large-scale applications.

    2. **Kinematic Wave Tracking (KWT)**: Tracks individual runoff pulses
       through the network using kinematic wave approximation. More
       physically based but computationally intensive.

    3. **Diffusive Wave (DW)**: Solves the diffusive wave equation for
       each river segment. Most accurate for backwater effects but slowest.

Design Rationale:
    MizuRoute addresses the need for consistent routing across models:
    - Hydrological models often have inconsistent or simplified routing
    - MizuRoute provides unified routing for any runoff source
    - Enables routing SUMMA, FUSE, GR, HYPE, or any model through same network
    - Supports continental-scale applications (used in NWM, ISIMIP)

Spatial Structure:
    - Segments: River reach elements with properties (length, slope, width)
    - HRUs: Hydrologic response units contributing runoff to segments
    - Network Topology: Upstream-downstream connectivity (segId, downSegId)
    - Remapping: Maps source model HRUs to mizuRoute HRUs when grids differ

Key Components:
    MizuRoutePreProcessor: Network topology setup, remapping file generation
    MizuRouteRunner: Model execution with routing scheme selection
    MizuRouteConfigMixin: Configuration access helpers for coupled models

Configuration Parameters:
    SETTINGS_MIZU_TOPOLOGY: Path to network topology NetCDF file
    SETTINGS_MIZU_WITHIN_BASIN: Within-basin routing option
    SETTINGS_MIZU_NEEDS_REMAP: Whether HRU remapping is required
    SETTINGS_MIZU_OUTPUT_VARS: Variables to output (streamflow, etc.)
    MIZU_FROM_MODEL: Source model for runoff (SUMMA, FUSE, GR, HYPE, etc.)

Typical Workflow:
    1. Generate river network topology from stream shapefile
    2. Create HRU-to-segment mapping
    3. Generate remapping file if source model uses different spatial units
    4. Configure routing scheme (IRF recommended for large domains)
    5. Run source hydrological model to generate runoff
    6. Execute mizuRoute via MizuRouteRunner
    7. Extract routed streamflow at gauge locations

Integration Patterns:
    - Coupled: Run as ROUTING_MODEL after HYDROLOGICAL_MODEL
    - Standalone: Route pre-computed runoff files
    - Multi-model: Route outputs from multiple hydrological models

Limitations and Considerations:
    - Network topology must be prepared in advance (river segments, connectivity)
    - Remapping adds preprocessing complexity when model grids differ
    - KWT and DW schemes are slower but more accurate than IRF
    - Lake/reservoir routing requires additional configuration
"""

from .mixins import MizuRouteConfigMixin
from .preprocessor import MizuRoutePreProcessor
from .runner import MizuRouteRunner

__all__ = [
    'MizuRoutePreProcessor',
    'MizuRouteRunner',
    'MizuRouteConfigMixin',
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry

from .config import MizuRouteConfigAdapter

ModelRegistry.register_config_adapter('MIZUROUTE')(MizuRouteConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import MizuRouteResultExtractor

ModelRegistry.register_result_extractor('MIZUROUTE')(MizuRouteResultExtractor)

# Register preprocessor with ModelRegistry
ModelRegistry.register_preprocessor('MIZUROUTE')(MizuRoutePreProcessor)

# Register runner with ModelRegistry
ModelRegistry.register_runner('MIZUROUTE', method_name='run_mizuroute')(MizuRouteRunner)
