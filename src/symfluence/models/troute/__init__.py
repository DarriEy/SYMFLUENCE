"""
TRoute Model Utilities.

This package contains components for the t-route model integration:
- Preprocessor: Handles spatial and data preprocessing
- Runner: Manages model execution
- Postprocessor: Extracts routed streamflow results
- Extractor: Advanced result analysis utilities
- Config: Configuration adapter with auto-generated defaults
- Plotter: 4-panel routing diagnostics visualization

t-route is NOAA's channel routing model that provides:
- Multiple routing methods (Muskingum-Cunge, diffusive wave)
- Integration with NWM and other hydrologic models
- Support for large-scale river network routing
"""

from .config import TRouteConfigAdapter
from .extractor import TRouteResultExtractor
from .plotter import TRoutePlotter
from .postprocessor import TRoutePostProcessor
from .preprocessor import TRoutePreProcessor
from .runner import TRouteRunner

__all__ = [
    'TRoutePreProcessor',
    'TRouteRunner',
    'TRoutePostProcessor',
    'TRouteResultExtractor',
    'TRouteConfigAdapter',
    'TRoutePlotter',
]

# Register all TRoute components via unified registry
from symfluence.core.registry import model_manifest

model_manifest(
    "TROUTE",
    config_adapter=TRouteConfigAdapter,
    result_extractor=TRouteResultExtractor,
    plotter=TRoutePlotter,
    build_instructions_module="symfluence.models.troute.build_instructions",
)

# Register calibration components
try:
    from .calibration import TRouteModelOptimizer  # noqa: F401
except ImportError:
    pass
