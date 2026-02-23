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

# Register with ModelRegistry
from symfluence.models.registry import ModelRegistry

ModelRegistry.register_config_adapter('TROUTE')(TRouteConfigAdapter)
ModelRegistry.register_result_extractor('TROUTE')(TRouteResultExtractor)

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Register calibration components
try:
    from .calibration import TRouteModelOptimizer  # noqa: F401
except ImportError:
    pass
