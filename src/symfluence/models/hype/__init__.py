"""
HYPE (HYdrological Predictions for the Environment) package.

This package contains components for running and managing HYPE model simulations
using the generalized pipeline pattern.

Components:
- HYPEPreProcessor: Main preprocessor orchestrating the pipeline
- HYPERunner: Model execution handler
- HYPEPostProcessor: Output processing and analysis
- HYPEForcingProcessor: Forcing data processing (hourly to daily conversion)
- HYPEConfigManager: Configuration file generation (info.txt, par.txt, filedir.txt)
- HYPEGeoDataManager: Geographic data file generation (GeoData.txt, GeoClass.txt, ForcKey.txt)
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
