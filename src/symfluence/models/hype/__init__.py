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
