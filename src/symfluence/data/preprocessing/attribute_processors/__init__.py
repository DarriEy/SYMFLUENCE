"""
Attribute processors package.

Provides modular attribute processing functionality split into specialized processors:
- BaseAttributeProcessor: Shared infrastructure
- ElevationProcessor: DEM, slope, aspect processing
- GeologyProcessor: Geological and hydrogeological attributes
- SoilProcessor: Soil properties
- LandCoverProcessor: Land cover and vegetation
- ClimateProcessor: Climate data
- HydrologyProcessor: Hydrological attributes
"""

from .base import BaseAttributeProcessor
from .elevation import ElevationProcessor
from .geology import GeologyProcessor
from .soil import SoilProcessor
from .landcover import LandCoverProcessor
from .climate import ClimateProcessor
from .hydrology import HydrologyProcessor

__all__ = [
    'BaseAttributeProcessor',
    'ElevationProcessor',
    'GeologyProcessor',
    'SoilProcessor',
    'LandCoverProcessor',
    'ClimateProcessor',
    'HydrologyProcessor',
]
