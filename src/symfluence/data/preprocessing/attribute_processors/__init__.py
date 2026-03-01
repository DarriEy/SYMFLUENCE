# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

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
from .climate import ClimateProcessor
from .elevation import ElevationProcessor
from .geology import GeologyProcessor
from .hydrology import HydrologyProcessor
from .landcover import LandCoverProcessor
from .soil import SoilProcessor

__all__ = [
    'BaseAttributeProcessor',
    'ElevationProcessor',
    'GeologyProcessor',
    'SoilProcessor',
    'LandCoverProcessor',
    'ClimateProcessor',
    'HydrologyProcessor',
]
