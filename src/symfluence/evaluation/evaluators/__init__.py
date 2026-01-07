"""
SYMFLUENCE Model Evaluators Package

This package contains evaluators for different hydrological variables including:
- Streamflow (routed and non-routed)
- Snow (SWE, SCA, depth)
- Groundwater (depth, GRACE TWS)
- Evapotranspiration (ET, latent heat)
- Soil moisture (point, SMAP, ESA)
"""

from .base import ModelEvaluator
from .et import ETEvaluator
from .streamflow import StreamflowEvaluator
from .gr_streamflow import GRStreamflowEvaluator
from .hype_streamflow import HYPEStreamflowEvaluator
from .soil_moisture import SoilMoistureEvaluator
from .snow import SnowEvaluator
from .groundwater import GroundwaterEvaluator
from .tws import TWSEvaluator

__all__ = [
    "ModelEvaluator",
    "ETEvaluator",
    "StreamflowEvaluator",
    "GRStreamflowEvaluator",
    "HYPEStreamflowEvaluator",
    "SoilMoistureEvaluator",
    "SnowEvaluator",
    "GroundwaterEvaluator",
    "TWSEvaluator"
]
