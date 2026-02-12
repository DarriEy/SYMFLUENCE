"""
SYMFLUENCE Model Evaluators Package

This package contains base evaluators for different hydrological variables including:
- Streamflow (routed and non-routed)
- Snow (SWE, SCA, depth)
- Groundwater (depth, GRACE TWS)
- Evapotranspiration (ET, latent heat)
- Soil moisture (point, SMAP, ESA)

Model-specific streamflow evaluators (GR, HYPE, RHESSys) have been moved to
symfluence.optimization.calibration_targets for consistency with the calibration
target pattern. Use those modules for model-specific calibration.
"""

from .base import ModelEvaluator
from .et import ETEvaluator
from .streamflow import StreamflowEvaluator
from .soil_moisture import SoilMoistureEvaluator
from .snow import SnowEvaluator
from .groundwater import GroundwaterEvaluator
from .tws import TWSEvaluator

__all__ = [
    "ModelEvaluator",
    "ETEvaluator",
    "StreamflowEvaluator",
    "SoilMoistureEvaluator",
    "SnowEvaluator",
    "GroundwaterEvaluator",
    "TWSEvaluator"
]
