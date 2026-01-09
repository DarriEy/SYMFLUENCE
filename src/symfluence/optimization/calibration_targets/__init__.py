"""
Calibration Targets

Calibration target classes that handle data loading, processing, and metric
calculation for specific variables during the optimization/calibration process.

Each calibration target is responsible for:
- Loading observed data for a specific variable
- Extracting simulated data from model outputs
- Calculating objective metrics for calibration

Base calibration targets (aliases from evaluation.evaluators):
- CalibrationTarget: Base class for all calibration targets
- ETTarget: Evapotranspiration calibration target
- StreamflowTarget: Streamflow calibration target
- SoilMoistureTarget: Soil moisture calibration target
- SnowTarget: Snow calibration target
- GroundwaterTarget: Groundwater calibration target
- TWSTarget: Terrestrial water storage calibration target
- MultivariateTarget: Multivariate calibration combining multiple variables

Model-specific calibration targets:
- NgenStreamflowTarget: NextGen model streamflow calibration
- FUSEStreamflowTarget: FUSE model streamflow calibration
- FUSESnowTarget: FUSE model snow calibration
"""

from .base import (
    CalibrationTarget,
    ETTarget,
    StreamflowTarget,
    GRStreamflowTarget,
    HYPEStreamflowTarget,
    SoilMoistureTarget,
    SnowTarget,
    GroundwaterTarget,
    TWSTarget,
    MultivariateTarget,
)
from .ngen_calibration_targets import NgenStreamflowTarget
from .fuse_calibration_targets import FUSEStreamflowTarget, FUSESnowTarget
from .rhessys_calibration_targets import RHESSysStreamflowTarget

__all__ = [
    # Base targets
    'CalibrationTarget',
    'ETTarget',
    'StreamflowTarget',
    'GRStreamflowTarget',
    'HYPEStreamflowTarget',
    'SoilMoistureTarget',
    'SnowTarget',
    'GroundwaterTarget',
    'TWSTarget',
    'MultivariateTarget',
    # Model-specific targets
    'NgenStreamflowTarget',
    'FUSEStreamflowTarget',
    'FUSESnowTarget',
    'RHESSysStreamflowTarget',
]
