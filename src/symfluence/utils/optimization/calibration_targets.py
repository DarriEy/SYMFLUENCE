#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SYMFLUENCE Calibration Targets

This module provides calibration targets for different hydrologic variables.
These targets handle data loading, processing, and metric calculation for specific variables
during the optimization/calibration process.

Note: This module has been refactored to use the centralized evaluators in 
symfluence.utils.evaluation.evaluators. The classes here are aliases for backward compatibility.
"""

from symfluence.utils.evaluation.evaluators import (
    ModelEvaluator as CalibrationTarget,
    ETEvaluator as ETTarget,
    StreamflowEvaluator as StreamflowTarget,
    SoilMoistureEvaluator as SoilMoistureTarget,
    SnowEvaluator as SnowTarget,
    GroundwaterEvaluator as GroundwaterTarget,
    TWSEvaluator as TWSTarget
)

# Re-export for backward compatibility
__all__ = [
    'CalibrationTarget',
    'ETTarget',
    'StreamflowTarget',
    'SoilMoistureTarget',
    'SnowTarget',
    'GroundwaterTarget',
    'TWSTarget'
]