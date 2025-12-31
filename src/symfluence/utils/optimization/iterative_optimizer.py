#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SYMFLUENCE Optimizer (Legacy Entry Point)

This module is now a wrapper around the modular optimizer sub-package.
Please migrate to importing from symfluence.utils.optimization.optimizers.* 
and symfluence.utils.optimization.core.* in the future.
"""

import logging
import warnings

# Backward compatible imports from core
from symfluence.utils.optimization.core.parameter_manager import ParameterManager
from symfluence.utils.optimization.core.model_executor import ModelExecutor, fix_summa_time_precision
from symfluence.utils.optimization.core.results_manager import ResultsManager

# Backward compatible imports from optimizers
from symfluence.utils.optimization.optimizers.base_optimizer import BaseOptimizer
from symfluence.utils.optimization.optimizers.dds_optimizer import DDSOptimizer
from symfluence.utils.optimization.optimizers.de_optimizer import DEOptimizer
from symfluence.utils.optimization.optimizers.pso_optimizer import PSOOptimizer
from symfluence.utils.optimization.optimizers.nsga2_optimizer import NSGA2Optimizer
from symfluence.utils.optimization.optimizers.async_dds_optimizer import AsyncDDSOptimizer
from symfluence.utils.optimization.optimizers.population_dds_optimizer import PopulationDDSOptimizer
from symfluence.utils.optimization.optimizers.sceua_optimizer import SCEUAOptimizer
from symfluence.utils.optimization.calibration_targets import (
    CalibrationTarget, StreamflowTarget, SnowTarget, SoilMoistureTarget, ETTarget, GroundwaterTarget, TWSTarget
)

# Re-export everything for backward compatibility
__all__ = [
    'ParameterManager',
    'ModelExecutor',
    'fix_summa_time_precision',
    'ResultsManager',
    'BaseOptimizer',
    'DDSOptimizer',
    'DEOptimizer',
    'PSOOptimizer',
    'NSGA2Optimizer',
    'AsyncDDSOptimizer',
    'PopulationDDSOptimizer',
    'SCEUAOptimizer',
    'CalibrationTarget',
    'StreamflowTarget',
    'SnowTarget',
    'SoilMoistureTarget',
    'ETTarget',
    'GroundwaterTarget',
    'TWSTarget'
]

# Warn about deprecation
# warnings.warn(
#     "Importing from symfluence.utils.optimization.iterative_optimizer is deprecated. "
#     "Use symfluence.utils.optimization.optimizers or .core instead.",
#     DeprecationWarning, stacklevel=2
# )
