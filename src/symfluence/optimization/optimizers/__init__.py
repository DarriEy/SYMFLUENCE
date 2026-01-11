"""
Optimization Algorithms Package

This module provides optimization infrastructure for hydrological model calibration.

RECOMMENDED: New Architecture (Model-Agnostic)
=============================================
For new code, use the model-specific optimizers from model_optimizers/:

    >>> from symfluence.optimization.model_optimizers import SUMMAModelOptimizer
    >>> optimizer = SUMMAModelOptimizer(config, logger)
    >>> results_path = optimizer.run_dds()  # or run_pso(), run_de(), etc.

Available model-specific optimizers:
    - SUMMAModelOptimizer: SUMMA hydrological model
    - FUSEModelOptimizer: FUSE model
    - NgenModelOptimizer: NextGen framework
    - GRModelOptimizer: GR4J/GR6J models
    - HYPEModelOptimizer: HYPE model
    - RHESSysModelOptimizer: RHESSys model
    - MESHModelOptimizer: MESH model

These use the clean BaseModelOptimizer base class with pure algorithm
implementations from the algorithms/ subpackage.

DEPRECATED: Legacy Architecture
==============================
The following classes are deprecated and will be removed in a future version:
    - BaseOptimizer: Contains SUMMA-specific code
    - DDSOptimizer, PSOOptimizer, DEOptimizer, etc.

These classes mix model-specific code (SUMMA) with algorithm logic.
They issue DeprecationWarning when instantiated.

Migration:
    OLD:
        optimizer = DDSOptimizer(config, logger)
        optimizer.run_optimization()

    NEW:
        optimizer = SUMMAModelOptimizer(config, logger)
        optimizer.run_dds()
"""

# New architecture - recommended
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer

# Algorithm implementations (pure, model-agnostic)
from symfluence.optimization.optimizers.algorithms import (
    get_algorithm,
    list_algorithms,
    OptimizationAlgorithm,
    DDSAlgorithm,
    PSOAlgorithm,
    DEAlgorithm,
    SCEUAAlgorithm,
    NSGA2Algorithm,
)

# Legacy classes (deprecated - issue DeprecationWarning when used)
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer
from symfluence.optimization.optimizers.dds_optimizer import DDSOptimizer
from symfluence.optimization.optimizers.de_optimizer import DEOptimizer
from symfluence.optimization.optimizers.pso_optimizer import PSOOptimizer
from symfluence.optimization.optimizers.nsga2_optimizer import NSGA2Optimizer
from symfluence.optimization.optimizers.async_dds_optimizer import AsyncDDSOptimizer
from symfluence.optimization.optimizers.population_dds_optimizer import PopulationDDSOptimizer
from symfluence.optimization.optimizers.sceua_optimizer import SCEUAOptimizer

__all__ = [
    # New architecture (recommended)
    'BaseModelOptimizer',
    'get_algorithm',
    'list_algorithms',
    'OptimizationAlgorithm',
    'DDSAlgorithm',
    'PSOAlgorithm',
    'DEAlgorithm',
    'SCEUAAlgorithm',
    'NSGA2Algorithm',
    # Legacy (deprecated)
    'BaseOptimizer',
    'DDSOptimizer',
    'DEOptimizer',
    'PSOOptimizer',
    'NSGA2Optimizer',
    'AsyncDDSOptimizer',
    'PopulationDDSOptimizer',
    'SCEUAOptimizer',
]
