"""
Optimization Algorithms Package

This module provides optimization infrastructure for hydrological model calibration.

Architecture
============
Use the model-specific optimizers from model_optimizers/:

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

Note (v0.5.12)
==============
Legacy optimizer classes (DDSOptimizer, PSOOptimizer, DEOptimizer, etc.)
have been removed in this version. These classes mixed model-specific code
with algorithm logic and have been superseded by the model-agnostic
BaseModelOptimizer architecture.

Migration from legacy code:
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
]
