# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization module for SYMFLUENCE.

This module provides optimization infrastructure for hydrological model calibration,
including support for multiple models (SUMMA, FUSE, NGEN) and various optimization
algorithms (DDS, PSO, SCE-UA, DE, ADAM, LBFGS).

Main Components:
    - OptimizerRegistry: Central registry for model-specific optimizers and workers
    - BaseModelOptimizer: Abstract base class for model-specific optimizers
    - BaseWorker: Abstract base class for parallel worker implementations
    - ObjectiveRegistry: Registry for objective functions and metrics

Model Optimizers:
    - SUMMAModelOptimizer: Optimizer for SUMMA model
    - FUSEModelOptimizer: Optimizer for FUSE model
    - NgenModelOptimizer: Optimizer for NextGen model

Usage:
    >>> from symfluence.optimization import OptimizerRegistry
    >>> optimizer_cls = OptimizerRegistry.get_optimizer('FUSE')
    >>> optimizer = optimizer_cls(config, logger)
    >>> results = optimizer.run_pso()
"""

# Trigger objective registration
from . import objectives
from .objectives import ObjectiveRegistry
from .optimizers.base_model_optimizer import BaseModelOptimizer
from .registry import OptimizerRegistry
from .workers.base_worker import BaseWorker, WorkerResult, WorkerTask

try:
    from .objectives import multivariate
except ImportError:
    pass

# Import model optimizers to trigger registration with OptimizerRegistry
from . import model_optimizers

# Gradient-based optimization utilities
from .gradient import EMA, AdamW, CosineAnnealingWarmRestarts, CosineDecay

__all__ = [
    # Registries
    "OptimizerRegistry",
    "ObjectiveRegistry",
    # Base classes
    "BaseModelOptimizer",
    "BaseWorker",
    "WorkerTask",
    "WorkerResult",
    # Gradient-based optimization utilities
    "AdamW",
    "CosineAnnealingWarmRestarts",
    "CosineDecay",
    "EMA",
]
