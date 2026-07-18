# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization Mixins

Reusable mixin classes that provide common functionality for optimizers:
- ParallelExecutionMixin: Parallel processing infrastructure
- ResultsTrackingMixin: Results persistence and tracking
- RetryExecutionMixin: Retry logic with exponential backoff
- GradientOptimizationMixin: ADAM/LBFGS gradient-based optimization
- SUMMAOptimizerMixin: SUMMA-specific functionality (extracted from legacy BaseOptimizer)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .gradient_optimization import GradientOptimizationMixin
from .parallel_execution import ParallelExecutionMixin
from .results_tracking import ResultsTrackingMixin
from .retry_execution import RetryExecutionMixin

# SUMMAOptimizerMixin is resolved lazily (PEP 562): it imports the SUMMA model
# package, which must not load when generic optimizer bases are imported at
# plugin-discovery time.


def __getattr__(name: str):
    if name == 'SUMMAOptimizerMixin':
        from .summa_optimizer_mixin import SUMMAOptimizerMixin
        globals()[name] = SUMMAOptimizerMixin
        return SUMMAOptimizerMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .summa_optimizer_mixin import SUMMAOptimizerMixin

__all__ = [
    'ParallelExecutionMixin',
    'ResultsTrackingMixin',
    'RetryExecutionMixin',
    'GradientOptimizationMixin',
    'SUMMAOptimizerMixin',
]
