# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: generic optimizer mixins moved to
``symfluence.core.calibration.mixins``. ``SUMMAOptimizerMixin`` stays here
(lazily) because it wraps the SUMMA model package, which core must not load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from symfluence.core.calibration.mixins import (
    GradientOptimizationMixin,
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
)


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
