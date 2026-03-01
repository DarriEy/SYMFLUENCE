# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HBV Calibration Module.

Provides calibration support for HBV-96 model including:
- Optimizer for iterative calibration
- Worker for distributed calibration
- Parameter manager for bounds and transformations
- Gradient-based optimization support (when JAX available)
"""

from .optimizer import HBVModelOptimizer
from .parameter_manager import HBVParameterManager, get_hbv_calibration_bounds
from .worker import HBVWorker

__all__ = [
    'HBVModelOptimizer',
    'HBVWorker',
    'HBVParameterManager',
    'get_hbv_calibration_bounds',
]
