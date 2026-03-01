# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
jFUSE Calibration Components.

Provides worker and parameter management for jFUSE model calibration
with native gradient support via JAX autodiff.
"""

from .parameter_manager import JFUSEParameterManager, get_jfuse_calibration_bounds
from .worker import JFUSEWorker

__all__ = [
    'JFUSEWorker',
    'JFUSEParameterManager',
    'get_jfuse_calibration_bounds',
]
