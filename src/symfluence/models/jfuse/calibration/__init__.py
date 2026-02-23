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
