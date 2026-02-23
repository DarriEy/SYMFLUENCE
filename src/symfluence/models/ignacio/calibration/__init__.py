"""
IGNACIO Calibration Support.

Provides calibration worker, parameter manager, and optimizer for
IGNACIO fire model FBP parameter calibration using spatial metrics
(IoU/Dice) as objective functions.
"""

from .optimizer import IGNACIOModelOptimizer
from .parameter_manager import IGNACIOParameterManager
from .worker import IGNACIOWorker

__all__ = ['IGNACIOWorker', 'IGNACIOParameterManager', 'IGNACIOModelOptimizer']
