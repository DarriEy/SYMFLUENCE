"""
IGNACIO Calibration Support.

Provides calibration worker, parameter manager, and optimizer for
IGNACIO fire model FBP parameter calibration using spatial metrics
(IoU/Dice) as objective functions.
"""

from .worker import IGNACIOWorker
from .parameter_manager import IGNACIOParameterManager
from .optimizer import IGNACIOModelOptimizer

__all__ = ['IGNACIOWorker', 'IGNACIOParameterManager', 'IGNACIOModelOptimizer']
