"""
HYPE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.hype.calibration.optimizer

    Please update imports to:
        from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer

__all__ = ['HYPEModelOptimizer']
