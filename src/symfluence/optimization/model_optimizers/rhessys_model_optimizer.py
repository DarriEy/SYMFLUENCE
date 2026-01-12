"""
RHESSYS Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.rhessys.calibration.optimizer

    Please update imports to:
        from symfluence.models.rhessys.calibration.optimizer import RHESSysModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.rhessys.calibration.optimizer import RHESSysModelOptimizer

__all__ = ['RHESSysModelOptimizer']
