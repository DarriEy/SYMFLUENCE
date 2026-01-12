"""
SUMMA Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.summa.calibration.optimizer

    Please update imports to:
        from symfluence.models.summa.calibration.optimizer import SUMMAModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.summa.calibration.optimizer import SUMMAModelOptimizer

__all__ = ['SUMMAModelOptimizer']
