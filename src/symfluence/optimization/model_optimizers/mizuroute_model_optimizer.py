"""
MIZUROUTE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.mizuroute.calibration.optimizer
    
    Please update imports to:
        from symfluence.models.mizuroute.calibration.optimizer import mizurouteModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.mizuroute.calibration.optimizer import mizurouteModelOptimizer

__all__ = ['mizurouteModelOptimizer']
