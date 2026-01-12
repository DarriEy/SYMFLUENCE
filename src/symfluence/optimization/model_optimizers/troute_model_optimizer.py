"""
TROUTE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.troute.calibration.optimizer
    
    Please update imports to:
        from symfluence.models.troute.calibration.optimizer import trouteModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.troute.calibration.optimizer import trouteModelOptimizer

__all__ = ['trouteModelOptimizer']
