"""
MESH Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.mesh.calibration.optimizer

    Please update imports to:
        from symfluence.models.mesh.calibration.optimizer import MESHModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.mesh.calibration.optimizer import MESHModelOptimizer

__all__ = ['MESHModelOptimizer']
