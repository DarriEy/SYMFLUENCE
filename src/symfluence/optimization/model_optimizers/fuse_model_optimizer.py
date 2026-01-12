"""
FUSE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.fuse.calibration.optimizer

    Please update imports to:
        from symfluence.models.fuse.calibration.optimizer import FUSEModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.fuse.calibration.optimizer import FUSEModelOptimizer
from symfluence.models.fuse.calibration.parameter_manager import FUSEParameterManager
from symfluence.models.fuse.calibration.worker import FUSEWorker

__all__ = ['FUSEModelOptimizer', 'FUSEParameterManager', 'FUSEWorker']
