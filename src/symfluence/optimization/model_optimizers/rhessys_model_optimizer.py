"""
RHESSYS Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.rhessys.calibration.optimizer

    Please update imports to:
        from symfluence.models.rhessys.calibration.optimizer import RHESSysModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.rhessys.calibration.optimizer import RHESSysModelOptimizer
from symfluence.models.rhessys.calibration.parameter_manager import RHESSysParameterManager
from symfluence.models.rhessys.calibration.worker import RHESSysWorker

__all__ = ['RHESSysModelOptimizer', 'RHESSysParameterManager', 'RHESSysWorker']
