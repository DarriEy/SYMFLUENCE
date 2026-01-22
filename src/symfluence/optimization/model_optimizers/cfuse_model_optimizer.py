"""
cFUSE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.cfuse.calibration.optimizer

    Please update imports to:
        from symfluence.models.cfuse.calibration.optimizer import CFUSEModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.cfuse.calibration.optimizer import CFUSEModelOptimizer
from symfluence.models.cfuse.calibration.parameter_manager import CFUSEParameterManager
from symfluence.models.cfuse.calibration.worker import CFUSEWorker

__all__ = ['CFUSEModelOptimizer', 'CFUSEParameterManager', 'CFUSEWorker']
