"""
jFUSE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.jfuse.calibration.optimizer

    Please update imports to:
        from symfluence.models.jfuse.calibration.optimizer import JFUSEModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.jfuse.calibration.optimizer import JFUSEModelOptimizer
from symfluence.models.jfuse.calibration.parameter_manager import JFUSEParameterManager
from symfluence.models.jfuse.calibration.worker import JFUSEWorker

__all__ = ['JFUSEModelOptimizer', 'JFUSEParameterManager', 'JFUSEWorker']
