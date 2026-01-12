"""
HYPE Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.hype.calibration.optimizer

    Please update imports to:
        from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer
from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager
from symfluence.models.hype.calibration.worker import HYPEWorker

__all__ = ['HYPEModelOptimizer', 'HYPEParameterManager', 'HYPEWorker']
