"""
NGEN Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.ngen.calibration.optimizer

    Please update imports to:
        from symfluence.models.ngen.calibration.optimizer import NgenModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.ngen.calibration.optimizer import NgenModelOptimizer
from symfluence.models.ngen.calibration.parameter_manager import NgenParameterManager
from symfluence.models.ngen.calibration.worker import NgenWorker

__all__ = ['NgenModelOptimizer', 'NgenParameterManager', 'NgenWorker']
