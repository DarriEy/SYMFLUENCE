
"""
MESH Model Optimizer (Backward Compatibility)

.. deprecated::
    Moved to symfluence.models.mesh.calibration.optimizer
"""

import logging
logger = logging.getLogger(__name__)
logger.debug("Importing mesh_model_optimizer wrapper")
from symfluence.models.mesh.calibration.optimizer import MESHModelOptimizer
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.mesh.calibration.worker import MESHWorker

__all__ = ['MESHModelOptimizer', 'MESHParameterManager', 'MESHWorker']

