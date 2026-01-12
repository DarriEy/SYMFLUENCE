"""
GNN Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.gnn.calibration.optimizer

    Please update imports to:
        from symfluence.models.gnn.calibration.optimizer import GNNModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.gnn.calibration.optimizer import GNNModelOptimizer
from symfluence.models.gnn.calibration.parameter_manager import GNNParameterManager
from symfluence.models.gnn.calibration.worker import GNNWorker

__all__ = ['GNNModelOptimizer', 'GNNParameterManager', 'GNNWorker']
