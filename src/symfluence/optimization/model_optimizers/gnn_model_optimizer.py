"""
GNN Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.gnn.calibration.optimizer

    Please update imports to:
        from symfluence.models.gnn.calibration.optimizer import GNNModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.gnn.calibration.optimizer import GNNModelOptimizer

__all__ = ['GNNModelOptimizer']
