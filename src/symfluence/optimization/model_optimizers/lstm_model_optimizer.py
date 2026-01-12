"""
LSTM Model Optimizer (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.lstm.calibration.optimizer

    Please update imports to:
        from symfluence.models.lstm.calibration.optimizer import LSTMModelOptimizer
"""

# Backward compatibility re-export
from symfluence.models.lstm.calibration.optimizer import LSTMModelOptimizer

__all__ = ['LSTMModelOptimizer']
