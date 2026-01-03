"""
LSTM (Flow and Snow Hydrological LSTM) package.

This package contains components for running and managing LSTM neural network model simulations.
Refactored to follow the modular structure of other SYMFLUENCE models.
"""

from .runner import LSTMRunner
from .preprocessor import LSTMPreprocessor
from .postprocessor import LSTMPostprocessor
from .model import LSTMModel

# Alias for backward compatibility
FLASH = LSTMRunner
FlashRunner = LSTMRunner
FlashPreprocessor = LSTMPreprocessor
FlashPostprocessor = LSTMPostprocessor

__all__ = [
    'LSTMRunner',
    'LSTMPreprocessor',
    'LSTMPostprocessor',
    'LSTMModel',
    'FLASH',
    'FlashRunner',
    'FlashPreprocessor',
    'FlashPostprocessor'
]
