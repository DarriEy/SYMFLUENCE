"""
LSTM (Flow and Snow Hydrological LSTM) package.

This package contains components for running and managing LSTM neural network model simulations.
Refactored to follow the modular structure of other SYMFLUENCE models.
"""

from .runner import LSTMRunner
from .preprocessor import LSTMPreprocessor
from .postprocessor import LSTMPostprocessor
from .model import LSTMModel
from .visualizer import visualize_lstm

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
    'visualize_lstm',
    'FLASH',
    'FlashRunner',
    'FlashPreprocessor',
    'FlashPostprocessor'
]

# Register config adapter with ModelRegistry (includes defaults registration)
from symfluence.models.registry import ModelRegistry
from .config import LSTMConfigAdapter
ModelRegistry.register_config_adapter('LSTM')(LSTMConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import LSTMResultExtractor
ModelRegistry.register_result_extractor('LSTM')(LSTMResultExtractor)

# Register preprocessor with ModelRegistry
ModelRegistry.register_preprocessor('LSTM')(LSTMPreprocessor)

# Register runner with ModelRegistry
ModelRegistry.register_runner('LSTM')(LSTMRunner)

# Register postprocessor with ModelRegistry
ModelRegistry.register_postprocessor('LSTM')(LSTMPostprocessor)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import LSTMPlotter  # noqa: F401
