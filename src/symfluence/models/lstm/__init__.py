"""Long Short-Term Memory (LSTM) Neural Network Hydrological Model.

This module implements an LSTM-based deep learning model for hydrological prediction,
specifically designed for streamflow forecasting from meteorological forcing data.
The model uses recurrent neural networks to learn temporal patterns in precipitation,
temperature, and other forcing variables to predict runoff and streamflow.

Model Architecture:
    1. **Input Processing**: Forcing data (precipitation, temperature, etc.) is
       normalized and organized into sequences with a configurable lookback window.

    2. **LSTM Layers**: Multiple stacked LSTM layers process the temporal sequences,
       learning long-term dependencies in hydrological processes like snowmelt,
       soil moisture dynamics, and baseflow recession.

    3. **Attention Mechanism** (optional): Self-attention layer highlights important
       timesteps in the input sequence, improving predictions during events.

    4. **Output Layer**: Fully connected layer produces streamflow predictions.

Design Rationale:
    The LSTM approach addresses limitations of process-based models:
    - Learns complex nonlinear relationships directly from data
    - Captures long-term memory effects (e.g., snow accumulation over winter)
    - Requires no explicit parameterization of hydrological processes
    - Can be trained on historical observations without physical equations
    - Particularly effective when process understanding is incomplete

Key Components:
    LSTMRunner: Main orchestrator for model workflow (preprocessing, training, simulation)
    LSTMPreprocessor: Data loading, normalization, sequence preparation
    LSTMPostprocessor: Output denormalization, result formatting, saving
    LSTMModel: PyTorch model architecture (LSTM + optional attention)

Configuration Parameters:
    LSTM_HIDDEN_SIZE: Hidden state dimension per LSTM layer (default: 128)
    LSTM_NUM_LAYERS: Number of stacked LSTM layers (default: 3)
    LSTM_LOOKBACK: Input sequence length in days (default: 700)
    LSTM_EPOCHS: Maximum training epochs (default: 300)
    LSTM_BATCH_SIZE: Training batch size (default: 64)
    LSTM_LEARNING_RATE: Adam optimizer learning rate (default: 0.001)
    LSTM_LEARNING_PATIENCE: Early stopping patience epochs (default: 30)
    LSTM_DROPOUT: Dropout rate for regularization (default: 0.2)
    LSTM_L2_REGULARIZATION: Weight decay coefficient (default: 1e-6)
    LSTM_USE_ATTENTION: Enable attention mechanism (default: True)
    LSTM_USE_SNOW: Include snow cover as input feature (default: False)
    LSTM_LOAD: Load pre-trained model instead of training (default: False)

Typical Workflow:
    1. Initialize LSTMRunner with configuration
    2. Load forcing data (precipitation, temperature) and streamflow observations
    3. Preprocess data: normalize, create sequences with lookback window
    4. Train model on historical data (or load pre-trained weights)
    5. Generate predictions for simulation period
    6. Postprocess: denormalize outputs, save results

Limitations and Considerations:
    - Requires substantial historical data for training (typically 5+ years)
    - Model is a black box; physical interpretability is limited
    - Performance degrades outside training data distribution (extrapolation)
    - GPU recommended for training; CPU inference is feasible
    - Legacy aliases (FLASH, FlashRunner) maintained for backward compatibility
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

# Register runner with ModelRegistry (method_name must match the actual method in runner.py)
ModelRegistry.register_runner('LSTM', method_name='run_lstm')(LSTMRunner)

# Register postprocessor with ModelRegistry
ModelRegistry.register_postprocessor('LSTM')(LSTMPostprocessor)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import LSTMPlotter  # noqa: F401
