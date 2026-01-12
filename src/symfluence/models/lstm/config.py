"""
LSTM Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the LSTM (Long Short-Term Memory) neural network model.

This module registers LSTM-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import LSTMConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# LSTM Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('LSTM')
class LSTMDefaults:
    """Default configuration values for LSTM model."""

    # Model architecture
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    LSTM_L2_REGULARIZATION = 0.0

    # Training settings
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_LEARNING_RATE = 0.001
    LSTM_LEARNING_PATIENCE = 10
    LSTM_LOOKBACK = 365

    # Model features
    LSTM_USE_ATTENTION = False
    LSTM_USE_SNOW = False
    LSTM_TRAIN_THROUGH_ROUTING = False

    # Model loading
    LSTM_LOAD = False


# ============================================================================
# LSTM Field Transformers (Flat to Nested Mapping)
# ============================================================================

LSTM_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'LSTM_LOAD': ('model', 'lstm', 'load'),
    'LSTM_HIDDEN_SIZE': ('model', 'lstm', 'hidden_size'),
    'LSTM_NUM_LAYERS': ('model', 'lstm', 'num_layers'),
    'LSTM_EPOCHS': ('model', 'lstm', 'epochs'),
    'LSTM_BATCH_SIZE': ('model', 'lstm', 'batch_size'),
    'LSTM_LEARNING_RATE': ('model', 'lstm', 'learning_rate'),
    'LSTM_LEARNING_PATIENCE': ('model', 'lstm', 'learning_patience'),
    'LSTM_LOOKBACK': ('model', 'lstm', 'lookback'),
    'LSTM_DROPOUT': ('model', 'lstm', 'dropout'),
    'LSTM_L2_REGULARIZATION': ('model', 'lstm', 'l2_regularization'),
    'LSTM_USE_ATTENTION': ('model', 'lstm', 'use_attention'),
    'LSTM_USE_SNOW': ('model', 'lstm', 'use_snow'),
    'LSTM_TRAIN_THROUGH_ROUTING': ('model', 'lstm', 'train_through_routing'),
}


# ============================================================================
# LSTM Config Adapter
# ============================================================================

class LSTMConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for LSTM model."""

    def __init__(self, model_name: str = 'LSTM'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for LSTM configuration."""
        return LSTMConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for LSTM."""
        return {
            k: v for k, v in vars(LSTMDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for LSTM."""
        return LSTM_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate LSTM-specific configuration."""
        # Validate hyperparameters are within reasonable ranges
        hidden_size = config.get('LSTM_HIDDEN_SIZE', 128)
        if hidden_size <= 0 or hidden_size > 2048:
            raise ConfigValidationError(
                f"LSTM_HIDDEN_SIZE must be between 1 and 2048, got {hidden_size}"
            )

        num_layers = config.get('LSTM_NUM_LAYERS', 2)
        if num_layers <= 0 or num_layers > 10:
            raise ConfigValidationError(
                f"LSTM_NUM_LAYERS must be between 1 and 10, got {num_layers}"
            )

        dropout = config.get('LSTM_DROPOUT', 0.2)
        if dropout < 0.0 or dropout >= 1.0:
            raise ConfigValidationError(
                f"LSTM_DROPOUT must be between 0.0 and 1.0, got {dropout}"
            )

        learning_rate = config.get('LSTM_LEARNING_RATE', 0.001)
        if learning_rate <= 0.0 or learning_rate > 1.0:
            raise ConfigValidationError(
                f"LSTM_LEARNING_RATE must be between 0.0 and 1.0, got {learning_rate}"
            )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for LSTM."""
        return []  # LSTM has sensible defaults for all parameters
