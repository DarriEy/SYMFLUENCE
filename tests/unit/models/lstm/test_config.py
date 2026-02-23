"""
Tests for LSTM configuration.
"""

import pytest
from symfluence.core.config.models import LSTMConfig, SymfluenceConfig


class TestLSTMConfig:
    """Tests for LSTM configuration."""

    def test_default_lstm_hidden_size(self, lstm_config):
        """Test default LSTM hidden size."""
        assert lstm_config.model.lstm.hidden_size == 64

    def test_default_lstm_num_layers(self, lstm_config):
        """Test default LSTM number of layers."""
        assert lstm_config.model.lstm.num_layers == 2

    def test_lstm_epochs_configured(self, lstm_config):
        """Test LSTM epochs configuration."""
        assert lstm_config.model.lstm.epochs == 10

    def test_lstm_batch_size_configured(self, lstm_config):
        """Test LSTM batch size configuration."""
        assert lstm_config.model.lstm.batch_size == 32

    def test_lstm_learning_rate_configured(self, lstm_config):
        """Test LSTM learning rate configuration."""
        assert lstm_config.model.lstm.learning_rate == 0.001

    def test_lstm_lookback_configured(self, lstm_config):
        """Test LSTM lookback window configuration."""
        assert lstm_config.model.lstm.lookback == 365

    def test_lstm_dropout_configured(self, lstm_config):
        """Test LSTM dropout configuration."""
        assert lstm_config.model.lstm.dropout == 0.1

    def test_lstm_load_default(self, lstm_config):
        """Test LSTM load checkpoint default."""
        assert lstm_config.model.lstm.load is False


class TestLSTMConfigValidation:
    """Tests for LSTM configuration validation."""

    def test_valid_lstm_config_passes(self, temp_dir):
        """Test that valid LSTM config passes validation."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'LSTM',
            'FORCING_DATASET': 'ERA5',
            'LSTM_HIDDEN_SIZE': 128,
            'LSTM_NUM_LAYERS': 3,
        }
        config = SymfluenceConfig(**config_dict)
        assert config.model.lstm.hidden_size == 128
        assert config.model.lstm.num_layers == 3

    def test_lstm_config_with_defaults(self, temp_dir):
        """Test LSTM config uses sensible defaults."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'LSTM',
            'FORCING_DATASET': 'ERA5',
        }
        config = SymfluenceConfig(**config_dict)
        # Check defaults are applied
        assert config.model.lstm.hidden_size is not None
        assert config.model.lstm.num_layers is not None
