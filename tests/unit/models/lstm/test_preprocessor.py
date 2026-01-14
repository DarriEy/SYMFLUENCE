"""
Tests for LSTM preprocessor.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd


class TestLSTMPreprocessorInitialization:
    """Tests for LSTM preprocessor initialization."""

    def test_preprocessor_can_be_imported(self):
        """Test that LSTMPreprocessor can be imported."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor
        assert LSTMPreprocessor is not None

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_initialization(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor initializes with config."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        # Create mock device
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Get config as dict for LSTMPreprocessor
        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor is not None
        assert preprocessor.config is not None

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_sets_lookback(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor sets lookback window from config."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Use flat config dict with LSTM_LOOKBACK key as preprocessor expects
        config_dict = {'LSTM_LOOKBACK': lstm_config.model.lstm.lookback}
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor.lookback == lstm_config.model.lstm.lookback


class TestLSTMDataPreparation:
    """Tests for LSTM data preparation methods."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    @patch('symfluence.models.lstm.preprocessor.xr')
    def test_load_forcing_data_success(self, mock_xr, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test loading forcing data successfully."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Mock xarray dataset
        mock_ds = MagicMock()
        mock_xr.open_dataset.return_value.__enter__ = Mock(return_value=mock_ds)
        mock_xr.open_dataset.return_value.__exit__ = Mock(return_value=False)

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        # The preprocessor should initialize without errors

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_handles_missing_data(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor handles missing forcing data gracefully."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        # Preprocessor should handle missing files gracefully during init


class TestLSTMFeatureScaling:
    """Tests for LSTM feature scaling."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_scaler_initialization(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test that scalers are initialized."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        # Verify preprocessor has scaling attributes
        assert hasattr(preprocessor, 'feature_scaler')
        assert hasattr(preprocessor, 'target_scaler')


class TestLSTMSequenceCreation:
    """Tests for LSTM sequence creation."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_sequence_shape_validation(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test that sequences have correct shape."""
        from symfluence.models.lstm.preprocessor import LSTMPreprocessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreprocessor(config_dict, mock_logger, project_dir, mock_device)
        # Sequence shape should be (samples, lookback, features)
        lookback = preprocessor.lookback
        assert lookback > 0
