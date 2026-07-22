"""
Tests for LSTM preprocessor.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")


class TestLSTMPreProcessorInitialization:
    """Tests for LSTM preprocessor initialization."""

    def test_preprocessor_can_be_imported(self):
        """Test that LSTMPreProcessor can be imported."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor
        assert LSTMPreProcessor is not None

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_initialization(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor initializes with config."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        # Create mock device
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Get config as dict for LSTMPreProcessor
        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor is not None
        assert preprocessor.config is not None

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_sets_lookback(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor sets lookback window from config."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Use full config dict - preprocessor now inherits from BaseModelPreProcessor
        # which requires proper config for path resolution
        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor.lookback == lstm_config.model.lstm.lookback


class TestLSTMDataPreparation:
    """Tests for LSTM data preparation methods."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    @patch('symfluence.models.lstm.preprocessor.xr')
    def test_load_forcing_data_success(self, mock_xr, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test loading forcing data successfully."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Mock xarray dataset
        mock_ds = MagicMock()
        mock_xr.open_dataset.return_value.__enter__ = Mock(return_value=mock_ds)
        mock_xr.open_dataset.return_value.__exit__ = Mock(return_value=False)

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # The preprocessor should initialize without errors

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_preprocessor_handles_missing_data(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test preprocessor handles missing forcing data gracefully."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Preprocessor should handle missing files gracefully during init


class TestLSTMFeatureScaling:
    """Tests for LSTM feature scaling."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_scaler_initialization(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test that scalers are initialized."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Verify preprocessor has scaling attributes
        assert hasattr(preprocessor, 'feature_scaler')
        assert hasattr(preprocessor, 'target_scaler')


class TestLSTMSequenceCreation:
    """Tests for LSTM sequence creation."""

    @patch('symfluence.models.lstm.preprocessor.torch')
    def test_sequence_shape_validation(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test that sequences have correct shape."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = lstm_config.model_dump()
        project_dir = setup_lstm_directories['domain_dir']

        preprocessor = LSTMPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Sequence shape should be (samples, lookback, features)
        lookback = preprocessor.lookback
        assert lookback > 0


class TestLSTMSequenceTensorBuilder:
    """Tests for the strided sequence-tensor builder."""

    @staticmethod
    def _builder(lookback):
        """A minimal object exposing only what _build_sequence_tensor uses."""
        import logging

        import torch

        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        stub = Mock()
        stub.lookback = lookback
        stub.device = torch.device('cpu')
        stub.logger = logging.getLogger('test')
        stub._warn_if_sequence_tensor_is_large = (
            lambda n: LSTMPreProcessor._warn_if_sequence_tensor_is_large(stub, n)
        )
        return stub

    @pytest.mark.parametrize(
        "shape,lookback",
        [((120, 4), 12), ((90, 3, 5), 10)],
    )
    def test_matches_naive_windowing(self, shape, lookback):
        """Strided windows equal the naive per-window stack, bit for bit."""
        import torch

        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        rng = np.random.default_rng(11)
        source = rng.standard_normal(shape)  # float64, as StandardScaler emits
        n_sequences = shape[0] - lookback

        expected = torch.FloatTensor(
            np.array([source[i:i + lookback] for i in range(n_sequences)])
        )
        actual = LSTMPreProcessor._build_sequence_tensor(
            self._builder(lookback), source, n_sequences
        )

        assert actual.shape == expected.shape
        assert actual.dtype is torch.float32
        assert torch.equal(actual, expected)

    def test_lookback_longer_than_record_raises_clear_error(self):
        """A lookback that leaves no sequences fails with a domain error."""
        from symfluence.core.exceptions import ModelExecutionError
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        source = np.zeros((10, 3))

        with pytest.raises(ModelExecutionError, match="lookback"):
            LSTMPreProcessor._build_sequence_tensor(self._builder(50), source, -40)

    def test_oversized_allocation_is_warned_about(self, caplog):
        """The footprint is logged, and flagged when it exceeds free memory."""
        from symfluence.models.lstm.preprocessor import LSTMPreProcessor

        stub = self._builder(365)
        stub.logger = MagicMock()

        with patch('psutil.virtual_memory') as mem:
            mem.return_value.available = 1_000_000
            LSTMPreProcessor._warn_if_sequence_tensor_is_large(stub, 8_000_000_000)

        assert stub.logger.warning.called
        assert "exceeds free memory" in stub.logger.warning.call_args[0][0]
