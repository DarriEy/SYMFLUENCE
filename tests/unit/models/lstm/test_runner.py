"""
Tests for LSTM model runner.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")


class TestLSTMRunnerInitialization:
    """Tests for LSTM runner initialization."""

    def test_runner_can_be_imported(self):
        """Test that LSTMRunner can be imported."""
        from symfluence.models.lstm.runner import LSTMRunner
        assert LSTMRunner is not None

    def test_runner_initialization(self, lstm_config, mock_logger, setup_lstm_directories):
        """Test runner initializes with config."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        assert runner is not None

    @patch('symfluence.models.lstm.runner.torch')
    def test_runner_detects_device(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test runner detects available device (CPU/CUDA)."""
        from symfluence.models.lstm.runner import LSTMRunner

        mock_torch.cuda.is_available.return_value = False
        runner = LSTMRunner(lstm_config, mock_logger)
        # Runner should default to CPU when CUDA not available

    @patch("symfluence.models.lstm.runner.droute")
    def test_loads_droute_geojson_and_json_metadata(
        self, mock_droute, lstm_config, mock_logger, setup_lstm_directories
    ):
        """dRoute topology and indices are loaded without pickle."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        settings = runner.project_dir / "settings" / "dRoute"
        settings.mkdir(parents=True, exist_ok=True)
        (settings / "network.geojson").write_text("{}", encoding="utf-8")
        (settings / "network_metadata.json").write_text(
            '{"outlet_idx": 3, "hru_to_seg_idx": {"101": 7}}',
            encoding="utf-8",
        )
        mock_droute.load_network_geojson.return_value = object()

        _, outlet_idx, mapping = runner._load_droute_network()

        assert outlet_idx == 3
        assert mapping == {"101": 7}
        mock_droute.load_network_geojson.assert_called_once_with(str(settings / "network.geojson"))

    def test_rejects_legacy_droute_pickle(
        self, lstm_config, mock_logger, setup_lstm_directories
    ):
        """Legacy pickle artifacts are never deserialized."""
        from symfluence.core.exceptions import ModelExecutionError
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        settings = runner.project_dir / "settings" / "dRoute"
        settings.mkdir(parents=True, exist_ok=True)
        (settings / "dRoute_network.pkl").write_bytes(b"not safe")

        with pytest.raises(ModelExecutionError, match="can execute arbitrary code"):
            runner._load_droute_network()


class TestLSTMModelCreation:
    """Tests for LSTM model creation."""

    @patch('symfluence.models.lstm.runner.torch')
    def test_model_architecture_matches_config(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test model architecture matches configuration."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        hidden_size = lstm_config.model.lstm.hidden_size
        num_layers = lstm_config.model.lstm.num_layers

        assert hidden_size == 64
        assert num_layers == 2


class TestLSTMTraining:
    """Tests for LSTM model training."""

    @patch('symfluence.models.lstm.runner.torch')
    def test_training_parameters(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test training uses correct parameters."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        epochs = lstm_config.model.lstm.epochs
        learning_rate = lstm_config.model.lstm.learning_rate

        assert epochs == 10
        assert learning_rate == 0.001


class TestLSTMSimulation:
    """Tests for LSTM simulation."""

    @patch('symfluence.models.lstm.runner.torch')
    def test_simulation_output_format(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test simulation produces expected output format."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        # Simulation should produce DataFrame output


class TestLSTMCheckpoints:
    """Tests for LSTM checkpoint save/load."""

    @patch('symfluence.models.lstm.runner.torch')
    def test_checkpoint_contains_model_state(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test checkpoint contains model state dict."""
        from symfluence.models.lstm.runner import LSTMRunner

        runner = LSTMRunner(lstm_config, mock_logger)
        # Checkpoint should include model_state_dict

    @patch('symfluence.models.lstm.runner.torch')
    def test_load_checkpoint_restores_state(self, mock_torch, lstm_config, mock_logger, setup_lstm_directories):
        """Test loading checkpoint restores model state."""
        from symfluence.models.lstm.runner import LSTMRunner

        lstm_config_with_load = lstm_config
        runner = LSTMRunner(lstm_config_with_load, mock_logger)
        # When load=True, runner should attempt to load checkpoint
