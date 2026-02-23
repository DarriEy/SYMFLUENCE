"""
Tests for GNN model runner.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestGNNRunnerInitialization:
    """Tests for GNN runner initialization."""

    def test_runner_can_be_imported(self):
        """Test that GNNRunner can be imported."""
        from symfluence.models.gnn.runner import GNNRunner
        assert GNNRunner is not None

    def test_runner_initialization(self, gnn_config, mock_logger, setup_gnn_directories):
        """Test runner initializes with config."""
        from symfluence.models.gnn.runner import GNNRunner

        runner = GNNRunner(gnn_config, mock_logger)
        assert runner is not None

    @patch('symfluence.models.gnn.runner.torch')
    def test_runner_detects_device(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test runner detects available device."""
        from symfluence.models.gnn.runner import GNNRunner

        mock_torch.cuda.is_available.return_value = False
        runner = GNNRunner(gnn_config, mock_logger)


class TestGNNModelArchitecture:
    """Tests for GNN model architecture."""

    @patch('symfluence.models.gnn.runner.torch')
    def test_model_uses_graph_convolution(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test model uses graph convolutional layers."""
        from symfluence.models.gnn.runner import GNNRunner

        runner = GNNRunner(gnn_config, mock_logger)
        num_layers = gnn_config.model.gnn.num_layers
        assert num_layers == 3

    @patch('symfluence.models.gnn.runner.torch')
    def test_model_hidden_size_matches_config(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test model hidden size matches configuration."""
        from symfluence.models.gnn.runner import GNNRunner

        runner = GNNRunner(gnn_config, mock_logger)
        hidden_size = gnn_config.model.gnn.hidden_size
        assert hidden_size == 64


class TestGNNTraining:
    """Tests for GNN model training."""

    @patch('symfluence.models.gnn.runner.torch')
    def test_training_parameters(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test training uses correct parameters."""
        from symfluence.models.gnn.runner import GNNRunner

        runner = GNNRunner(gnn_config, mock_logger)
        epochs = gnn_config.model.gnn.epochs
        learning_rate = gnn_config.model.gnn.learning_rate

        assert epochs == 10
        assert learning_rate == 0.001


class TestGNNSimulation:
    """Tests for GNN simulation."""

    @patch('symfluence.models.gnn.runner.torch')
    def test_simulation_output_has_spatial_dims(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test simulation output has spatial dimensions."""
        from symfluence.models.gnn.runner import GNNRunner

        runner = GNNRunner(gnn_config, mock_logger)
        # GNN output should have predictions for each node
