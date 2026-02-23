"""
Tests for GNN configuration.
"""

import pytest

from symfluence.core.config.models import SymfluenceConfig


class TestGNNConfig:
    """Tests for GNN configuration."""

    def test_gnn_hidden_size_configured(self, gnn_config):
        """Test GNN hidden size configuration."""
        assert gnn_config.model.gnn.hidden_size == 64

    def test_gnn_num_layers_configured(self, gnn_config):
        """Test GNN number of layers configuration."""
        assert gnn_config.model.gnn.num_layers == 3

    def test_gnn_epochs_configured(self, gnn_config):
        """Test GNN epochs configuration."""
        assert gnn_config.model.gnn.epochs == 10

    def test_gnn_batch_size_configured(self, gnn_config):
        """Test GNN batch size configuration."""
        assert gnn_config.model.gnn.batch_size == 32

    def test_gnn_learning_rate_configured(self, gnn_config):
        """Test GNN learning rate configuration."""
        assert gnn_config.model.gnn.learning_rate == 0.001

    def test_gnn_dropout_configured(self, gnn_config):
        """Test GNN dropout configuration."""
        assert gnn_config.model.gnn.dropout == 0.1

    def test_gnn_load_default(self, gnn_config):
        """Test GNN load checkpoint default."""
        assert gnn_config.model.gnn.load is False


class TestGNNConfigValidation:
    """Tests for GNN configuration validation."""

    def test_valid_gnn_config_passes(self, temp_dir):
        """Test that valid GNN config passes validation."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'GNN',
            'FORCING_DATASET': 'ERA5',
            'GNN_HIDDEN_SIZE': 128,
            'GNN_NUM_LAYERS': 4,
        }
        config = SymfluenceConfig(**config_dict)
        assert config.model.gnn.hidden_size == 128
        assert config.model.gnn.num_layers == 4

    def test_gnn_requires_distributed_domain(self, gnn_config):
        """Test that GNN is typically used with distributed domain."""
        # GNN benefits from spatial structure
        # Note: 'delineate' is auto-mapped to 'semidistributed' for backward compatibility
        assert gnn_config.domain.definition_method in ('semidistributed', 'distributed', 'delineate')
