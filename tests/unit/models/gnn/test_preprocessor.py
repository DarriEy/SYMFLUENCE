"""
Tests for GNN preprocessor.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGNNPreProcessorInitialization:
    """Tests for GNN preprocessor initialization."""

    def test_preprocessor_can_be_imported(self):
        """Test that GNNPreProcessor can be imported."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor
        assert GNNPreProcessor is not None

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_preprocessor_initialization(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test preprocessor initializes with config."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor is not None


class TestGNNGraphLoading:
    """Tests for GNN graph structure loading."""

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_graph_structure_initialization(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test graph structure is initialized from river network."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # GNN should have adjacency matrix attribute
        assert hasattr(preprocessor, 'adj_matrix')

    @patch('symfluence.models.gnn.preprocessor.torch')
    @patch('symfluence.models.gnn.preprocessor.gpd')
    def test_load_river_network_for_adjacency(self, mock_gpd, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test loading river network to build adjacency matrix."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        mock_gdf = MagicMock()
        mock_gpd.read_file.return_value = mock_gdf

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # River network defines graph edges
        assert hasattr(preprocessor, 'node_mapping')


class TestGNNFeaturePreparation:
    """Tests for GNN feature preparation."""

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_node_features_preparation(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test node features are prepared correctly."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Each node (catchment) should have HRU mapping
        assert hasattr(preprocessor, 'hru_to_node')

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_edge_features_preparation(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test edge features are prepared if needed."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Should have outlet indices for target assignment
        assert hasattr(preprocessor, 'outlet_indices')
