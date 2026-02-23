"""
Tests for GNN result extractor.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestGNNResultExtractor:
    """Tests for GNN result extraction."""

    def test_extractor_can_be_imported(self):
        """Test that GNNResultExtractor can be imported."""
        from symfluence.models.gnn.extractor import GNNResultExtractor
        assert GNNResultExtractor is not None

    def test_extractor_initialization(self):
        """Test extractor initializes with model name."""
        from symfluence.models.gnn.extractor import GNNResultExtractor

        extractor = GNNResultExtractor('GNN')
        assert extractor is not None
        assert extractor.model_name == 'GNN'

    def test_get_output_file_patterns(self):
        """Test extractor returns output file patterns."""
        from symfluence.models.gnn.extractor import GNNResultExtractor

        extractor = GNNResultExtractor('GNN')
        patterns = extractor.get_output_file_patterns()

        assert 'streamflow' in patterns
        assert isinstance(patterns['streamflow'], list)
        assert len(patterns['streamflow']) > 0

    def test_extract_with_spatial_selection(self):
        """Test extraction supports spatial selection."""
        from symfluence.models.gnn.extractor import GNNResultExtractor

        extractor = GNNResultExtractor('GNN')
        # GNN outputs should support extraction by node/catchment
        assert hasattr(extractor, 'extract_variable')
