"""
Tests for LSTM result extractor.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestLSTMResultExtractor:
    """Tests for LSTM result extraction."""

    def test_extractor_can_be_imported(self):
        """Test that LSTMResultExtractor can be imported."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor
        assert LSTMResultExtractor is not None

    def test_extractor_initialization(self):
        """Test extractor initializes with model name."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')
        assert extractor is not None
        assert extractor.model_name == 'LSTM'

    def test_get_output_file_patterns(self):
        """Test extractor returns output file patterns."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')
        patterns = extractor.get_output_file_patterns()

        assert 'streamflow' in patterns
        assert isinstance(patterns['streamflow'], list)
        assert len(patterns['streamflow']) > 0

    def test_get_variable_names(self):
        """Test extractor maps variable names correctly."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')
        var_names = extractor.get_variable_names('streamflow')

        assert 'predicted_streamflow' in var_names or 'streamflow' in var_names


class TestLSTMDataExtraction:
    """Tests for data extraction from LSTM outputs."""

    def test_extract_variable_netcdf(self):
        """Test extracting variables from LSTM NetCDF output."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')

        # Verify extractor has extract_variable method
        assert hasattr(extractor, 'extract_variable')

    def test_spatial_aggregation_method(self):
        """Test LSTM spatial aggregation method."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')
        method = extractor.get_spatial_aggregation_method('streamflow')

        # LSTM uses outlet_selection for selecting specific HRU/outlet
        assert method in ['weighted_sum', 'selection', 'sum', 'mean', 'outlet_selection']

    def test_unit_conversion_required(self):
        """Test if LSTM outputs require unit conversion."""
        from symfluence.models.lstm.extractor import LSTMResultExtractor

        extractor = LSTMResultExtractor('LSTM')
        requires = extractor.requires_unit_conversion('streamflow')

        # Should return a boolean
        assert isinstance(requires, bool)
