"""
Tests for Model Result Extractors.

Verifies that model-specific result extraction adapters are properly
registered and provide expected interfaces.
"""

import pytest
from symfluence.models.base import ModelResultExtractor
from symfluence.models.registry import ModelRegistry


class TestResultExtractorRegistration:
    """Test that result extractors are properly registered."""

    def test_summa_extractor_registered(self):
        """SUMMA extractor should be registered."""
        extractor = ModelRegistry.get_result_extractor('SUMMA')
        assert extractor is not None
        assert isinstance(extractor, ModelResultExtractor)

    def test_mizuroute_extractor_registered(self):
        """mizuRoute extractor should be registered."""
        extractor = ModelRegistry.get_result_extractor('MIZUROUTE')
        assert extractor is not None
        assert isinstance(extractor, ModelResultExtractor)

    def test_ngen_extractor_registered(self):
        """NGEN extractor should be registered."""
        extractor = ModelRegistry.get_result_extractor('NGEN')
        assert extractor is not None
        assert isinstance(extractor, ModelResultExtractor)

    def test_hype_extractor_registered(self):
        """HYPE extractor should be registered."""
        extractor = ModelRegistry.get_result_extractor('HYPE')
        assert extractor is not None
        assert isinstance(extractor, ModelResultExtractor)

    def test_gr_extractor_registered(self):
        """GR extractor should be registered."""
        extractor = ModelRegistry.get_result_extractor('GR')
        assert extractor is not None
        assert isinstance(extractor, ModelResultExtractor)

    def test_list_result_extractors(self):
        """Should be able to list all registered extractors."""
        extractors = ModelRegistry.list_result_extractors()
        assert isinstance(extractors, list)
        assert 'SUMMA' in extractors
        assert 'MIZUROUTE' in extractors
        assert 'NGEN' in extractors
        assert 'HYPE' in extractors
        assert 'GR' in extractors

    def test_has_result_extractor(self):
        """Should correctly check if model has extractor."""
        assert ModelRegistry.has_result_extractor('SUMMA') is True
        assert ModelRegistry.has_result_extractor('MIZUROUTE') is True
        assert ModelRegistry.has_result_extractor('NONEXISTENT') is False


class TestSUMMAExtractor:
    """Test SUMMA result extractor."""

    def test_summa_provides_file_patterns(self):
        """SUMMA extractor should provide file patterns."""
        extractor = ModelRegistry.get_result_extractor('SUMMA')
        patterns = extractor.get_output_file_patterns()

        assert isinstance(patterns, dict)
        assert 'streamflow' in patterns
        assert isinstance(patterns['streamflow'], list)
        assert len(patterns['streamflow']) > 0

    def test_summa_provides_variable_names(self):
        """SUMMA extractor should provide variable names."""
        extractor = ModelRegistry.get_result_extractor('SUMMA')
        var_names = extractor.get_variable_names('streamflow')

        assert isinstance(var_names, list)
        assert 'averageRoutedRunoff' in var_names or 'scalarTotalRunoff' in var_names

    def test_summa_requires_unit_conversion(self):
        """SUMMA should indicate unit conversion needed for streamflow."""
        extractor = ModelRegistry.get_result_extractor('SUMMA')
        assert extractor.requires_unit_conversion('streamflow') is True

    def test_summa_spatial_aggregation_method(self):
        """SUMMA should indicate spatial aggregation method."""
        extractor = ModelRegistry.get_result_extractor('SUMMA')
        method = extractor.get_spatial_aggregation_method('streamflow')
        assert method == 'weighted'


class TestMizuRouteExtractor:
    """Test mizuRoute result extractor."""

    def test_mizuroute_provides_file_patterns(self):
        """mizuRoute extractor should provide file patterns."""
        extractor = ModelRegistry.get_result_extractor('MIZUROUTE')
        patterns = extractor.get_output_file_patterns()

        assert isinstance(patterns, dict)
        assert 'streamflow' in patterns
        assert any('mizuRoute' in p for p in patterns['streamflow'])

    def test_mizuroute_provides_variable_names(self):
        """mizuRoute extractor should provide variable names."""
        extractor = ModelRegistry.get_result_extractor('MIZUROUTE')
        var_names = extractor.get_variable_names('streamflow')

        assert isinstance(var_names, list)
        assert 'IRFroutedRunoff' in var_names or 'KWTroutedRunoff' in var_names

    def test_mizuroute_no_unit_conversion(self):
        """mizuRoute outputs are already in m³/s."""
        extractor = ModelRegistry.get_result_extractor('MIZUROUTE')
        assert extractor.requires_unit_conversion('streamflow') is False


class TestNGENExtractor:
    """Test NGEN result extractor."""

    def test_ngen_provides_file_patterns(self):
        """NGEN extractor should provide file patterns."""
        extractor = ModelRegistry.get_result_extractor('NGEN')
        patterns = extractor.get_output_file_patterns()

        assert isinstance(patterns, dict)
        assert 'streamflow' in patterns
        assert any('troute' in p.lower() or 'nexus' in p.lower() for p in patterns['streamflow'])

    def test_ngen_provides_variable_names(self):
        """NGEN extractor should provide variable names."""
        extractor = ModelRegistry.get_result_extractor('NGEN')
        var_names = extractor.get_variable_names('streamflow')

        assert isinstance(var_names, list)
        assert len(var_names) > 0


class TestHYPEExtractor:
    """Test HYPE result extractor."""

    def test_hype_provides_file_patterns(self):
        """HYPE extractor should provide file patterns."""
        extractor = ModelRegistry.get_result_extractor('HYPE')
        patterns = extractor.get_output_file_patterns()

        assert isinstance(patterns, dict)
        assert 'streamflow' in patterns
        assert 'timeCOUT.txt' in patterns['streamflow']

    def test_hype_provides_variable_names(self):
        """HYPE extractor should provide variable names."""
        extractor = ModelRegistry.get_result_extractor('HYPE')
        var_names = extractor.get_variable_names('streamflow')

        assert isinstance(var_names, list)
        assert 'COUT' in var_names or 'streamflow' in var_names


class TestGRExtractor:
    """Test GR result extractor."""

    def test_gr_provides_file_patterns(self):
        """GR extractor should provide file patterns."""
        extractor = ModelRegistry.get_result_extractor('GR')
        patterns = extractor.get_output_file_patterns()

        assert isinstance(patterns, dict)
        assert 'streamflow' in patterns
        assert 'GR_results.csv' in patterns['streamflow']

    def test_gr_provides_variable_names(self):
        """GR extractor should provide variable names."""
        extractor = ModelRegistry.get_result_extractor('GR')
        var_names = extractor.get_variable_names('streamflow')

        assert isinstance(var_names, list)
        assert 'Qsim' in var_names or 'Q' in var_names


class TestExtractorInterface:
    """Test that all extractors implement required interface."""

    @pytest.mark.parametrize('model_name', ['SUMMA', 'MIZUROUTE', 'NGEN', 'HYPE', 'GR'])
    def test_extractor_has_required_methods(self, model_name):
        """All extractors should implement required interface methods."""
        extractor = ModelRegistry.get_result_extractor(model_name)

        assert hasattr(extractor, 'get_output_file_patterns')
        assert hasattr(extractor, 'get_variable_names')
        assert hasattr(extractor, 'extract_variable')
        assert hasattr(extractor, 'requires_unit_conversion')
        assert hasattr(extractor, 'get_spatial_aggregation_method')

    @pytest.mark.parametrize('model_name', ['SUMMA', 'MIZUROUTE', 'NGEN', 'HYPE', 'GR'])
    def test_extractor_methods_are_callable(self, model_name):
        """All extractor methods should be callable."""
        extractor = ModelRegistry.get_result_extractor(model_name)

        assert callable(extractor.get_output_file_patterns)
        assert callable(extractor.get_variable_names)
        assert callable(extractor.extract_variable)
        assert callable(extractor.requires_unit_conversion)
        assert callable(extractor.get_spatial_aggregation_method)

    @pytest.mark.parametrize('model_name', ['SUMMA', 'MIZUROUTE', 'NGEN', 'HYPE', 'GR'])
    def test_extractor_returns_expected_types(self, model_name):
        """Extractors should return expected types."""
        extractor = ModelRegistry.get_result_extractor(model_name)

        # get_output_file_patterns should return dict
        patterns = extractor.get_output_file_patterns()
        assert isinstance(patterns, dict)

        # get_variable_names should return list
        var_names = extractor.get_variable_names('streamflow')
        assert isinstance(var_names, list)

        # requires_unit_conversion should return bool
        needs_conversion = extractor.requires_unit_conversion('streamflow')
        assert isinstance(needs_conversion, bool)

        # get_spatial_aggregation_method can return str or None
        method = extractor.get_spatial_aggregation_method('streamflow')
        assert method is None or isinstance(method, str)
