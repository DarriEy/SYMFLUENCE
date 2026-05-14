"""Tests for CWatM result extractor."""


class TestCWatMResultExtractor:
    """Tests for CWatM result extractor."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.cwatm.extractor import CWatMResultExtractor
        assert CWatMResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import symfluence.models.cwatm  # noqa: F401
        from symfluence.core.registries import R
        assert 'CWATM' in R.result_extractors

    def test_output_file_patterns(self):
        from symfluence.models.cwatm.extractor import CWatMResultExtractor
        extractor = CWatMResultExtractor('CWATM')
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
        assert 'et' in patterns
        assert 'snow' in patterns
        assert 'groundwater' in patterns

    def test_variable_names_streamflow(self):
        from symfluence.models.cwatm.extractor import CWatMResultExtractor
        extractor = CWatMResultExtractor('CWATM')
        names = extractor.get_variable_names('streamflow')
        assert 'discharge' in names

    def test_no_unit_conversion_needed(self):
        from symfluence.models.cwatm.extractor import CWatMResultExtractor
        extractor = CWatMResultExtractor('CWATM')
        assert extractor.requires_unit_conversion('streamflow') is False

    def test_spatial_aggregation_method(self):
        from symfluence.models.cwatm.extractor import CWatMResultExtractor
        extractor = CWatMResultExtractor('CWATM')
        assert extractor.get_spatial_aggregation_method('streamflow') == 'max'
        assert extractor.get_spatial_aggregation_method('et') == 'sum'
        assert extractor.get_spatial_aggregation_method('soil_moisture') == 'mean'
