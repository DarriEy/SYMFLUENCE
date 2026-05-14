"""Tests for PCR-GLOBWB result extractor."""


class TestPCRGLOBWBResultExtractor:
    """Tests for PCR-GLOBWB result extractor."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        assert PCRGLOBWBResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import symfluence.models.pcrglobwb  # noqa: F401
        from symfluence.core.registries import R
        assert 'PCRGLOBWB' in R.result_extractors

    def test_output_file_patterns(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        extractor = PCRGLOBWBResultExtractor('PCRGLOBWB')
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
        assert 'et' in patterns
        assert 'snow' in patterns
        assert 'soil_moisture' in patterns
        assert 'groundwater' in patterns
        assert 'runoff' in patterns

    def test_variable_names_streamflow(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        extractor = PCRGLOBWBResultExtractor('PCRGLOBWB')
        names = extractor.get_variable_names('streamflow')
        assert 'discharge' in names

    def test_variable_names_et(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        extractor = PCRGLOBWBResultExtractor('PCRGLOBWB')
        names = extractor.get_variable_names('et')
        assert 'totalEvaporation' in names

    def test_no_unit_conversion_needed(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        extractor = PCRGLOBWBResultExtractor('PCRGLOBWB')
        assert extractor.requires_unit_conversion('streamflow') is False
        assert extractor.requires_unit_conversion('et') is False

    def test_spatial_aggregation_method(self):
        from symfluence.models.pcrglobwb.extractor import PCRGLOBWBResultExtractor
        extractor = PCRGLOBWBResultExtractor('PCRGLOBWB')
        assert extractor.get_spatial_aggregation_method('streamflow') == 'max'
        assert extractor.get_spatial_aggregation_method('et') == 'sum'
        assert extractor.get_spatial_aggregation_method('soil_moisture') == 'mean'
