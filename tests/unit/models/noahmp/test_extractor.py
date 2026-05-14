"""Tests for Noah-MP result extractor."""


class TestNoahMPResultExtractor:
    """Tests for Noah-MP result extractor."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        assert NoahMPResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import symfluence.models.noahmp  # noqa: F401
        from symfluence.core.registries import R
        assert 'NOAHMP' in R.result_extractors

    def test_output_file_patterns(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
        assert 'et' in patterns
        assert 'snow' in patterns
        assert 'soil_moisture' in patterns
        assert 'energy' in patterns

    def test_variable_names_streamflow(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        names = extractor.get_variable_names('streamflow')
        assert 'SFCRNOFF' in names
        assert 'UGDRNOFF' in names

    def test_variable_names_et(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        names = extractor.get_variable_names('et')
        assert 'ECAN' in names
        assert 'ETRAN' in names
        assert 'EDIR' in names

    def test_variable_names_snow(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        names = extractor.get_variable_names('snow')
        assert 'SNEQV' in names
        assert 'SNOWH' in names

    def test_variable_names_energy(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        names = extractor.get_variable_names('energy')
        assert 'LH' in names
        assert 'FSH' in names

    def test_no_unit_conversion_needed(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        assert extractor.requires_unit_conversion('streamflow') is False

    def test_spatial_aggregation_method(self):
        from symfluence.models.noahmp.extractor import NoahMPResultExtractor
        extractor = NoahMPResultExtractor('NOAHMP')
        assert extractor.get_spatial_aggregation_method('streamflow') == 'sum'
        assert extractor.get_spatial_aggregation_method('et') == 'sum'
        assert extractor.get_spatial_aggregation_method('soil_moisture') == 'mean'
