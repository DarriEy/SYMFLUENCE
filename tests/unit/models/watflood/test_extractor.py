"""Tests for WATFLOOD result extractor."""

import pandas as pd
import pytest


class TestWATFLOODResultExtractor:
    """Tests for WATFLOOD result extractor."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor
        assert WATFLOODResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        import symfluence.models.watflood  # noqa: F401 — trigger registration
        from symfluence.core.registries import R
        assert 'WATFLOOD' in R.result_extractors

    def test_output_file_patterns(self):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor
        extractor = WATFLOODResultExtractor('WATFLOOD')
        patterns = extractor.get_output_file_patterns()
        assert 'streamflow' in patterns
        assert any('tb0' in p for p in patterns['streamflow'])

    def test_variable_names_streamflow(self):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor
        extractor = WATFLOODResultExtractor('WATFLOOD')
        names = extractor.get_variable_names('streamflow')
        assert 'QO' in names
        assert 'QSIM' in names

    def test_requires_unit_conversion(self):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor
        extractor = WATFLOODResultExtractor('WATFLOOD')
        assert extractor.requires_unit_conversion('streamflow') is False
        assert extractor.requires_unit_conversion('et') is True

    def test_spatial_aggregation_method(self):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor
        extractor = WATFLOODResultExtractor('WATFLOOD')
        assert extractor.get_spatial_aggregation_method('streamflow') == 'sum'
        assert extractor.get_spatial_aggregation_method('et') == 'mean'


class TestTb0Parsing:
    """Tests for .tb0 file parsing."""

    def test_parse_tb0_file(self, tmp_path):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor

        tb0_content = """:ColumnMetaData
:EndColumnMetaData
2005 1 1 0 12.5
2005 1 2 0 15.3
2005 1 3 0 11.8
"""
        tb0_file = tmp_path / "spl_test.tb0"
        tb0_file.write_text(tb0_content)

        extractor = WATFLOODResultExtractor('WATFLOOD')
        result = extractor._parse_tb0_file(tb0_file, 'streamflow')

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(12.5)
        assert result.iloc[2] == pytest.approx(11.8)
        assert result.index[0] == pd.Timestamp(2005, 1, 1, 0)


class TestCsvParsing:
    """Tests for CSV output parsing."""

    def test_extract_csv_streamflow(self, tmp_path):
        from symfluence.models.watflood.extractor import WATFLOODResultExtractor

        csv_content = """date,QSIM,precip
2005-01-01,12.5,3.2
2005-01-02,15.3,0.0
2005-01-03,11.8,1.1
"""
        csv_file = tmp_path / "CHARM_dly.csv"
        csv_file.write_text(csv_content)

        extractor = WATFLOODResultExtractor('WATFLOOD')
        result = extractor.extract_variable(csv_file, 'streamflow')

        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(12.5)
        assert result.iloc[1] == pytest.approx(15.3)
