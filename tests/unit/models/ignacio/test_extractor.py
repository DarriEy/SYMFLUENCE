"""Tests for IGNACIO result extractor."""

import pytest
import json
import tempfile
from pathlib import Path


class TestIGNACIOResultExtractorImport:
    """Tests for IGNACIO result extractor import and registration."""

    def test_extractor_can_be_imported(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        assert IGNACIOResultExtractor is not None

    def test_extractor_registered_with_registry(self):
        from symfluence.models.registry import ModelRegistry
        assert 'IGNACIO' in ModelRegistry._result_extractors


class TestIGNACIOOutputPatterns:
    """Tests for IGNACIO output file patterns."""

    def test_output_file_patterns(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        patterns = extractor.get_output_file_patterns()
        assert 'burned_area' in patterns
        assert 'summary' in patterns

    def test_burned_area_patterns(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        patterns = extractor.get_output_file_patterns()['burned_area']
        assert any('*.shp' in p for p in patterns)

    def test_summary_patterns(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        patterns = extractor.get_output_file_patterns()['summary']
        assert any('ignacio_summary.json' in p for p in patterns)


class TestIGNACIOVariableNames:
    """Tests for IGNACIO variable name mappings."""

    def test_burned_area_names(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        names = extractor.get_variable_names('burned_area')
        assert 'total_area_ha' in names

    def test_fire_intensity_names(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        names = extractor.get_variable_names('fire_intensity')
        assert 'hfi' in names

    def test_rate_of_spread_names(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        names = extractor.get_variable_names('rate_of_spread')
        assert 'ros' in names


class TestIGNACIOJSONExtraction:
    """Tests for IGNACIO JSON extraction."""

    def test_extract_from_summary_json(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'ignacio_summary.json'
            data = {
                'statistics': {
                    'total_area_ha': 150.5,
                    'n_perimeters': 3,
                }
            }
            json_path.write_text(json.dumps(data))

            extractor = IGNACIOResultExtractor()
            series = extractor.extract_variable(json_path, 'burned_area')
            assert len(series) == 1
            assert series.iloc[0] == 150.5

    def test_extract_iou_from_validation(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'ignacio_summary.json'
            data = {
                'statistics': {},
                'observed_validation': {
                    'iou': 0.75,
                    'dice': 0.86,
                }
            }
            json_path.write_text(json.dumps(data))

            extractor = IGNACIOResultExtractor()
            series = extractor.extract_variable(json_path, 'iou')
            assert series.iloc[0] == 0.75

    def test_extract_missing_variable_raises(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'ignacio_summary.json'
            json_path.write_text(json.dumps({'statistics': {}}))

            extractor = IGNACIOResultExtractor()
            with pytest.raises(ValueError, match="not found"):
                extractor.extract_variable(json_path, 'nonexistent')


class TestIGNACIOExtractorProperties:
    """Tests for IGNACIO extractor properties."""

    def test_no_unit_conversion_needed(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        assert extractor.requires_unit_conversion('burned_area') is False

    def test_spatial_aggregation_method(self):
        from symfluence.models.ignacio.extractor import IGNACIOResultExtractor
        extractor = IGNACIOResultExtractor()
        assert extractor.get_spatial_aggregation_method('burned_area') == 'sum'
