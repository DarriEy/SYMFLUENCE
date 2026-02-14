"""Tests for SourceMetadata dataclass."""

import pytest
from symfluence.data.model_ready.source_metadata import SourceMetadata


class TestSourceMetadata:
    """Tests for SourceMetadata round-trip and attribute handling."""

    def test_to_netcdf_attrs_basic(self):
        meta = SourceMetadata(source='ERA5', processing='area-weighted remapping')
        attrs = meta.to_netcdf_attrs()
        assert attrs['source_source'] == 'ERA5'
        assert attrs['source_processing'] == 'area-weighted remapping'
        # Optional None fields should be absent
        assert 'source_source_doi' not in attrs
        assert 'source_url' not in attrs

    def test_to_netcdf_attrs_full(self):
        meta = SourceMetadata(
            source='USGS NWIS',
            processing='QC filtered',
            acquisition_date='2025-01-15',
            source_doi='10.5066/F7P55KJN',
            original_units='cfs',
            version='2.1',
            url='https://waterservices.usgs.gov/nwis/',
        )
        attrs = meta.to_netcdf_attrs()
        assert len(attrs) == 7
        assert attrs['source_version'] == '2.1'
        assert attrs['source_original_units'] == 'cfs'

    def test_from_netcdf_attrs_roundtrip(self):
        original = SourceMetadata(
            source='ERA5',
            processing='bilinear interpolation',
            acquisition_date='2024-06-01',
            source_doi='10.24381/cds.adbb2d47',
        )
        attrs = original.to_netcdf_attrs()
        restored = SourceMetadata.from_netcdf_attrs(attrs)
        assert restored.source == original.source
        assert restored.processing == original.processing
        assert restored.acquisition_date == original.acquisition_date
        assert restored.source_doi == original.source_doi

    def test_from_netcdf_attrs_unknown_keys_ignored(self):
        attrs = {
            'source_source': 'RDRS',
            'source_processing': 'merged',
            'unrelated_key': 'value',
            'source_nonexistent_field': 'ignored',
        }
        meta = SourceMetadata.from_netcdf_attrs(attrs)
        assert meta.source == 'RDRS'
        assert meta.processing == 'merged'

    def test_from_netcdf_attrs_missing_source(self):
        attrs = {'source_processing': 'test'}
        meta = SourceMetadata.from_netcdf_attrs(attrs)
        assert meta.source == 'unknown'

    def test_empty_strings_omitted(self):
        meta = SourceMetadata(source='ERA5')
        attrs = meta.to_netcdf_attrs()
        # Empty default strings should not appear
        assert 'source_processing' not in attrs
        assert 'source_acquisition_date' not in attrs
