"""Tests for ObservationsNetCDFBuilder."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

netCDF4 = pytest.importorskip('netCDF4')

from symfluence.data.model_ready.observations_builder import ObservationsNetCDFBuilder


def _write_streamflow_csv(path: Path, domain: str = 'test') -> None:
    """Create a minimal streamflow CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'datetime': dates,
        'discharge_cms': np.random.uniform(1, 10, 30),
    })
    df.to_csv(path, index=False)


def _write_snow_csv(path: Path) -> None:
    """Create a minimal snow CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    df = pd.DataFrame({'swe_mm': np.random.uniform(0, 50, 30)}, index=dates)
    df.index.name = 'datetime'
    df.to_csv(path)


class TestObservationsNetCDFBuilder:
    """Tests for observations store construction."""

    def test_build_single_streamflow(self, tmp_path):
        obs = tmp_path / 'observations' / 'streamflow' / 'preprocessed'
        _write_streamflow_csv(obs / 'test_streamflow_processed.csv')

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is not None
        assert result.exists()

        # Verify group structure
        ds = netCDF4.Dataset(str(result), 'r')
        assert 'streamflow' in ds.groups
        grp = ds.groups['streamflow']
        assert 'time' in grp.dimensions
        assert 'gauge' in grp.dimensions
        assert 'discharge_cms' in grp.variables
        assert grp.variables['time'].units == 'days since 1970-01-01'
        ds.close()

    def test_build_with_snow(self, tmp_path):
        obs = tmp_path / 'observations'
        _write_streamflow_csv(
            obs / 'streamflow' / 'preprocessed' / 'test_streamflow_processed.csv'
        )
        _write_snow_csv(obs / 'snow' / 'swe' / 'processed' / 'test_swe_processed.csv')

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is not None

        ds = netCDF4.Dataset(str(result), 'r')
        assert 'streamflow' in ds.groups
        assert 'snow' in ds.groups
        ds.close()

    def test_build_no_observations(self, tmp_path):
        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is None

    def test_build_empty_dir(self, tmp_path):
        (tmp_path / 'observations' / 'streamflow').mkdir(parents=True)
        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is None

    def test_global_attrs(self, tmp_path):
        obs = tmp_path / 'observations' / 'streamflow' / 'preprocessed'
        _write_streamflow_csv(obs / 'test_streamflow_processed.csv')

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        assert ds.Conventions == 'CF-1.8'
        assert ds.domain_name == 'test'
        ds.close()

    def test_variable_has_cf_attrs(self, tmp_path):
        obs = tmp_path / 'observations' / 'streamflow' / 'preprocessed'
        _write_streamflow_csv(obs / 'test_streamflow_processed.csv')

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        v = ds.groups['streamflow'].variables['discharge_cms']
        assert v.standard_name == 'water_volume_transport_in_river_channel'
        assert v.units == 'm3 s-1'
        ds.close()

    def test_source_metadata_on_variable(self, tmp_path):
        obs = tmp_path / 'observations' / 'streamflow' / 'preprocessed'
        _write_streamflow_csv(obs / 'test_streamflow_processed.csv')

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
            config_dict={'STREAMFLOW_DATA_PROVIDER': 'USGS'},
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        v = ds.groups['streamflow'].variables['discharge_cms']
        assert v.source_source == 'USGS'
        ds.close()

    def test_multi_gauge(self, tmp_path):
        multi_dir = tmp_path / 'multi_gauge_obs'
        multi_dir.mkdir(parents=True)
        for gid in ['001', '002']:
            dates = pd.date_range('2020-01-01', periods=10, freq='D')
            df = pd.DataFrame({
                'datetime': dates,
                'discharge_cms': np.random.uniform(1, 10, 10),
            })
            df.to_csv(multi_dir / f'ID_{gid}.csv', index=False)

        builder = ObservationsNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
            config_dict={'MULTI_GAUGE_OBS_DIR': str(multi_dir)},
        )
        # Also need observations dir to exist for the build guard
        (tmp_path / 'observations').mkdir()
        result = builder.build()
        assert result is not None

        ds = netCDF4.Dataset(str(result), 'r')
        grp = ds.groups['streamflow']
        assert grp.dimensions['gauge'].size == 2
        ds.close()
