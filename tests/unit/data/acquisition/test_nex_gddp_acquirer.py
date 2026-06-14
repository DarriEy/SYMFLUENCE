# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the NEX-GDDP-CMIP6 acquisition handler hardening.

Pins the fixes from the 2026-06 parity-validation sweep:

1. The handler hardcoded a ``_v2.0`` filename suffix and, on 404, silently
   dropped the variable — producing partial forcing with no error. Now it
   falls back ``_v2.0`` -> ``_v1.1`` -> unsuffixed and raises when any
   requested variable yields zero files.
2. Dead code (discarded strftime results) removed.
3. When DOMAIN_MEAN_ELEV_M is unset, the fabricated ``airpres`` is a constant
   sea-level 101325 Pa — a prominent warning is now emitted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.nex_gddp import NEXGDDPCHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]


def _make_config(tmp_path, **extra):
    cfg = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2010-01-01 00:00',
        'EXPERIMENT_TIME_END': '2010-01-31 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'NEX-GDDP-CMIP6',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 86400,
        'BOUNDING_BOX_COORDS': '52/-116/50/-114',
        'NEX_MODELS': ['ACCESS-CM2'],
        'NEX_SCENARIOS': ['historical'],
        'NEX_VARIABLES': ['tas'],
        'NEX_ENSEMBLES': ['r1i1p1f1'],
    }
    cfg.update(extra)
    return cfg


def _make_handler(tmp_path, **extra):
    return NEXGDDPCHandler(_make_config(tmp_path, **extra), logging.getLogger('test_nex'))


def _mock_response(status_code, content=b''):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = 'not found' if status_code != 200 else ''
    resp.iter_content.return_value = [content]
    return resp


class TestSuffixFallbackChain:
    def test_falls_back_v20_v11_unsuffixed(self, tmp_path):
        """404 on _v2.0 and _v1.1 must fall through to the unsuffixed name."""
        handler = _make_handler(tmp_path)
        cache = tmp_path / 'cache'
        cache.mkdir()
        responses = [_mock_response(404), _mock_response(404), _mock_response(200, b'NCDF')]
        with patch('symfluence.data.acquisition.handlers.nex_gddp.requests.get',
                   side_effect=responses) as mock_get:
            import datetime as dt
            result = handler._fetch_year_chunk(
                'https://ncss.example/grid', cache, 'tas', 'ACCESS-CM2', 'historical',
                'r1i1p1f1', 'gn', 2010, dt.date(2010, 1, 1), dt.date(2010, 12, 31),
                50.0, 52.0, -116.0, -114.0)

        assert result is not None
        assert result.exists()
        urls = [call.args[0] for call in mock_get.call_args_list]
        assert urls[0].endswith('_2010_v2.0.nc')
        assert urls[1].endswith('_2010_v1.1.nc')
        assert urls[2].endswith('_2010.nc')

    def test_returns_none_when_all_suffixes_fail(self, tmp_path):
        handler = _make_handler(tmp_path)
        cache = tmp_path / 'cache'
        cache.mkdir()
        with patch('symfluence.data.acquisition.handlers.nex_gddp.requests.get',
                   return_value=_mock_response(404)) as mock_get:
            import datetime as dt
            result = handler._fetch_year_chunk(
                'https://ncss.example/grid', cache, 'tas', 'ACCESS-CM2', 'historical',
                'r1i1p1f1', 'gn', 2010, dt.date(2010, 1, 1), dt.date(2010, 12, 31),
                50.0, 52.0, -116.0, -114.0)

        assert result is None
        assert mock_get.call_count == 3

    def test_cached_file_short_circuits(self, tmp_path):
        handler = _make_handler(tmp_path)
        cache = tmp_path / 'cache'
        cache.mkdir()
        cached = cache / 'tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_2010_v2.0_20100101-20101231.nc'
        cached.write_bytes(b'NCDF')
        with patch('symfluence.data.acquisition.handlers.nex_gddp.requests.get') as mock_get:
            import datetime as dt
            result = handler._fetch_year_chunk(
                'https://ncss.example/grid', cache, 'tas', 'ACCESS-CM2', 'historical',
                'r1i1p1f1', 'gn', 2010, dt.date(2010, 1, 1), dt.date(2010, 12, 31),
                50.0, 52.0, -116.0, -114.0)
        assert result == cached
        mock_get.assert_not_called()


class TestMissingVariableAggregation:
    def test_download_raises_when_variable_gets_zero_files(self, tmp_path):
        """All-404 must raise, not silently produce partial/empty forcing."""
        handler = _make_handler(tmp_path, NEX_VARIABLES=['tas', 'pr'])
        with patch('symfluence.data.acquisition.handlers.nex_gddp.requests.get',
                   return_value=_mock_response(404)):
            with pytest.raises(RuntimeError, match='tas|pr'):
                handler.download(tmp_path / 'out')


def _write_synthetic_year(path: Path, year: int):
    """Write a tiny NEX-GDDP-style tas file covering January of the year."""
    times = pd.date_range(f'{year}-01-01T12:00:00', f'{year}-01-31T12:00:00', freq='D')
    lats = np.array([50.5, 51.0])
    lons = np.array([-115.5, -115.0])
    tas = 270.0 + np.random.default_rng(0).standard_normal(
        (len(times), len(lats), len(lons))).astype('float32')
    ds = xr.Dataset(
        {'tas': (('time', 'lat', 'lon'), tas)},
        coords={'time': times, 'lat': lats, 'lon': lons},
    )
    ds.to_netcdf(path)


class TestSyntheticPressureWarning:
    def _run_download(self, tmp_path, caplog, **extra):
        handler = _make_handler(tmp_path, **extra)
        out_dir = tmp_path / 'out'
        out_dir.mkdir()

        def fake_fetch(ncss_base, var_cache_dir, var, model_name, scenario_name,
                       member, grid_label, year, chunk_start, chunk_end,
                       lat_min, lat_max, lon_min, lon_max):
            path = var_cache_dir / f'{var}_{year}.nc'
            if not path.exists():
                _write_synthetic_year(path, year)
            return path

        with patch.object(NEXGDDPCHandler, '_fetch_year_chunk', side_effect=fake_fetch):
            with caplog.at_level(logging.WARNING):
                handler.download(out_dir)
        return out_dir

    def test_warns_and_uses_sea_level_constant_when_elevation_unset(self, tmp_path, caplog):
        out_dir = self._run_download(tmp_path, caplog)
        warnings = '\n'.join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert 'DOMAIN_MEAN_ELEV_M' in warnings
        assert 'SEA-LEVEL' in warnings

        monthly = sorted(out_dir.glob('NEXGDDP_all_*.nc'))
        assert monthly, 'no monthly outputs written'
        with xr.open_dataset(monthly[0]) as ds:
            assert float(ds['airpres'].mean()) == pytest.approx(101325.0)

    def test_no_warning_when_elevation_configured(self, tmp_path, caplog):
        out_dir = self._run_download(tmp_path, caplog, DOMAIN_MEAN_ELEV_M=1000.0)
        warnings = '\n'.join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert 'SEA-LEVEL' not in warnings

        monthly = sorted(out_dir.glob('NEXGDDP_all_*.nc'))
        with xr.open_dataset(monthly[0]) as ds:
            expected = 101325.0 * np.exp(-1000.0 / 8400.0)
            assert float(ds['airpres'].mean()) == pytest.approx(expected, rel=1e-4)
