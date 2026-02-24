"""
Tests for Daymet spatial subsetting (OPeNDAP + client-side fallback).

Verifies that the Daymet handler correctly converts bounding boxes to the
native Lambert Conformal Conic projection and subsets data accordingly,
instead of downloading full North American continental files.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.daymet import DaymetAcquirer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bow at Banff bbox (the original issue)
BOW_BBOX = {
    'lat_min': 51.0,
    'lat_max': 51.5,
    'lon_min': -116.0,
    'lon_max': -115.4,
}


def _make_acquirer(bbox=None, tmp_path=None):
    """Create a DaymetAcquirer with the given bbox, bypassing full config."""
    import pandas as pd

    bbox = bbox or BOW_BBOX
    config = {
        'DOMAIN_NAME': 'test_bow',
        'DATA_DIR': str(tmp_path or '/tmp/daymet_test'),
        'BOUNDING_BOX_COORDS': (
            f"{bbox['lat_max']}/{bbox['lon_min']}/"
            f"{bbox['lat_min']}/{bbox['lon_max']}"
        ),
        'EXPERIMENT_TIME_START': '2020-01-01',
        'EXPERIMENT_TIME_END': '2020-12-31',
    }
    logger = MagicMock()
    acq = DaymetAcquirer(config, logger)
    # Dict configs don't coerce to typed config, so set attributes directly
    acq.bbox = bbox
    acq.start_date = pd.Timestamp('2020-01-01')
    acq.end_date = pd.Timestamp('2020-12-31')
    return acq


def _make_daymet_dataset(lcc_bbox, n_time=10):
    """Create a synthetic Daymet-like dataset in LCC coordinates.

    Builds a small grid that covers the LCC bounding box with 1 km spacing,
    including 2D lat/lon auxiliary coordinates.
    """
    from pyproj import Transformer

    # Build x/y grid that covers the LCC bbox.
    # y is descending (north→south) to match real Daymet data.
    x = np.arange(lcc_bbox['x_min'] - 2000, lcc_bbox['x_max'] + 2000, 1000.0)
    y = np.arange(lcc_bbox['y_max'] + 2000, lcc_bbox['y_min'] - 2000, -1000.0)
    time = np.arange(n_time)

    rng = np.random.default_rng(42)
    data = rng.uniform(-10, 30, size=(n_time, len(y), len(x))).astype(np.float32)

    # Build 2D lat/lon from the x/y grid (inverse LCC transform)
    transformer = Transformer.from_crs(
        DaymetAcquirer.DAYMET_CRS, "EPSG:4326", always_xy=True
    )
    xx, yy = np.meshgrid(x, y)
    lon_2d, lat_2d = transformer.transform(xx, yy)

    ds = xr.Dataset(
        {
            'tmax': (['time', 'y', 'x'], data),
        },
        coords={
            'x': x,
            'y': y,
            'time': time,
            'lat': (['y', 'x'], lat_2d),
            'lon': (['y', 'x'], lon_2d),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Tests: bbox → LCC conversion
# ---------------------------------------------------------------------------

class TestBboxToLcc:
    """Test geographic → LCC bounding box conversion."""

    def test_returns_four_keys(self):
        acq = _make_acquirer()
        lcc = acq._bbox_to_lcc()
        assert set(lcc.keys()) == {'x_min', 'x_max', 'y_min', 'y_max'}

    def test_x_min_less_than_x_max(self):
        acq = _make_acquirer()
        lcc = acq._bbox_to_lcc()
        assert lcc['x_min'] < lcc['x_max']

    def test_y_min_less_than_y_max(self):
        acq = _make_acquirer()
        lcc = acq._bbox_to_lcc()
        assert lcc['y_min'] < lcc['y_max']

    def test_includes_buffer(self):
        """Verify that the LCC bbox includes a 1 km buffer beyond the corners."""
        from pyproj import Transformer

        acq = _make_acquirer()
        lcc = acq._bbox_to_lcc()

        transformer = Transformer.from_crs(
            "EPSG:4326", DaymetAcquirer.DAYMET_CRS, always_xy=True
        )
        # SW corner
        x_sw, y_sw = transformer.transform(
            BOW_BBOX['lon_min'], BOW_BBOX['lat_min']
        )
        assert lcc['x_min'] < x_sw
        assert lcc['y_min'] < y_sw

    def test_different_bboxes_give_different_results(self):
        acq1 = _make_acquirer(BOW_BBOX)
        acq2 = _make_acquirer({
            'lat_min': 35.0, 'lat_max': 36.0,
            'lon_min': -90.0, 'lon_max': -89.0,
        })
        lcc1 = acq1._bbox_to_lcc()
        lcc2 = acq2._bbox_to_lcc()
        assert lcc1['x_min'] != lcc2['x_min']


# ---------------------------------------------------------------------------
# Tests: OPeNDAP subsetting
# ---------------------------------------------------------------------------

class TestOpendapSubset:
    """Test the two-step OPeNDAP subsetting (coords then data)."""

    def test_success_writes_subsetted_file(self, tmp_path):
        """Coords fetched via OPeNDAP, data via session, subset saved."""
        acq = _make_acquirer(tmp_path=tmp_path)
        lcc_bbox = acq._bbox_to_lcc()
        ds_full = _make_daymet_dataset(lcc_bbox, n_time=5)

        # Step 1 mock: coord-only dataset via OPeNDAP
        ds_coords = xr.Dataset(coords={
            'x': ds_full.coords['x'],
            'y': ds_full.coords['y'],
            'time': ds_full.coords['time'],
        })

        # Step 3 mock: session downloads subset as .nc4
        nc_bytes_path = tmp_path / '_response.nc'
        ds_full.to_netcdf(nc_bytes_path, engine='h5netcdf')
        nc_bytes = nc_bytes_path.read_bytes()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = nc_bytes

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        var_file = tmp_path / 'daymet_tmax_2020.nc'

        # Mock only intercepts OPeNDAP URL; local file reads pass through
        real_open = xr.open_dataset

        def selective_mock(path_or_url, *args, **kwargs):
            if isinstance(path_or_url, str) and path_or_url.startswith('http'):
                return ds_coords
            return real_open(path_or_url, *args, **kwargs)

        with patch('xarray.open_dataset', side_effect=selective_mock), \
             patch.object(acq, '_get_earthdata_session', return_value=mock_session), \
             patch.object(DaymetAcquirer, '_ensure_dodsrc'):
            result = acq._download_opendap_subset(
                'tmax', 2020, var_file, lcc_bbox
            )

        assert result is True
        assert var_file.exists()
        assert mock_session.get.call_count == 1

        ds_saved = xr.open_dataset(var_file)
        assert ds_saved.sizes['x'] > 0
        assert ds_saved.sizes['y'] > 0
        ds_saved.close()

    def test_returns_false_on_failure(self, tmp_path):
        """When OPeNDAP fails, returns False."""
        acq = _make_acquirer(tmp_path=tmp_path)
        lcc_bbox = acq._bbox_to_lcc()
        var_file = tmp_path / 'daymet_tmax_2020.nc'

        with patch('xarray.open_dataset', side_effect=OSError("auth failed")), \
             patch('time.sleep'), \
             patch.object(DaymetAcquirer, '_ensure_dodsrc'):
            result = acq._download_opendap_subset(
                'tmax', 2020, var_file, lcc_bbox
            )

        assert result is False
        assert not var_file.exists()

    def test_returns_false_on_empty_subset(self, tmp_path):
        """When no grid cells fall within the bbox, returns False."""
        acq = _make_acquirer(tmp_path=tmp_path)
        lcc_bbox = acq._bbox_to_lcc()

        # Create a coordinate dataset with axes far from the bbox
        far_away_bbox = {
            'x_min': lcc_bbox['x_max'] + 100000,
            'x_max': lcc_bbox['x_max'] + 200000,
            'y_min': lcc_bbox['y_max'] + 100000,
            'y_max': lcc_bbox['y_max'] + 200000,
        }
        ds = _make_daymet_dataset(far_away_bbox, n_time=5)
        ds_coords = xr.Dataset(coords={
            'x': ds.coords['x'], 'y': ds.coords['y'], 'time': ds.coords['time'],
        })
        var_file = tmp_path / 'daymet_tmax_2020.nc'

        with patch('xarray.open_dataset', return_value=ds_coords), \
             patch.object(DaymetAcquirer, '_ensure_dodsrc'):
            result = acq._download_opendap_subset(
                'tmax', 2020, var_file, lcc_bbox
            )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: OPeNDAP retry behaviour
# ---------------------------------------------------------------------------

class TestOpendapRetries:
    """Test that OPeNDAP subsetting retries on transient failures."""

    def test_succeeds_on_second_attempt(self, tmp_path):
        """Transient failure on first attempt, success on second."""
        acq = _make_acquirer(tmp_path=tmp_path)
        lcc_bbox = acq._bbox_to_lcc()
        ds_full = _make_daymet_dataset(lcc_bbox, n_time=5)
        ds_coords = xr.Dataset(coords={
            'x': ds_full.coords['x'], 'y': ds_full.coords['y'],
            'time': ds_full.coords['time'],
        })
        var_file = tmp_path / 'daymet_tmax_2020.nc'

        # Prepare mock session response
        nc_bytes_path = tmp_path / '_resp.nc'
        ds_full.to_netcdf(nc_bytes_path, engine='h5netcdf')
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = nc_bytes_path.read_bytes()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        real_open = xr.open_dataset
        call_count = {'n': 0}

        def flaky_open(path_or_url, *args, **kwargs):
            if not isinstance(path_or_url, str) or not path_or_url.startswith('http'):
                return real_open(path_or_url, *args, **kwargs)
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise OSError("transient server error")
            return ds_coords

        with patch('xarray.open_dataset', side_effect=flaky_open), \
             patch('time.sleep'), \
             patch.object(acq, '_get_earthdata_session', return_value=mock_session), \
             patch.object(DaymetAcquirer, '_ensure_dodsrc'):
            result = acq._download_opendap_subset(
                'tmax', 2020, var_file, lcc_bbox, max_retries=3,
            )

        assert result is True
        assert var_file.exists()

    def test_fails_after_all_retries(self, tmp_path):
        """Returns False after exhausting all retry attempts."""
        acq = _make_acquirer(tmp_path=tmp_path)
        lcc_bbox = acq._bbox_to_lcc()
        var_file = tmp_path / 'daymet_tmax_2020.nc'

        with patch('xarray.open_dataset', side_effect=OSError("persistent")), \
             patch('time.sleep'), \
             patch.object(DaymetAcquirer, '_ensure_dodsrc'):
            result = acq._download_opendap_subset(
                'tmax', 2020, var_file, lcc_bbox, max_retries=2,
            )

        assert result is False
        assert not var_file.exists()
        acq.logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _download_gridded orchestration
# ---------------------------------------------------------------------------

class TestDownloadGridded:
    """Test the top-level _download_gridded orchestration."""

    def test_calls_opendap_subset(self, tmp_path):
        """OPeNDAP is attempted for each variable."""
        acq = _make_acquirer(tmp_path=tmp_path)
        output_file = tmp_path / 'daymet_tmax_tmin_prcp_20200101_20201231.nc'

        with patch.object(acq, '_download_opendap_subset', return_value=True) as mock_opendap:
            acq._download_gridded(output_file, ['tmax'])

        assert mock_opendap.call_count == 1

    def test_logs_error_when_opendap_fails(self, tmp_path):
        """When OPeNDAP fails, logs an error with credential guidance."""
        acq = _make_acquirer(tmp_path=tmp_path)
        output_file = tmp_path / 'daymet_tmax_tmin_prcp_20200101_20201231.nc'

        with patch.object(acq, '_download_opendap_subset', return_value=False):
            acq._download_gridded(output_file, ['tmax'])

        acq.logger.error.assert_called_once()
        msg = acq.logger.error.call_args[0][0]
        assert 'dodsrc' in msg

    def test_skips_existing_files(self, tmp_path):
        """Existing per-variable files are not re-downloaded."""
        acq = _make_acquirer(tmp_path=tmp_path)
        output_file = tmp_path / 'daymet_tmax_tmin_prcp_20200101_20201231.nc'

        # Pre-create the var file
        existing = tmp_path / 'daymet_tmax_2020.nc'
        existing.touch()

        with patch.object(acq, '_download_opendap_subset') as mock_opendap:
            acq._download_gridded(output_file, ['tmax'])

        assert mock_opendap.call_count == 0

    def test_no_bbox_logs_error(self, tmp_path):
        """Without a bounding box, logs an error and returns."""
        acq = _make_acquirer(tmp_path=tmp_path)
        acq.bbox = {}
        output_file = tmp_path / 'daymet_out.nc'

        acq._download_gridded(output_file, ['tmax'])
        acq.logger.error.assert_called_once()

    def test_multiple_variables_and_years(self, tmp_path):
        """Processes each variable x year combination."""
        import pandas as pd
        acq = _make_acquirer(tmp_path=tmp_path)
        # Override dates to span 2 years
        acq.start_date = pd.Timestamp('2019-06-01')
        acq.end_date = pd.Timestamp('2020-06-30')
        output_file = tmp_path / 'daymet_out.nc'

        with patch.object(acq, '_download_opendap_subset', return_value=True) as mock_opendap:
            acq._download_gridded(output_file, ['tmax', 'prcp'])

        # 2 years x 2 variables = 4 calls
        assert mock_opendap.call_count == 4
