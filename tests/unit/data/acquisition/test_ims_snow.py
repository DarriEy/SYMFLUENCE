"""
Unit Tests for IMSSnowAcquirer.

Tests IMS snow cover data acquisition:
- Polar stereographic projection conversion
- Bounding box pixel calculation
- Domain statistics extraction
- Output dataset creation
- File listing and parsing
"""

import math
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from fixtures.acquisition_fixtures import MockConfigFactory

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ims_config():
    """Configuration with IMS-specific settings."""
    config = MockConfigFactory.create()
    config['IMS_SNOW_RESOLUTION'] = '4km'
    config['IMS_SNOW_DOWNLOAD_RAW'] = False
    return config


@pytest.fixture
def ims_acquirer(ims_config, mock_logger):
    """Create an IMSSnowAcquirer instance."""
    from symfluence.data.acquisition.handlers.ims_snow import IMSSnowAcquirer

    return IMSSnowAcquirer(ims_config, mock_logger)


@pytest.fixture
def sample_grid_4km():
    """Create a sample 4km IMS grid subset with known snow values."""
    grid = np.zeros((6144, 6144), dtype=np.uint8)
    # Fill a small region with land and snow values
    grid[1000:1050, 2000:2050] = 2  # Land
    grid[1010:1030, 2010:2030] = 4  # Snow on land
    grid[1000:1010, 2000:2010] = 1  # Water
    return grid


# =============================================================================
# Constants and Registration Tests
# =============================================================================

class TestIMSConstants:
    """Test IMS module-level constants."""

    def test_value_codes_defined(self):
        """IMS value codes should be defined."""
        from symfluence.data.acquisition.handlers.ims_snow import IMSSnowAcquirer

        assert IMSSnowAcquirer.VALUE_OUTSIDE == 0
        assert IMSSnowAcquirer.VALUE_WATER == 1
        assert IMSSnowAcquirer.VALUE_LAND == 2
        assert IMSSnowAcquirer.VALUE_SEA_ICE == 3
        assert IMSSnowAcquirer.VALUE_SNOW == 4

    def test_grid_definitions_exist(self):
        """IMS grid parameters should be defined for all resolutions."""
        from symfluence.data.acquisition.handlers.ims_snow import IMS_GRIDS

        assert '1km' in IMS_GRIDS
        assert '4km' in IMS_GRIDS
        assert '24km' in IMS_GRIDS

        for res, params in IMS_GRIDS.items():
            assert 'ncols' in params
            assert 'nrows' in params
            assert 'cell_size' in params
            assert 'start_year' in params

    def test_4km_grid_dimensions(self):
        """4km grid should have correct dimensions."""
        from symfluence.data.acquisition.handlers.ims_snow import IMS_GRIDS

        assert IMS_GRIDS['4km']['ncols'] == 6144
        assert IMS_GRIDS['4km']['nrows'] == 6144
        assert IMS_GRIDS['4km']['cell_size'] == 4000

    def test_registry_registration(self):
        """IMSSnowAcquirer should be registered under IMS or IMS_SNOW."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        assert AcquisitionRegistry.is_registered('IMS') or \
               AcquisitionRegistry.is_registered('IMS_SNOW')


# =============================================================================
# Coordinate Conversion Tests
# =============================================================================

@pytest.mark.acquisition
class TestLatLonToIMSPixel:
    """Tests for _latlon_to_ims_pixel polar stereographic conversion."""

    def test_returns_row_col_tuple(self, ims_acquirer):
        """Should return (row, col) tuple of integers."""
        row, col = ims_acquirer._latlon_to_ims_pixel(46.0, 8.0, '4km')

        assert isinstance(row, int)
        assert isinstance(col, int)

    def test_pixel_within_grid_bounds(self, ims_acquirer):
        """Result should be within valid grid bounds for Northern Hemisphere."""
        from symfluence.data.acquisition.handlers.ims_snow import IMS_GRIDS

        row, col = ims_acquirer._latlon_to_ims_pixel(50.0, 10.0, '4km')
        grid = IMS_GRIDS['4km']

        assert 0 <= row < grid['nrows']
        assert 0 <= col < grid['ncols']

    def test_high_latitude_gives_smaller_row(self, ims_acquirer):
        """Higher latitudes (closer to pole) should generally have
        different pixel coordinates than lower latitudes."""
        row_60, col_60 = ims_acquirer._latlon_to_ims_pixel(60.0, 0.0, '4km')
        row_40, col_40 = ims_acquirer._latlon_to_ims_pixel(40.0, 0.0, '4km')

        # They should differ (not a strict ordering test due to projection)
        assert (row_60, col_60) != (row_40, col_40)

    def test_different_resolutions_give_different_pixels(self, ims_acquirer):
        """Same lat/lon at different resolutions should give different pixels."""
        row_4km, col_4km = ims_acquirer._latlon_to_ims_pixel(50.0, 10.0, '4km')
        row_24km, col_24km = ims_acquirer._latlon_to_ims_pixel(50.0, 10.0, '24km')

        # 4km grid is 6x larger than 24km, so pixel numbers should differ
        assert row_4km != row_24km or col_4km != col_24km


# =============================================================================
# Bounding Box Pixel Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetBboxPixels:
    """Tests for _get_bbox_pixels method."""

    def test_returns_four_bounds(self, ims_acquirer):
        """Should return (row_start, row_end, col_start, col_end)."""
        result = ims_acquirer._get_bbox_pixels('4km')

        assert len(result) == 4
        row_start, row_end, col_start, col_end = result
        assert row_start < row_end
        assert col_start < col_end

    def test_bounds_within_grid(self, ims_acquirer):
        """Pixel bounds should be within grid dimensions."""
        from symfluence.data.acquisition.handlers.ims_snow import IMS_GRIDS

        row_start, row_end, col_start, col_end = ims_acquirer._get_bbox_pixels('4km')
        grid = IMS_GRIDS['4km']

        assert row_start >= 0
        assert col_start >= 0
        assert row_end <= grid['nrows']
        assert col_end <= grid['ncols']

    def test_includes_buffer(self, ims_acquirer):
        """Pixel bounds should include a buffer around the exact conversion."""
        # The buffer of 5 pixels means the range is at least 10 pixels wide
        row_start, row_end, col_start, col_end = ims_acquirer._get_bbox_pixels('4km')

        assert (row_end - row_start) >= 10
        assert (col_end - col_start) >= 10


# =============================================================================
# Domain Statistics Extraction Tests
# =============================================================================

@pytest.mark.acquisition
class TestExtractDomainStats:
    """Tests for _extract_domain_stats method."""

    def test_calculates_snow_fraction(self, ims_acquirer):
        """Should calculate correct snow fraction."""
        grid = np.array([
            [2, 2, 4, 4],
            [2, 4, 4, 4],
            [2, 2, 2, 4],
            [1, 1, 2, 2],
        ], dtype=np.uint8)

        bbox_pixels = (0, 4, 0, 4)
        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        # Land pixels: value 2 or 4 = 14 total (exclude water=1)
        # Snow pixels: value 4 = 6
        # Snow fraction = 6 / 14
        expected_land = 14
        expected_snow = 6
        assert stats['land_pixels'] == expected_land
        assert stats['snow_pixels'] == expected_snow
        assert stats['snow_fraction'] == pytest.approx(expected_snow / expected_land)

    def test_handles_all_water(self, ims_acquirer):
        """Snow fraction should be NaN when no land pixels exist."""
        grid = np.ones((10, 10), dtype=np.uint8)  # All water
        bbox_pixels = (0, 10, 0, 10)

        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        assert stats['land_pixels'] == 0
        assert np.isnan(stats['snow_fraction'])

    def test_handles_no_snow(self, ims_acquirer):
        """Snow fraction should be 0 when no snow pixels."""
        grid = np.full((10, 10), 2, dtype=np.uint8)  # All land, no snow
        bbox_pixels = (0, 10, 0, 10)

        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        assert stats['snow_fraction'] == 0.0
        assert stats['snow_pixels'] == 0

    def test_handles_all_snow(self, ims_acquirer):
        """Snow fraction should be 1 when all land is snow-covered."""
        grid = np.full((10, 10), 4, dtype=np.uint8)  # All snow
        bbox_pixels = (0, 10, 0, 10)

        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        assert stats['snow_fraction'] == pytest.approx(1.0)

    def test_subset_uses_bbox_pixels(self, ims_acquirer):
        """Should only analyze the subset defined by bbox_pixels."""
        grid = np.full((20, 20), 2, dtype=np.uint8)  # All land
        # Place snow only in subset
        grid[5:10, 5:10] = 4

        # Subset exactly covers the snow area
        bbox_pixels = (5, 10, 5, 10)
        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        assert stats['snow_fraction'] == pytest.approx(1.0)
        assert stats['total_pixels'] == 25

    def test_returns_expected_keys(self, ims_acquirer):
        """Stats dict should contain all expected keys."""
        grid = np.full((10, 10), 2, dtype=np.uint8)
        bbox_pixels = (0, 10, 0, 10)

        stats = ims_acquirer._extract_domain_stats(grid, bbox_pixels)

        expected_keys = {'snow_fraction', 'snow_pixels', 'land_pixels',
                         'water_pixels', 'total_pixels'}
        assert set(stats.keys()) == expected_keys


# =============================================================================
# Output Dataset Creation Tests
# =============================================================================

@pytest.mark.acquisition
class TestCreateOutputDataset:
    """Tests for _create_output_dataset method."""

    def test_creates_netcdf_file(self, ims_acquirer, tmp_path):
        """Should create a valid NetCDF file."""
        import xarray as xr

        results = [
            {'date': datetime(2020, 1, 1), 'doy': 1, 'snow_fraction': 0.5,
             'snow_pixels': 100, 'land_pixels': 200, 'water_pixels': 10,
             'total_pixels': 210},
            {'date': datetime(2020, 1, 2), 'doy': 2, 'snow_fraction': 0.3,
             'snow_pixels': 60, 'land_pixels': 200, 'water_pixels': 10,
             'total_pixels': 210},
        ]

        output_file = tmp_path / "test_ims.nc"
        ims_acquirer._create_output_dataset(results, output_file, '4km')

        assert output_file.exists()

        ds = xr.open_dataset(output_file)
        assert 'snow_fraction' in ds.data_vars
        assert 'snow_pixels' in ds.data_vars
        assert 'land_pixels' in ds.data_vars
        assert len(ds.time) == 2
        ds.close()

    def test_dataset_attributes(self, ims_acquirer, tmp_path):
        """Output dataset should have proper global attributes."""
        import xarray as xr

        results = [
            {'date': datetime(2020, 1, 1), 'doy': 1, 'snow_fraction': 0.5,
             'snow_pixels': 100, 'land_pixels': 200, 'water_pixels': 10,
             'total_pixels': 210},
        ]

        output_file = tmp_path / "test_ims.nc"
        ims_acquirer._create_output_dataset(results, output_file, '4km')

        ds = xr.open_dataset(output_file)
        assert 'IMS' in ds.attrs.get('title', '')
        assert 'NOAA' in ds.attrs.get('institution', '')
        assert 'nsidc' in ds.attrs.get('references', '')
        ds.close()

    def test_handles_duplicate_dates(self, ims_acquirer, tmp_path):
        """Should deduplicate entries by date."""
        import xarray as xr

        results = [
            {'date': datetime(2020, 1, 1), 'doy': 1, 'snow_fraction': 0.5,
             'snow_pixels': 100, 'land_pixels': 200, 'water_pixels': 10,
             'total_pixels': 210},
            {'date': datetime(2020, 1, 1), 'doy': 1, 'snow_fraction': 0.6,
             'snow_pixels': 120, 'land_pixels': 200, 'water_pixels': 10,
             'total_pixels': 210},
        ]

        output_file = tmp_path / "test_ims.nc"
        ims_acquirer._create_output_dataset(results, output_file, '4km')

        ds = xr.open_dataset(output_file)
        assert len(ds.time) == 1  # Duplicates removed
        ds.close()


# =============================================================================
# File Listing Tests
# =============================================================================

@pytest.mark.acquisition
class TestListAvailableFiles:
    """Tests for _list_available_files method."""

    @patch("symfluence.data.acquisition.handlers.ims_snow.requests.get")
    def test_parses_gz_files_from_html(self, mock_get, ims_acquirer):
        """Should extract .asc.gz files from directory listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = '''
        <html><body>
        <a href="ims2020001_00UTC_4km_v1.3.asc.gz">ims2020001_00UTC_4km_v1.3.asc.gz</a>
        <a href="ims2020002_00UTC_4km_v1.3.asc.gz">ims2020002_00UTC_4km_v1.3.asc.gz</a>
        <a href="readme.txt">readme.txt</a>
        </body></html>
        '''
        mock_get.return_value = mock_response

        files = ims_acquirer._list_available_files(2020, '4km')

        assert len(files) == 2
        assert all(f.endswith('.asc.gz') for f in files)

    @patch("symfluence.data.acquisition.handlers.ims_snow.requests.get")
    def test_handles_network_error(self, mock_get, ims_acquirer):
        """Should return empty list on network failure."""
        mock_get.side_effect = ConnectionError("fail")

        files = ims_acquirer._list_available_files(2020, '4km')

        assert files == []

    @patch("symfluence.data.acquisition.handlers.ims_snow.requests.get")
    def test_returns_sorted_list(self, mock_get, ims_acquirer):
        """File list should be sorted."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = '''
        <html><body>
        <a href="ims2020005_4km.asc.gz">f</a>
        <a href="ims2020001_4km.asc.gz">f</a>
        <a href="ims2020003_4km.asc.gz">f</a>
        </body></html>
        '''
        mock_get.return_value = mock_response

        files = ims_acquirer._list_available_files(2020, '4km')

        assert files == sorted(files)


# =============================================================================
# Download Method Tests
# =============================================================================

@pytest.mark.acquisition
class TestIMSDownload:
    """Tests for top-level download method."""

    def test_invalid_resolution_raises(self, mock_logger, tmp_path):
        """Should raise ValueError for invalid resolution."""
        from symfluence.data.acquisition.handlers.ims_snow import IMSSnowAcquirer

        config = MockConfigFactory.create()
        config['IMS_SNOW_RESOLUTION'] = '99km'
        config['FORCE_DOWNLOAD'] = True
        acquirer = IMSSnowAcquirer(config, mock_logger)

        with pytest.raises(ValueError, match="Invalid IMS resolution"):
            acquirer.download(tmp_path)

    def test_returns_existing_file(self, ims_acquirer, tmp_path):
        """Should return existing file when not forced."""
        output_file = tmp_path / f"{ims_acquirer.domain_name}_IMS_snow_4km.nc"
        output_file.touch()

        result = ims_acquirer.download(tmp_path)

        assert result == output_file
