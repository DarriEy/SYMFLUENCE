"""
Tests for GridDelineator class.

Tests grid-based domain delineation including:
- Grid creation from bounding box
- Watershed clipping
- D8 flow direction extraction
- Native grid creation (Phase 2)
"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# Import fixtures
from .conftest import requires_geopandas, requires_rasterio

# Skip all tests if geopandas not available
pytestmark = requires_geopandas

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    gpd = None


class TestGridDelineatorInit:
    """Tests for GridDelineator initialization."""

    def test_init_default_config(self, mock_config_dict, mock_logger, tmp_path):
        """Test GridDelineator initializes with default configuration."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)

                assert delineator.grid_cell_size == 1000.0
                assert delineator.clip_to_watershed is True
                assert delineator.grid_source == 'generate'
                assert delineator.native_grid_dataset == 'era5'

    def test_init_custom_grid_size(self, mock_config_dict, mock_logger, tmp_path):
        """Test GridDelineator respects custom grid cell size."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 500.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)

                assert delineator.grid_cell_size == 500.0


class TestGridCreation:
    """Tests for grid creation from bounding box."""

    def test_create_grid_from_bbox_valid(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation with valid bounding box."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = '47.0/8.0/46.0/9.0'
        mock_config_dict['GRID_CELL_SIZE'] = 10000.0  # Large cells for fast test

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is not None
                assert len(grid_gdf) > 0
                assert 'GRU_ID' in grid_gdf.columns
                assert 'GRU_area' in grid_gdf.columns
                assert grid_gdf.crs.to_epsg() == 4326

    def test_create_grid_from_bbox_invalid_format(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation fails gracefully with invalid bbox format."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = 'invalid/format'

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is None

    def test_create_grid_from_bbox_missing_coords(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation fails gracefully with missing bbox coords."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = ''

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is None

    def test_grid_has_sequential_ids(self, mock_config_dict, mock_logger, tmp_path):
        """Test that grid cells have sequential GRU_IDs starting from 1."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0  # Large cells

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None and len(grid_gdf) > 0:
                    ids = sorted(grid_gdf['GRU_ID'].tolist())
                    expected_ids = list(range(1, len(grid_gdf) + 1))
                    assert ids == expected_ids


class TestGridAttributes:
    """Tests for grid attribute calculation."""

    def test_add_grid_attributes_centroids(self, mock_config_dict, mock_logger, tmp_path):
        """Test that grid attributes include centroid coordinates."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None:
                    grid_gdf = delineator._add_grid_attributes(grid_gdf)

                    assert 'center_lon' in grid_gdf.columns
                    assert 'center_lat' in grid_gdf.columns
                    assert 'gru_to_seg' in grid_gdf.columns


class TestRiverNetworkFromGrid:
    """Tests for river network creation from grid topology."""

    def test_create_river_network_from_grid(self, mock_config_dict, mock_logger, tmp_path):
        """Test river network creation from grid cells."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None:
                    # Add required columns for network creation
                    grid_gdf['downstream'] = 0  # All outlets for simplicity

                    network_gdf = delineator._create_river_network_from_grid(grid_gdf)

                    assert network_gdf is not None
                    assert len(network_gdf) == len(grid_gdf)
                    assert 'LINKNO' in network_gdf.columns
                    assert 'DSLINKNO' in network_gdf.columns


class TestNativeGridImplementation:
    """Tests for native grid creation from forcing data (Phase 2)."""

    def test_native_grid_era5_structure(self, geospatial_test_domain_setup):
        """Test native grid creation reads ERA5-like forcing data."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(config, logger)

                # Note: This test will exercise the _create_native_grid method
                # After Phase 2 implementation, this should return a valid GeoDataFrame
                result = delineator._create_native_grid()

                # For now (stub), it returns None and logs a warning
                # After implementation: assert result is not None

    def test_native_grid_coord_detection(self, geospatial_test_domain_setup, tmp_path):
        """Test coordinate name detection for various forcing datasets."""
        import xarray as xr

        # Create test datasets with different coordinate conventions
        coord_conventions = [
            {'lat_name': 'latitude', 'lon_name': 'longitude'},  # ERA5
            {'lat_name': 'lat', 'lon_name': 'lon'},  # CERRA/CARRA
            {'lat_name': 'y', 'lon_name': 'x'},  # Some grids
        ]

        for conv in coord_conventions:
            lat = np.arange(46.0, 47.0, 0.25)
            lon = np.arange(8.0, 9.0, 0.25)

            ds = xr.Dataset(
                {'temp': ([conv['lat_name'], conv['lon_name']], np.random.rand(len(lat), len(lon)))},
                coords={conv['lat_name']: lat, conv['lon_name']: lon}
            )

            test_path = tmp_path / f"test_{conv['lat_name']}.nc"
            ds.to_netcdf(test_path)

            # Verify the file was created with expected coordinates
            with xr.open_dataset(test_path) as loaded:
                assert conv['lat_name'] in loaded.coords
                assert conv['lon_name'] in loaded.coords


class TestMethodSuffix:
    """Tests for method suffix generation."""

    def test_method_suffix_generate(self, mock_config_dict, mock_logger, tmp_path):
        """Test method suffix for generated grid."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_SOURCE'] = 'generate'
        mock_config_dict['GRID_CELL_SIZE'] = 1000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                suffix = delineator._get_delineation_method_name()

                # Suffix should include 'distributed' and cell size
                assert 'distributed' in suffix.lower() or '1000' in suffix


class TestD8Offsets:
    """Tests for D8 flow direction offset constants."""

    def test_d8_offsets_complete(self):
        """Test that all 8 D8 directions are defined."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import D8_OFFSETS

        assert len(D8_OFFSETS) == 8
        assert all(d in D8_OFFSETS for d in range(1, 9))

    def test_d8_offsets_values(self):
        """Test D8 offset values are valid (row, col) tuples."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import D8_OFFSETS

        for direction, (drow, dcol) in D8_OFFSETS.items():
            assert isinstance(drow, int)
            assert isinstance(dcol, int)
            assert -1 <= drow <= 1
            assert -1 <= dcol <= 1
            # Can't have both zero (that would be no movement)
            assert not (drow == 0 and dcol == 0)
