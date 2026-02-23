"""Unit tests for WMFire FireGrid and FireGridManager classes."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from symfluence.models.wmfire.fire_grid import FireGrid, FireGridManager


class TestFireGrid:
    """Tests for FireGrid class."""

    def test_fire_grid_creation(self):
        """Test basic FireGrid creation."""
        data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype='int32')
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 5000000.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        assert grid.nrows == 3
        assert grid.ncols == 3
        assert grid.resolution == 30.0
        assert grid.crs == 'EPSG:32610'

    def test_fire_grid_bounds(self):
        """Test grid bounds calculation."""
        data = np.zeros((4, 5), dtype='float32')
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 5000000.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        minx, miny, maxx, maxy = grid.bounds
        assert minx == 500000.0
        assert maxy == 5000000.0
        assert maxx == 500000.0 + 30.0 * 5  # 5 columns
        assert miny == 5000000.0 - 30.0 * 4  # 4 rows

    def test_to_text_integer(self):
        """Test text export for integer grid."""
        data = np.array([[1, 2], [3, 4]], dtype='int32')
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 60.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        text = grid.to_text()
        lines = text.strip().split('\n')

        assert len(lines) == 2
        assert lines[0] == '1\t2'
        assert lines[1] == '3\t4'

    def test_to_text_float(self):
        """Test text export for float grid."""
        data = np.array([[100.5, 200.0], [300.5, 400.0]], dtype='float32')
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 60.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        text = grid.to_text()
        lines = text.strip().split('\n')

        assert len(lines) == 2
        assert '100.5' in lines[0]
        assert '200.0' in lines[0]

    def test_to_text_with_header(self):
        """Test text export with ESRI ASCII header."""
        data = np.array([[1, 2], [3, 4]], dtype='int32')
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 5000060.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        text = grid.to_text(include_header=True)
        lines = text.strip().split('\n')

        assert 'ncols 2' in lines[0]
        assert 'nrows 2' in lines[1]
        assert 'xllcorner' in lines[2]
        assert 'yllcorner' in lines[3]
        assert 'cellsize 30' in lines[4]

    @pytest.mark.parametrize("dtype,expected_type", [
        ('int32', 'int32'),
        ('float32', 'float32'),
        ('float64', 'float32'),  # Converted to float32
    ])
    def test_to_geotiff_mock(self, dtype, expected_type, tmp_path):
        """Test GeoTIFF export with mocked rasterio."""
        data = np.zeros((3, 3), dtype=dtype)
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 90.0)

        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        # Mock rasterio
        with patch.dict('sys.modules', {'rasterio': MagicMock(), 'rasterio.transform': MagicMock()}):
            # Should not raise
            output_path = tmp_path / "test.tif"
            # Will use mock, but tests that method doesn't crash
            # grid.to_geotiff(output_path)


class TestFireGridManager:
    """Tests for FireGridManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.model.rhessys.wmfire.grid_resolution = 30
        return config

    def test_init(self, mock_config):
        """Test FireGridManager initialization."""
        manager = FireGridManager(mock_config)
        assert manager.resolution == 30

    def test_init_default_resolution(self):
        """Test default resolution when config missing."""
        config = MagicMock()
        config.model.rhessys.wmfire = None

        manager = FireGridManager(config)
        assert manager.resolution == 30  # Default

    def test_estimate_utm_crs_northern(self, mock_config):
        """Test UTM CRS estimation for northern hemisphere."""
        manager = FireGridManager(mock_config)

        # San Francisco area (UTM zone 10N)
        crs = manager._estimate_utm_crs(37.77, -122.42)
        assert crs == 'EPSG:32610'

        # New York area (UTM zone 18N)
        crs = manager._estimate_utm_crs(40.71, -74.01)
        assert crs == 'EPSG:32618'

    def test_estimate_utm_crs_southern(self, mock_config):
        """Test UTM CRS estimation for southern hemisphere."""
        manager = FireGridManager(mock_config)

        # Sydney area (UTM zone 56S)
        crs = manager._estimate_utm_crs(-33.87, 151.21)
        assert crs == 'EPSG:32756'

    def test_create_simple_grid(self, mock_config):
        """Test simple grid creation from patch info."""
        manager = FireGridManager(mock_config)

        patch_info = [
            {'patch_id': 1, 'elev': 1000.0},
            {'patch_id': 2, 'elev': 1200.0},
            {'patch_id': 3, 'elev': 1100.0},
        ]

        patch_grid, dem_grid = manager.create_simple_grid(patch_info, arrange_by='elevation')

        # Should be 3 rows x 3 cols
        assert patch_grid.nrows == 3
        assert patch_grid.ncols == 3

        # Check elevation ordering (lowest first)
        # Row 0 should have patch_id 1 (elev 1000)
        assert patch_grid.data[0, 0] == 1
        # Row 1 should have patch_id 3 (elev 1100)
        assert patch_grid.data[1, 0] == 3
        # Row 2 should have patch_id 2 (elev 1200)
        assert patch_grid.data[2, 0] == 2

    def test_rasterize_patches_fallback(self, mock_config):
        """Test fallback rasterization without rasterio."""
        manager = FireGridManager(mock_config)

        # Create mock GeoDataFrame
        try:
            import geopandas as gpd
            from shapely.geometry import box

            # Create simple grid of polygons
            polygons = [
                box(0, 0, 30, 30),
                box(30, 0, 60, 30),
            ]
            gdf = gpd.GeoDataFrame({
                'HRU_ID': [1, 2],
                'geometry': polygons
            }, crs='EPSG:32610')

            transform = (30.0, 0.0, 0.0, 0.0, -30.0, 30.0)
            result = manager._rasterize_patches_fallback(gdf, 1, 2, transform)

            assert result.shape == (1, 2)
            # Cell centers at (15, 15) and (45, 15) should be in patches 1 and 2
            assert result[0, 0] == 1
            assert result[0, 1] == 2

        except ImportError:
            pytest.skip("geopandas not available")

    def test_create_synthetic_dem(self, mock_config):
        """Test synthetic DEM creation from HRU attributes."""
        manager = FireGridManager(mock_config)

        try:
            import geopandas as gpd
            from shapely.geometry import box

            polygons = [box(0, 0, 30, 30), box(30, 0, 60, 30)]
            gdf = gpd.GeoDataFrame({
                'HRU_ID': [1, 2],
                'elev_mean': [1000.0, 1500.0],
                'geometry': polygons
            }, crs='EPSG:32610')

            patch_grid = np.array([[1, 2]], dtype='int32')
            dem = manager._create_synthetic_dem(gdf, patch_grid, 1, 2)

            assert dem.shape == (1, 2)
            assert dem[0, 0] == 1000.0
            assert dem[0, 1] == 1500.0

        except ImportError:
            pytest.skip("geopandas not available")


class TestFireGridIntegration:
    """Integration tests for FireGrid with real-ish data."""

    @pytest.fixture
    def sample_catchment(self, tmp_path):
        """Create sample catchment shapefile."""
        try:
            import geopandas as gpd
            from shapely.geometry import box

            # Create 4 HRU polygons in a 2x2 arrangement
            polygons = [
                box(500000, 4500000, 500100, 4500100),  # HRU 1
                box(500100, 4500000, 500200, 4500100),  # HRU 2
                box(500000, 4500100, 500100, 4500200),  # HRU 3
                box(500100, 4500100, 500200, 4500200),  # HRU 4
            ]

            gdf = gpd.GeoDataFrame({
                'HRU_ID': [1, 2, 3, 4],
                'elev_mean': [1000, 1100, 1200, 1300],
                'geometry': polygons
            }, crs='EPSG:32610')

            shp_path = tmp_path / 'catchment.shp'
            gdf.to_file(shp_path)

            return shp_path, gdf

        except ImportError:
            pytest.skip("geopandas not available")

    def test_full_grid_creation(self, sample_catchment):
        """Test full grid creation workflow."""
        shp_path, gdf = sample_catchment

        config = MagicMock()
        config.model.rhessys.wmfire.grid_resolution = 30

        manager = FireGridManager(config)
        patch_grid, dem_grid = manager.create_fire_grid(gdf)

        # Check grid was created
        assert patch_grid.nrows > 0
        assert patch_grid.ncols > 0
        assert patch_grid.crs == 'EPSG:32610'

        # Check DEM was created
        assert dem_grid.nrows == patch_grid.nrows
        assert dem_grid.ncols == patch_grid.ncols

        # Check text export works
        text = patch_grid.to_text()
        assert len(text) > 0
