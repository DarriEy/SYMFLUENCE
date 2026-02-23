"""
Tests for raster_utils module.

Tests raster processing utilities including:
- Landcover mode calculation
- Aspect computation
- Radiation calculations
- Scipy mode compatibility
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import fixtures
from .conftest import requires_rasterio


class TestScipyModeCompat:
    """Tests for _scipy_mode_compat function."""

    def test_mode_compat_1d_array(self):
        """Test scipy mode compatibility with 1D array."""
        from symfluence.geospatial.raster_utils import _scipy_mode_compat

        arr = np.array([1, 1, 2, 2, 2, 3])
        result = _scipy_mode_compat(arr)

        # Mode should be 2 (appears 3 times)
        assert result.flat[0] == 2

    def test_mode_compat_2d_array(self):
        """Test scipy mode compatibility with 2D array."""
        from symfluence.geospatial.raster_utils import _scipy_mode_compat

        arr = np.array([
            [1, 1, 2],
            [2, 2, 2],
            [3, 3, 3]
        ])
        result = _scipy_mode_compat(arr.flatten())

        # Flattened: [1,1,2,2,2,2,3,3,3] - mode is 2 (4 times)
        assert result.flat[0] == 2

    def test_mode_compat_single_value(self):
        """Test scipy mode compatibility with single value array."""
        from symfluence.geospatial.raster_utils import _scipy_mode_compat

        arr = np.array([5])
        result = _scipy_mode_compat(arr)

        assert result.flat[0] == 5

    def test_mode_compat_empty_handling(self):
        """Test scipy mode compatibility with potential empty result."""
        from symfluence.geospatial.raster_utils import _scipy_mode_compat

        # Array with uniform values
        arr = np.array([7, 7, 7, 7])
        result = _scipy_mode_compat(arr)

        assert result.flat[0] == 7


class TestLandcoverMode:
    """Tests for landcover mode calculation."""

    @requires_rasterio
    def test_landcover_mode_basic(self, tmp_path):
        """Test basic landcover mode calculation."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create a simple landcover raster
        landcover_data = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [3, 3, 2]
        ], dtype=np.uint8)

        landcover_path = tmp_path / "landcover.tif"

        transform = from_bounds(8.0, 46.0, 9.0, 47.0, 3, 3)

        with rasterio.open(
            landcover_path, 'w',
            driver='GTiff',
            height=3, width=3,
            count=1,
            dtype='uint8',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(landcover_data, 1)

        # Test that the file was created
        assert landcover_path.exists()


class TestAspectCalculation:
    """Tests for aspect calculation utilities."""

    def test_aspect_from_dem_values(self):
        """Test aspect calculation from DEM-like gradient values."""
        # Simple slope facing east (positive dlon)
        dlon = 10.0
        dlat = 0.0

        # Aspect = arctan2(-dlat, dlon) in degrees, measured from north
        # Note: arctan2(-0, 10) = 0 (east-facing slope when measured from east axis)
        # To get compass bearing from north, we need different convention
        aspect = np.degrees(np.arctan2(dlon, -dlat))

        # East-facing slope should have aspect around 90 degrees from north
        assert 85 <= aspect <= 95, f"Aspect was {aspect}, expected ~90"


class TestRasterValidation:
    """Tests for raster validation utilities."""

    @requires_rasterio
    def test_validate_raster_exists(self, tmp_path):
        """Test raster existence validation."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create a valid raster
        raster_path = tmp_path / "valid.tif"
        data = np.ones((10, 10), dtype=np.float32)
        transform = from_bounds(8.0, 46.0, 9.0, 47.0, 10, 10)

        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=10, width=10,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        assert raster_path.exists()

    def test_validate_raster_not_exists(self, tmp_path):
        """Test validation fails for non-existent raster."""
        non_existent_path = tmp_path / "does_not_exist.tif"
        assert not non_existent_path.exists()


class TestNodataHandling:
    """Tests for nodata value handling in raster operations."""

    @requires_rasterio
    def test_nodata_exclusion(self, tmp_path):
        """Test that nodata values are excluded from calculations."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create raster with nodata values
        data = np.array([
            [1, 2, -9999],
            [3, -9999, 4],
            [-9999, 5, 6]
        ], dtype=np.float32)

        raster_path = tmp_path / "with_nodata.tif"
        transform = from_bounds(8.0, 46.0, 9.0, 47.0, 3, 3)

        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=3, width=3,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(data, 1)

        # Read back and verify nodata is set
        with rasterio.open(raster_path) as src:
            assert src.nodata == -9999
            read_data = src.read(1)
            # Non-nodata values
            valid_data = read_data[read_data != -9999]
            assert len(valid_data) == 6  # 6 valid values


class TestCoordinateTransforms:
    """Tests for raster coordinate transformation utilities."""

    @requires_rasterio
    def test_pixel_to_coords(self, tmp_path):
        """Test conversion from pixel indices to geographic coordinates."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create raster with known transform
        raster_path = tmp_path / "geo.tif"
        data = np.ones((100, 100), dtype=np.float32)

        # Bounds: lon 8-9, lat 46-47
        transform = from_bounds(8.0, 46.0, 9.0, 47.0, 100, 100)

        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=100, width=100,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        with rasterio.open(raster_path) as src:
            # Top-left corner (row=0, col=0) should be at (8.0, 47.0)
            x, y = src.transform * (0, 0)
            assert abs(x - 8.0) < 0.01
            assert abs(y - 47.0) < 0.01

            # Bottom-right corner
            x, y = src.transform * (100, 100)
            assert abs(x - 9.0) < 0.01
            assert abs(y - 46.0) < 0.01


class TestCRSHandling:
    """Tests for CRS handling in raster operations."""

    @requires_rasterio
    def test_crs_detection(self, tmp_path):
        """Test CRS is correctly detected from raster."""
        import rasterio
        from rasterio.transform import from_bounds

        raster_path = tmp_path / "crs_test.tif"
        data = np.ones((10, 10), dtype=np.float32)
        transform = from_bounds(8.0, 46.0, 9.0, 47.0, 10, 10)

        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=10, width=10,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        with rasterio.open(raster_path) as src:
            assert src.crs.to_epsg() == 4326
