"""
Tests for DEM stream burning.

Tests the StreamBurner processor and the base delineator integration
(_condition_dem, _burn_streams_into_dem, _find_stream_burn_source).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .conftest import requires_rasterio

pytestmark = requires_rasterio

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None

try:
    import geopandas as gpd
    from shapely.geometry import LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_logger():
    """Provide a mock logger."""
    logger = MagicMock()
    return logger


@pytest.fixture
def small_dem(tmp_path):
    """
    Create a small synthetic DEM GeoTIFF for testing.

    Returns (dem_path, dem_array, profile).
    """
    height, width = 50, 50
    elevation = np.full((height, width), 100.0, dtype=np.float32)
    # Add a valley down the centre
    elevation[:, 20:30] = 80.0
    nodata = -9999.0

    transform = from_bounds(8.0, 46.0, 9.0, 47.0, width, height)
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': nodata,
    }

    dem_path = tmp_path / 'dem.tif'
    with rasterio.open(dem_path, 'w', **profile) as dst:
        dst.write(elevation, 1)

    return dem_path, elevation, profile


@pytest.fixture
def small_dem_with_nodata(tmp_path):
    """DEM with some nodata pixels on the stream path."""
    height, width = 50, 50
    elevation = np.full((height, width), 100.0, dtype=np.float32)
    nodata = -9999.0
    # Set nodata in a band where the stream will cross
    elevation[10:15, :] = nodata

    transform = from_bounds(8.0, 46.0, 9.0, 47.0, width, height)
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': nodata,
    }

    dem_path = tmp_path / 'dem_nodata.tif'
    with rasterio.open(dem_path, 'w', **profile) as dst:
        dst.write(elevation, 1)

    return dem_path, elevation, profile


@pytest.fixture
def stream_shapefile(tmp_path):
    """
    Create a simple stream shapefile crossing the DEM grid.

    A single line from (8.2, 46.8) to (8.8, 46.2) — diagonal across the DEM.
    """
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas/shapely not available")

    line = LineString([(8.2, 46.8), (8.5, 46.5), (8.8, 46.2)])
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[line], crs='EPSG:4326')
    shp_path = tmp_path / 'streams.shp'
    gdf.to_file(shp_path)
    return shp_path


@pytest.fixture
def stream_geopackage(tmp_path):
    """Stream file in GeoPackage format."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas/shapely not available")

    line = LineString([(8.2, 46.8), (8.8, 46.2)])
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[line], crs='EPSG:4326')
    gpkg_path = tmp_path / 'streams.gpkg'
    gdf.to_file(gpkg_path, driver='GPKG')
    return gpkg_path


@pytest.fixture
def stream_different_crs(tmp_path):
    """Stream file in a different CRS (UTM zone 32N) to test reprojection."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas/shapely not available")

    line = LineString([(8.2, 46.8), (8.8, 46.2)])
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[line], crs='EPSG:4326')
    gdf_utm = gdf.to_crs('EPSG:32632')
    shp_path = tmp_path / 'streams_utm.shp'
    gdf_utm.to_file(shp_path)
    return shp_path


@pytest.fixture
def empty_stream_shapefile(tmp_path):
    """Stream shapefile with no features."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas/shapely not available")

    gdf = gpd.GeoDataFrame({'id': []}, geometry=[], crs='EPSG:4326')
    gdf = gdf.set_geometry('geometry')
    shp_path = tmp_path / 'streams_empty.shp'
    gdf.to_file(shp_path)
    return shp_path


# =============================================================================
# StreamBurner Tests
# =============================================================================

class TestStreamBurner:
    """Tests for the StreamBurner processor."""

    def test_burn_lowers_stream_pixels(self, small_dem, stream_shapefile, tmp_path, mock_logger):
        """Burned pixels should be original - burn_depth."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, _ = small_dem
        output = tmp_path / 'burned.tif'
        burn_depth = 5.0

        burner = StreamBurner(mock_logger)
        result = burner.burn_streams(dem_path, stream_shapefile, output, burn_depth)

        assert result.exists()

        with rasterio.open(result) as src:
            burned = src.read(1)

        # Some pixels must have been lowered
        diff = original - burned
        assert diff.max() == pytest.approx(burn_depth, abs=0.01)
        # Burned pixels should be exactly burn_depth lower
        burned_mask = diff > 0
        assert burned_mask.any()
        np.testing.assert_allclose(diff[burned_mask], burn_depth, atol=0.01)

    def test_non_stream_pixels_unchanged(self, small_dem, stream_shapefile, tmp_path, mock_logger):
        """Non-stream pixels must remain identical to the original DEM."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, _ = small_dem
        output = tmp_path / 'burned.tif'

        burner = StreamBurner(mock_logger)
        burner.burn_streams(dem_path, stream_shapefile, output, 5.0)

        with rasterio.open(output) as src:
            burned = src.read(1)

        unchanged_mask = (original - burned) == 0
        np.testing.assert_array_equal(burned[unchanged_mask], original[unchanged_mask])

    def test_nodata_respected(self, small_dem_with_nodata, stream_shapefile, tmp_path, mock_logger):
        """Nodata pixels must not be modified even if they overlap with streams."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, profile = small_dem_with_nodata
        output = tmp_path / 'burned_nodata.tif'
        nodata = profile['nodata']

        burner = StreamBurner(mock_logger)
        burner.burn_streams(dem_path, stream_shapefile, output, 10.0)

        with rasterio.open(output) as src:
            burned = src.read(1)

        # Nodata pixels should remain exactly nodata
        nodata_mask = original == nodata
        np.testing.assert_array_equal(burned[nodata_mask], nodata)

    def test_crs_reprojection(self, small_dem, stream_different_crs, tmp_path, mock_logger):
        """Streams in a different CRS should be reprojected before burning."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, _ = small_dem
        output = tmp_path / 'burned_reproject.tif'

        burner = StreamBurner(mock_logger)
        result = burner.burn_streams(dem_path, stream_different_crs, output, 5.0)

        assert result.exists()
        with rasterio.open(result) as src:
            burned = src.read(1)

        # Should have burned some pixels even after reprojection
        assert (original - burned).max() > 0

    def test_geopackage_input(self, small_dem, stream_geopackage, tmp_path, mock_logger):
        """GeoPackage stream files should work."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, _, _ = small_dem
        output = tmp_path / 'burned_gpkg.tif'

        burner = StreamBurner(mock_logger)
        result = burner.burn_streams(dem_path, stream_geopackage, output, 5.0)
        assert result.exists()

    def test_empty_streams_returns_unmodified(self, small_dem, empty_stream_shapefile, tmp_path, mock_logger):
        """Empty stream file should produce an unmodified copy of the DEM."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, _ = small_dem
        output = tmp_path / 'burned_empty.tif'

        burner = StreamBurner(mock_logger)
        result = burner.burn_streams(dem_path, empty_stream_shapefile, output, 5.0)

        with rasterio.open(result) as src:
            burned = src.read(1)

        np.testing.assert_array_equal(burned, original)

    def test_missing_dem_raises(self, stream_shapefile, tmp_path, mock_logger):
        """Missing DEM should raise FileNotFoundError."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        burner = StreamBurner(mock_logger)
        with pytest.raises(FileNotFoundError, match="DEM file not found"):
            burner.burn_streams(
                tmp_path / 'nonexistent.tif',
                stream_shapefile,
                tmp_path / 'out.tif',
            )

    def test_missing_stream_file_raises(self, small_dem, tmp_path, mock_logger):
        """Missing stream file should raise FileNotFoundError."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, _, _ = small_dem
        burner = StreamBurner(mock_logger)
        with pytest.raises(FileNotFoundError, match="Stream file not found"):
            burner.burn_streams(
                dem_path,
                tmp_path / 'no_streams.shp',
                tmp_path / 'out.tif',
            )

    def test_custom_burn_depth(self, small_dem, stream_shapefile, tmp_path, mock_logger):
        """A custom burn depth should be applied correctly."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, original, _ = small_dem
        output = tmp_path / 'burned_deep.tif'
        depth = 20.0

        burner = StreamBurner(mock_logger)
        burner.burn_streams(dem_path, stream_shapefile, output, depth)

        with rasterio.open(output) as src:
            burned = src.read(1)

        diff = original - burned
        burned_mask = diff > 0
        assert burned_mask.any()
        np.testing.assert_allclose(diff[burned_mask], depth, atol=0.01)

    def test_output_has_lzw_compression(self, small_dem, stream_shapefile, tmp_path, mock_logger):
        """Output file should use LZW compression."""
        from symfluence.geospatial.geofabric.processors.stream_burner import StreamBurner

        dem_path, _, _ = small_dem
        output = tmp_path / 'burned_lzw.tif'

        burner = StreamBurner(mock_logger)
        burner.burn_streams(dem_path, stream_shapefile, output, 5.0)

        with rasterio.open(output) as src:
            assert src.compression == rasterio.enums.Compression.lzw


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestDEMConditioningConfig:
    """Tests for DEM conditioning config fields and validators."""

    def test_default_method_is_none(self):
        """Default DEM_CONDITIONING_METHOD should be 'none'."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig()
        assert cfg.dem_conditioning_method == 'none'

    def test_valid_method_burn_streams(self):
        """'burn_streams' should be accepted."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig(DEM_CONDITIONING_METHOD='burn_streams')
        assert cfg.dem_conditioning_method == 'burn_streams'

    def test_method_case_insensitive(self):
        """Method names should be case-insensitive."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig(DEM_CONDITIONING_METHOD='Burn_Streams')
        assert cfg.dem_conditioning_method == 'burn_streams'

    def test_invalid_method_raises(self):
        """Invalid conditioning method should raise ValueError."""
        from pydantic import ValidationError

        from symfluence.core.config.models.domain import DelineationConfig
        with pytest.raises(ValidationError, match="DEM_CONDITIONING_METHOD"):
            DelineationConfig(DEM_CONDITIONING_METHOD='invalid_method')

    def test_default_burn_depth(self):
        """Default burn depth should be 5.0."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig()
        assert cfg.stream_burn_depth == 5.0

    def test_positive_burn_depth_required(self):
        """Non-positive burn depth should raise ValueError."""
        from pydantic import ValidationError

        from symfluence.core.config.models.domain import DelineationConfig
        with pytest.raises(ValidationError, match="STREAM_BURN_DEPTH"):
            DelineationConfig(STREAM_BURN_DEPTH=0)

    def test_negative_burn_depth_rejected(self):
        """Negative burn depth should raise ValueError."""
        from pydantic import ValidationError

        from symfluence.core.config.models.domain import DelineationConfig
        with pytest.raises(ValidationError, match="STREAM_BURN_DEPTH"):
            DelineationConfig(STREAM_BURN_DEPTH=-3.0)

    def test_default_burn_source(self):
        """Default burn source should be 'auto'."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig()
        assert cfg.stream_burn_source == 'auto'

    def test_default_custom_path(self):
        """Default custom path should be 'default'."""
        from symfluence.core.config.models.domain import DelineationConfig
        cfg = DelineationConfig()
        assert cfg.stream_burn_custom_path == 'default'


# =============================================================================
# Base Delineator Integration Tests
# =============================================================================

class TestConditionDem:
    """Tests for _condition_dem() on the base delineator."""

    def test_returns_original_when_none(self, mock_config_dict, mock_logger, tmp_path):
        """When method='none', _condition_dem returns self.dem_path."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['DEM_CONDITIONING_METHOD'] = 'none'

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                result = d._condition_dem()
                assert result == d.dem_path

    def test_unknown_method_returns_original(self, mock_config_dict, mock_logger, tmp_path):
        """Unknown methods should warn and return the raw DEM."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['DEM_CONDITIONING_METHOD'] = 'unknown_thing'

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                result = d._condition_dem()
                assert result == d.dem_path


class TestFindStreamBurnSource:
    """Tests for _find_stream_burn_source()."""

    def test_custom_path_found(self, mock_config_dict, mock_logger, tmp_path):
        """Custom path should be returned if it exists."""
        stream_file = tmp_path / 'my_streams.shp'
        stream_file.touch()
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['STREAM_BURN_CUSTOM_PATH'] = str(stream_file)

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                result = d._find_stream_burn_source('custom')
                assert result == stream_file

    def test_custom_path_missing_returns_none(self, mock_config_dict, mock_logger, tmp_path):
        """Custom path that doesn't exist should return None."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['STREAM_BURN_CUSTOM_PATH'] = str(tmp_path / 'missing.shp')

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                result = d._find_stream_burn_source('custom')
                assert result is None

    def test_merit_source_finds_shp(self, mock_config_dict, mock_logger, tmp_path):
        """MERIT source should find *rivernet*.shp files."""
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                # Create files under the delineator's actual project_dir
                merit_dir = d.project_dir / 'attributes' / 'geofabric' / 'merit'
                merit_dir.mkdir(parents=True, exist_ok=True)
                stream_file = merit_dir / 'basin_rivernet_v1.shp'
                stream_file.touch()

                result = d._find_stream_burn_source('merit')
                assert result == stream_file

    def test_tdx_source_finds_parquet(self, mock_config_dict, mock_logger, tmp_path):
        """TDX source should find *rivers*.parquet files."""
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                tdx_dir = d.project_dir / 'attributes' / 'geofabric' / 'tdx'
                tdx_dir.mkdir(parents=True, exist_ok=True)
                stream_file = tdx_dir / 'tdx_rivers_v1.parquet'
                stream_file.touch()

                result = d._find_stream_burn_source('tdx')
                assert result == stream_file

    def test_auto_searches_all(self, mock_config_dict, mock_logger, tmp_path):
        """Auto mode should search across all geofabric dirs."""
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                tdx_dir = d.project_dir / 'attributes' / 'geofabric' / 'tdx'
                tdx_dir.mkdir(parents=True, exist_ok=True)
                stream_file = tdx_dir / 'some_rivers_data.parquet'
                stream_file.touch()

                result = d._find_stream_burn_source('auto')
                assert result == stream_file

    def test_auto_falls_back_to_river_network(self, mock_config_dict, mock_logger, tmp_path):
        """Auto mode should fall back to shapefiles/river_network/ dir."""
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                rn_dir = d.project_dir / 'shapefiles' / 'river_network'
                rn_dir.mkdir(parents=True, exist_ok=True)
                stream_file = rn_dir / 'test_domain_riverNetwork_delineate.shp'
                stream_file.touch()

                result = d._find_stream_burn_source('auto')
                assert result == stream_file

    def test_auto_returns_none_when_nothing_found(self, mock_config_dict, mock_logger, tmp_path):
        """Auto mode should return None when no stream files exist."""
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
                d = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                result = d._find_stream_burn_source('auto')
                assert result is None
