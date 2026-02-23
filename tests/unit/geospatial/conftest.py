"""
Shared fixtures for geospatial unit tests.

Provides common test fixtures for delineation, raster operations,
and coordinate utilities tests.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

# Import geospatial fixtures from fixtures module
from fixtures.geospatial_fixtures import (
    GEOPANDAS_AVAILABLE,
    synthetic_dem_raster,
    synthetic_forcing_netcdf_with_grid,
    synthetic_pour_point_gdf,
    synthetic_river_network_gdf,
    synthetic_watershed_gdf,
)

# =============================================================================
# Mock Logger Fixture
# =============================================================================

@pytest.fixture
def mock_logger():
    """
    Provide a mock logger that captures log messages.

    Returns a mock logger with info, warning, error, debug methods
    that store messages for assertion.
    """
    logger = MagicMock(spec=logging.Logger)

    # Store messages for verification
    logger.messages = {
        'info': [],
        'warning': [],
        'error': [],
        'debug': [],
    }

    def capture_info(msg, *args, **kwargs):
        logger.messages['info'].append(msg)

    def capture_warning(msg, *args, **kwargs):
        logger.messages['warning'].append(msg)

    def capture_error(msg, *args, **kwargs):
        logger.messages['error'].append(msg)

    def capture_debug(msg, *args, **kwargs):
        logger.messages['debug'].append(msg)

    logger.info.side_effect = capture_info
    logger.warning.side_effect = capture_warning
    logger.error.side_effect = capture_error
    logger.debug.side_effect = capture_debug

    return logger


# =============================================================================
# Mock Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_config_dict(tmp_path):
    """
    Provide a mock configuration dictionary for delineator tests.

    Creates a configuration suitable for GridDelineator, LumpedDelineator,
    and other geospatial components.
    """
    # Create required directories
    (tmp_path / "shapefiles" / "pour_point").mkdir(parents=True, exist_ok=True)
    (tmp_path / "shapefiles" / "river_basins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "shapefiles" / "river_network").mkdir(parents=True, exist_ok=True)
    (tmp_path / "dem").mkdir(parents=True, exist_ok=True)
    (tmp_path / "forcing").mkdir(parents=True, exist_ok=True)
    (tmp_path / "taudem-interim-files").mkdir(parents=True, exist_ok=True)

    # Create a minimal DEM file placeholder (touch)
    dem_path = tmp_path / "dem" / "test_dem.tif"
    dem_path.touch()

    return {
        'DOMAIN_NAME': 'test_domain',
        'PROJECT_DIR': str(tmp_path),
        'DEM_PATH': str(dem_path),
        'DEM_NAME': 'test_dem.tif',
        'POUR_POINT_SHP_PATH': str(tmp_path / "shapefiles" / "pour_point"),
        'POUR_POINT_SHP_NAME': 'test_domain_pourPoint.shp',
        'BOUNDING_BOX_COORDS': '47.0/8.0/46.0/9.0',  # lat_max/lon_min/lat_min/lon_max
        'GRID_CELL_SIZE': 1000.0,
        'CLIP_GRID_TO_WATERSHED': True,
        'GRID_SOURCE': 'generate',
        'NATIVE_GRID_DATASET': 'era5',
        'FORCING_PATH': str(tmp_path / "forcing"),
        'FORCING_DIR': str(tmp_path / "forcing"),
        'FORCING_DATASET': 'era5',
        'NUM_PROCESSES': 1,
        'CLEANUP_INTERMEDIATE_FILES': True,
        'DELINEATION_METHOD': 'stream_threshold',
        'OUTPUT_BASINS_PATH': 'default',
        'OUTPUT_RIVERS_PATH': 'default',
        'TAUDEM_DIR': '/usr/local/bin',  # Default TauDEM location
        'DATA_DIR': str(tmp_path / "data"),
    }


@pytest.fixture
def mock_config_object(mock_config_dict, tmp_path):
    """
    Provide a mock configuration object with nested attribute access.

    Mimics the config object structure used by SYMFLUENCE delineators.
    """
    class MockConfig:
        def __init__(self, d: Dict[str, Any]):
            self._dict = d

        def get(self, key, default=None):
            return self._dict.get(key, default)

        def __getitem__(self, key):
            return self._dict[key]

        def __contains__(self, key):
            return key in self._dict

        def to_dict(self, flatten: bool = True) -> Dict[str, Any]:
            return self._dict.copy()

    class MockDomain:
        def __init__(self, d: Dict[str, Any]):
            self.name = d.get('DOMAIN_NAME', 'test_domain')
            self.bounding_box_coords = d.get('BOUNDING_BOX_COORDS', '47.0/8.0/46.0/9.0')
            self.definition_method = 'lumped'
            self.grid_cell_size = d.get('GRID_CELL_SIZE', 1000.0)
            self.clip_grid_to_watershed = d.get('CLIP_GRID_TO_WATERSHED', True)
            self.grid_source = d.get('GRID_SOURCE', 'generate')
            self.native_grid_dataset = d.get('NATIVE_GRID_DATASET', 'era5')
            self.subset_from_geofabric = False
            self.delineation = MockDelineation(d)

    class MockDelineation:
        def __init__(self, d: Dict[str, Any]):
            self.method = d.get('DELINEATION_METHOD', 'stream_threshold')
            self.cleanup_intermediate_files = d.get('CLEANUP_INTERMEDIATE_FILES', True)
            self.routing = 'lumped'
            self.geofabric_type = 'na'
            self.delineate_coastal_watersheds = False
            self.delineate_by_pourpoint = True

    class MockPaths:
        def __init__(self, d: Dict[str, Any], tmp_path: Path):
            self.project_dir = tmp_path
            self.dem_path = d.get('DEM_PATH', 'default')
            self.pour_point_path = d.get('POUR_POINT_SHP_PATH', 'default')
            self.pour_point_name = d.get('POUR_POINT_SHP_NAME', 'default')
            self.output_basins_path = d.get('OUTPUT_BASINS_PATH', 'default')
            self.output_rivers_path = d.get('OUTPUT_RIVERS_PATH', 'default')
            self.river_basins_name = 'default'
            self.forcing_path = d.get('FORCING_DIR', str(tmp_path / "forcing"))

    class MockSystem:
        def __init__(self, d: Dict[str, Any]):
            self.num_processes = d.get('NUM_PROCESSES', 1)
            self.taudem_dir = d.get('TAUDEM_DIR', '/usr/local/bin')

    class MockForcing:
        def __init__(self, d: Dict[str, Any]):
            self.dataset = d.get('FORCING_DATASET', 'era5')

    # Create the config object
    config = MockConfig(mock_config_dict)
    config.domain = MockDomain(mock_config_dict)
    config.paths = MockPaths(mock_config_dict, tmp_path)
    config.system = MockSystem(mock_config_dict)
    config.forcing = MockForcing(mock_config_dict)

    return config


# =============================================================================
# Geospatial Test Domain Fixture
# =============================================================================

@pytest.fixture
def geospatial_test_domain_setup(tmp_path, mock_config_dict, mock_logger):
    """
    Provide a complete test domain setup for geospatial tests.

    Creates directory structure, synthetic data files, and configuration.
    Returns a dictionary with all paths and components.
    """
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")

    # Create directory structure
    shapefiles_dir = tmp_path / "shapefiles"
    pour_point_dir = shapefiles_dir / "pour_point"
    river_basins_dir = shapefiles_dir / "river_basins"
    river_network_dir = shapefiles_dir / "river_network"
    dem_dir = tmp_path / "attributes" / "dem"
    forcing_dir = tmp_path / "forcing"
    interim_dir = tmp_path / "taudem-interim-files"

    for d in [pour_point_dir, river_basins_dir, river_network_dir, dem_dir, forcing_dir, interim_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create pour point shapefile
    pour_point = synthetic_pour_point_gdf(lat=46.5, lon=8.5)
    pour_point_path = pour_point_dir / "test_domain_pourPoint.shp"
    pour_point.to_file(pour_point_path)

    # Create synthetic watershed and river network
    basins = synthetic_watershed_gdf(n_basins=5, lat_center=46.5, lon_center=8.5)
    rivers = synthetic_river_network_gdf(basins)

    basins_path = river_basins_dir / "test_domain_riverBasins_test.shp"
    rivers_path = river_network_dir / "test_domain_riverNetwork_test.shp"

    basins.to_file(basins_path)
    rivers.to_file(rivers_path)

    # Create DEM if rasterio available
    dem_path = dem_dir / "test_domain_dem.tif"
    dem_path_result, dem_array, dem_meta = synthetic_dem_raster(
        lat_range=(46.0, 47.0),
        lon_range=(8.0, 9.0),
        resolution=0.01,  # Coarser for faster tests
        output_path=dem_path,
    )

    # Create forcing data
    forcing_path = forcing_dir / "era5_test.nc"
    _, forcing_ds = synthetic_forcing_netcdf_with_grid(
        lat_range=(46.0, 47.0),
        lon_range=(8.0, 9.0),
        resolution=0.25,
        output_path=forcing_path,
    )

    # Update config with actual paths
    config = mock_config_dict.copy()
    config['PROJECT_DIR'] = str(tmp_path)
    config['DEM_PATH'] = str(dem_path_result) if dem_path_result else 'default'
    config['POUR_POINT_SHP_PATH'] = str(pour_point_dir)
    config['FORCING_DIR'] = str(forcing_dir)

    return {
        'tmp_path': tmp_path,
        'config': config,
        'logger': mock_logger,
        'dem_path': dem_path_result,
        'dem_array': dem_array,
        'dem_meta': dem_meta,
        'pour_point_path': pour_point_path,
        'pour_point_gdf': pour_point,
        'basins_path': basins_path,
        'basins_gdf': basins,
        'rivers_path': rivers_path,
        'rivers_gdf': rivers,
        'forcing_path': forcing_path,
        'forcing_ds': forcing_ds,
        'shapefiles_dir': shapefiles_dir,
        'interim_dir': interim_dir,
    }


# =============================================================================
# Skip Markers
# =============================================================================

requires_geopandas = pytest.mark.skipif(
    not GEOPANDAS_AVAILABLE,
    reason="geopandas not available"
)

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

requires_rasterio = pytest.mark.skipif(
    not RASTERIO_AVAILABLE,
    reason="rasterio not available"
)


# =============================================================================
# TauDEM Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_taudem_executor():
    """
    Provide a mock TauDEM executor for tests that don't need actual TauDEM.

    Returns a MagicMock that can be configured for specific test scenarios.
    """
    mock_taudem = MagicMock()
    mock_taudem.taudem_dir = Path("/usr/local/bin")
    mock_taudem.get_mpi_command.return_value = None  # No MPI by default
    mock_taudem.run_command.return_value = 0  # Success by default

    return mock_taudem


@pytest.fixture
def mock_gdal_processor(mock_logger):
    """
    Provide a mock GDAL processor for tests.

    Returns a MagicMock that simulates GDAL operations.
    """
    mock_gdal = MagicMock()
    mock_gdal.raster_to_polygon.return_value = True
    mock_gdal.run_gdal_processing.return_value = True

    return mock_gdal
