"""
Shared fixtures for MESH model tests.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mesh_config(temp_dir):
    """Create a MESH-specific configuration."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'mesh_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'delineate',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'MESH',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'CALIBRATION_PERIOD': '2020-01-01, 2020-06-30',
        'MESH_SPATIAL_MODE': 'distributed',
        'MESH_PARAMS_TO_CALIBRATE': 'ZSNL,MANN,RCHARG',
        'MESH_SPINUP_DAYS': 365,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def setup_mesh_directories(temp_dir, mesh_config):
    """Set up directory structure for MESH testing."""
    data_dir = mesh_config.system.data_dir
    domain_dir = data_dir / f"domain_{mesh_config.domain.name}"

    # Create necessary directories
    forcing_dir = domain_dir / 'forcing' / 'MESH_input'
    basin_avg_dir = domain_dir / 'forcing' / 'basin_averaged_data'
    observations_dir = domain_dir / 'observations' / 'streamflow' / 'preprocessed'
    simulations_dir = domain_dir / 'simulations' / 'MESH'
    shapefiles_dir = domain_dir / 'shapefiles' / 'river_basins'
    river_network_dir = domain_dir / 'shapefiles' / 'river_network'
    settings_dir = domain_dir / 'settings' / 'MESH'
    attributes_dir = domain_dir / 'attributes' / 'gistool-outputs'
    installs_dir = mesh_config.system.code_dir / 'installs' / 'mesh' / 'bin'

    for d in [forcing_dir, basin_avg_dir, observations_dir, simulations_dir,
              shapefiles_dir, river_network_dir, settings_dir, attributes_dir, installs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy executable
    mesh_exe = installs_dir / 'mesh.exe'
    mesh_exe.touch()

    return {
        'data_dir': data_dir,
        'domain_dir': domain_dir,
        'forcing_dir': forcing_dir,
        'basin_avg_dir': basin_avg_dir,
        'observations_dir': observations_dir,
        'simulations_dir': simulations_dir,
        'shapefiles_dir': shapefiles_dir,
        'river_network_dir': river_network_dir,
        'settings_dir': settings_dir,
        'attributes_dir': attributes_dir,
        'installs_dir': installs_dir,
        'mesh_exe': mesh_exe,
    }


@pytest.fixture
def sample_mesh_output_csv(setup_mesh_directories):
    """Create a sample MESH output CSV file."""
    forcing_dir = setup_mesh_directories['forcing_dir']
    output_file = forcing_dir / 'MESH_output_streamflow.csv'

    # MESH format: DAY, YEAR, QOMEAS1, QOSIM1, ...
    csv_content = """DAY, YEAR, QOMEAS1, QOSIM1, QOMEAS2, QOSIM2
1, 2020, 10.5, 11.2, 5.3, 5.1
2, 2020, 11.0, 10.8, 5.5, 5.4
3, 2020, 12.5, 12.1, 6.0, 5.9
4, 2020, 15.0, 14.5, 7.0, 6.8
5, 2020, 13.0, 13.2, 6.5, 6.3
"""
    output_file.write_text(csv_content)
    return output_file


@pytest.fixture
def sample_river_shapefile(setup_mesh_directories):
    """Create mock river network shapefile path."""
    river_dir = setup_mesh_directories['river_network_dir']
    river_shp = river_dir / 'test_domain_riverNetwork_delineate.shp'
    river_shp.touch()
    # Create sidecar files
    for ext in ['.shx', '.dbf', '.prj']:
        (river_dir / f'test_domain_riverNetwork_delineate{ext}').touch()
    return river_shp


@pytest.fixture
def sample_basin_shapefile(setup_mesh_directories):
    """Create mock river basins shapefile path."""
    basin_dir = setup_mesh_directories['shapefiles_dir']
    basin_shp = basin_dir / 'test_domain_riverBasins_delineate.shp'
    basin_shp.touch()
    # Create sidecar files
    for ext in ['.shx', '.dbf', '.prj']:
        (basin_dir / f'test_domain_riverBasins_delineate{ext}').touch()
    return basin_shp


@pytest.fixture
def mock_meshflow():
    """Mock meshflow module for testing without the dependency."""
    with patch.dict('sys.modules', {'meshflow': MagicMock()}):
        yield
