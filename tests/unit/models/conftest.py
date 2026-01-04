"""
Shared fixtures for model preprocessor tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def base_config(temp_dir):
    """Create a base configuration for testing."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_run',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def summa_config(temp_dir):
    """Create a SUMMA-specific configuration."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_run',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'SETTINGS_SUMMA_PATH': 'default',
        'SETTINGS_SUMMA_FILEMANAGER': 'fileManager.txt',
        'SETTINGS_SUMMA_COLDSTATE': 'coldState.nc',
        'SETTINGS_SUMMA_TRIALPARAMS': 'trialParams.nc',
        'SETTINGS_SUMMA_ATTRIBUTES': 'attributes.nc',
        'CATCHMENT_SHP_HRUID': 'hruId',
        'CATCHMENT_SHP_GRUID': 'gruId',
        'FORCING_MEASUREMENT_HEIGHT': 3.0,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def fuse_config(temp_dir):
    """Create a FUSE-specific configuration."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_run',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'FUSE',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'SETTINGS_FUSE_PATH': 'default',
        'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
        'FUSE_SPATIAL_MODE': 'lumped',
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def setup_test_directories(temp_dir, base_config):
    """Set up common test directory structure."""
    data_dir = base_config.system.data_dir
    code_dir = base_config.system.code_dir
    domain_dir = data_dir / f"domain_{base_config.domain.name}"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Create base settings directories for models
    for model in ['SUMMA', 'FUSE', 'GR', 'HYPE']:
        base_settings = code_dir / 'src' / 'symfluence' / 'resources' / 'base_settings' / model
        base_settings.mkdir(parents=True, exist_ok=True)

        # Create dummy settings files
        (base_settings / f'{model}_settings.txt').write_text('# Test settings')

    return {
        'data_dir': data_dir,
        'code_dir': code_dir,
        'domain_dir': domain_dir,
    }


@pytest.fixture
def mock_forcing_data(setup_test_directories, base_config):
    """Create mock forcing data files."""
    domain_dir = setup_test_directories['domain_dir']
    forcing_dir = domain_dir / 'forcing' / 'merged_data'
    forcing_dir.mkdir(parents=True, exist_ok=True)

    # Create mock NetCDF file path (we won't actually create the file)
    forcing_file = forcing_dir / 'era5_merged.nc'

    return {
        'forcing_dir': forcing_dir,
        'forcing_file': forcing_file,
    }


@pytest.fixture
def mock_shapefile_data(setup_test_directories, base_config):
    """Create mock shapefile directory structure."""
    domain_dir = setup_test_directories['domain_dir']

    # Create shapefile directories
    catchment_dir = domain_dir / 'shapefiles' / 'catchment'
    river_dir = domain_dir / 'shapefiles' / 'river_network'
    forcing_shp_dir = domain_dir / 'shapefiles' / 'forcing'

    for shp_dir in [catchment_dir, river_dir, forcing_shp_dir]:
        shp_dir.mkdir(parents=True, exist_ok=True)

    return {
        'catchment_dir': catchment_dir,
        'river_dir': river_dir,
        'forcing_shp_dir': forcing_shp_dir,
    }
