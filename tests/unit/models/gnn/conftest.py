"""
Shared fixtures for GNN model tests.
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
def gnn_config(temp_dir):
    """Create a GNN-specific configuration."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'gnn_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'delineate',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'GNN',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'CALIBRATION_PERIOD': '2020-01-01, 2020-06-30',
        'GNN_HIDDEN_SIZE': 64,
        'GNN_NUM_LAYERS': 3,
        'GNN_EPOCHS': 10,
        'GNN_BATCH_SIZE': 32,
        'GNN_LEARNING_RATE': 0.001,
        'GNN_DROPOUT': 0.1,
        'GNN_LOAD': False,
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
def setup_gnn_directories(temp_dir, gnn_config):
    """Set up directory structure for GNN testing."""
    data_dir = gnn_config.system.data_dir
    domain_dir = data_dir / f"domain_{gnn_config.domain.name}"

    # Create necessary directories
    forcing_dir = domain_dir / 'forcing' / 'merged_data'
    observations_dir = domain_dir / 'observations' / 'streamflow' / 'preprocessed'
    simulations_dir = domain_dir / 'simulations' / 'GNN'
    shapefiles_dir = domain_dir / 'shapefiles' / 'river_network'

    for d in [forcing_dir, observations_dir, simulations_dir, shapefiles_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        'data_dir': data_dir,
        'domain_dir': domain_dir,
        'forcing_dir': forcing_dir,
        'observations_dir': observations_dir,
        'simulations_dir': simulations_dir,
        'shapefiles_dir': shapefiles_dir,
    }


@pytest.fixture
def mock_torch():
    """Mock torch module for testing without PyTorch."""
    with patch.dict('sys.modules', {'torch': MagicMock(), 'torch.nn': MagicMock(), 'torch_geometric': MagicMock()}):
        yield
