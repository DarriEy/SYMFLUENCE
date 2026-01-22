"""
Shared fixtures for LSTM model tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def lstm_config(temp_dir):
    """Create an LSTM-specific configuration."""
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
        'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'lstm_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'LSTM',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'CALIBRATION_PERIOD': '2020-01-01, 2020-06-30',
        'LSTM_HIDDEN_SIZE': 64,
        'LSTM_NUM_LAYERS': 2,
        'LSTM_EPOCHS': 10,
        'LSTM_BATCH_SIZE': 32,
        'LSTM_LEARNING_RATE': 0.001,
        'LSTM_LOOKBACK': 365,
        'LSTM_DROPOUT': 0.1,
        'LSTM_LOAD': False,
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
def setup_lstm_directories(temp_dir, lstm_config):
    """Set up directory structure for LSTM testing."""
    data_dir = lstm_config.system.data_dir
    domain_dir = data_dir / f"domain_{lstm_config.domain.name}"

    # Create necessary directories
    forcing_dir = domain_dir / 'forcing' / 'merged_data'
    observations_dir = domain_dir / 'observations' / 'streamflow' / 'preprocessed'
    simulations_dir = domain_dir / 'simulations' / 'LSTM'

    for d in [forcing_dir, observations_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        'data_dir': data_dir,
        'domain_dir': domain_dir,
        'forcing_dir': forcing_dir,
        'observations_dir': observations_dir,
        'simulations_dir': simulations_dir,
    }


@pytest.fixture
def mock_torch():
    """Mock torch module for testing without PyTorch."""
    with patch.dict('sys.modules', {'torch': MagicMock(), 'torch.nn': MagicMock()}):
        yield
