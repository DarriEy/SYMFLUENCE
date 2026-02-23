from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.data_manager import DataManager


@pytest.fixture
def mock_config(tmp_path):
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'DOMAIN_DEFINITION_METHOD': 'subset',
        'SUB_GRID_DISCRETIZATION': 'GRU',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': '3600',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'STREAMFLOW_DATA_PROVIDER': 'LOCAL',
        'STREAMFLOW_RAW_PATH': 'default',
        'STREAMFLOW_PROCESSED_PATH': 'default',
        'STREAMFLOW_RAW_NAME': 'test_flow.csv',
        'ADDITIONAL_OBSERVATIONS': []
    }
    return SymfluenceConfig(**config_dict)

def test_data_manager_initialization(mock_config, mock_logger):
    """Test that DataManager can be initialized with the new structure."""
    dm = DataManager(mock_config, mock_logger)
    assert dm.project_dir == Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain"

@patch('symfluence.data.acquisition.observed_processor.ObservedDataProcessor.process_streamflow_data')
@patch('symfluence.data.acquisition.observed_processor.ObservedDataProcessor.process_fluxnet_data')
def test_process_observed_data_calls(mock_flux, mock_flow, mock_config, mock_logger):
    """Test that process_observed_data correctly calls the processor methods."""
    dm = DataManager(mock_config, mock_logger)
    dm.process_observed_data()

    mock_flow.assert_called_once()
    mock_flux.assert_called_once()

def test_imports_from_data_utils_fail():
    """Verify that we can no longer import from the removed data_utils."""
    with pytest.raises(ImportError):
        from symfluence.data.acquisition.data_utils import ObservedDataProcessor
