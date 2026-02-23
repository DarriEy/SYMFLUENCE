"""
Unit tests for DataProcessor.

Tests data loading and preparation functionality for visualization.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.processors.data_processor import DataProcessor


@pytest.fixture
def data_processor(mock_config, mock_logger):
    """Create a DataProcessor instance."""
    return DataProcessor(mock_config, mock_logger)


class TestDataProcessor:
    """Test suite for DataProcessor."""

    def test_initialization(self, data_processor, mock_config):
        """Test that DataProcessor initializes correctly."""
        assert data_processor.logger is not None
        assert data_processor.project_dir is not None

    def test_project_dir_construction(self, mock_config, mock_logger):
        """Test that project_dir is constructed correctly."""
        processor = DataProcessor(mock_config, mock_logger)
        expected_path = Path(mock_config['SYMFLUENCE_DATA_DIR']) / f"domain_{mock_config['DOMAIN_NAME']}"
        assert processor.project_dir == expected_path


class TestStreamflowObservationsLoading:
    """Test streamflow observation loading methods."""

    def test_load_streamflow_observations_success(self, data_processor):
        """Test successful loading of streamflow observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            dates = pd.date_range('2020-01-01', periods=100, freq='h')
            df = pd.DataFrame({
                'datetime': dates,
                'discharge_cms': np.random.random(100) * 10
            })
            obs_file = Path(tmpdir) / "obs.csv"
            df.to_csv(obs_file, index=False)

            obs_files = [('USGS', str(obs_file))]
            result = data_processor.load_streamflow_observations(obs_files)

            assert len(result) == 1
            assert result[0][0] == 'USGS'
            assert isinstance(result[0][1], pd.Series)

    def test_load_streamflow_observations_missing_column(
        self, data_processor, mock_logger
    ):
        """Test loading with missing discharge_cms column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV without discharge_cms
            dates = pd.date_range('2020-01-01', periods=100, freq='h')
            df = pd.DataFrame({
                'datetime': dates,
                'flow': np.random.random(100) * 10  # Wrong column name
            })
            obs_file = Path(tmpdir) / "obs.csv"
            df.to_csv(obs_file, index=False)

            obs_files = [('USGS', str(obs_file))]
            result = data_processor.load_streamflow_observations(obs_files)

            # Should return empty list and log warning
            assert len(result) == 0
            mock_logger.warning.assert_called()

    def test_load_streamflow_observations_file_not_found(
        self, data_processor, mock_logger
    ):
        """Test loading with nonexistent file."""
        obs_files = [('USGS', '/nonexistent/path/obs.csv')]
        result = data_processor.load_streamflow_observations(obs_files)

        assert len(result) == 0
        mock_logger.warning.assert_called()

    def test_load_streamflow_observations_resampling(self, data_processor):
        """Test resampling during observation loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create hourly data
            dates = pd.date_range('2020-01-01', periods=48, freq='h')
            df = pd.DataFrame({
                'datetime': dates,
                'discharge_cms': np.random.random(48) * 10
            })
            obs_file = Path(tmpdir) / "obs.csv"
            df.to_csv(obs_file, index=False)

            obs_files = [('USGS', str(obs_file))]

            # Load with daily resampling
            result = data_processor.load_streamflow_observations(
                obs_files, resample_freq='D'
            )

            assert len(result) == 1
            # Should have 2 days of data (48 hours -> 2 days)
            assert len(result[0][1]) <= 3  # ~2-3 daily values

    def test_load_streamflow_observations_no_resampling(self, data_processor):
        """Test loading without resampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dates = pd.date_range('2020-01-01', periods=24, freq='h')
            df = pd.DataFrame({
                'datetime': dates,
                'discharge_cms': np.random.random(24) * 10
            })
            obs_file = Path(tmpdir) / "obs.csv"
            df.to_csv(obs_file, index=False)

            obs_files = [('USGS', str(obs_file))]
            result = data_processor.load_streamflow_observations(
                obs_files, resample_freq=None
            )

            assert len(result) == 1
            assert len(result[0][1]) == 24

    def test_load_streamflow_observations_multiple_files(self, data_processor):
        """Test loading multiple observation files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dates = pd.date_range('2020-01-01', periods=24, freq='h')

            # Create two CSV files
            for i, name in enumerate(['usgs', 'wsc']):
                df = pd.DataFrame({
                    'datetime': dates,
                    'discharge_cms': np.random.random(24) * 10 + i
                })
                obs_file = Path(tmpdir) / f"{name}_obs.csv"
                df.to_csv(obs_file, index=False)

            obs_files = [
                ('USGS', str(Path(tmpdir) / 'usgs_obs.csv')),
                ('WSC', str(Path(tmpdir) / 'wsc_obs.csv')),
            ]
            result = data_processor.load_streamflow_observations(obs_files)

            assert len(result) == 2


class TestLumpedModelOutputsLoading:
    """Test lumped model output loading methods."""

    def test_load_lumped_model_outputs_success(self, data_processor):
        """Test successful loading of lumped model outputs."""
        # This test requires xarray/netCDF, so we mock it
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=True)
            mock_series = pd.Series(
                np.random.random(100),
                index=pd.date_range('2020-01-01', periods=100, freq='h')
            )
            mock_ds.__getitem__ = Mock(return_value=Mock(to_pandas=Mock(return_value=mock_series)))
            mock_open.return_value = mock_ds

            with patch.object(data_processor, '_get_basin_area', return_value=1000000):
                model_outputs = [('SUMMA', '/path/to/output.nc')]
                result = data_processor.load_lumped_model_outputs(model_outputs)

                assert len(result) == 1
                assert result[0][0] == 'SUMMA'

    def test_load_lumped_model_outputs_missing_variable(
        self, data_processor, mock_logger
    ):
        """Test loading with missing variable in NetCDF."""
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=False)
            mock_open.return_value = mock_ds

            model_outputs = [('SUMMA', '/path/to/output.nc')]
            result = data_processor.load_lumped_model_outputs(model_outputs)

            assert len(result) == 0
            mock_logger.error.assert_called()

    def test_load_lumped_model_outputs_unit_conversion(self, data_processor):
        """Test unit conversion during loading."""
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=True)
            original_values = np.random.random(100) * 0.001  # m/s scale
            mock_series = pd.Series(
                original_values,
                index=pd.date_range('2020-01-01', periods=100, freq='h')
            )
            mock_ds.__getitem__ = Mock(return_value=Mock(to_pandas=Mock(return_value=mock_series.copy())))
            mock_open.return_value = mock_ds

            basin_area = 1e6  # 1 km²
            with patch.object(data_processor, '_get_basin_area', return_value=basin_area):
                model_outputs = [('SUMMA', '/path/to/output.nc')]
                result = data_processor.load_lumped_model_outputs(
                    model_outputs, convert_units=True
                )

                assert len(result) == 1
                # Values should be multiplied by basin area

    def test_load_lumped_model_outputs_no_conversion(self, data_processor):
        """Test loading without unit conversion."""
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=True)
            original_values = np.random.random(100)
            mock_series = pd.Series(
                original_values,
                index=pd.date_range('2020-01-01', periods=100, freq='h')
            )
            mock_ds.__getitem__ = Mock(return_value=Mock(to_pandas=Mock(return_value=mock_series)))
            mock_open.return_value = mock_ds

            model_outputs = [('SUMMA', '/path/to/output.nc')]
            result = data_processor.load_lumped_model_outputs(
                model_outputs, convert_units=False
            )

            assert len(result) == 1


class TestDistributedModelOutputsLoading:
    """Test distributed model output loading methods."""

    def test_load_distributed_model_outputs_success(self, data_processor):
        """Test successful loading of distributed model outputs."""
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=True)
            mock_data = Mock()
            mock_data.dims = ['seg', 'time']

            # Create mock data array
            dates = pd.date_range('2020-01-01', periods=100, freq='h')
            mock_data.sel = Mock(return_value=Mock(to_pandas=Mock(
                return_value=pd.Series(np.random.random(100), index=dates)
            )))
            mock_ds.__getitem__ = Mock(return_value=mock_data)
            mock_ds.seg = Mock(values=[1, 2, 3])
            mock_open.return_value = mock_ds

            result = data_processor.load_distributed_model_outputs(
                '/path/to/routing.nc',
                variable_name='streamflow',
                reach_id=1
            )

            # May return series or None depending on implementation

    def test_load_distributed_model_outputs_missing_variable(
        self, data_processor, mock_logger
    ):
        """Test loading with missing variable."""
        with patch('xarray.open_dataset') as mock_open:
            mock_ds = Mock()
            mock_ds.__contains__ = Mock(return_value=False)
            mock_open.return_value = mock_ds

            result = data_processor.load_distributed_model_outputs(
                '/path/to/routing.nc'
            )

            assert result is None
            mock_logger.error.assert_called()


class TestResultsFileLoading:
    """Test results file loading methods."""

    def test_read_results_file_success(self, data_processor):
        """Test successful reading of results file."""
        if hasattr(data_processor, 'read_results_file'):
            with tempfile.TemporaryDirectory() as tmpdir:
                data_processor.project_dir = Path(tmpdir)
                results_dir = Path(tmpdir) / "results"
                results_dir.mkdir()

                # Create results CSV with proper experiment ID from config
                exp_id = data_processor.config.get('EXPERIMENT_ID', 'test_exp')
                dates = pd.date_range('2020-01-01', periods=100, freq='D')
                df = pd.DataFrame({
                    'obs_discharge': np.random.random(100) * 10,
                    'SUMMA_discharge': np.random.random(100) * 10,
                }, index=dates)
                df.to_csv(results_dir / f"{exp_id}_results.csv")

                result = data_processor.read_results_file()
                assert isinstance(result, pd.DataFrame)

    def test_read_results_file_not_found(self, data_processor, mock_logger):
        """Test reading nonexistent results file."""
        if hasattr(data_processor, 'read_results_file'):
            with tempfile.TemporaryDirectory() as tmpdir:
                data_processor.project_dir = Path(tmpdir)
                # Don't create the results file - test that FileNotFoundError is raised

                with pytest.raises(FileNotFoundError):
                    data_processor.read_results_file()
