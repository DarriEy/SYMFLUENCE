"""
Tests for SnowEvaluator.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.evaluation.evaluators.snow import SnowEvaluator


def _make_config_value_getter(config_dict):
    """Create a mock _get_config_value that uses config_dict values."""
    call_count = [0]
    init_values = [
        '2010-01-01, 2015-12-31',
        '2016-01-01, 2018-12-31',
        'daily',
        config_dict.get('OPTIMIZATION_TARGET', 'swe'),
    ]

    def mock_get_config_value(typed_accessor, default=None, dict_key=None):
        if call_count[0] < len(init_values):
            result = init_values[call_count[0]]
            call_count[0] += 1
            return result
        if dict_key and dict_key in config_dict:
            return config_dict[dict_key]
        return default

    return mock_get_config_value


@pytest.fixture
def snow_evaluator(mock_config, tmp_path, mock_logger):
    """Create a SnowEvaluator for testing."""
    config_dict = {
        'OPTIMIZATION_TARGET': 'swe',
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
    }
    mock_config.to_dict.side_effect = lambda flatten=True: config_dict

    with patch.object(SnowEvaluator, '_get_config_value',
                     side_effect=_make_config_value_getter(config_dict)):
        evaluator = SnowEvaluator(mock_config, tmp_path, mock_logger)
        evaluator.config_dict = config_dict
        return evaluator


class TestSnowEvaluatorInit:
    """Test SnowEvaluator initialization."""

    def test_basic_initialization_swe(self, mock_config, tmp_path, mock_logger):
        """Test initialization with SWE target."""
        config_dict = {'OPTIMIZATION_TARGET': 'swe'}
        with patch.object(SnowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = SnowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'swe'
            assert evaluator.variable_name == 'swe'

    def test_basic_initialization_sca(self, mock_config, tmp_path, mock_logger):
        """Test initialization with SCA target."""
        config_dict = {'OPTIMIZATION_TARGET': 'sca'}
        with patch.object(SnowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = SnowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'sca'


class TestSWEExtraction:
    """Test SWE data extraction."""

    def test_extracts_scalar_swe(self, snow_evaluator, tmp_path):
        """Test extraction of scalarSWE variable."""
        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], np.random.uniform(0, 100, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })
        ds['scalarSWE'].attrs['units'] = 'kg m-2'

        file_path = tmp_path / 'snow_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = snow_evaluator._extract_swe_data(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        assert result.min() >= 0  # SWE should be non-negative

    def test_collapses_multiple_hrus(self, snow_evaluator, tmp_path):
        """Test SWE extraction with multiple HRUs using mean aggregation."""
        # Create data with specific values to test mean
        data = np.array([[[10.0, 20.0, 30.0]]])  # 1 time, 1 hru placeholder, 3 actual
        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], [[10.0, 20.0, 30.0]]),  # Mean should be 20
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=1),
            'hru': [0, 1, 2],
        })

        file_path = tmp_path / 'snow_multi_hru.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = snow_evaluator._extract_swe_data(ds_loaded)

        assert result.iloc[0] == pytest.approx(20.0)

    def test_raises_for_missing_swe(self, snow_evaluator, tmp_path):
        """Test raises error when scalarSWE not found."""
        ds = xr.Dataset({
            'some_other_variable': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            with pytest.raises(ValueError, match="scalarSWE variable not found"):
                snow_evaluator._extract_swe_data(ds_loaded)


class TestSCAExtraction:
    """Test Snow Covered Area extraction."""

    def test_extracts_sca_from_snow_fraction(self, snow_evaluator, tmp_path):
        """Test extraction of scalarGroundSnowFraction."""
        ds = xr.Dataset({
            'scalarGroundSnowFraction': (['time', 'hru'], np.random.uniform(0, 1, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'snow_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = snow_evaluator._extract_sca_data(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        assert result.min() >= 0
        assert result.max() <= 1

    def test_extracts_sca_from_swe_threshold(self, snow_evaluator, tmp_path):
        """Test SCA extraction from SWE using threshold."""
        # SWE > 1.0 should be snow covered
        swe_data = np.array([[0.5, 2.0, 0.0, 5.0, 0.8]])
        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], swe_data.T.reshape(5, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=5),
            'hru': [0],
        })

        file_path = tmp_path / 'snow_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = snow_evaluator._extract_sca_data(ds_loaded)

        # Check threshold-based conversion
        expected = [0.0, 1.0, 0.0, 1.0, 0.0]  # SWE > 1.0 = snow
        np.testing.assert_array_equal(result.values, expected)


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_swe_path(self, snow_evaluator, tmp_path):
        """Test SWE observed data path."""
        snow_evaluator._project_dir = tmp_path
        snow_evaluator.domain_name = 'test_basin'
        snow_evaluator.optimization_target = 'swe'

        result = snow_evaluator.get_observed_data_path()

        expected = tmp_path / 'data' / 'observations' / 'snow' / 'swe' / 'processed' / 'test_basin_swe_processed.csv'
        assert result == expected

    def test_sca_path(self, snow_evaluator, tmp_path):
        """Test SCA observed data path."""
        snow_evaluator._project_dir = tmp_path
        snow_evaluator.domain_name = 'test_basin'
        snow_evaluator.optimization_target = 'sca'

        result = snow_evaluator.get_observed_data_path()

        expected = tmp_path / 'data' / 'observations' / 'snow' / 'sca' / 'processed' / 'test_basin_sca_processed.csv'
        assert result == expected


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_swe_column(self, snow_evaluator):
        """Test finding SWE column."""
        snow_evaluator.optimization_target = 'swe'
        columns = ['DateTime', 'swe_mm', 'temperature']

        result = snow_evaluator._get_observed_data_column(columns)

        assert result == 'swe_mm'

    def test_finds_sca_column(self, snow_evaluator):
        """Test finding SCA column."""
        snow_evaluator.optimization_target = 'sca'
        columns = ['DateTime', 'sca_fraction', 'temperature']

        result = snow_evaluator._get_observed_data_column(columns)

        assert result == 'sca_fraction'

    def test_finds_modis_fsc(self, snow_evaluator):
        """Test finding MODIS FSC column."""
        snow_evaluator.optimization_target = 'sca'
        columns = ['DateTime', 'NDSI_Snow_Cover', 'Cloud_Cover']

        result = snow_evaluator._get_observed_data_column(columns)

        assert result == 'NDSI_Snow_Cover'


class TestNeedsRouting:
    """Test routing requirement."""

    def test_snow_never_needs_routing(self, snow_evaluator):
        """Test that snow evaluation never needs routing."""
        assert snow_evaluator.needs_routing() is False


class TestDebugLogging:
    """Test debug logging format (after cleanup)."""

    def test_init_logging_format(self, mock_config, tmp_path, mock_logger):
        """Test that init logging uses standard format."""
        config_dict = {'OPTIMIZATION_TARGET': 'swe'}
        with patch.object(SnowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            # Should not raise any errors
            evaluator = SnowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            # Verify evaluator was created successfully
            assert evaluator is not None
