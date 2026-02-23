"""
Tests for base ModelEvaluator class.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from symfluence.evaluation.evaluators.base import ModelEvaluator


class ConcreteEvaluator(ModelEvaluator):
    """Concrete implementation for testing abstract base class."""

    def get_simulation_files(self, sim_dir: Path):
        return list(sim_dir.glob('*.nc'))

    def extract_simulated_data(self, sim_files, **kwargs):
        return pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))

    def get_observed_data_path(self):
        return self.project_dir / 'obs.csv'

    def needs_routing(self):
        return False

    def _get_observed_data_column(self, columns):
        return columns[0] if columns else None


class TestModelEvaluatorInit:
    """Test ModelEvaluator initialization."""

    def test_init_with_typed_config(self, mock_config, tmp_path, mock_logger):
        """Test initialization with typed SymfluenceConfig."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value='2010-01-01, 2015-12-31'):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            assert evaluator.config == mock_config
            assert evaluator._project_dir == tmp_path

    def test_init_parses_calibration_period(self, mock_config, tmp_path, mock_logger):
        """Test that calibration period is parsed correctly."""
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',  # calibration_period
            '2016-01-01, 2018-12-31',  # evaluation_period
            'daily',  # calibration_timestep
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            assert evaluator.calibration_period[0] == pd.Timestamp('2010-01-01')
            assert evaluator.calibration_period[1] == pd.Timestamp('2015-12-31')

    def test_init_handles_invalid_timestep(self, mock_config, tmp_path, mock_logger):
        """Test that invalid calibration_timestep defaults to 'native'."""
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',
            '2016-01-01, 2018-12-31',
            'invalid_timestep',
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            assert evaluator.eval_timestep == 'native'


class TestCalculatePeriodMetrics:
    """Test _calculate_period_metrics method."""

    def test_calculates_metrics_for_calibration_period(self, mock_config, tmp_path, mock_logger):
        """Test metrics calculation for calibration period."""
        mock_config.evaluation.spinup_years = None
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2010-12-31',
            '2011-01-01, 2011-12-31',
            'native',
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

        # Create test data (outside patch so real _get_config_value is used)
        time_idx = pd.date_range('2010-01-01', periods=365, freq='D')
        obs_data = pd.Series(np.random.uniform(10, 100, 365), index=time_idx)
        sim_data = pd.Series(np.random.uniform(10, 100, 365), index=time_idx)

        period = (pd.Timestamp('2010-01-01'), pd.Timestamp('2010-12-31'))
        metrics = evaluator._calculate_period_metrics(obs_data, sim_data, period, 'Test')

        assert 'Test_KGE' in metrics
        assert 'Test_NSE' in metrics
        assert 'Test_RMSE' in metrics

    def test_rounds_both_indices_consistently(self, mock_config, tmp_path, mock_logger):
        """Test that both obs and sim indices are rounded consistently (bug fix)."""
        mock_config.evaluation.spinup_years = None
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2010-01-31',
            '',
            'native',
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

        # Create data with aligned timestamps (after rounding)
        # Using the same timestamps ensures they align properly after rounding
        time_idx = pd.date_range('2010-01-01', periods=10, freq='D')

        obs_data = pd.Series(np.random.rand(10), index=time_idx)
        sim_data = pd.Series(np.random.rand(10), index=time_idx)

        period = (pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-31'))
        metrics = evaluator._calculate_period_metrics(obs_data, sim_data, period, 'Test')

        # Should find common indices
        assert 'Test_KGE' in metrics

    def test_handles_same_timezone_data(self, mock_config, tmp_path, mock_logger):
        """Test handling of timezone-naive data."""
        mock_config.evaluation.spinup_years = None
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2010-01-10',
            '',
            'native',
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

        # Create timezone-naive data for both series
        time_idx = pd.date_range('2010-01-01', periods=10, freq='D')

        obs_data = pd.Series(np.random.rand(10), index=time_idx)
        sim_data = pd.Series(np.random.rand(10), index=time_idx)

        period = (pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-10'))
        metrics = evaluator._calculate_period_metrics(obs_data, sim_data, period, 'Test')

        # Should calculate metrics successfully
        assert 'Test_KGE' in metrics

    def test_returns_empty_dict_for_no_overlap(self, mock_config, tmp_path, mock_logger):
        """Test returns empty dict when no common time indices."""
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2010-01-31',
            '',
            'native',
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            obs_idx = pd.date_range('2010-01-01', periods=10, freq='D')
            sim_idx = pd.date_range('2011-01-01', periods=10, freq='D')  # Different year

            obs_data = pd.Series(np.arange(10, dtype=float), index=obs_idx)
            sim_data = pd.Series(np.arange(10, dtype=float), index=sim_idx)

            period = (pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-31'))
            metrics = evaluator._calculate_period_metrics(obs_data, sim_data, period, 'Test')

            assert metrics == {}


class TestCollapseSpatialDims:
    """Test _collapse_spatial_dims method."""

    def test_collapses_single_hru(self, mock_config, tmp_path, mock_logger):
        """Test collapsing single HRU dimension."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = xr.DataArray(
                np.random.rand(10, 1),
                dims=['time', 'hru'],
                coords={'time': pd.date_range('2010-01-01', periods=10), 'hru': [0]}
            )

            result = evaluator._collapse_spatial_dims(data)

            assert isinstance(result, pd.Series)
            assert len(result) == 10
            assert 'hru' not in result.index.names

    def test_collapses_multiple_hrus_with_mean(self, mock_config, tmp_path, mock_logger):
        """Test collapsing multiple HRUs using mean aggregation."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = xr.DataArray(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2 time, 3 hru
                dims=['time', 'hru'],
                coords={'time': pd.date_range('2010-01-01', periods=2), 'hru': [0, 1, 2]}
            )

            result = evaluator._collapse_spatial_dims(data, aggregate='mean')

            assert isinstance(result, pd.Series)
            assert len(result) == 2
            # Mean of [1, 2, 3] = 2.0, Mean of [4, 5, 6] = 5.0
            assert result.iloc[0] == pytest.approx(2.0)
            assert result.iloc[1] == pytest.approx(5.0)

    def test_collapses_multiple_dims(self, mock_config, tmp_path, mock_logger):
        """Test collapsing multiple spatial dimensions."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = xr.DataArray(
                np.random.rand(10, 2, 3),
                dims=['time', 'hru', 'gru'],
                coords={
                    'time': pd.date_range('2010-01-01', periods=10),
                    'hru': [0, 1],
                    'gru': [0, 1, 2]
                }
            )

            result = evaluator._collapse_spatial_dims(data)

            assert isinstance(result, pd.Series)
            assert len(result) == 10

    def test_sum_aggregation(self, mock_config, tmp_path, mock_logger):
        """Test sum aggregation method."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                dims=['time', 'hru'],
                coords={'time': pd.date_range('2010-01-01', periods=2), 'hru': [0, 1]}
            )

            result = evaluator._collapse_spatial_dims(data, aggregate='sum')

            assert result.iloc[0] == pytest.approx(3.0)  # 1 + 2
            assert result.iloc[1] == pytest.approx(7.0)  # 3 + 4

    def test_first_aggregation(self, mock_config, tmp_path, mock_logger):
        """Test first aggregation method."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                dims=['time', 'hru'],
                coords={'time': pd.date_range('2010-01-01', periods=2), 'hru': [0, 1]}
            )

            result = evaluator._collapse_spatial_dims(data, aggregate='first')

            assert result.iloc[0] == pytest.approx(1.0)
            assert result.iloc[1] == pytest.approx(3.0)


class TestResampleToTimestep:
    """Test _resample_to_timestep method."""

    def test_resample_hourly_to_daily(self, mock_config, tmp_path, mock_logger):
        """Test resampling hourly data to daily."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            # Create hourly data
            hourly_idx = pd.date_range('2010-01-01', periods=48, freq='h')
            hourly_data = pd.Series(np.arange(48, dtype=float), index=hourly_idx)

            result = evaluator._resample_to_timestep(hourly_data, 'daily')

            assert len(result) == 2  # 2 days

    def test_native_returns_unchanged(self, mock_config, tmp_path, mock_logger):
        """Test that 'native' timestep returns data unchanged."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = pd.Series([1, 2, 3], index=pd.date_range('2010-01-01', periods=3, freq='D'))
            result = evaluator._resample_to_timestep(data, 'native')

            pd.testing.assert_series_equal(result, data)


class TestParseDateRange:
    """Test _parse_date_range method."""

    def test_parses_valid_date_range(self, mock_config, tmp_path, mock_logger):
        """Test parsing valid date range string."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            start, end = evaluator._parse_date_range('2010-01-01, 2015-12-31')

            assert start == pd.Timestamp('2010-01-01')
            assert end == pd.Timestamp('2015-12-31')

    def test_returns_none_for_empty_string(self, mock_config, tmp_path, mock_logger):
        """Test returning None for empty string."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            start, end = evaluator._parse_date_range('')

            assert start is None
            assert end is None

    def test_handles_invalid_date_format(self, mock_config, tmp_path, mock_logger):
        """Test handling invalid date format."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            start, end = evaluator._parse_date_range('invalid, dates')

            assert start is None
            assert end is None


class TestAlignSeries:
    """Test align_series method."""

    def test_aligns_series_with_spinup(self, mock_config, tmp_path, mock_logger):
        """Test series alignment with spinup years removed."""
        mock_config.evaluation.spinup_years = None
        with patch.object(ModelEvaluator, '_get_config_value', side_effect=[
            '',  # calibration_period
            '',  # evaluation_period
            '',  # calibration_timestep
        ]):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

        # Set config_dict override for align_series to use (outside patch)
        evaluator.config_dict = {
            'EVALUATION_SPINUP_YEARS': 1,
            'CALIBRATION_PERIOD': '2010-01-01, 2012-12-31',
        }

        # 3 years of data
        time_idx = pd.date_range('2010-01-01', periods=365 * 3, freq='D')
        sim = pd.Series(np.random.rand(365 * 3), index=time_idx)
        obs = pd.Series(np.random.rand(365 * 3), index=time_idx)

        sim_aligned, obs_aligned = evaluator.align_series(sim, obs)

        # First year should be removed
        assert sim_aligned.index[0] >= pd.Timestamp('2011-01-01')

    def test_handles_empty_series(self, mock_config, tmp_path, mock_logger):
        """Test handling empty series."""
        mock_config.to_dict.return_value = {}

        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            sim = pd.Series(dtype=float)
            obs = pd.Series(dtype=float)

            sim_aligned, obs_aligned = evaluator.align_series(sim, obs)

            assert sim_aligned.empty
            assert obs_aligned.empty


class TestValidateData:
    """Test _validate_data method for data quality validation."""

    def test_returns_false_for_none_data(self, mock_config, tmp_path, mock_logger):
        """Test that None data returns (False, error_message)."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            is_valid, error_msg = evaluator._validate_data(None, 'test')

            assert is_valid is False
            assert 'None' in error_msg

    def test_returns_false_for_empty_data(self, mock_config, tmp_path, mock_logger):
        """Test that empty Series returns (False, error_message)."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            empty_data = pd.Series(dtype=float)
            is_valid, error_msg = evaluator._validate_data(empty_data, 'test')

            assert is_valid is False
            assert 'empty' in error_msg

    def test_returns_false_for_all_nan_data(self, mock_config, tmp_path, mock_logger):
        """Test that all-NaN data returns (False, error_message)."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            nan_data = pd.Series([np.nan, np.nan, np.nan])
            is_valid, error_msg = evaluator._validate_data(nan_data, 'test')

            assert is_valid is False
            assert 'only NaN' in error_msg

    def test_returns_false_for_insufficient_points(self, mock_config, tmp_path, mock_logger):
        """Test that data with too few valid points returns (False, error_message)."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            few_points = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
            # Default min_valid_points is 10
            is_valid, error_msg = evaluator._validate_data(few_points, 'test')

            assert is_valid is False
            assert 'insufficient' in error_msg

    def test_returns_true_for_valid_data(self, mock_config, tmp_path, mock_logger):
        """Test that valid data returns (True, None)."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            valid_data = pd.Series(np.random.rand(100))
            is_valid, error_msg = evaluator._validate_data(valid_data, 'test')

            assert is_valid is True
            assert error_msg is None

    def test_warns_for_constant_data(self, mock_config, tmp_path, mock_logger, caplog):
        """Test that constant data logs a warning but still returns True."""
        import logging
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            with caplog.at_level(logging.WARNING):
                constant_data = pd.Series([5.0] * 100)
                is_valid, error_msg = evaluator._validate_data(constant_data, 'test')

            assert is_valid is True
            assert error_msg is None
            # Logger should have been called with warning about constant data
            assert any('constant' in record.message for record in caplog.records)

    def test_custom_min_valid_points(self, mock_config, tmp_path, mock_logger):
        """Test with custom minimum valid points threshold."""
        with patch.object(ModelEvaluator, '_get_config_value', return_value=''):
            evaluator = ConcreteEvaluator(mock_config, tmp_path, mock_logger)

            data = pd.Series([1.0, 2.0, 3.0])
            # With min_valid_points=3, this should pass
            is_valid, error_msg = evaluator._validate_data(data, 'test', min_valid_points=3)

            assert is_valid is True
            assert error_msg is None
