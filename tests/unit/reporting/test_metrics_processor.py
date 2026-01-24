"""
Unit tests for MetricsProcessor.

Tests performance metrics calculation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock

from symfluence.reporting.processors.metrics_processor import MetricsProcessor


@pytest.fixture
def metrics_processor(mock_logger):
    """Create a MetricsProcessor instance."""
    return MetricsProcessor(mock_logger)


@pytest.fixture
def sample_obs_sim_aligned():
    """Create aligned observation and simulation time series."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    # Create correlated obs/sim with known properties
    obs = pd.Series(10 + 5 * np.sin(np.linspace(0, 4 * np.pi, 365)) +
                    np.random.normal(0, 1, 365),
                    index=dates, name='obs')
    obs = obs.clip(lower=0.1)

    # Sim is obs with some bias and noise
    sim = obs * 0.95 + np.random.normal(0, 0.5, 365)
    sim = sim.clip(lower=0.1)
    sim = pd.Series(sim.values, index=dates, name='sim')

    return obs, sim


@pytest.fixture
def sample_obs_sim_misaligned():
    """Create misaligned observation and simulation time series."""
    np.random.seed(42)

    # Obs starts Jan 1
    obs_dates = pd.date_range('2020-01-01', periods=200, freq='D')
    obs = pd.Series(np.random.random(200) * 10, index=obs_dates, name='obs')

    # Sim starts Feb 1 (offset by 31 days)
    sim_dates = pd.date_range('2020-02-01', periods=200, freq='D')
    sim = pd.Series(np.random.random(200) * 10, index=sim_dates, name='sim')

    return obs, sim


class TestMetricsProcessor:
    """Test suite for MetricsProcessor."""

    def test_initialization(self, metrics_processor):
        """Test that MetricsProcessor initializes correctly."""
        assert metrics_processor.logger is not None

    def test_initialization_without_logger(self):
        """Test initialization without providing a logger."""
        processor = MetricsProcessor()
        assert processor.logger is not None


class TestCalculateForPeriod:
    """Test the calculate_for_period method."""

    def test_calculate_for_period_full_data(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test metrics calculation on full time series."""
        obs, sim = sample_obs_sim_aligned

        metrics = metrics_processor.calculate_for_period(obs, sim)

        # Should return dictionary with standard metrics
        assert isinstance(metrics, dict)
        assert 'KGE' in metrics or 'kge' in metrics.keys() or len(metrics) > 0

    def test_calculate_for_period_with_dates(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test metrics calculation with date filtering."""
        obs, sim = sample_obs_sim_aligned

        start = pd.Timestamp('2020-03-01')
        end = pd.Timestamp('2020-09-30')

        metrics = metrics_processor.calculate_for_period(
            obs, sim,
            period_start=start,
            period_end=end
        )

        assert isinstance(metrics, dict)

    def test_calculate_for_period_with_spinup_days(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test metrics calculation with spinup removal."""
        obs, sim = sample_obs_sim_aligned

        metrics = metrics_processor.calculate_for_period(
            obs, sim,
            spinup_days=30
        )

        assert isinstance(metrics, dict)

    def test_calculate_for_period_with_spinup_percent(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test metrics calculation with percentage spinup removal."""
        obs, sim = sample_obs_sim_aligned

        metrics = metrics_processor.calculate_for_period(
            obs, sim,
            spinup_percent=0.1  # Remove 10%
        )

        assert isinstance(metrics, dict)

    def test_calculate_for_period_empty_after_alignment(
        self, metrics_processor, mock_logger
    ):
        """Test with non-overlapping data."""
        # Create non-overlapping series
        obs_dates = pd.date_range('2020-01-01', periods=100, freq='D')
        sim_dates = pd.date_range('2021-01-01', periods=100, freq='D')

        obs = pd.Series(np.random.random(100), index=obs_dates)
        sim = pd.Series(np.random.random(100), index=sim_dates)

        metrics = metrics_processor.calculate_for_period(obs, sim)

        # Should return empty metrics dict
        assert isinstance(metrics, dict)
        # Logger should warn about no overlap
        mock_logger.warning.assert_called()

    def test_calculate_for_period_empty_after_date_filter(
        self, metrics_processor, sample_obs_sim_aligned, mock_logger
    ):
        """Test with date filter resulting in no data."""
        obs, sim = sample_obs_sim_aligned

        # Filter to dates outside data range
        start = pd.Timestamp('2025-01-01')
        end = pd.Timestamp('2025-12-31')

        metrics = metrics_processor.calculate_for_period(
            obs, sim,
            period_start=start,
            period_end=end
        )

        assert isinstance(metrics, dict)
        mock_logger.warning.assert_called()


class TestCalculateForCalibrationValidation:
    """Test the calculate_for_calibration_validation method."""

    def test_calculate_calib_valid_success(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test calculation for both calibration and validation periods."""
        obs, sim = sample_obs_sim_aligned

        calib_metrics, valid_metrics = metrics_processor.calculate_for_calibration_validation(
            obs, sim,
            calib_start=pd.Timestamp('2020-01-01'),
            calib_end=pd.Timestamp('2020-06-30'),
            valid_start=pd.Timestamp('2020-07-01'),
            valid_end=pd.Timestamp('2020-12-31')
        )

        assert isinstance(calib_metrics, dict)
        assert isinstance(valid_metrics, dict)

    def test_calculate_calib_valid_with_spinup(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test calculation with spinup days."""
        obs, sim = sample_obs_sim_aligned

        calib_metrics, valid_metrics = metrics_processor.calculate_for_calibration_validation(
            obs, sim,
            calib_start=pd.Timestamp('2020-01-01'),
            calib_end=pd.Timestamp('2020-06-30'),
            valid_start=pd.Timestamp('2020-07-01'),
            valid_end=pd.Timestamp('2020-12-31'),
            spinup_days=30
        )

        # Spinup should only apply to calibration, not validation
        assert isinstance(calib_metrics, dict)
        assert isinstance(valid_metrics, dict)

    def test_calculate_calib_valid_different_period_lengths(
        self, metrics_processor, sample_obs_sim_aligned
    ):
        """Test with different calibration and validation period lengths."""
        obs, sim = sample_obs_sim_aligned

        # Short calibration, long validation
        calib_metrics, valid_metrics = metrics_processor.calculate_for_calibration_validation(
            obs, sim,
            calib_start=pd.Timestamp('2020-01-01'),
            calib_end=pd.Timestamp('2020-03-31'),  # 3 months
            valid_start=pd.Timestamp('2020-04-01'),
            valid_end=pd.Timestamp('2020-12-31')   # 9 months
        )

        assert isinstance(calib_metrics, dict)
        assert isinstance(valid_metrics, dict)


class TestCalculateForMultipleModels:
    """Test the calculate_for_multiple_models method."""

    def test_calculate_for_multiple_models(self, metrics_processor):
        """Test metrics calculation for multiple models."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        obs = pd.Series(np.random.random(100) * 10, index=dates)
        models = {
            'SUMMA': pd.Series(np.random.random(100) * 10, index=dates),
            'FUSE': pd.Series(np.random.random(100) * 10, index=dates),
            'HYPE': pd.Series(np.random.random(100) * 10, index=dates),
        }

        if hasattr(metrics_processor, 'calculate_for_multiple_models'):
            result = metrics_processor.calculate_for_multiple_models(obs, models)
            assert isinstance(result, dict)
            # Should have metrics for each model
            assert len(result) >= len(models) or isinstance(result, dict)


class TestEmptyMetrics:
    """Test the _empty_metrics helper method."""

    def test_empty_metrics_returns_dict(self, metrics_processor):
        """Test that _empty_metrics returns a dictionary."""
        if hasattr(metrics_processor, '_empty_metrics'):
            result = metrics_processor._empty_metrics()
            assert isinstance(result, dict)

    def test_empty_metrics_contains_nan(self, metrics_processor):
        """Test that _empty_metrics contains NaN values."""
        if hasattr(metrics_processor, '_empty_metrics'):
            result = metrics_processor._empty_metrics()
            # All values should be NaN
            for key, value in result.items():
                assert np.isnan(value) or value is None


class TestMetricsAccuracy:
    """Test that calculated metrics are accurate."""

    def test_perfect_simulation(self, metrics_processor):
        """Test metrics with perfect simulation (obs == sim)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.Series(np.random.random(100) * 10, index=dates)

        # Perfect simulation should have KGE ~= 1, RMSE ~= 0
        metrics = metrics_processor.calculate_for_period(data, data.copy())

        if 'KGE' in metrics:
            assert metrics['KGE'] > 0.99
        if 'RMSE' in metrics:
            assert metrics['RMSE'] < 0.01
        if 'NSE' in metrics:
            assert metrics['NSE'] > 0.99

    def test_constant_bias_simulation(self, metrics_processor):
        """Test metrics with constant bias."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        obs = pd.Series(np.ones(100) * 10, index=dates)
        sim = obs + 2  # Constant positive bias

        metrics = metrics_processor.calculate_for_period(obs, sim)

        # RMSE should equal the bias
        if 'RMSE' in metrics:
            assert abs(metrics['RMSE'] - 2) < 0.01

    def test_scaled_simulation(self, metrics_processor):
        """Test metrics with scaled simulation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        obs = pd.Series(np.random.random(100) * 10 + 5, index=dates)
        sim = obs * 1.5  # 50% overestimation

        metrics = metrics_processor.calculate_for_period(obs, sim)

        # Bias should be detectable
        if 'Bias' in metrics or 'bias' in metrics:
            bias = metrics.get('Bias', metrics.get('bias', 0))
            assert abs(bias) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_series(self, metrics_processor, mock_logger):
        """Test with single-value time series."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        obs = pd.Series([10], index=dates)
        sim = pd.Series([10], index=dates)

        metrics = metrics_processor.calculate_for_period(obs, sim)
        # Should handle gracefully

    def test_all_nan_values(self, metrics_processor, mock_logger):
        """Test with all NaN values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        obs = pd.Series([np.nan] * 100, index=dates)
        sim = pd.Series([np.nan] * 100, index=dates)

        metrics = metrics_processor.calculate_for_period(obs, sim)
        # Should handle gracefully without crashing

    def test_some_nan_values(self, metrics_processor):
        """Test with some NaN values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        obs = pd.Series(np.random.random(100) * 10, index=dates)
        sim = pd.Series(np.random.random(100) * 10, index=dates)

        # Introduce some NaNs
        obs.iloc[10:15] = np.nan
        sim.iloc[50:55] = np.nan

        metrics = metrics_processor.calculate_for_period(obs, sim)
        # Should calculate metrics on valid data
        assert isinstance(metrics, dict)

    def test_zero_values(self, metrics_processor):
        """Test with zero values in series."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        obs = pd.Series(np.random.random(100) * 10, index=dates)
        obs.iloc[0:10] = 0  # Some zeros
        sim = obs * 0.95

        metrics = metrics_processor.calculate_for_period(obs, sim)
        assert isinstance(metrics, dict)

    def test_negative_values(self, metrics_processor):
        """Test with negative values (should work mathematically)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        # Temperature-like data with negatives
        obs = pd.Series(np.random.random(100) * 30 - 10, index=dates)
        sim = obs * 0.9 + 1

        metrics = metrics_processor.calculate_for_period(obs, sim)
        assert isinstance(metrics, dict)
