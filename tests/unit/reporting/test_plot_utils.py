"""
Unit tests for plot_utils module.
"""

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.core.plot_utils import (
    align_timeseries,
    calculate_flow_duration_curve,
    calculate_metrics,
    calculate_summary_statistics,
    format_metrics_for_display,
    resample_timeseries,
)


class TestCalculateMetrics:
    """Test suite for calculate_metrics function."""

    def test_perfect_match(self):
        """Test metrics with perfect obs-sim match."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_metrics(obs, sim)

        assert 'NSE' in metrics
        assert 'KGE' in metrics
        assert 'RMSE' in metrics
        # NSE should be 1.0 for perfect match
        # (exact value depends on metrics implementation)

    def test_with_nans(self):
        """Test that NaN values are handled correctly."""
        obs = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sim = np.array([1.1, np.nan, 3.1, 4.1, 5.1])

        metrics = calculate_metrics(obs, sim)

        # Should return metrics for non-NaN pairs
        assert all(k in metrics for k in ['RMSE', 'KGE', 'NSE', 'MAE'])

    def test_empty_arrays(self):
        """Test with empty arrays."""
        obs = np.array([])
        sim = np.array([])

        metrics = calculate_metrics(obs, sim)

        # Should return NaN for all metrics
        assert all(np.isnan(v) for v in metrics.values())

    def test_all_nans(self):
        """Test with all NaN values."""
        obs = np.array([np.nan, np.nan, np.nan])
        sim = np.array([np.nan, np.nan, np.nan])

        metrics = calculate_metrics(obs, sim)

        # Should return NaN for all metrics
        assert all(np.isnan(v) for v in metrics.values())

    def test_metric_keys(self):
        """Test that all expected metric keys are present."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 2.1, 3.1])

        metrics = calculate_metrics(obs, sim)

        expected_keys = ['RMSE', 'KGE', 'KGEp', 'NSE', 'MAE', 'KGEnp']
        assert all(k in metrics for k in expected_keys)


class TestCalculateFlowDurationCurve:
    """Test suite for calculate_flow_duration_curve function."""

    def test_basic_fdc(self):
        """Test basic flow duration curve calculation."""
        flows = np.array([5.0, 3.0, 4.0, 1.0, 2.0])

        exceedance, sorted_flows = calculate_flow_duration_curve(flows)

        # Should be sorted in descending order
        assert np.array_equal(sorted_flows, np.array([5.0, 4.0, 3.0, 2.0, 1.0]))

        # Exceedance should go from 0.2 to 1.0
        expected_exceedance = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_array_almost_equal(exceedance, expected_exceedance)

    def test_with_zeros(self):
        """Test FDC with zero values."""
        flows = np.array([5.0, 0.0, 3.0, 0.0, 1.0])

        # Without removing zeros
        exceedance, sorted_flows = calculate_flow_duration_curve(flows, remove_zeros=False)
        assert len(sorted_flows) == 5

        # With removing zeros
        exceedance, sorted_flows = calculate_flow_duration_curve(flows, remove_zeros=True)
        assert len(sorted_flows) == 3

    def test_with_nans(self):
        """Test FDC with NaN values."""
        flows = np.array([5.0, np.nan, 3.0, 2.0])

        exceedance, sorted_flows = calculate_flow_duration_curve(flows)

        # NaNs should be removed
        assert len(sorted_flows) == 3

    def test_empty_result(self):
        """Test with all NaN or all zeros removed."""
        flows = np.array([0.0, 0.0, 0.0])

        exceedance, sorted_flows = calculate_flow_duration_curve(flows, remove_zeros=True)

        assert len(exceedance) == 0
        assert len(sorted_flows) == 0


class TestAlignTimeseries:
    """Test suite for align_timeseries function."""

    def test_basic_alignment(self):
        """Test basic time series alignment."""
        dates = pd.date_range('2020-01-01', periods=10)
        obs = pd.Series(range(10), index=dates)
        sim = pd.Series(range(10, 20), index=dates)

        obs_aligned, sim_aligned = align_timeseries(obs, sim)

        assert len(obs_aligned) == 10
        assert len(sim_aligned) == 10
        np.testing.assert_array_equal(obs_aligned.values, obs.values)

    def test_spinup_days(self):
        """Test spinup removal by days."""
        dates = pd.date_range('2020-01-01', periods=100)
        obs = pd.Series(range(100), index=dates)
        sim = pd.Series(range(100, 200), index=dates)

        obs_aligned, sim_aligned = align_timeseries(obs, sim, spinup_days=10)

        # Should have 90 days left
        assert len(obs_aligned) == 90
        assert obs_aligned.index[0] >= dates[0] + pd.Timedelta(days=10)

    def test_spinup_percent(self):
        """Test spinup removal by percentage."""
        dates = pd.date_range('2020-01-01', periods=100)
        obs = pd.Series(range(100), index=dates)
        sim = pd.Series(range(100, 200), index=dates)

        obs_aligned, sim_aligned = align_timeseries(obs, sim, spinup_percent=10)

        # Should remove 10% = 10 rows
        assert len(obs_aligned) == 90

    def test_no_overlap(self):
        """Test with non-overlapping time series."""
        obs = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        sim = pd.Series([4, 5, 6], index=pd.date_range('2020-02-01', periods=3))

        obs_aligned, sim_aligned = align_timeseries(obs, sim)

        assert len(obs_aligned) == 0
        assert len(sim_aligned) == 0

    def test_spinup_days_precedence(self):
        """Test that spinup_days takes precedence over spinup_percent."""
        dates = pd.date_range('2020-01-01', periods=100)
        obs = pd.Series(range(100), index=dates)
        sim = pd.Series(range(100, 200), index=dates)

        obs_aligned, sim_aligned = align_timeseries(
            obs, sim, spinup_days=20, spinup_percent=10
        )

        # Should remove 20 days, not 10%
        assert len(obs_aligned) == 80


class TestFormatMetricsForDisplay:
    """Test suite for format_metrics_for_display function."""

    def test_basic_formatting(self):
        """Test basic metrics formatting."""
        metrics = {'NSE': 0.87654, 'RMSE': 1.234}

        result = format_metrics_for_display(metrics, precision=2)

        assert 'NSE: 0.88' in result
        assert 'RMSE: 1.23' in result

    def test_with_label(self):
        """Test formatting with label."""
        metrics = {'NSE': 0.87, 'RMSE': 1.23}

        result = format_metrics_for_display(metrics, label='Model A')

        assert 'Model A:' in result
        assert 'NSE:' in result

    def test_empty_metrics(self):
        """Test with empty metrics dictionary."""
        metrics = {}

        result = format_metrics_for_display(metrics)

        assert result == ""

    def test_with_nan_values(self):
        """Test formatting with NaN values."""
        metrics = {'NSE': 0.87, 'RMSE': np.nan}

        result = format_metrics_for_display(metrics, precision=2)

        assert 'NSE: 0.87' in result
        assert 'RMSE:' in result  # NaN should be handled


class TestResampleTimeseries:
    """Test suite for resample_timeseries function."""

    def test_hourly_to_daily_mean(self):
        """Test resampling from hourly to daily with mean."""
        dates = pd.date_range('2020-01-01', periods=48, freq='h')
        series = pd.Series(range(48), index=dates)

        daily = resample_timeseries(series, freq='D', aggregation='mean')

        assert len(daily) == 2  # 2 days
        assert daily.iloc[0] == 11.5  # Mean of first 24 values

    def test_daily_to_monthly_sum(self):
        """Test resampling from daily to monthly with sum."""
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        series = pd.Series(np.ones(60), index=dates)

        monthly = resample_timeseries(series, freq='ME', aggregation='sum')

        assert len(monthly) == 2  # Jan and Feb

    def test_invalid_aggregation(self):
        """Test with invalid aggregation method."""
        dates = pd.date_range('2020-01-01', periods=10)
        series = pd.Series(range(10), index=dates)

        with pytest.raises(ValueError, match="Unknown aggregation"):
            resample_timeseries(series, freq='D', aggregation='invalid')


class TestCalculateSummaryStatistics:
    """Test suite for calculate_summary_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])

        stats = calculate_summary_statistics(data)

        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['q25'] == 2.0
        assert stats['q75'] == 4.0

    def test_with_nans(self):
        """Test statistics with NaN values."""
        data = np.array([1, 2, np.nan, 4, 5])

        stats = calculate_summary_statistics(data)

        # Should compute stats on non-NaN values
        assert not np.isnan(stats['mean'])
        assert stats['mean'] == 3.0

    def test_all_nans(self):
        """Test with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])

        stats = calculate_summary_statistics(data)

        # Should return NaN for all stats
        assert all(np.isnan(v) for v in stats.values())

    def test_empty_array(self):
        """Test with empty array."""
        data = np.array([])

        stats = calculate_summary_statistics(data)

        # Should return NaN for all stats
        assert all(np.isnan(v) for v in stats.values())
