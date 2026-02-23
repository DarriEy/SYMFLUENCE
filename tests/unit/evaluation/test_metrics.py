"""
Tests for metrics module - performance metrics for hydrological evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from symfluence.evaluation.metrics import (
    _apply_transformation,
    _clean_data,
    bias,
    calculate_all_metrics,
    calculate_metrics,
    correlation,
    get_metric_function,
    interpret_metric,
    kge,
    kge_np,
    kge_prime,
    list_available_metrics,
    log_nse,
    mae,
    mare,
    nrmse,
    nse,
    pbias,
    r_squared,
    rmse,
    volumetric_efficiency,
)


class TestDataCleaning:
    """Test data cleaning and transformation utilities."""

    def test_clean_data_removes_nans(self):
        """Test that _clean_data removes NaN values."""
        obs = np.array([1.0, 2.0, np.nan, 4.0])
        sim = np.array([1.1, 2.1, 3.1, np.nan])

        obs_clean, sim_clean = _clean_data(obs, sim)

        assert len(obs_clean) == 2
        assert len(sim_clean) == 2
        assert not np.any(np.isnan(obs_clean))
        assert not np.any(np.isnan(sim_clean))

    def test_apply_transformation_identity(self):
        """Test that transfo=1.0 returns original arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 2.1, 3.1])

        obs_trans, sim_trans = _apply_transformation(obs, sim, transfo=1.0)

        np.testing.assert_array_equal(obs_trans, obs)
        np.testing.assert_array_equal(sim_trans, sim)

    def test_apply_transformation_cleans_invalid_values(self):
        """Test that transformation cleans NaN/Inf values after transformation."""
        obs = np.array([1.0, 2.0, 3.0, 0.0])  # 0.0 will cause issues with negative exponents
        sim = np.array([1.1, 2.1, 3.1, 0.1])

        obs_trans, sim_trans = _apply_transformation(obs, sim, transfo=-1.0)

        # Should not contain NaN or Inf
        assert not np.any(np.isnan(obs_trans))
        assert not np.any(np.isnan(sim_trans))
        assert not np.any(np.isinf(obs_trans))
        assert not np.any(np.isinf(sim_trans))

    def test_apply_transformation_power(self):
        """Test power transformation."""
        obs = np.array([1.0, 4.0, 9.0])
        sim = np.array([1.0, 4.0, 9.0])

        obs_trans, sim_trans = _apply_transformation(obs, sim, transfo=0.5)

        np.testing.assert_array_almost_equal(obs_trans, [1.0, 2.0, 3.0])


class TestNSE:
    """Test Nash-Sutcliffe Efficiency metric."""

    def test_nse_perfect_fit(self):
        """Test NSE returns 1.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nse(obs, sim)

        assert result == pytest.approx(1.0)

    def test_nse_mean_prediction(self):
        """Test NSE returns 0.0 when simulation is mean of observations."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean of obs

        result = nse(obs, sim)

        assert result == pytest.approx(0.0)

    def test_nse_handles_nan(self):
        """Test NSE handles NaN values."""
        obs = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        result = nse(obs, sim)

        assert not np.isnan(result)

    def test_nse_empty_array(self):
        """Test NSE returns NaN for empty arrays."""
        obs = np.array([])
        sim = np.array([])

        result = nse(obs, sim)

        assert np.isnan(result)

    def test_nse_with_transformation(self):
        """Test NSE with power transformation."""
        obs = np.array([1.0, 4.0, 9.0, 16.0])
        sim = np.array([1.0, 4.0, 9.0, 16.0])

        result = nse(obs, sim, transfo=0.5)

        assert result == pytest.approx(1.0)


class TestKGE:
    """Test Kling-Gupta Efficiency metric."""

    def test_kge_perfect_fit(self):
        """Test KGE returns 1.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kge(obs, sim)

        assert result == pytest.approx(1.0)

    def test_kge_return_components(self):
        """Test KGE returns components when requested."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kge(obs, sim, return_components=True)

        assert isinstance(result, dict)
        assert 'KGE' in result
        assert 'r' in result
        assert 'alpha' in result
        assert 'beta' in result
        assert result['KGE'] == pytest.approx(1.0)
        assert result['r'] == pytest.approx(1.0)
        assert result['alpha'] == pytest.approx(1.0)
        assert result['beta'] == pytest.approx(1.0)

    def test_kge_handles_nan(self):
        """Test KGE handles NaN values."""
        obs = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        result = kge(obs, sim)

        assert not np.isnan(result)

    def test_kge_constant_series(self):
        """Test KGE handles constant series (zero std)."""
        obs = np.array([1.0, 1.0, 1.0, 1.0])
        sim = np.array([1.0, 1.0, 1.0, 1.0])

        result = kge(obs, sim)

        # With zero std, correlation is undefined
        assert np.isnan(result)


class TestKGEPrime:
    """Test modified KGE (KGE') metric."""

    def test_kge_prime_perfect_fit(self):
        """Test KGE' returns 1.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kge_prime(obs, sim)

        assert result == pytest.approx(1.0)


class TestErrorMetrics:
    """Test error metrics (RMSE, MAE, etc.)."""

    def test_rmse_perfect_fit(self):
        """Test RMSE returns 0.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])

        result = rmse(obs, sim)

        assert result == pytest.approx(0.0)

    def test_rmse_known_value(self):
        """Test RMSE calculation with known values."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 3.0, 4.0])

        result = rmse(obs, sim)

        # RMSE = sqrt(mean([1, 1, 1])) = 1.0
        assert result == pytest.approx(1.0)

    def test_mae_perfect_fit(self):
        """Test MAE returns 0.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])

        result = mae(obs, sim)

        assert result == pytest.approx(0.0)

    def test_nrmse_perfect_fit(self):
        """Test NRMSE returns 0.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nrmse(obs, sim)

        assert result == pytest.approx(0.0)


class TestBiasMetrics:
    """Test bias metrics."""

    def test_bias_no_bias(self):
        """Test bias returns 0.0 when no bias."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])

        result = bias(obs, sim)

        assert result == pytest.approx(0.0)

    def test_bias_positive(self):
        """Test positive bias (overestimation)."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 3.0, 4.0])

        result = bias(obs, sim)

        assert result == pytest.approx(1.0)

    def test_pbias_no_bias(self):
        """Test PBIAS returns 0.0 when no bias."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])

        result = pbias(obs, sim)

        assert result == pytest.approx(0.0)


class TestCorrelationMetrics:
    """Test correlation metrics."""

    def test_correlation_perfect(self):
        """Test correlation returns 1.0 for perfect correlation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = correlation(obs, sim)

        assert result == pytest.approx(1.0)

    def test_correlation_negative(self):
        """Test negative correlation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        result = correlation(obs, sim)

        assert result == pytest.approx(-1.0)

    def test_r_squared(self):
        """Test R² calculation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = r_squared(obs, sim)

        assert result == pytest.approx(1.0)


class TestLogNSE:
    """Test log-transformed NSE."""

    def test_log_nse_perfect_fit(self):
        """Test logNSE for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = log_nse(obs, sim)

        assert result == pytest.approx(1.0)

    def test_log_nse_handles_zeros(self):
        """Test logNSE handles zero values via epsilon."""
        obs = np.array([0.0, 1.0, 2.0, 3.0])
        sim = np.array([0.0, 1.0, 2.0, 3.0])

        result = log_nse(obs, sim)

        assert not np.isnan(result)


class TestCalculateAllMetrics:
    """Test calculate_all_metrics function."""

    def test_returns_all_expected_metrics(self):
        """Test that calculate_all_metrics returns all expected keys."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        result = calculate_all_metrics(obs, sim)

        expected_keys = [
            'NSE', 'logNSE', 'KGE', 'KGEp', 'KGEnp', 'VE',
            'RMSE', 'NRMSE', 'MAE', 'MARE',
            'PBIAS', 'bias', 'correlation', 'R2',
            'r', 'alpha', 'beta'
        ]

        for key in expected_keys:
            assert key in result


class TestCalculateMetrics:
    """Test calculate_metrics function."""

    def test_calculate_selected_metrics(self):
        """Test calculating only selected metrics."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        result = calculate_metrics(obs, sim, metrics=['KGE', 'NSE'])

        assert 'KGE' in result
        assert 'NSE' in result
        assert len(result) == 2


class TestMetricRegistry:
    """Test metric registry functions."""

    def test_get_metric_function(self):
        """Test getting a metric function by name."""
        func = get_metric_function('KGE')
        assert func is not None
        assert callable(func)

    def test_get_metric_function_case_insensitive(self):
        """Test case-insensitive lookup."""
        func = get_metric_function('kge')
        assert func is not None

    def test_list_available_metrics(self):
        """Test listing available metrics."""
        metrics_list = list_available_metrics()

        assert 'KGE' in metrics_list
        assert 'NSE' in metrics_list
        assert 'RMSE' in metrics_list


class TestInterpretMetric:
    """Test metric interpretation."""

    def test_interpret_kge_excellent(self):
        """Test interpretation of excellent KGE."""
        result = interpret_metric('KGE', 0.95)

        assert 'Excellent' in result

    def test_interpret_kge_nan(self):
        """Test interpretation of NaN metric."""
        result = interpret_metric('KGE', np.nan)

        assert 'NaN' in result
