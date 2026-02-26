"""
Property-based tests for numerical modules using Hypothesis.

These tests verify mathematical invariants that must hold for *any* valid input,
not just hand-picked examples.  They complement the existing example-based tests
by exploring the space of possible inputs systematically.

Modules covered:
    - symfluence.evaluation.metrics_core   (performance metrics)
    - symfluence.evaluation.metrics_hydrograph  (hydrograph signatures)
    - symfluence.evaluation.metric_transformer  (optimisation direction)
    - symfluence.geospatial.coordinate_utils    (lon/lat normalisation)
"""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from symfluence.evaluation.metric_transformer import MetricTransformer
from symfluence.evaluation.metrics_core import (
    _apply_transformation,
    _clean_data,
    bias,
    correlation,
    kge,
    kge_np,
    kge_prime,
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
from symfluence.evaluation.metrics_hydrograph import (
    baseflow_index,
    flow_duration_curve_metrics,
    peak_timing_error,
    recession_constant,
)
from symfluence.geospatial.coordinate_utils import (
    BoundingBox,
    normalize_longitude,
    validate_bbox,
)

# ---------------------------------------------------------------------------
# Shared Hypothesis strategies
# ---------------------------------------------------------------------------

# Streamflow-like values: strictly positive, finite, realistic magnitude
_streamflow_element = st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False)

# General hydrological values (may include zero, but no negatives for most metrics)
_positive_element = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)

# Signed values (e.g. bias, temperature)
_signed_element = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


def _paired_streamflow_arrays(min_size: int = 10, max_size: int = 500):
    """Strategy: two aligned arrays of positive streamflow-like values."""
    n = st.shared(st.integers(min_value=min_size, max_value=max_size), key="n")
    return st.tuples(
        n.flatmap(lambda sz: arrays(np.float64, sz, elements=_streamflow_element)),
        n.flatmap(lambda sz: arrays(np.float64, sz, elements=_streamflow_element)),
    )


def _paired_positive_arrays(min_size: int = 10, max_size: int = 500):
    """Strategy: two aligned arrays of non-negative values (may include 0)."""
    n = st.shared(st.integers(min_value=min_size, max_value=max_size), key="n_pos")
    return st.tuples(
        n.flatmap(lambda sz: arrays(np.float64, sz, elements=_positive_element)),
        n.flatmap(lambda sz: arrays(np.float64, sz, elements=_positive_element)),
    )


def _single_streamflow_array(min_size: int = 30, max_size: int = 500):
    """Strategy: single array of positive streamflow-like values."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda sz: arrays(np.float64, sz, elements=_streamflow_element)
    )


# ===================================================================
# SECTION 1: Core Metric Mathematical Invariants
# ===================================================================

pytestmark = [pytest.mark.unit, pytest.mark.property]


class TestNSEProperties:
    """Property-based tests for Nash-Sutcliffe Efficiency."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_nse_upper_bound(self, data):
        """NSE must never exceed 1.0."""
        obs, sim = data
        result = nse(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_nse_perfect_prediction_is_one(self, obs):
        """NSE(x, x) must always equal 1.0 when variance > 0."""
        assume(np.std(obs) > 1e-10)
        result = nse(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_nse_mean_prediction_is_zero(self, data):
        """NSE of mean(obs) as simulation must be 0.0."""
        obs, _ = data
        assume(np.std(obs) > 1e-10)
        sim_mean = np.full_like(obs, np.mean(obs))
        result = nse(obs, sim_mean)
        assert result == pytest.approx(0.0, abs=1e-8)

    @given(obs=arrays(np.float64, st.integers(10, 100), elements=_streamflow_element))
    @settings(max_examples=100, deadline=None)
    def test_nse_constant_observation_is_nan(self, obs):
        """NSE is undefined (NaN) when observations have near-zero variance."""
        constant = np.full_like(obs, obs[0])
        sim = constant + 1.0
        result = nse(constant, sim)
        assert np.isnan(result)


class TestKGEProperties:
    """Property-based tests for Kling-Gupta Efficiency."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_kge_upper_bound(self, data):
        """KGE must never exceed 1.0."""
        obs, sim = data
        result = kge(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_kge_perfect_prediction_is_one(self, obs):
        """KGE(x, x) = 1.0 when there is variance."""
        assume(np.std(obs) > 1e-10)
        result = kge(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_kge_components_at_perfect_prediction(self, obs):
        """All KGE components should be 1.0 for perfect prediction."""
        assume(np.std(obs) > 1e-10)
        result = kge(obs, obs, return_components=True)
        assert result['r'] == pytest.approx(1.0, abs=1e-10)
        assert result['alpha'] == pytest.approx(1.0, abs=1e-10)
        assert result['beta'] == pytest.approx(1.0, abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_kge_component_ranges(self, data):
        """KGE correlation in [-1,1], alpha >= 0, beta >= 0 for positive data."""
        obs, sim = data
        result = kge(obs, sim, return_components=True)
        if not np.isnan(result['r']):
            assert -1.0 - 1e-10 <= result['r'] <= 1.0 + 1e-10
        if not np.isnan(result['alpha']):
            assert result['alpha'] >= -1e-10
        if not np.isnan(result['beta']):
            assert result['beta'] >= -1e-10

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_kge_decomposition_consistency(self, data):
        """KGE value must equal 1 - sqrt((r-1)^2 + (a-1)^2 + (b-1)^2)."""
        obs, sim = data
        result = kge(obs, sim, return_components=True)
        if np.isnan(result['KGE']):
            return
        r, a, b = result['r'], result['alpha'], result['beta']
        expected = 1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
        assert result['KGE'] == pytest.approx(expected, abs=1e-10)


class TestKGEPrimeProperties:
    """Property-based tests for modified KGE (KGE')."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_kge_prime_upper_bound(self, data):
        """KGE' must never exceed 1.0."""
        obs, sim = data
        result = kge_prime(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_kge_prime_perfect_prediction_is_one(self, obs):
        """KGE'(x, x) = 1.0 when there is variance."""
        assume(np.std(obs) > 1e-10)
        assume(np.mean(obs) > 1e-10)
        result = kge_prime(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)


class TestKGENonParametricProperties:
    """Property-based tests for non-parametric KGE."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=100, deadline=None)
    def test_kge_np_upper_bound(self, data):
        """KGEnp must never exceed 1.0."""
        obs, sim = data
        result = kge_np(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=100, deadline=None)
    def test_kge_np_perfect_prediction_is_one(self, obs):
        """KGEnp(x, x) = 1.0 for non-constant positive data."""
        assume(np.std(obs) > 1e-10)
        assume(np.mean(obs) > 1e-10)
        result = kge_np(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)


class TestRMSEProperties:
    """Property-based tests for Root Mean Square Error."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_rmse_non_negative(self, data):
        """RMSE must always be >= 0."""
        obs, sim = data
        result = rmse(obs, sim)
        if not np.isnan(result):
            assert result >= -1e-15

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_rmse_perfect_prediction_is_zero(self, obs):
        """RMSE(x, x) = 0.0."""
        result = rmse(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-12)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_rmse_symmetry(self, data):
        """RMSE(obs, sim) == RMSE(sim, obs) for untransformed data."""
        obs, sim = data
        assert rmse(obs, sim) == pytest.approx(rmse(sim, obs), abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_rmse_geq_mae(self, data):
        """RMSE >= MAE (Cauchy-Schwarz inequality)."""
        obs, sim = data
        r = rmse(obs, sim)
        m = mae(obs, sim)
        if not (np.isnan(r) or np.isnan(m)):
            assert r >= m - 1e-10

    @given(
        obs=arrays(np.float64, st.integers(5, 100), elements=_streamflow_element),
        k=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_rmse_scale_equivariance(self, obs, k):
        """RMSE(k*obs, k*sim) = k * RMSE(obs, sim) for positive k."""
        sim = obs * 1.1  # fixed relative error
        result_original = rmse(obs, sim)
        result_scaled = rmse(k * obs, k * sim)
        if not (np.isnan(result_original) or np.isnan(result_scaled)):
            assert result_scaled == pytest.approx(k * result_original, rel=1e-8)


class TestMAEProperties:
    """Property-based tests for Mean Absolute Error."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_mae_non_negative(self, data):
        """MAE must always be >= 0."""
        obs, sim = data
        result = mae(obs, sim)
        if not np.isnan(result):
            assert result >= -1e-15

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_mae_perfect_prediction_is_zero(self, obs):
        """MAE(x, x) = 0.0."""
        result = mae(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-12)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_mae_symmetry(self, data):
        """MAE(obs, sim) == MAE(sim, obs) for untransformed data."""
        obs, sim = data
        assert mae(obs, sim) == pytest.approx(mae(sim, obs), abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_mae_triangle_inequality(self, data):
        """MAE(obs, sim3) <= MAE(obs, sim1) + MAE(sim1, sim3)."""
        obs, sim1 = data
        # Create a third series as midpoint
        sim3 = (obs + sim1) / 2.0
        lhs = mae(obs, sim3)
        rhs = mae(obs, sim1) + mae(sim1, sim3)
        if not (np.isnan(lhs) or np.isnan(rhs)):
            assert lhs <= rhs + 1e-8


class TestNRMSEProperties:
    """Property-based tests for Normalized RMSE."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_nrmse_non_negative(self, data):
        """NRMSE must always be >= 0."""
        obs, sim = data
        result = nrmse(obs, sim)
        if not np.isnan(result):
            assert result >= -1e-15

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=100, deadline=None)
    def test_nrmse_perfect_prediction_is_zero(self, obs):
        """NRMSE(x, x) = 0.0 when std(obs) > 0."""
        assume(np.std(obs) > 1e-10)
        result = nrmse(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-12)


class TestBiasProperties:
    """Property-based tests for bias and percent bias."""

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_bias_perfect_prediction_is_zero(self, obs):
        """bias(x, x) = 0.0."""
        result = bias(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-12)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_bias_antisymmetry(self, data):
        """bias(obs, sim) = -bias(sim, obs)."""
        obs, sim = data
        b1 = bias(obs, sim)
        b2 = bias(sim, obs)
        if not (np.isnan(b1) or np.isnan(b2)):
            assert b1 == pytest.approx(-b2, abs=1e-10)

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_pbias_perfect_prediction_is_zero(self, obs):
        """pbias(x, x) = 0.0."""
        assume(np.sum(obs) > 1e-10)
        result = pbias(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_pbias_sign_convention(self, data):
        """Positive pbias means overestimation (sum(sim) > sum(obs))."""
        obs, sim = data
        assume(np.sum(obs) > 1e-10)
        result = pbias(obs, sim)
        if not np.isnan(result):
            if np.sum(sim) > np.sum(obs):
                assert result > -1e-10
            elif np.sum(sim) < np.sum(obs):
                assert result < 1e-10


class TestCorrelationProperties:
    """Property-based tests for correlation and R²."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_correlation_bounded(self, data):
        """Pearson r must be in [-1, 1]."""
        obs, sim = data
        result = correlation(obs, sim)
        if not np.isnan(result):
            assert -1.0 - 1e-10 <= result <= 1.0 + 1e-10

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_correlation_symmetry(self, data):
        """correlation(obs, sim) == correlation(sim, obs)."""
        obs, sim = data
        r1 = correlation(obs, sim)
        r2 = correlation(sim, obs)
        if not (np.isnan(r1) or np.isnan(r2)):
            assert r1 == pytest.approx(r2, abs=1e-10)

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_correlation_perfect_is_one(self, obs):
        """correlation(x, x) = 1.0 when variance > 0."""
        assume(np.std(obs) > 1e-10)
        result = correlation(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_r_squared_bounded(self, data):
        """R² must be in [0, 1]."""
        obs, sim = data
        result = r_squared(obs, sim)
        if not np.isnan(result):
            assert -1e-10 <= result <= 1.0 + 1e-10

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_r_squared_equals_correlation_squared(self, data):
        """R² must equal r²."""
        obs, sim = data
        r = correlation(obs, sim)
        r2 = r_squared(obs, sim)
        if not (np.isnan(r) or np.isnan(r2)):
            assert r2 == pytest.approx(r ** 2, abs=1e-10)


class TestLogNSEProperties:
    """Property-based tests for log-transformed NSE."""

    @given(obs=arrays(np.float64, st.integers(10, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_log_nse_perfect_prediction_is_one(self, obs):
        """log_nse(x, x) = 1.0 for positive data with variance."""
        assume(np.std(obs) > 1e-10)
        assume(np.all(obs > 0))
        result = log_nse(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-8)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_log_nse_upper_bound(self, data):
        """log_nse must never exceed 1.0."""
        obs, sim = data
        result = log_nse(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10


class TestMAREProperties:
    """Property-based tests for Mean Absolute Relative Error."""

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_mare_non_negative(self, data):
        """MARE must always be >= 0."""
        obs, sim = data
        result = mare(obs, sim)
        if not np.isnan(result):
            assert result >= -1e-15

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_mare_perfect_prediction_is_zero(self, obs):
        """MARE(x, x) = 0.0."""
        result = mare(obs, obs)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestVolumetricEfficiencyProperties:
    """Property-based tests for Volumetric Efficiency."""

    @given(obs=arrays(np.float64, st.integers(5, 200), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_ve_perfect_prediction_is_one(self, obs):
        """VE(x, x) = 1.0 for positive data."""
        assume(np.sum(obs) > 1e-10)
        result = volumetric_efficiency(obs, obs)
        assert result == pytest.approx(1.0, abs=1e-10)

    @given(data=_paired_streamflow_arrays())
    @settings(max_examples=200, deadline=None)
    def test_ve_upper_bound(self, data):
        """VE must never exceed 1.0."""
        obs, sim = data
        result = volumetric_efficiency(obs, sim)
        if not np.isnan(result):
            assert result <= 1.0 + 1e-10


class TestDataCleaningProperties:
    """Property-based tests for data cleaning utilities."""

    @given(data=_paired_streamflow_arrays(min_size=5))
    @settings(max_examples=200, deadline=None)
    def test_clean_data_preserves_length_alignment(self, data):
        """After cleaning, obs and sim must have same length."""
        obs, sim = data
        obs_clean, sim_clean = _clean_data(obs, sim)
        assert len(obs_clean) == len(sim_clean)

    @given(data=_paired_streamflow_arrays(min_size=5))
    @settings(max_examples=200, deadline=None)
    def test_clean_data_no_nans(self, data):
        """After cleaning, neither array should contain NaN."""
        obs, sim = data
        obs_clean, sim_clean = _clean_data(obs, sim)
        assert not np.any(np.isnan(obs_clean))
        assert not np.any(np.isnan(sim_clean))

    @given(obs=arrays(np.float64, st.integers(5, 100), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_clean_data_nan_free_input_unchanged(self, obs):
        """NaN-free inputs should pass through with same length."""
        sim = obs * 1.1
        obs_clean, sim_clean = _clean_data(obs, sim)
        assert len(obs_clean) == len(obs)


class TestTransformationProperties:
    """Property-based tests for power transformation."""

    @given(obs=arrays(np.float64, st.integers(5, 100), elements=_streamflow_element))
    @settings(max_examples=200, deadline=None)
    def test_identity_transformation(self, obs):
        """transfo=1.0 must return arrays unchanged."""
        sim = obs * 1.1
        obs_t, sim_t = _apply_transformation(obs, sim, transfo=1.0)
        np.testing.assert_array_equal(obs_t, obs)

    @given(
        obs=arrays(np.float64, st.integers(5, 100), elements=_streamflow_element),
        transfo=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_transformation_no_nans_or_infs(self, obs, transfo):
        """Transformation of positive data with positive exponent must not produce NaN/Inf."""
        sim = obs * 1.1
        obs_t, sim_t = _apply_transformation(obs, sim, transfo=transfo)
        assert not np.any(np.isnan(obs_t))
        assert not np.any(np.isinf(obs_t))
        assert not np.any(np.isnan(sim_t))
        assert not np.any(np.isinf(sim_t))


# ===================================================================
# SECTION 2: Hydrograph Signature Properties
# ===================================================================


class TestBaseflowIndexProperties:
    """Property-based tests for Baseflow Index."""

    @given(flow=_single_streamflow_array(min_size=30))
    @settings(max_examples=200, deadline=None)
    def test_bfi_bounded_zero_one(self, flow):
        """BFI must be in [0, 1] for positive flow data."""
        result = baseflow_index(flow)
        if not np.isnan(result):
            assert -1e-10 <= result <= 1.0 + 1e-10

    @given(
        flow=_single_streamflow_array(min_size=30),
        alpha=st.floats(min_value=0.8, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_bfi_bounded_any_alpha(self, flow, alpha):
        """BFI bounded in [0, 1] regardless of filter parameter."""
        result = baseflow_index(flow, filter_param=alpha)
        if not np.isnan(result):
            assert -1e-10 <= result <= 1.0 + 1e-10


class TestRecessionConstantProperties:
    """Property-based tests for recession constant."""

    @given(flow=_single_streamflow_array(min_size=50))
    @settings(max_examples=200, deadline=None)
    def test_recession_constant_bounded(self, flow):
        """Recession constant K must be in (0, 1] when defined."""
        result = recession_constant(flow)
        if not np.isnan(result):
            assert 0.0 < result <= 1.0 + 1e-10


class TestPeakTimingErrorProperties:
    """Property-based tests for peak timing error."""

    @given(data=_paired_streamflow_arrays(min_size=30))
    @settings(max_examples=100, deadline=None)
    def test_n_peaks_non_negative(self, data):
        """Number of peaks must be >= 0."""
        obs, sim = data
        result = peak_timing_error(obs, sim)
        assert result['n_peaks'] >= 0

    @given(obs=_single_streamflow_array(min_size=30))
    @settings(max_examples=100, deadline=None)
    def test_perfect_timing_bounded_error(self, obs):
        """Peak timing error of identical series is bounded by window half-width.

        The algorithm detects *local* peaks in observed data but searches for
        the *global* max in a ±5-step simulated window.  When a local peak is
        not the window's global max, a non-zero timing offset results even for
        identical series.  The error is bounded by the half-window size (5).
        """
        result = peak_timing_error(obs, obs)
        if result['n_peaks'] > 0:
            assert abs(result['mean_timing_error']) <= 5
            assert result['abs_timing_error'] <= 5


class TestFlowDurationCurveProperties:
    """Property-based tests for FDC metrics."""

    @given(data=_paired_streamflow_arrays(min_size=50))
    @settings(max_examples=100, deadline=None)
    def test_fdc_perfect_prediction_zero_bias(self, data):
        """FDC biases should be ~0 for identical series."""
        obs, _ = data
        result = flow_duration_curve_metrics(obs, obs)
        for key in ['fdc_bias_low', 'fdc_bias_mid', 'fdc_bias_high']:
            if not np.isnan(result[key]):
                assert result[key] == pytest.approx(0.0, abs=1e-8)


# ===================================================================
# SECTION 3: Metric Transformer Properties
# ===================================================================


class TestMetricTransformerProperties:
    """Property-based tests for the optimisation direction transformer."""

    @given(value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_roundtrip_maximize_metrics(self, value):
        """Transform then inverse_transform for maximize metrics must be identity."""
        for metric in ['KGE', 'NSE', 'R2', 'correlation']:
            transformed = MetricTransformer.transform_for_maximization(metric, value)
            recovered = MetricTransformer.inverse_transform(metric, transformed)
            assert recovered == pytest.approx(value, abs=1e-12)

    @given(value=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_roundtrip_minimize_metrics(self, value):
        """Transform then inverse_transform for minimize metrics recovers positive value."""
        for metric in ['RMSE', 'MAE', 'NRMSE', 'MARE']:
            transformed = MetricTransformer.transform_for_maximization(metric, value)
            recovered = MetricTransformer.inverse_transform(metric, transformed)
            assert recovered == pytest.approx(value, abs=1e-12)

    @given(
        v_good=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        v_bad=st.floats(min_value=-0.5, max_value=0.49, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ordering_preserved_maximize(self, v_good, v_bad):
        """Better KGE must remain better after transformation."""
        t_good = MetricTransformer.transform_for_maximization('KGE', v_good)
        t_bad = MetricTransformer.transform_for_maximization('KGE', v_bad)
        assert t_good > t_bad

    @given(
        v_good=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        v_bad=st.floats(min_value=10.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ordering_preserved_minimize(self, v_good, v_bad):
        """Lower RMSE (better) must become higher after transformation."""
        t_good = MetricTransformer.transform_for_maximization('RMSE', v_good)
        t_bad = MetricTransformer.transform_for_maximization('RMSE', v_bad)
        assert t_good > t_bad

    @given(value=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_pbias_sign_symmetry(self, value):
        """Both +x and -x PBIAS must produce the same transformed value."""
        t_pos = MetricTransformer.transform_for_maximization('PBIAS', value)
        t_neg = MetricTransformer.transform_for_maximization('PBIAS', -value)
        assert t_pos == t_neg

    @given(value=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_pbias_zero_is_best(self, value):
        """Zero PBIAS must always produce higher transformed value than non-zero."""
        t_zero = MetricTransformer.transform_for_maximization('PBIAS', 0.0)
        t_nonzero = MetricTransformer.transform_for_maximization('PBIAS', value)
        assert t_zero >= t_nonzero

    def test_none_passthrough_all_metrics(self):
        """None must pass through for every metric type."""
        for metric in ['KGE', 'RMSE', 'PBIAS']:
            assert MetricTransformer.transform_for_maximization(metric, None) is None
            assert MetricTransformer.inverse_transform(metric, None) is None


# ===================================================================
# SECTION 4: Coordinate Utilities Properties
# ===================================================================


class TestLongitudeNormalizationProperties:
    """Property-based tests for longitude normalisation."""

    @given(lon=st.floats(min_value=-360.0, max_value=720.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=500, deadline=None)
    def test_normalize_to_0_360_range(self, lon):
        """Result must be in [0, 360)."""
        result = normalize_longitude(lon, target_range='0-360')
        assert 0.0 <= result < 360.0 + 1e-10

    @given(lon=st.floats(min_value=-360.0, max_value=720.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=500, deadline=None)
    def test_normalize_to_minus180_180_range(self, lon):
        """Result must be in [-180, 180)."""
        result = normalize_longitude(lon, target_range='-180-180')
        assert -180.0 - 1e-10 <= result < 180.0 + 1e-10

    @given(lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=500, deadline=None)
    def test_normalize_idempotent_0_360(self, lon):
        """Normalising to 0-360 twice must give the same result."""
        once = normalize_longitude(lon, '0-360')
        twice = normalize_longitude(once, '0-360')
        assert once == pytest.approx(twice, abs=1e-10)

    @given(lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=500, deadline=None)
    def test_normalize_idempotent_minus180_180(self, lon):
        """Normalising to -180-180 twice must give the same result."""
        once = normalize_longitude(lon, '-180-180')
        twice = normalize_longitude(once, '-180-180')
        assert once == pytest.approx(twice, abs=1e-10)

    @given(lon=st.floats(min_value=-180.0, max_value=179.9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=500, deadline=None)
    def test_roundtrip_normalisation(self, lon):
        """0-360 -> -180-180 -> 0-360 must preserve the value."""
        via_360 = normalize_longitude(lon, '0-360')
        back_180 = normalize_longitude(via_360, '-180-180')
        back_360 = normalize_longitude(back_180, '0-360')
        assert via_360 == pytest.approx(back_360, abs=1e-10)


class TestBoundingBoxProperties:
    """Property-based tests for BoundingBox."""

    @given(
        lat_a=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lat_b=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_bbox_auto_sorts_latitude(self, lat_a, lat_b, lon_min, lon_max):
        """BoundingBox must auto-sort latitudes so lat_min <= lat_max."""
        bbox = BoundingBox(lat_a, lat_b, lon_min, lon_max)
        assert bbox.lat_min <= bbox.lat_max

    @given(
        lat_min=st.floats(min_value=-89.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        lat_max=st.floats(min_value=0.01, max_value=89.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        pad_deg=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_bbox_pad_expands_or_stays(self, lat_min, lat_max, lon_min, lon_max, pad_deg):
        """Padding a bbox must not shrink it."""
        bbox = BoundingBox(lat_min, lat_max, lon_min, lon_max)
        padded = bbox.pad(pad_deg)
        assert padded.lat_min <= bbox.lat_min + 1e-10
        assert padded.lat_max >= bbox.lat_max - 1e-10

    @given(
        lat_min=st.floats(min_value=-89.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        lat_max=st.floats(min_value=0.01, max_value=89.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_bbox_pad_zero_is_identity(self, lat_min, lat_max, lon_min, lon_max):
        """Padding by 0 degrees must return an equivalent bbox."""
        bbox = BoundingBox(lat_min, lat_max, lon_min, lon_max)
        padded = bbox.pad(0.0)
        assert padded.lat_min == pytest.approx(bbox.lat_min, abs=1e-10)
        assert padded.lat_max == pytest.approx(bbox.lat_max, abs=1e-10)
        assert padded.lon_min == pytest.approx(bbox.lon_min, abs=1e-10)
        assert padded.lon_max == pytest.approx(bbox.lon_max, abs=1e-10)

    @given(
        lat_min=st.floats(min_value=-85.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        lat_max=st.floats(min_value=0.01, max_value=85.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-170.0, max_value=170.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-170.0, max_value=170.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_bbox_latitude_clamp_at_poles(self, lat_min, lat_max, lon_min, lon_max):
        """Latitude must never exceed [-90, 90] after padding."""
        bbox = BoundingBox(lat_min, lat_max, lon_min, lon_max)
        padded = bbox.pad(100.0)  # extreme padding
        assert padded.lat_min >= -90.0
        assert padded.lat_max <= 90.0

    @given(
        lat_min=st.floats(min_value=-89.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        lat_max=st.floats(min_value=0.01, max_value=89.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_bbox_to_dict_roundtrip(self, lat_min, lat_max, lon_min, lon_max):
        """to_dict() must contain the original coordinates."""
        bbox = BoundingBox(lat_min, lat_max, lon_min, lon_max)
        d = bbox.to_dict()
        assert d['lat_min'] == bbox.lat_min
        assert d['lat_max'] == bbox.lat_max
        assert d['lon_min'] == bbox.lon_min
        assert d['lon_max'] == bbox.lon_max


class TestValidateBboxProperties:
    """Property-based tests for bbox validation."""

    @given(
        lat_min=st.floats(min_value=-90.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        lat_max=st.floats(min_value=0.01, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon_min=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        lon_max=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_valid_bbox_always_passes(self, lat_min, lat_max, lon_min, lon_max):
        """A correctly constructed bbox must always validate."""
        bbox = {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max,
        }
        assert validate_bbox(bbox) is True

    @given(
        lat=st.floats(min_value=91.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_invalid_latitude_rejected(self, lat):
        """Latitude outside [-90, 90] must be rejected."""
        bbox = {'lat_min': 0.0, 'lat_max': lat, 'lon_min': 0.0, 'lon_max': 1.0}
        with pytest.raises(ValueError):
            validate_bbox(bbox)
