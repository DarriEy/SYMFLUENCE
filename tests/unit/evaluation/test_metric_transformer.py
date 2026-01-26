"""Tests for MetricTransformer utility."""

import numpy as np
import pytest

from symfluence.evaluation.metric_transformer import MetricTransformer


class TestMetricTransformerDirection:
    """Test metric direction detection."""

    def test_maximize_metrics_direction(self):
        """Test that maximize metrics are correctly identified."""
        maximize_metrics = ['KGE', 'NSE', 'logNSE', 'KGEp', 'KGEnp', 'VE', 'correlation', 'R2']
        for metric in maximize_metrics:
            assert MetricTransformer.get_direction(metric) == 'maximize', \
                f"{metric} should be a maximize metric"

    def test_minimize_metrics_direction(self):
        """Test that minimize metrics are correctly identified."""
        minimize_metrics = ['RMSE', 'MAE', 'NRMSE', 'MARE', 'PBIAS', 'bias']
        for metric in minimize_metrics:
            assert MetricTransformer.get_direction(metric) == 'minimize', \
                f"{metric} should be a minimize metric"

    def test_case_insensitive_lookup(self):
        """Test that metric lookup works with aliases defined in METRIC_REGISTRY.

        Note: Only metrics with explicit lowercase aliases (kge, nse, etc.) support
        case-insensitive lookup. Other metrics must use their canonical case (e.g., RMSE).
        """
        # Metrics with lowercase aliases work case-insensitively
        assert MetricTransformer.get_direction('kge') == 'maximize'
        assert MetricTransformer.get_direction('KGE') == 'maximize'
        assert MetricTransformer.get_direction('nse') == 'maximize'
        assert MetricTransformer.get_direction('NSE') == 'maximize'

        # Metrics without lowercase aliases need canonical case
        assert MetricTransformer.get_direction('RMSE') == 'minimize'
        assert MetricTransformer.get_direction('MAE') == 'minimize'
        assert MetricTransformer.get_direction('PBIAS') == 'minimize'

    def test_unknown_metric_defaults_to_maximize(self):
        """Unknown metrics should default to maximize (safer for fitness tracking)."""
        assert MetricTransformer.get_direction('UnknownMetric') == 'maximize'
        assert MetricTransformer.get_direction('CustomFitness') == 'maximize'

    def test_is_minimize_metric(self):
        """Test the is_minimize_metric helper."""
        assert MetricTransformer.is_minimize_metric('RMSE') is True
        assert MetricTransformer.is_minimize_metric('KGE') is False


class TestTransformForMaximization:
    """Test metric value transformation for maximization."""

    def test_maximize_metrics_unchanged(self):
        """Maximize metrics should not be transformed."""
        assert MetricTransformer.transform_for_maximization('KGE', 0.85) == 0.85
        assert MetricTransformer.transform_for_maximization('NSE', 0.9) == 0.9
        assert MetricTransformer.transform_for_maximization('R2', 0.95) == 0.95
        assert MetricTransformer.transform_for_maximization('correlation', 0.88) == 0.88

    def test_minimize_metrics_negated(self):
        """Minimize metrics should be negated."""
        assert MetricTransformer.transform_for_maximization('RMSE', 10.0) == -10.0
        assert MetricTransformer.transform_for_maximization('MAE', 5.0) == -5.0
        assert MetricTransformer.transform_for_maximization('NRMSE', 0.5) == -0.5
        assert MetricTransformer.transform_for_maximization('MARE', 0.2) == -0.2

    def test_signed_minimize_metrics_use_abs(self):
        """Signed minimize metrics (PBIAS, bias) should use -abs(value)."""
        # Positive PBIAS (overestimation)
        assert MetricTransformer.transform_for_maximization('PBIAS', 20.0) == -20.0
        # Negative PBIAS (underestimation) - should also become -20
        assert MetricTransformer.transform_for_maximization('PBIAS', -20.0) == -20.0
        # Both +10% and -10% bias are equally bad
        assert (MetricTransformer.transform_for_maximization('PBIAS', 10.0) ==
                MetricTransformer.transform_for_maximization('PBIAS', -10.0))

        # Same for bias
        assert MetricTransformer.transform_for_maximization('bias', 5.0) == -5.0
        assert MetricTransformer.transform_for_maximization('bias', -5.0) == -5.0

    def test_none_value_passthrough(self):
        """None values should be passed through unchanged."""
        assert MetricTransformer.transform_for_maximization('KGE', None) is None
        assert MetricTransformer.transform_for_maximization('RMSE', None) is None

    def test_nan_value_passthrough(self):
        """NaN values should be passed through unchanged."""
        result = MetricTransformer.transform_for_maximization('KGE', np.nan)
        assert np.isnan(result)
        result = MetricTransformer.transform_for_maximization('RMSE', np.nan)
        assert np.isnan(result)

    def test_zero_value(self):
        """Zero values should be handled correctly."""
        assert MetricTransformer.transform_for_maximization('RMSE', 0.0) == 0.0
        assert MetricTransformer.transform_for_maximization('PBIAS', 0.0) == 0.0
        assert MetricTransformer.transform_for_maximization('KGE', 0.0) == 0.0

    def test_negative_maximize_metrics(self):
        """Negative maximize metric values should not be changed."""
        # NSE can be negative (worse than mean model)
        assert MetricTransformer.transform_for_maximization('NSE', -0.5) == -0.5
        assert MetricTransformer.transform_for_maximization('KGE', -1.0) == -1.0


class TestInverseTransform:
    """Test inverse transformation back to original metric space."""

    def test_maximize_metrics_unchanged(self):
        """Maximize metrics should not be inverse transformed."""
        assert MetricTransformer.inverse_transform('KGE', 0.85) == 0.85
        assert MetricTransformer.inverse_transform('NSE', 0.9) == 0.9

    def test_minimize_metrics_negated_back(self):
        """Minimize metrics should be negated back."""
        assert MetricTransformer.inverse_transform('RMSE', -10.0) == 10.0
        assert MetricTransformer.inverse_transform('MAE', -5.0) == 5.0

    def test_roundtrip_maximize(self):
        """Transform followed by inverse should return original for maximize metrics."""
        original = 0.85
        transformed = MetricTransformer.transform_for_maximization('KGE', original)
        recovered = MetricTransformer.inverse_transform('KGE', transformed)
        assert recovered == original

    def test_roundtrip_minimize(self):
        """Transform followed by inverse should return original for minimize metrics."""
        original = 10.0
        transformed = MetricTransformer.transform_for_maximization('RMSE', original)
        recovered = MetricTransformer.inverse_transform('RMSE', transformed)
        assert recovered == original

    def test_none_passthrough(self):
        """None should pass through inverse transform."""
        assert MetricTransformer.inverse_transform('KGE', None) is None

    def test_nan_passthrough(self):
        """NaN should pass through inverse transform."""
        result = MetricTransformer.inverse_transform('RMSE', np.nan)
        assert np.isnan(result)


class TestTransformObjectives:
    """Test multi-objective transformation."""

    def test_transform_mixed_objectives(self):
        """Test transforming a mix of maximize and minimize objectives."""
        names = ['KGE', 'RMSE']
        values = [0.85, 10.0]
        transformed = MetricTransformer.transform_objectives(names, values)

        assert transformed[0] == 0.85  # KGE unchanged
        assert transformed[1] == -10.0  # RMSE negated

    def test_transform_all_maximize(self):
        """Test transforming all maximize objectives."""
        names = ['KGE', 'NSE']
        values = [0.85, 0.90]
        transformed = MetricTransformer.transform_objectives(names, values)

        assert transformed[0] == 0.85
        assert transformed[1] == 0.90

    def test_transform_all_minimize(self):
        """Test transforming all minimize objectives."""
        names = ['RMSE', 'MAE']
        values = [10.0, 5.0]
        transformed = MetricTransformer.transform_objectives(names, values)

        assert transformed[0] == -10.0
        assert transformed[1] == -5.0

    def test_length_mismatch_raises(self):
        """Length mismatch between names and values should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            MetricTransformer.transform_objectives(['KGE', 'RMSE'], [0.85])

    def test_empty_lists(self):
        """Empty lists should return empty list."""
        assert MetricTransformer.transform_objectives([], []) == []


class TestOptimizationIntegration:
    """Test that transformer produces correct optimization behavior."""

    def test_better_kge_stays_better_after_transform(self):
        """Higher KGE should remain higher after transformation."""
        good = MetricTransformer.transform_for_maximization('KGE', 0.9)
        bad = MetricTransformer.transform_for_maximization('KGE', 0.5)
        assert good > bad

    def test_lower_rmse_becomes_higher_after_transform(self):
        """Lower RMSE should become higher after transformation."""
        good = MetricTransformer.transform_for_maximization('RMSE', 5.0)  # Better (lower)
        bad = MetricTransformer.transform_for_maximization('RMSE', 20.0)  # Worse (higher)
        assert good > bad  # After transform, good should be > bad

    def test_zero_pbias_best_after_transform(self):
        """Zero PBIAS should give highest transformed value."""
        zero = MetricTransformer.transform_for_maximization('PBIAS', 0.0)
        pos = MetricTransformer.transform_for_maximization('PBIAS', 10.0)
        neg = MetricTransformer.transform_for_maximization('PBIAS', -10.0)

        assert zero > pos
        assert zero > neg
        assert pos == neg  # Both equally bad
