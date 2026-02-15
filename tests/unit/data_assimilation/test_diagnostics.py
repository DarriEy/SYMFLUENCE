"""Tests for DA diagnostics."""

import numpy as np
import pytest

from symfluence.data_assimilation.diagnostics import (
    rank_histogram,
    crps,
    spread_error_ratio,
    innovation_consistency,
    open_loop_comparison,
)


class TestRankHistogram:

    def test_shape(self):
        n_timesteps, n_members = 100, 10
        preds = np.random.randn(n_timesteps, n_members)
        obs = np.random.randn(n_timesteps)

        hist = rank_histogram(preds, obs)
        assert hist.shape == (n_members + 1,)
        assert hist.sum() == n_timesteps

    def test_handles_nan(self):
        preds = np.random.randn(10, 5)
        obs = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0])

        hist = rank_histogram(preds, obs)
        assert hist.sum() == 7  # 10 - 3 NaN


class TestCRPS:

    def test_perfect_ensemble(self):
        """CRPS should be zero for perfect ensemble centered on obs."""
        n_timesteps = 50
        obs = np.ones(n_timesteps) * 5.0
        # All members equal to obs
        preds = np.ones((n_timesteps, 20)) * 5.0

        score = crps(preds, obs)
        assert score < 0.01

    def test_worse_ensemble_higher_crps(self):
        """Worse ensemble should have higher CRPS."""
        n_timesteps = 100
        rng = np.random.default_rng(42)
        obs = rng.normal(5.0, 1.0, n_timesteps)

        # Good ensemble: close to truth
        good_preds = obs[:, None] + rng.normal(0, 0.1, (n_timesteps, 20))
        # Bad ensemble: far from truth
        bad_preds = obs[:, None] + rng.normal(5, 3.0, (n_timesteps, 20))

        crps_good = crps(good_preds, obs)
        crps_bad = crps(bad_preds, obs)

        assert crps_good < crps_bad


class TestSpreadErrorRatio:

    def test_well_calibrated(self):
        """Ratio should be finite and positive for a reasonable ensemble."""
        rng = np.random.default_rng(42)
        n = 500
        obs = rng.normal(5.0, 1.0, n)
        # Ensemble centered on obs with some spread
        preds = obs[:, None] + rng.normal(0, 1.0, (n, 50))

        ratio = spread_error_ratio(preds, obs)
        # Just check it's a finite positive number
        assert ratio > 0.0
        assert np.isfinite(ratio)


class TestInnovationConsistency:

    def test_consistent(self):
        rng = np.random.default_rng(42)
        n = 100
        obs_error_std = 0.5
        ens_stds = np.ones(n) * 1.0
        # Innovation variance should be ~ ens_std^2 + obs_std^2 = 1.25
        innovations = rng.normal(0, np.sqrt(1.25), n)

        ratio = innovation_consistency(innovations, ens_stds, obs_error_std)
        assert 0.5 < ratio < 2.0


class TestOpenLoopComparison:

    def test_da_improvement(self):
        rng = np.random.default_rng(42)
        n = 100
        obs = rng.normal(5.0, 1.0, n)

        # DA closer to truth
        da_preds = obs + rng.normal(0, 0.3, n)
        # Open loop further
        ol_preds = obs + rng.normal(2.0, 1.0, n)

        result = open_loop_comparison(da_preds, ol_preds, obs)

        assert result['da_rmse'] < result['ol_rmse']
        assert result['rmse_improvement'] > 0
