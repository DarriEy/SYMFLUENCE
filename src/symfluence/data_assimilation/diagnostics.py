"""
Data assimilation diagnostics.

Provides verification metrics for ensemble forecasting quality:
- Rank histogram (reliability)
- CRPS (continuous ranked probability score)
- Spread-error ratio
- Innovation consistency
- Open-loop comparison
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def rank_histogram(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute rank histogram for ensemble reliability assessment.

    A uniform histogram indicates a well-calibrated ensemble.

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Histogram counts of shape (n_members + 1,).
    """
    valid = ~np.isnan(observations)
    preds = ensemble_predictions[valid]
    obs = observations[valid]

    n_members = preds.shape[1]
    ranks = np.zeros(len(obs), dtype=int)

    for i in range(len(obs)):
        ranks[i] = np.searchsorted(np.sort(preds[i]), obs[i])

    histogram = np.bincount(ranks, minlength=n_members + 1)
    return histogram


def crps(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Compute mean Continuous Ranked Probability Score (CRPS).

    Lower CRPS indicates better probabilistic forecast quality.

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Mean CRPS value.
    """
    valid = ~np.isnan(observations)
    preds = ensemble_predictions[valid]
    obs = observations[valid]

    n_timesteps, n_members = preds.shape
    crps_values = np.zeros(n_timesteps)

    for t in range(n_timesteps):
        # Sort ensemble
        sorted_ens = np.sort(preds[t])

        # CRPS decomposition: |x_i - obs| term
        abs_diff = np.mean(np.abs(sorted_ens - obs[t]))

        # Inter-member spread term
        spread = 0.0
        for i in range(n_members):
            for j in range(i + 1, n_members):
                spread += np.abs(sorted_ens[i] - sorted_ens[j])
        spread /= (n_members * (n_members - 1)) if n_members > 1 else 1.0

        crps_values[t] = abs_diff - 0.5 * spread

    return float(np.mean(crps_values))


def spread_error_ratio(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Compute the spread-error ratio.

    A well-calibrated ensemble has a ratio close to 1.0:
    - ratio > 1: ensemble is over-dispersive (too much spread)
    - ratio < 1: ensemble is under-dispersive (too little spread)

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Spread-error ratio.
    """
    valid = ~np.isnan(observations)
    preds = ensemble_predictions[valid]
    obs = observations[valid]

    # Spread: mean ensemble standard deviation
    spread = np.mean(np.std(preds, axis=1))

    # Error: RMSE of ensemble mean
    ens_mean = np.mean(preds, axis=1)
    rmse = np.sqrt(np.mean((ens_mean - obs) ** 2))

    if rmse < 1e-12:
        return float('inf')

    return float(spread / rmse)


def innovation_consistency(
    innovations: np.ndarray,
    ensemble_stds: np.ndarray,
    obs_error_std: float,
) -> float:
    """Check consistency of innovations with expected variance.

    Normalized innovations should have unit variance. Returns the
    ratio of actual to expected innovation variance.

    Args:
        innovations: obs - predicted_mean, shape (n_timesteps,).
        ensemble_stds: Ensemble spread, shape (n_timesteps,).
        obs_error_std: Observation error standard deviation.

    Returns:
        Ratio of actual/expected innovation variance (should be ~1.0).
    """
    valid = ~np.isnan(innovations)
    innov = innovations[valid]
    stds = ensemble_stds[valid]

    expected_var = stds ** 2 + obs_error_std ** 2
    actual_var = innov ** 2

    ratio = np.mean(actual_var) / np.mean(expected_var) if np.mean(expected_var) > 1e-12 else float('inf')
    return float(ratio)


def open_loop_comparison(
    da_predictions: np.ndarray,
    open_loop_predictions: np.ndarray,
    observations: np.ndarray,
) -> Dict[str, float]:
    """Compare DA performance against open-loop (no assimilation) baseline.

    Args:
        da_predictions: DA ensemble mean, shape (n_timesteps,).
        open_loop_predictions: Open-loop predictions, shape (n_timesteps,).
        observations: Observations, shape (n_timesteps,).

    Returns:
        Dictionary with RMSE and correlation for both DA and open-loop.
    """
    valid = ~np.isnan(observations) & ~np.isnan(da_predictions) & ~np.isnan(open_loop_predictions)

    da = da_predictions[valid]
    ol = open_loop_predictions[valid]
    obs = observations[valid]

    if len(obs) == 0:
        return {'da_rmse': np.nan, 'ol_rmse': np.nan, 'da_corr': np.nan, 'ol_corr': np.nan}

    da_rmse = float(np.sqrt(np.mean((da - obs) ** 2)))
    ol_rmse = float(np.sqrt(np.mean((ol - obs) ** 2)))

    da_corr = float(np.corrcoef(da, obs)[0, 1]) if len(obs) > 1 else np.nan
    ol_corr = float(np.corrcoef(ol, obs)[0, 1]) if len(obs) > 1 else np.nan

    return {
        'da_rmse': da_rmse,
        'ol_rmse': ol_rmse,
        'da_corr': da_corr,
        'ol_corr': ol_corr,
        'rmse_improvement': float((ol_rmse - da_rmse) / ol_rmse * 100) if ol_rmse > 1e-12 else 0.0,
    }
