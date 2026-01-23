#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hydrological Performance Metrics

This module provides standardized performance metrics for hydrological model evaluation.

Metrics Implemented
-------------------
Efficiency metrics:
    - NSE: Nash-Sutcliffe Efficiency
    - logNSE: Log-transformed Nash-Sutcliffe Efficiency
    - KGE: Kling-Gupta Efficiency
    - KGE': Modified Kling-Gupta Efficiency (uses CV instead of std)
    - KGEnp: Non-parametric Kling-Gupta Efficiency
    - VE: Volumetric Efficiency

Error metrics:
    - RMSE: Root Mean Square Error
    - NRMSE: Normalized Root Mean Square Error
    - MAE: Mean Absolute Error
    - MARE: Mean Absolute Relative Error

Bias metrics:
    - bias: Mean error (sim - obs)
    - PBIAS: Percent bias

Correlation metrics:
    - correlation: Pearson or Spearman correlation coefficient
    - r_squared: Coefficient of determination (R²)

All functions handle NaN values automatically and support optional transformations.

Usage
-----
>>> from symfluence.evaluation.metrics import kge, nse, calculate_all_metrics
>>> metrics = calculate_all_metrics(observed, simulated)
>>> print(f"KGE: {metrics['KGE']:.3f}, NSE: {metrics['NSE']:.3f}")

References
----------
Nash & Sutcliffe (1970), Gupta et al. (2009), Kling et al. (2012),
Pool et al. (2018), Criss & Winston (2008)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, List, Callable, Optional, cast
from dataclasses import dataclass
from scipy import stats

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core metric functions
    'nse',
    'log_nse',
    'kge',
    'kge_prime',
    'kge_np',
    'rmse',
    'nrmse',
    'mae',
    'mare',
    'bias',
    'pbias',
    'correlation',
    'r_squared',
    'volumetric_efficiency',
    # Hydrograph signature metrics
    'peak_timing_error',
    'recession_constant',
    'baseflow_index',
    'flow_duration_curve_metrics',
    'hydrograph_signatures',
    # Convenience functions
    'calculate_all_metrics',
    'calculate_metrics',
    'get_metric_function',
    'list_available_metrics',
    # Registry and metadata
    'METRIC_REGISTRY',
    'MetricInfo',
    # Helper functions
    'interpret_metric',
]


# =============================================================================
# Metric Metadata
# =============================================================================

@dataclass(frozen=True)
class MetricInfo:
    """Metadata for a performance metric.

    Attributes
    ----------
    name : str
        Short name/identifier for the metric
    full_name : str
        Full descriptive name
    range : tuple
        (min, max) possible values, use float('inf') for unbounded
    optimal : float
        Optimal/perfect value
    direction : str
        'maximize' or 'minimize'
    units : str
        Units or 'dimensionless'
    description : str
        Brief description of what the metric measures
    reference : str
        Academic reference
    """
    name: str
    full_name: str
    range: Tuple[float, float]
    optimal: float
    direction: str
    units: str
    description: str
    reference: str


def _clean_data(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and align observed and simulated data by removing NaN values.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values

    Returns
    -------
    tuple of np.ndarray
        Cleaned (observed, simulated) arrays with NaN values removed
    """
    obs = np.array(observed)
    sim = np.array(simulated)

    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[valid_mask]
    sim_clean = sim[valid_mask]

    return obs_clean, sim_clean


def _apply_transformation(
    observed: np.ndarray,
    simulated: np.ndarray,
    transfo: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply power transformation to observed and simulated data.

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    simulated : np.ndarray
        Simulated values
    transfo : float, optional
        Transformation exponent (default: 1.0 for no transformation)
        If negative, adds epsilon before transformation

    Returns
    -------
    tuple of np.ndarray
        Transformed (observed, simulated) arrays with NaN/Inf values removed
    """
    if transfo == 1.0:
        return observed, simulated

    # Add small epsilon if using negative transformation (e.g., for inverse/log-like transforms)
    if transfo < 0:
        epsilon = np.mean(observed) / 100
        obs_trans = (epsilon + observed) ** transfo
        sim_trans = (epsilon + simulated) ** transfo
    else:
        obs_trans = observed ** transfo
        sim_trans = simulated ** transfo

    # Clean NaN/Inf values that may result from transformation
    valid_mask = ~(np.isnan(obs_trans) | np.isnan(sim_trans) |
                   np.isinf(obs_trans) | np.isinf(sim_trans))
    return obs_trans[valid_mask], sim_trans[valid_mask]


def nse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency.

    NSE = 1 - [sum((obs - sim)²) / sum((obs - mean(obs))²)]

    Range: (-∞, 1], where 1 is perfect fit, 0 means model is as good as mean,
    negative values mean model is worse than mean.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency value, or np.nan if calculation fails

    References
    ----------
    Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through
    conceptual models part I—A discussion of principles. Journal of hydrology,
    10(3), 282-290.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    mean_obs = np.mean(obs)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - mean_obs) ** 2)

    if denominator == 0:
        return np.nan

    return 1.0 - (numerator / denominator)


def kge(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
    return_components: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Calculate Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r - 1)² + (alpha - 1)² + (beta - 1)²)

    Where:
    - r = correlation coefficient
    - alpha = std(sim) / std(obs) (variability ratio)
    - beta = mean(sim) / mean(obs) (bias ratio)

    Range: (-∞, 1], where 1 is perfect fit.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)
    return_components : bool, optional
        If True, return dict with KGE and components (r, alpha, beta)

    Returns
    -------
    float or dict
        KGE value, or dict with KGE and components if return_components=True
        Returns np.nan if calculation fails

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of hydrology,
    377(1-2), 80-91.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        if return_components:
            return {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    # Calculate components
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)

    # Correlation
    if std_obs == 0 or std_sim == 0:
        r = np.nan
    else:
        r = np.corrcoef(obs, sim)[0, 1]

    # Variability ratio
    alpha = std_sim / std_obs if std_obs != 0 else np.nan

    # Bias ratio
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    # Calculate KGE
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge_value = np.nan
    else:
        kge_value = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    if return_components:
        return {
            'KGE': float(kge_value),
            'r': float(r),
            'alpha': float(alpha),
            'beta': float(beta)
        }

    return float(kge_value)


def kge_prime(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
    return_components: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Calculate modified Kling-Gupta Efficiency (KGE').

    Similar to KGE but uses coefficient of variation instead of standard deviation:
    KGE' = 1 - sqrt((r - 1)² + (gamma - 1)² + (beta - 1)²)

    Where:
    - r = correlation coefficient
    - gamma = CV(sim) / CV(obs) (variability ratio using coefficient of variation)
    - beta = mean(sim) / mean(obs) (bias ratio)

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)
    return_components : bool, optional
        If True, return dict with KGE' and components

    Returns
    -------
    float or dict
        KGE' value, or dict with components if return_components=True

    References
    ----------
    Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios. Journal of
    hydrology, 424, 264-277.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        if return_components:
            return {'KGEp': np.nan, 'r': np.nan, 'gamma': np.nan, 'beta': np.nan}
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    # Calculate components
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)

    # Correlation
    if std_obs == 0 or std_sim == 0:
        r = np.nan
    else:
        r = np.corrcoef(obs, sim)[0, 1]

    # Coefficient of variation ratio (gamma)
    cv_obs = std_obs / mean_obs if mean_obs != 0 else np.nan
    cv_sim = std_sim / mean_sim if mean_sim != 0 else np.nan
    gamma = cv_sim / cv_obs if cv_obs != 0 else np.nan

    # Bias ratio
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    # Calculate KGE'
    if np.isnan(r) or np.isnan(gamma) or np.isnan(beta):
        kgep_value = np.nan
    else:
        kgep_value = 1.0 - np.sqrt((r - 1)**2 + (gamma - 1)**2 + (beta - 1)**2)

    if return_components:
        return {
            'KGEp': float(kgep_value),
            'r': float(r),
            'gamma': float(gamma),
            'beta': float(beta)
        }

    return float(kgep_value)


def kge_np(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate Non-Parametric Kling-Gupta Efficiency (KGEnp).

    Uses Spearman correlation and flow duration curves instead of
    Pearson correlation and standard deviation.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        KGEnp value, or np.nan if calculation fails

    References
    ----------
    Pool, S., Vis, M., & Seibert, J. (2018). Evaluating model performance:
    towards a non-parametric variant of the Kling-Gupta efficiency.
    Hydrological Sciences Journal, 63(13-14), 1941-1953.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    # Spearman correlation instead of Pearson
    r_np = stats.spearmanr(obs, sim)[0]

    # Non-parametric alpha: based on normalized flow duration curves
    fdc_obs = np.sort(obs / (mean_obs * len(obs)))
    fdc_sim = np.sort(sim / (mean_sim * len(sim)))
    alpha_np = 1.0 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))

    # Beta remains the same (bias ratio)
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    if np.isnan(r_np) or np.isnan(alpha_np) or np.isnan(beta):
        return np.nan

    return float(1.0 - np.sqrt((r_np - 1)**2 + (alpha_np - 1)**2 + (beta - 1)**2))


def rmse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate Root Mean Square Error.

    RMSE = sqrt(mean((obs - sim)²))

    Range: [0, ∞), where 0 is perfect fit. Units same as input data.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        RMSE value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def nrmse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate Normalized Root Mean Square Error.

    NRMSE = RMSE / std(obs)

    Normalizes RMSE by standard deviation of observations.
    Range: [0, ∞), where 0 is perfect fit. Dimensionless.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        NRMSE value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    rmse_value = np.sqrt(np.mean((obs - sim) ** 2))
    std_obs = np.std(obs)

    if std_obs == 0:
        return np.nan

    return float(rmse_value / std_obs)


def mae(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate Mean Absolute Error.

    MAE = mean(|obs - sim|)

    Range: [0, ∞), where 0 is perfect fit. Units same as input data.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        MAE value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    return float(np.mean(np.abs(obs - sim)))


def bias(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate bias (mean error).

    Bias = mean(sim) - mean(obs)

    Range: (-∞, ∞), where 0 is no bias. Units same as input data.
    Positive values indicate overestimation, negative indicate underestimation.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        Bias value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    return float(np.mean(sim) - np.mean(obs))


def pbias(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> float:
    """
    Calculate percent bias.

    PBIAS = 100 * sum(sim - obs) / sum(obs)

    Range: (-∞, ∞), where 0 is no bias. Dimensionless percentage.
    Positive values indicate overestimation, negative indicate underestimation.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    float
        Percent bias value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    obs, sim = _apply_transformation(obs, sim, transfo)

    sum_obs = np.sum(obs)
    if sum_obs == 0:
        return np.nan

    return float(100.0 * (np.sum(sim) - sum_obs) / sum_obs)


def correlation(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    method: str = 'pearson'
) -> float:
    """
    Calculate correlation coefficient.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    method : str, optional
        Correlation method: 'pearson' or 'spearman' (default: 'pearson')

    Returns
    -------
    float
        Correlation coefficient [-1, 1], or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 2:
        return np.nan

    if method == 'pearson':
        if np.std(obs) == 0 or np.std(sim) == 0:
            return np.nan
        return float(np.corrcoef(obs, sim)[0, 1])
    elif method == 'spearman':
        return float(stats.spearmanr(obs, sim)[0])
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def log_nse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    epsilon: Optional[float] = None
) -> float:
    """
    Calculate log-transformed Nash-Sutcliffe Efficiency.

    logNSE = 1 - [sum((log(obs) - log(sim))²) / sum((log(obs) - mean(log(obs)))²)]

    Emphasizes low flow performance by applying logarithmic transformation.
    Range: (-∞, 1], where 1 is perfect fit.

    Parameters
    ----------
    observed : array-like
        Observed values (must be positive)
    simulated : array-like
        Simulated values (must be positive)
    epsilon : float, optional
        Small value added before log transform to handle zeros.
        Default: 1% of mean observed value.

    Returns
    -------
    float
        Log-transformed NSE value, or np.nan if calculation fails

    References
    ----------
    Krause, P., Boyle, D. P., & Bäse, F. (2005). Comparison of different
    efficiency criteria for hydrological model assessment. Advances in
    geosciences, 5, 89-97.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    # Add epsilon to handle zero values
    if epsilon is None:
        epsilon = np.mean(obs) * 0.01 if np.mean(obs) > 0 else 1e-6

    # Filter out non-positive values before log transform
    valid_mask = (obs > -epsilon) & (sim > -epsilon)
    obs = obs[valid_mask]
    sim = sim[valid_mask]

    if len(obs) == 0:
        return np.nan

    obs_log = np.log(obs + epsilon)
    sim_log = np.log(sim + epsilon)

    mean_obs_log = np.mean(obs_log)
    numerator = np.sum((obs_log - sim_log) ** 2)
    denominator = np.sum((obs_log - mean_obs_log) ** 2)

    if denominator == 0:
        return np.nan

    return float(1.0 - (numerator / denominator))


def r_squared(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate coefficient of determination (R²).

    R² = r² where r is Pearson correlation coefficient

    Range: [0, 1], where 1 indicates perfect linear relationship.
    Note: R² can be misleading if there is systematic bias.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values

    Returns
    -------
    float
        R² value, or np.nan if calculation fails
    """
    r = correlation(observed, simulated, method='pearson')
    if np.isnan(r):
        return np.nan
    return float(r ** 2)


def mare(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    epsilon: Optional[float] = None
) -> float:
    """
    Calculate Mean Absolute Relative Error.

    MARE = mean(|obs - sim| / obs)

    Range: [0, ∞), where 0 is perfect fit. Dimensionless.
    Emphasizes relative errors, particularly useful for comparing across scales.

    Parameters
    ----------
    observed : array-like
        Observed values (must be non-zero for meaningful results)
    simulated : array-like
        Simulated values
    epsilon : float, optional
        Small value added to denominator to avoid division by zero.
        Default: 1% of mean observed value.

    Returns
    -------
    float
        MARE value, or np.nan if calculation fails
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    if epsilon is None:
        epsilon = np.mean(np.abs(obs)) * 0.01 if np.mean(np.abs(obs)) > 0 else 1e-6

    # Avoid division by zero
    denominator = np.abs(obs) + epsilon
    relative_errors = np.abs(obs - sim) / denominator

    return float(np.mean(relative_errors))


def volumetric_efficiency(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Volumetric Efficiency (VE).

    VE = 1 - [sum(|obs - sim|) / sum(obs)]

    Range: (-∞, 1], where 1 is perfect fit.
    Measures the fraction of water delivered at the proper time.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values

    Returns
    -------
    float
        VE value, or np.nan if calculation fails

    References
    ----------
    Criss, R. E., & Winston, W. E. (2008). Do Nash values have value?
    Discussion and alternate proposals. Hydrological Processes, 22(14),
    2723-2725.
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    sum_obs = np.sum(obs)
    if sum_obs == 0:
        return np.nan

    return float(1.0 - np.sum(np.abs(obs - sim)) / sum_obs)


def calculate_all_metrics(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0
) -> Dict[str, float]:
    """
    Calculate all standard performance metrics in one call.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    dict
        Dictionary containing all metrics:
        - NSE: Nash-Sutcliffe Efficiency
        - KGE: Kling-Gupta Efficiency
        - KGEp: Modified KGE
        - RMSE: Root Mean Square Error
        - NRMSE: Normalized RMSE
        - MAE: Mean Absolute Error
        - PBIAS: Percent Bias
        - correlation: Pearson correlation
        - bias: Mean error
        - r: Correlation (KGE component)
        - alpha: Variability ratio (KGE component)
        - beta: Bias ratio (KGE component)
    """
    # Get KGE with components
    kge_result = cast(Dict[str, float], kge(observed, simulated, transfo, return_components=True))

    return {
        # Efficiency metrics
        'NSE': float(nse(observed, simulated, transfo)),
        'logNSE': float(log_nse(observed, simulated)),
        'KGE': float(kge_result['KGE']),
        'KGEp': float(kge_prime(observed, simulated, transfo)),
        'KGEnp': float(kge_np(observed, simulated, transfo)),
        'VE': float(volumetric_efficiency(observed, simulated)),
        # Error metrics
        'RMSE': float(rmse(observed, simulated, transfo)),
        'NRMSE': float(nrmse(observed, simulated, transfo)),
        'MAE': float(mae(observed, simulated, transfo)),
        'MARE': float(mare(observed, simulated)),
        # Bias metrics
        'PBIAS': float(pbias(observed, simulated, transfo)),
        'bias': float(bias(observed, simulated, transfo)),
        # Correlation metrics
        'correlation': float(correlation(observed, simulated)),
        'R2': float(r_squared(observed, simulated)),
        # KGE components
        'r': float(kge_result['r']),
        'alpha': float(kge_result['alpha']),
        'beta': float(kge_result['beta'])
    }


# =============================================================================
# Calculate Metrics (alias for backward compatibility)
# =============================================================================

def calculate_metrics(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    metrics: Optional[List[str]] = None,
    transfo: float = 1.0
) -> Dict[str, float]:
    """
    Calculate selected performance metrics.

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
    metrics : list of str, optional
        List of metric names to calculate. If None, calculates all.
        Valid names: NSE, logNSE, KGE, KGEp, KGEnp, VE, RMSE, NRMSE,
        MAE, MARE, PBIAS, bias, correlation, R2
    transfo : float, optional
        Power transformation exponent (default: 1.0)

    Returns
    -------
    dict
        Dictionary of requested metrics

    Examples
    --------
    >>> metrics = calculate_metrics(obs, sim, metrics=['KGE', 'NSE', 'RMSE'])
    >>> print(f"KGE: {metrics['KGE']:.3f}")
    """
    if metrics is None:
        return calculate_all_metrics(observed, simulated, transfo)

    result = {}
    for metric_name in metrics:
        func = get_metric_function(metric_name)
        if func is not None:
            try:
                if metric_name in ('correlation', 'R2', 'logNSE', 'MARE', 'VE'):
                    result[metric_name] = float(func(observed, simulated))
                else:
                    result[metric_name] = float(func(observed, simulated, transfo))
            except TypeError:
                # Function doesn't accept transfo parameter
                result[metric_name] = float(func(observed, simulated))
        else:
            result[metric_name] = np.nan

    return result


# =============================================================================
# Metric Registry
# =============================================================================

# Registry mapping metric names to their functions and metadata
METRIC_REGISTRY: Dict[str, Dict[str, Union[Callable, MetricInfo]]] = {
    'NSE': {
        'function': nse,
        'info': MetricInfo(
            name='NSE',
            full_name='Nash-Sutcliffe Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='Measures how well simulated values match observed variance',
            reference='Nash & Sutcliffe (1970)'
        )
    },
    'logNSE': {
        'function': log_nse,
        'info': MetricInfo(
            name='logNSE',
            full_name='Log-transformed Nash-Sutcliffe Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='NSE on log-transformed values, emphasizes low flows',
            reference='Krause et al. (2005)'
        )
    },
    'KGE': {
        'function': kge,
        'info': MetricInfo(
            name='KGE',
            full_name='Kling-Gupta Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='Decomposes NSE into correlation, variability, and bias',
            reference='Gupta et al. (2009)'
        )
    },
    'KGEp': {
        'function': kge_prime,
        'info': MetricInfo(
            name='KGEp',
            full_name='Modified Kling-Gupta Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='KGE using coefficient of variation instead of std',
            reference='Kling et al. (2012)'
        )
    },
    'KGEnp': {
        'function': kge_np,
        'info': MetricInfo(
            name='KGEnp',
            full_name='Non-parametric Kling-Gupta Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='KGE using Spearman correlation and flow duration curves',
            reference='Pool et al. (2018)'
        )
    },
    'VE': {
        'function': volumetric_efficiency,
        'info': MetricInfo(
            name='VE',
            full_name='Volumetric Efficiency',
            range=(float('-inf'), 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='Fraction of water delivered at the proper time',
            reference='Criss & Winston (2008)'
        )
    },
    'RMSE': {
        'function': rmse,
        'info': MetricInfo(
            name='RMSE',
            full_name='Root Mean Square Error',
            range=(0.0, float('inf')),
            optimal=0.0,
            direction='minimize',
            units='same as input',
            description='Average magnitude of simulation errors',
            reference='Standard'
        )
    },
    'NRMSE': {
        'function': nrmse,
        'info': MetricInfo(
            name='NRMSE',
            full_name='Normalized Root Mean Square Error',
            range=(0.0, float('inf')),
            optimal=0.0,
            direction='minimize',
            units='dimensionless',
            description='RMSE normalized by observed standard deviation',
            reference='Standard'
        )
    },
    'MAE': {
        'function': mae,
        'info': MetricInfo(
            name='MAE',
            full_name='Mean Absolute Error',
            range=(0.0, float('inf')),
            optimal=0.0,
            direction='minimize',
            units='same as input',
            description='Average absolute simulation error',
            reference='Standard'
        )
    },
    'MARE': {
        'function': mare,
        'info': MetricInfo(
            name='MARE',
            full_name='Mean Absolute Relative Error',
            range=(0.0, float('inf')),
            optimal=0.0,
            direction='minimize',
            units='dimensionless',
            description='Average relative simulation error',
            reference='Standard'
        )
    },
    'bias': {
        'function': bias,
        'info': MetricInfo(
            name='bias',
            full_name='Mean Error (Bias)',
            range=(float('-inf'), float('inf')),
            optimal=0.0,
            direction='minimize',
            units='same as input',
            description='Mean difference between simulated and observed',
            reference='Standard'
        )
    },
    'PBIAS': {
        'function': pbias,
        'info': MetricInfo(
            name='PBIAS',
            full_name='Percent Bias',
            range=(float('-inf'), float('inf')),
            optimal=0.0,
            direction='minimize',
            units='percent',
            description='Percentage difference in total volumes',
            reference='Standard'
        )
    },
    'correlation': {
        'function': correlation,
        'info': MetricInfo(
            name='correlation',
            full_name='Pearson Correlation Coefficient',
            range=(-1.0, 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='Linear correlation between observed and simulated',
            reference='Standard'
        )
    },
    'R2': {
        'function': r_squared,
        'info': MetricInfo(
            name='R2',
            full_name='Coefficient of Determination',
            range=(0.0, 1.0),
            optimal=1.0,
            direction='maximize',
            units='dimensionless',
            description='Proportion of variance explained by the model',
            reference='Standard'
        )
    },
}

# Add aliases for common alternative names
METRIC_REGISTRY['kge'] = METRIC_REGISTRY['KGE']
METRIC_REGISTRY['nse'] = METRIC_REGISTRY['NSE']
METRIC_REGISTRY['kge_prime'] = METRIC_REGISTRY['KGEp']
METRIC_REGISTRY['kge_np'] = METRIC_REGISTRY['KGEnp']
METRIC_REGISTRY['r_squared'] = METRIC_REGISTRY['R2']
METRIC_REGISTRY['log_nse'] = METRIC_REGISTRY['logNSE']


# =============================================================================
# Helper Functions
# =============================================================================

def get_metric_function(name: str) -> Optional[Callable]:
    """
    Get the function for a metric by name.

    Parameters
    ----------
    name : str
        Metric name (case-insensitive for common metrics)

    Returns
    -------
    callable or None
        The metric function, or None if not found

    Examples
    --------
    >>> kge_func = get_metric_function('KGE')
    >>> result = kge_func(observed, simulated)
    """
    if name in METRIC_REGISTRY:
        return cast(Callable, METRIC_REGISTRY[name]['function'])
    # Try lowercase
    if name.lower() in METRIC_REGISTRY:
        return cast(Callable, METRIC_REGISTRY[name.lower()]['function'])
    return None


def get_metric_info(name: str) -> Optional[MetricInfo]:
    """
    Get metadata for a metric by name.

    Parameters
    ----------
    name : str
        Metric name

    Returns
    -------
    MetricInfo or None
        Metric metadata, or None if not found
    """
    if name in METRIC_REGISTRY:
        return cast(MetricInfo, METRIC_REGISTRY[name]['info'])
    if name.lower() in METRIC_REGISTRY:
        return cast(MetricInfo, METRIC_REGISTRY[name.lower()]['info'])
    return None


def list_available_metrics() -> List[str]:
    """
    List all available metric names.

    Returns
    -------
    list of str
        Sorted list of metric names (excluding aliases)
    """
    # Return only primary names, not aliases
    primary_names = [
        'NSE', 'logNSE', 'KGE', 'KGEp', 'KGEnp', 'VE',
        'RMSE', 'NRMSE', 'MAE', 'MARE',
        'bias', 'PBIAS', 'correlation', 'R2'
    ]
    return primary_names


# =============================================================================
# Hydrograph Signature Metrics
# =============================================================================

def peak_timing_error(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    threshold_percentile: float = 95.0
) -> Dict[str, float]:
    """
    Calculate peak timing error metrics.

    Identifies peaks above threshold and computes timing differences.

    Parameters
    ----------
    observed : array-like
        Observed streamflow values
    simulated : array-like
        Simulated streamflow values
    time_index : array-like, optional
        Time index (if None, uses integer indices)
    threshold_percentile : float
        Percentile threshold for peak identification (default: 95)

    Returns
    -------
    dict
        Dictionary with:
        - mean_timing_error: Mean timing error (positive = late, negative = early)
        - abs_timing_error: Mean absolute timing error
        - n_peaks: Number of peaks identified
        - peak_magnitude_error: Mean relative error in peak magnitude
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 10:
        return {'mean_timing_error': np.nan, 'abs_timing_error': np.nan,
                'n_peaks': 0, 'peak_magnitude_error': np.nan}

    # Identify peaks in observed data
    threshold = np.percentile(obs, threshold_percentile)
    obs_peaks = []

    for i in range(1, len(obs) - 1):
        if obs[i] > threshold and obs[i] > obs[i-1] and obs[i] > obs[i+1]:
            obs_peaks.append(i)

    if len(obs_peaks) == 0:
        return {'mean_timing_error': np.nan, 'abs_timing_error': np.nan,
                'n_peaks': 0, 'peak_magnitude_error': np.nan}

    timing_errors = []
    magnitude_errors = []

    for obs_peak_idx in obs_peaks:
        # Search window around observed peak for simulated peak
        window_start = max(0, obs_peak_idx - 5)
        window_end = min(len(sim), obs_peak_idx + 6)

        sim_window = sim[window_start:window_end]
        if len(sim_window) == 0:
            continue

        sim_peak_local = np.argmax(sim_window)
        sim_peak_idx = window_start + sim_peak_local

        # Timing error (positive = simulated peak is late)
        timing_errors.append(sim_peak_idx - obs_peak_idx)

        # Magnitude error
        if obs[obs_peak_idx] > 0:
            magnitude_errors.append(
                (sim[sim_peak_idx] - obs[obs_peak_idx]) / obs[obs_peak_idx]
            )

    if len(timing_errors) == 0:
        return {'mean_timing_error': np.nan, 'abs_timing_error': np.nan,
                'n_peaks': 0, 'peak_magnitude_error': np.nan}

    return {
        'mean_timing_error': float(np.mean(timing_errors)),
        'abs_timing_error': float(np.mean(np.abs(timing_errors))),
        'n_peaks': len(timing_errors),
        'peak_magnitude_error': float(np.mean(magnitude_errors)) if magnitude_errors else np.nan
    }


def recession_constant(
    flow: Union[np.ndarray, pd.Series],
    method: str = 'linear'
) -> float:
    """
    Calculate recession constant (K) from streamflow data.

    The recession constant describes how quickly streamflow decreases
    during recession periods: Q(t) = Q0 * K^t

    Parameters
    ----------
    flow : array-like
        Streamflow time series
    method : str
        Method for fitting: 'linear' (log-linear regression) or 'ratio'

    Returns
    -------
    float
        Recession constant K (typically 0.9-0.99 for daily data)
        K closer to 1 = slower recession (more baseflow contribution)
    """
    flow = np.array(flow)
    flow = flow[~np.isnan(flow)]

    if len(flow) < 10:
        return np.nan

    # Identify recession periods (3+ consecutive decreasing values)
    recession_ratios = []

    i = 0
    while i < len(flow) - 3:
        # Check for recession start
        if flow[i+1] < flow[i] and flow[i+2] < flow[i+1] and flow[i+3] < flow[i+2]:
            # Find end of recession
            j = i + 1
            while j < len(flow) - 1 and flow[j+1] < flow[j]:
                j += 1

            # Calculate recession ratios for this period
            if j - i >= 3:
                for k in range(i, j):
                    if flow[k] > 0:
                        recession_ratios.append(flow[k+1] / flow[k])

            i = j
        else:
            i += 1

    if len(recession_ratios) < 5:
        return np.nan

    # Return median recession constant
    return float(np.median(recession_ratios))


def baseflow_index(
    flow: Union[np.ndarray, pd.Series],
    filter_param: float = 0.925
) -> float:
    """
    Calculate Baseflow Index (BFI) using digital filter method.

    BFI = baseflow volume / total flow volume

    Parameters
    ----------
    flow : array-like
        Streamflow time series
    filter_param : float
        Filter parameter (alpha), typically 0.925 for daily data.
        Higher values = more smoothing = higher baseflow.

    Returns
    -------
    float
        Baseflow index (0-1), higher values indicate groundwater-dominated system

    References
    ----------
    Eckhardt, K. (2005). How to construct recursive digital filters for
    baseflow separation. Hydrological Processes, 19(2), 507-515.
    """
    flow = np.array(flow)
    flow = flow[~np.isnan(flow)]

    if len(flow) < 10:
        return np.nan

    # Single-pass Lyne-Hollick filter
    alpha = filter_param
    quickflow = np.zeros_like(flow)
    quickflow[0] = 0

    for i in range(1, len(flow)):
        quickflow[i] = alpha * quickflow[i-1] + (1 + alpha) / 2 * (flow[i] - flow[i-1])
        quickflow[i] = max(0, min(quickflow[i], flow[i]))

    baseflow = flow - quickflow

    total_flow = np.sum(flow)
    if total_flow == 0:
        return np.nan

    return float(np.sum(baseflow) / total_flow)


def flow_duration_curve_metrics(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate Flow Duration Curve (FDC) based metrics.

    Parameters
    ----------
    observed : array-like
        Observed streamflow values
    simulated : array-like
        Simulated streamflow values

    Returns
    -------
    dict
        Dictionary with:
        - fdc_slope: Slope of FDC (log scale) - steeper = more flashy
        - fdc_bias_low: Bias in low flows (Q70-Q99)
        - fdc_bias_mid: Bias in mid flows (Q30-Q70)
        - fdc_bias_high: Bias in high flows (Q1-Q30)
    """
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 30:
        return {'fdc_slope': np.nan, 'fdc_bias_low': np.nan,
                'fdc_bias_mid': np.nan, 'fdc_bias_high': np.nan}

    # Sort flows (descending for FDC)
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    n = len(obs_sorted)

    # Exceedance probabilities (used implicitly via indices)
    _ = np.arange(1, n + 1) / (n + 1)  # noqa: F841

    # FDC slope (using log-transformed flows, Q20 to Q70)
    idx_20 = int(0.2 * n)
    idx_70 = int(0.7 * n)

    if obs_sorted[idx_20] > 0 and obs_sorted[idx_70] > 0:
        log_q20 = np.log10(obs_sorted[idx_20])
        log_q70 = np.log10(obs_sorted[idx_70])
        fdc_slope = (log_q20 - log_q70) / (0.7 - 0.2)
    else:
        fdc_slope = np.nan

    # Flow segment biases
    def calc_bias(start_pct, end_pct):
        start_idx = int(start_pct * n)
        end_idx = int(end_pct * n)
        if end_idx <= start_idx:
            return np.nan
        obs_seg = obs_sorted[start_idx:end_idx]
        sim_seg = sim_sorted[start_idx:end_idx]
        if np.sum(obs_seg) == 0:
            return np.nan
        return float((np.sum(sim_seg) - np.sum(obs_seg)) / np.sum(obs_seg) * 100)

    return {
        'fdc_slope': float(fdc_slope) if not np.isnan(fdc_slope) else np.nan,
        'fdc_bias_low': calc_bias(0.70, 0.99),   # Low flows
        'fdc_bias_mid': calc_bias(0.30, 0.70),   # Mid flows
        'fdc_bias_high': calc_bias(0.01, 0.30),  # High flows
    }


def hydrograph_signatures(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive hydrograph signature metrics.

    This is a convenience function that computes all signature metrics
    and compares observed vs simulated signatures.

    Parameters
    ----------
    observed : array-like
        Observed streamflow values
    simulated : array-like
        Simulated streamflow values
    time_index : array-like, optional
        Time index for peak timing calculations

    Returns
    -------
    dict
        Dictionary with all signature metrics and comparison values
    """
    obs, sim = _clean_data(observed, simulated)

    result = {}

    # Peak timing
    peak_metrics = peak_timing_error(obs, sim, time_index)
    result.update({f'peak_{k}': v for k, v in peak_metrics.items()})

    # Recession constants
    result['recession_k_obs'] = recession_constant(obs)
    result['recession_k_sim'] = recession_constant(sim)
    if not np.isnan(result['recession_k_obs']) and not np.isnan(result['recession_k_sim']):
        result['recession_k_error'] = result['recession_k_sim'] - result['recession_k_obs']
    else:
        result['recession_k_error'] = np.nan

    # Baseflow indices
    result['bfi_obs'] = baseflow_index(obs)
    result['bfi_sim'] = baseflow_index(sim)
    if not np.isnan(result['bfi_obs']) and not np.isnan(result['bfi_sim']):
        result['bfi_error'] = result['bfi_sim'] - result['bfi_obs']
    else:
        result['bfi_error'] = np.nan

    # FDC metrics
    fdc_metrics = flow_duration_curve_metrics(obs, sim)
    result.update(fdc_metrics)

    # Flow variability (coefficient of variation)
    if np.mean(obs) > 0:
        result['cv_obs'] = float(np.std(obs) / np.mean(obs))
    else:
        result['cv_obs'] = np.nan

    if np.mean(sim) > 0:
        result['cv_sim'] = float(np.std(sim) / np.mean(sim))
    else:
        result['cv_sim'] = np.nan

    return result


def interpret_metric(name: str, value: float) -> str:
    """
    Provide a human-readable interpretation of a metric value.

    Parameters
    ----------
    name : str
        Metric name
    value : float
        Metric value

    Returns
    -------
    str
        Interpretation string

    Examples
    --------
    >>> print(interpret_metric('KGE', 0.75))
    'KGE = 0.750: Good (0.75 is considered good performance)'
    """
    info = get_metric_info(name)
    if info is None:
        return f"{name} = {value:.3f}: Unknown metric"

    if np.isnan(value):
        return f"{name} = NaN: Could not be calculated (insufficient data or invalid values)"

    # Determine performance category based on metric type
    if info.direction == 'maximize':
        if info.optimal == 1.0:
            if value >= 0.9:
                category = "Excellent"
            elif value >= 0.75:
                category = "Good"
            elif value >= 0.5:
                category = "Satisfactory"
            elif value >= 0.0:
                category = "Poor"
            else:
                category = "Unsatisfactory"
        else:
            category = "N/A"
    else:  # minimize
        if info.optimal == 0.0:
            if value == 0:
                category = "Perfect"
            elif name == 'PBIAS':
                abs_val = abs(value)
                if abs_val < 10:
                    category = "Excellent"
                elif abs_val < 25:
                    category = "Good"
                elif abs_val < 50:
                    category = "Satisfactory"
                else:
                    category = "Poor"
            else:
                # For error metrics, interpretation depends on scale
                category = "See context"
        else:
            category = "N/A"

    return f"{name} = {value:.3f}: {category}"
