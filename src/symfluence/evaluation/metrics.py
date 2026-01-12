#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hydrological Performance Metrics

This module provides standardized performance metrics for model evaluation including:
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE, KGE', KGEnp)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Bias metrics (PBIAS, relative bias)
- Correlation metrics

All functions handle NaN values automatically and support optional transformations.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple
from scipy import stats


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
        Transformed (observed, simulated) arrays
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

    return obs_trans, sim_trans


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
    kge_result = kge(observed, simulated, transfo, return_components=True)

    return {
        'NSE': nse(observed, simulated, transfo),
        'KGE': kge_result['KGE'],
        'KGEp': kge_prime(observed, simulated, transfo),
        'KGEnp': kge_np(observed, simulated, transfo),
        'RMSE': rmse(observed, simulated, transfo),
        'NRMSE': nrmse(observed, simulated, transfo),
        'MAE': mae(observed, simulated, transfo),
        'PBIAS': pbias(observed, simulated, transfo),
        'bias': bias(observed, simulated, transfo),
        'correlation': correlation(observed, simulated),
        'r': kge_result['r'],
        'alpha': kge_result['alpha'],
        'beta': kge_result['beta']
    }
