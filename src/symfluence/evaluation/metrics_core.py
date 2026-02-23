"""Core hydrological performance metric implementations."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "_clean_data",
    "_apply_transformation",
    "_prepare_metric_inputs",
    "nse",
    "log_nse",
    "kge",
    "kge_prime",
    "kge_np",
    "rmse",
    "nrmse",
    "mae",
    "mare",
    "bias",
    "pbias",
    "correlation",
    "r_squared",
    "volumetric_efficiency",
    "calculate_all_metrics",
]


def _clean_data(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean and align observed and simulated data by removing NaN values."""
    obs = np.array(observed)
    sim = np.array(simulated)

    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    return obs[valid_mask], sim[valid_mask]


def _apply_transformation(
    observed: np.ndarray,
    simulated: np.ndarray,
    transfo: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply power transformation to observed and simulated data."""
    if transfo == 1.0:
        return observed, simulated

    if transfo < 0:
        epsilon = np.mean(observed) / 100
        obs_trans = (epsilon + observed) ** transfo
        sim_trans = (epsilon + simulated) ** transfo
    else:
        obs_trans = observed**transfo
        sim_trans = simulated**transfo

    valid_mask = ~(
        np.isnan(obs_trans)
        | np.isnan(sim_trans)
        | np.isinf(obs_trans)
        | np.isinf(sim_trans)
    )
    return obs_trans[valid_mask], sim_trans[valid_mask]


def _prepare_metric_inputs(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
    min_length: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return cleaned/transformed arrays when enough valid data exist."""
    obs, sim = _clean_data(observed, simulated)
    if len(obs) < min_length:
        return None

    obs, sim = _apply_transformation(obs, sim, transfo)
    if len(obs) < min_length:
        return None

    return obs, sim


def _near_zero(value: float, obs: np.ndarray) -> bool:
    """Check if value is near zero relative to the scale of observations."""
    scale = np.mean(np.abs(obs))
    if scale == 0:
        return True
    return abs(value) < 1e-10 * scale * scale * len(obs)


def _safe_pearson_correlation(obs: np.ndarray, sim: np.ndarray) -> float:
    """Return Pearson r, or NaN when variance is near zero."""
    if _near_zero(np.sum((obs - np.mean(obs)) ** 2), obs):
        return np.nan
    if _near_zero(np.sum((sim - np.mean(sim)) ** 2), sim):
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def nse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate Nash-Sutcliffe Efficiency."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    mean_obs = np.mean(obs)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - mean_obs) ** 2)

    if _near_zero(denominator, obs):
        return np.nan

    return 1.0 - (numerator / denominator)


def kge(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
    return_components: bool = False,
) -> Union[float, Dict[str, float]]:
    """Calculate Kling-Gupta Efficiency."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        if return_components:
            return {"KGE": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
        return np.nan
    obs, sim = prepared

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)

    r = _safe_pearson_correlation(obs, sim)
    alpha = std_sim / std_obs if std_obs != 0 else np.nan
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge_value = np.nan
    else:
        kge_value = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    if return_components:
        return {
            "KGE": float(kge_value),
            "r": float(r),
            "alpha": float(alpha),
            "beta": float(beta),
        }

    return float(kge_value)


def kge_prime(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
    return_components: bool = False,
) -> Union[float, Dict[str, float]]:
    """Calculate modified Kling-Gupta Efficiency (KGE')."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        if return_components:
            return {"KGEp": np.nan, "r": np.nan, "gamma": np.nan, "beta": np.nan}
        return np.nan
    obs, sim = prepared

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)

    r = _safe_pearson_correlation(obs, sim)

    cv_obs = std_obs / mean_obs if mean_obs != 0 else np.nan
    cv_sim = std_sim / mean_sim if mean_sim != 0 else np.nan
    gamma = cv_sim / cv_obs if cv_obs != 0 else np.nan

    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    if np.isnan(r) or np.isnan(gamma) or np.isnan(beta):
        kgep_value = np.nan
    else:
        kgep_value = 1.0 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    if return_components:
        return {
            "KGEp": float(kgep_value),
            "r": float(r),
            "gamma": float(gamma),
            "beta": float(beta),
        }

    return float(kgep_value)


def kge_np(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate Non-Parametric Kling-Gupta Efficiency (KGEnp)."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    r_np = stats.spearmanr(obs, sim)[0]

    fdc_obs = np.sort(obs) / np.sum(obs) if mean_obs != 0 else np.nan
    fdc_sim = np.sort(sim) / np.sum(sim) if mean_sim != 0 else np.nan
    alpha_np = 1.0 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))

    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan

    if np.isnan(r_np) or np.isnan(alpha_np) or np.isnan(beta):
        return np.nan

    return float(1.0 - np.sqrt((r_np - 1) ** 2 + (alpha_np - 1) ** 2 + (beta - 1) ** 2))


def rmse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate Root Mean Square Error."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def nrmse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate Normalized Root Mean Square Error."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    rmse_value = np.sqrt(np.mean((obs - sim) ** 2))
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    if _near_zero(denominator, obs):
        return np.nan

    std_obs = np.sqrt(denominator / len(obs))
    return float(rmse_value / std_obs)


def mae(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate Mean Absolute Error."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    return float(np.mean(np.abs(obs - sim)))


def bias(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate bias (mean error)."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    return float(np.mean(sim) - np.mean(obs))


def pbias(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    transfo: float = 1.0,
) -> float:
    """Calculate percent bias."""
    prepared = _prepare_metric_inputs(observed, simulated, transfo=transfo)
    if prepared is None:
        return np.nan
    obs, sim = prepared

    sum_obs = np.sum(obs)
    if sum_obs == 0:
        return np.nan

    return float(100.0 * (np.sum(sim) - sum_obs) / sum_obs)


def correlation(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    method: str = "pearson",
) -> float:
    """Calculate correlation coefficient."""
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 2:
        return np.nan

    if method == "pearson":
        return _safe_pearson_correlation(obs, sim)
    if method == "spearman":
        return float(stats.spearmanr(obs, sim)[0])
    raise ValueError(f"Unknown correlation method: {method}")


def log_nse(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    epsilon: Optional[float] = None,
) -> float:
    """Calculate log-transformed Nash-Sutcliffe Efficiency."""
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    if epsilon is None:
        epsilon = np.mean(obs) * 0.01 if np.mean(obs) > 0 else 1e-6

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

    if _near_zero(denominator, obs_log):
        return np.nan

    return float(1.0 - (numerator / denominator))


def r_squared(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate coefficient of determination (R²)."""
    r = correlation(observed, simulated, method="pearson")
    if np.isnan(r):
        return np.nan
    return float(r**2)


def mare(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    epsilon: Optional[float] = None,
) -> float:
    """Calculate Mean Absolute Relative Error."""
    obs, sim = _clean_data(observed, simulated)

    if len(obs) == 0:
        return np.nan

    if epsilon is None:
        epsilon = np.mean(np.abs(obs)) * 0.01 if np.mean(np.abs(obs)) > 0 else 1e-6

    denominator = np.abs(obs) + epsilon
    relative_errors = np.abs(obs - sim) / denominator

    return float(np.mean(relative_errors))


def volumetric_efficiency(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate Volumetric Efficiency (VE)."""
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
    transfo: float = 1.0,
) -> Dict[str, float]:
    """Calculate all standard performance metrics in one call."""
    kge_result = cast(Dict[str, float], kge(observed, simulated, transfo, return_components=True))

    return {
        "NSE": float(nse(observed, simulated, transfo)),
        "logNSE": float(log_nse(observed, simulated)),
        "KGE": float(kge_result["KGE"]),
        "KGEp": float(kge_prime(observed, simulated, transfo)),
        "KGEnp": float(kge_np(observed, simulated, transfo)),
        "VE": float(volumetric_efficiency(observed, simulated)),
        "RMSE": float(rmse(observed, simulated, transfo)),
        "NRMSE": float(nrmse(observed, simulated, transfo)),
        "MAE": float(mae(observed, simulated, transfo)),
        "MARE": float(mare(observed, simulated)),
        "PBIAS": float(pbias(observed, simulated, transfo)),
        "bias": float(bias(observed, simulated, transfo)),
        "correlation": float(correlation(observed, simulated)),
        "R2": float(r_squared(observed, simulated)),
        "r": float(kge_result["r"]),
        "alpha": float(kge_result["alpha"]),
        "beta": float(kge_result["beta"]),
    }
