"""Hydrograph-signature metric implementations."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics_core import _clean_data

__all__ = [
    "peak_timing_error",
    "recession_constant",
    "baseflow_index",
    "flow_duration_curve_metrics",
    "hydrograph_signatures",
]


def peak_timing_error(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, float]:
    """Calculate peak timing error metrics."""
    del time_index  # Currently unused; preserved for API compatibility.
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 10:
        return {
            "mean_timing_error": np.nan,
            "abs_timing_error": np.nan,
            "n_peaks": 0,
            "peak_magnitude_error": np.nan,
        }

    threshold = np.percentile(obs, threshold_percentile)
    obs_peaks = []

    for i in range(1, len(obs) - 1):
        if obs[i] > threshold and obs[i] > obs[i - 1] and obs[i] > obs[i + 1]:
            obs_peaks.append(i)

    if len(obs_peaks) == 0:
        return {
            "mean_timing_error": np.nan,
            "abs_timing_error": np.nan,
            "n_peaks": 0,
            "peak_magnitude_error": np.nan,
        }

    timing_errors = []
    magnitude_errors = []

    for obs_peak_idx in obs_peaks:
        window_start = max(0, obs_peak_idx - 5)
        window_end = min(len(sim), obs_peak_idx + 6)

        sim_window = sim[window_start:window_end]
        if len(sim_window) == 0:
            continue

        sim_peak_local = np.argmax(sim_window)
        sim_peak_idx = window_start + sim_peak_local

        timing_errors.append(sim_peak_idx - obs_peak_idx)

        if obs[obs_peak_idx] > 0:
            magnitude_errors.append((sim[sim_peak_idx] - obs[obs_peak_idx]) / obs[obs_peak_idx])

    if len(timing_errors) == 0:
        return {
            "mean_timing_error": np.nan,
            "abs_timing_error": np.nan,
            "n_peaks": 0,
            "peak_magnitude_error": np.nan,
        }

    return {
        "mean_timing_error": float(np.mean(timing_errors)),
        "abs_timing_error": float(np.mean(np.abs(timing_errors))),
        "n_peaks": len(timing_errors),
        "peak_magnitude_error": float(np.mean(magnitude_errors)) if magnitude_errors else np.nan,
    }


def recession_constant(
    flow: Union[np.ndarray, pd.Series],
    method: str = "linear",
) -> float:
    """Calculate recession constant (K) from streamflow data."""
    del method  # Reserved for future extension.

    flow_array = np.array(flow)
    flow_array = flow_array[~np.isnan(flow_array)]

    if len(flow_array) < 10:
        return np.nan

    recession_ratios = []

    i = 0
    while i < len(flow_array) - 3:
        if (
            flow_array[i + 1] < flow_array[i]
            and flow_array[i + 2] < flow_array[i + 1]
            and flow_array[i + 3] < flow_array[i + 2]
        ):
            j = i + 1
            while j < len(flow_array) - 1 and flow_array[j + 1] < flow_array[j]:
                j += 1

            if j - i >= 3:
                for k in range(i, j):
                    if flow_array[k] > 0:
                        recession_ratios.append(flow_array[k + 1] / flow_array[k])

            i = j
        else:
            i += 1

    if len(recession_ratios) < 5:
        return np.nan

    return float(np.median(recession_ratios))


def baseflow_index(
    flow: Union[np.ndarray, pd.Series],
    filter_param: float = 0.925,
) -> float:
    """Calculate Baseflow Index (BFI) using digital filter method."""
    flow_array = np.array(flow)
    flow_array = flow_array[~np.isnan(flow_array)]

    if len(flow_array) < 10:
        return np.nan

    alpha = filter_param
    quickflow = np.zeros_like(flow_array)
    quickflow[0] = 0

    for i in range(1, len(flow_array)):
        quickflow[i] = alpha * quickflow[i - 1] + (1 + alpha) / 2 * (flow_array[i] - flow_array[i - 1])
        quickflow[i] = max(0, min(quickflow[i], flow_array[i]))

    baseflow = flow_array - quickflow

    total_flow = np.sum(flow_array)
    if total_flow == 0:
        return np.nan

    return float(np.sum(baseflow) / total_flow)


def flow_duration_curve_metrics(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate Flow Duration Curve (FDC) based metrics."""
    obs, sim = _clean_data(observed, simulated)

    if len(obs) < 30:
        return {
            "fdc_slope": np.nan,
            "fdc_bias_low": np.nan,
            "fdc_bias_mid": np.nan,
            "fdc_bias_high": np.nan,
        }

    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    n = len(obs_sorted)

    _ = np.arange(1, n + 1) / (n + 1)

    idx_20 = int(0.2 * n)
    idx_70 = int(0.7 * n)

    if obs_sorted[idx_20] > 0 and obs_sorted[idx_70] > 0:
        log_q20 = np.log10(obs_sorted[idx_20])
        log_q70 = np.log10(obs_sorted[idx_70])
        fdc_slope = (log_q20 - log_q70) / (0.7 - 0.2)
    else:
        fdc_slope = np.nan

    def calc_bias(start_pct: float, end_pct: float) -> float:
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
        "fdc_slope": float(fdc_slope) if not np.isnan(fdc_slope) else np.nan,
        "fdc_bias_low": calc_bias(0.70, 0.99),
        "fdc_bias_mid": calc_bias(0.30, 0.70),
        "fdc_bias_high": calc_bias(0.01, 0.30),
    }


def hydrograph_signatures(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
) -> Dict[str, float]:
    """Calculate comprehensive hydrograph signature metrics."""
    obs, sim = _clean_data(observed, simulated)

    result: Dict[str, float] = {}

    peak_metrics = peak_timing_error(obs, sim, time_index)
    result.update({f"peak_{k}": v for k, v in peak_metrics.items()})

    result["recession_k_obs"] = recession_constant(obs)
    result["recession_k_sim"] = recession_constant(sim)
    if not np.isnan(result["recession_k_obs"]) and not np.isnan(result["recession_k_sim"]):
        result["recession_k_error"] = result["recession_k_sim"] - result["recession_k_obs"]
    else:
        result["recession_k_error"] = np.nan

    result["bfi_obs"] = baseflow_index(obs)
    result["bfi_sim"] = baseflow_index(sim)
    if not np.isnan(result["bfi_obs"]) and not np.isnan(result["bfi_sim"]):
        result["bfi_error"] = result["bfi_sim"] - result["bfi_obs"]
    else:
        result["bfi_error"] = np.nan

    fdc_metrics = flow_duration_curve_metrics(obs, sim)
    result.update(fdc_metrics)

    if np.mean(obs) > 0:
        result["cv_obs"] = float(np.std(obs) / np.mean(obs))
    else:
        result["cv_obs"] = np.nan

    if np.mean(sim) > 0:
        result["cv_sim"] = float(np.std(sim) / np.mean(sim))
    else:
        result["cv_sim"] = np.nan

    return result
