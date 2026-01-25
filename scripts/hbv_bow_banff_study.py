#!/usr/bin/env python3
"""
HBV Model Calibration Study: Bow at Banff

Demonstrates the HBV model calibration workflow comparing discrete (lax.scan)
and ODE (diffrax with adjoint) solver implementations.

Study Design:
- Catchment: Bow River at Banff (2210 km², glacierized, Alberta, Canada)
- Period: 2000-2009 (10 years)
- Calibration: 2000-2004 (5 years)
- Validation: 2005-2009 (5 years)
- Warmup: 365 days (excluded from metrics)
- Forcing: RDRS v2.1 (Regional Deterministic Reanalysis System)
- Observations: WSC gauge 05BB001

Usage:
    python scripts/hbv_bow_banff_study.py [--output-dir OUTPUT_DIR] [--wrr-figure]

Options:
    --output-dir    Output directory for results (default: ~/Desktop)
    --wrr-figure    Generate WRR publication figure with convergence and sensitivity

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import warnings
import xarray as xr

warnings.filterwarnings('ignore')

# JAX configuration
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# HBV imports
from symfluence.models.hbv.model import simulate, simulate_jax
from symfluence.models.hbv.parameters import (
    DEFAULT_PARAMS, HBVParameters,
    create_params_from_dict, scale_params_for_timestep
)
from symfluence.models.hbv.hbv_ode import simulate_ode_with_routing, HAS_DIFFRAX


# =============================================================================
# CONFIGURATION
# =============================================================================

CATCHMENT_NAME = "Bow at Banff"
CATCHMENT_AREA_KM2 = 2210.0
WSC_GAUGE = "05BB001"

DATA_ROOT = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data")
DOMAIN_DIR = DATA_ROOT / "domain_Bow_at_Banff_lumped_rdrs"
FORCING_FILE = DOMAIN_DIR / "forcing/HBV_input/Bow_at_Banff_lumped_rdrs_hbv_forcing_24h.nc"
OBS_FILE = DOMAIN_DIR / "observations/streamflow/preprocessed/Bow_at_Banff_lumped_rdrs_streamflow_processed.csv"

CAL_START = "2000-01-01"
CAL_END = "2004-12-31"
VAL_END = "2009-12-31"
WARMUP_DAYS = 365

PARAM_NAMES = ['tt', 'cfmax', 'sfcf', 'cfr', 'cwh', 'fc', 'lp', 'beta',
               'k0', 'k1', 'k2', 'uzl', 'perc', 'maxbas']

# Extended bounds for glacierized catchments
EXTENDED_BOUNDS = {
    'tt': (-3.0, 5.0),
    'cfmax': (0.5, 15.0),
    'sfcf': (0.5, 1.5),
    'cfr': (0.0, 0.1),
    'cwh': (0.0, 0.2),
    'fc': (50.0, 800.0),
    'lp': (0.3, 1.0),
    'beta': (1.0, 6.0),
    'k0': (0.01, 0.99),
    'k1': (0.001, 0.5),
    'k2': (0.0001, 0.1),
    'uzl': (0.0, 100.0),
    'perc': (0.0, 15.0),
    'maxbas': (1.0, 10.0),
}

# Optimizer settings
DE_MAXITER = 100
DE_POPSIZE = 15
DE_TOL = 0.001
DE_SEED = 42
LOCAL_RESTARTS = 3

# WRR Figure Settings (colorblind-friendly palette)
WRR_COLORS = {
    'observed': '#000000',      # Black
    'discrete': '#0072B2',      # Blue (colorblind safe)
    'ode': '#D55E00',           # Orange (colorblind safe)
    'cal_period': '#0072B2',    # Blue
    'val_period': '#E69F00',    # Yellow-orange
    'surface': 'viridis',       # Colorblind-safe colormap
}
WRR_FIGURE_WIDTH_MM = 170  # Full page width
WRR_FIGURE_HEIGHT_MM = 180
WRR_DPI = 600
WRR_FONT_SIZE = 8
WRR_TITLE_SIZE = 9


# =============================================================================
# CONVERGENCE TRACKING
# =============================================================================

@dataclass
class ConvergenceHistory:
    """Container for optimization convergence history."""
    iterations: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    params: List[np.ndarray] = field(default_factory=list)
    gradients: List[Optional[np.ndarray]] = field(default_factory=list)
    method: str = "unknown"

    def record(self, iteration: int, loss: float, params: np.ndarray,
               gradient: Optional[np.ndarray] = None) -> None:
        """Record a single optimization step."""
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.params.append(params.copy())
        self.gradients.append(gradient.copy() if gradient is not None else None)


class ConvergenceCallback:
    """Callback for tracking optimization convergence in differential_evolution."""

    def __init__(self, objective_fn: Callable, history: ConvergenceHistory):
        self.objective_fn = objective_fn
        self.history = history
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None

    def __call__(self, xk: np.ndarray, convergence: float = None) -> bool:
        """Called by differential_evolution at each iteration."""
        loss = float(self.objective_fn(xk))
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = xk.copy()
        self.history.record(self.iteration, self.best_loss, xk)
        self.iteration += 1
        return False  # False = continue optimization


class GradientDescentCallback:
    """Callback for tracking gradient descent optimization."""

    def __init__(self, objective_fn: Callable, history: ConvergenceHistory):
        self.objective_fn = objective_fn
        self.history = history
        self.iteration = 0

    def __call__(self, xk: np.ndarray) -> None:
        """Called by minimize at each iteration."""
        loss = float(self.objective_fn(xk))
        self.history.record(self.iteration, loss, xk)
        self.iteration += 1


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(sim: np.ndarray, obs: np.ndarray, warmup: int = 0) -> Dict[str, float]:
    """Calculate all performance metrics."""
    sim_eval = sim[warmup:]
    obs_eval = obs[warmup:]
    mask = ~np.isnan(obs_eval) & ~np.isnan(sim_eval)

    if mask.sum() < 10:
        return {'nse': -999, 'kge': -999, 'r': 0, 'alpha': 0, 'beta': 0, 'rmse': np.inf}

    sim_m, obs_m = sim_eval[mask], obs_eval[mask]

    # NSE
    ss_res = np.sum((sim_m - obs_m)**2)
    ss_tot = np.sum((obs_m - np.mean(obs_m))**2)
    nse = 1 - ss_res / (ss_tot + 1e-10)

    # KGE components
    r = np.corrcoef(sim_m, obs_m)[0, 1]
    alpha = np.std(sim_m) / (np.std(obs_m) + 1e-10)
    beta = np.mean(sim_m) / (np.mean(obs_m) + 1e-10)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # RMSE
    rmse = np.sqrt(np.mean((sim_m - obs_m)**2))

    return {'nse': nse, 'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse': rmse}


# =============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# =============================================================================

def compute_parameter_sensitivity(
    params_opt: Dict[str, float],
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    obs: np.ndarray,
    param_names: List[str],
    n_points: int = 15,
    warmup: int = 365
) -> Dict[str, Dict[str, Any]]:
    """
    Compute 1D sensitivity for parameters around optimum.

    Parameters
    ----------
    params_opt : dict
        Optimized parameters
    precip, temp, pet, obs : np.ndarray
        Forcing and observation data
    param_names : list
        Names of parameters to analyze
    n_points : int
        Number of points for each sweep
    warmup : int
        Warmup period in days

    Returns
    -------
    dict
        Dictionary with sensitivity curves for each parameter
    """
    from symfluence.models.hbv.model import simulate

    sensitivity = {}

    for param_name in param_names:
        if param_name not in EXTENDED_BOUNDS:
            continue

        bounds = EXTENDED_BOUNDS[param_name]
        opt_value = params_opt.get(param_name, DEFAULT_PARAMS.get(param_name))

        # Create sweep values around optimum (log-scale for some params)
        if param_name in ['k0', 'k1', 'k2', 'perc']:
            # Log-scale for rate parameters
            low = max(bounds[0], opt_value * 0.1)
            high = min(bounds[1], opt_value * 10)
            values = np.logspace(np.log10(low), np.log10(high), n_points)
        else:
            # Linear scale
            values = np.linspace(bounds[0], bounds[1], n_points)

        losses = []
        kges = []

        for value in values:
            test_params = params_opt.copy()
            test_params[param_name] = value

            try:
                runoff, _ = simulate(precip, temp, pet, test_params,
                                      use_jax=True, timestep_hours=24)
                metrics = calc_metrics(np.array(runoff), obs, warmup)
                losses.append(1 - metrics['kge'])  # Loss = 1 - KGE
                kges.append(metrics['kge'])
            except Exception:
                losses.append(np.nan)
                kges.append(np.nan)

        sensitivity[param_name] = {
            'values': values,
            'losses': np.array(losses),
            'kges': np.array(kges),
            'optimum': opt_value,
            'bounds': bounds
        }

    return sensitivity


def compute_2d_sensitivity(
    params_opt: Dict[str, float],
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    obs: np.ndarray,
    param1: str,
    param2: str,
    n_points: int = 15,
    warmup: int = 365
) -> Dict[str, Any]:
    """
    Compute 2D response surface for two parameters.

    Parameters
    ----------
    params_opt : dict
        Optimized parameters
    precip, temp, pet, obs : np.ndarray
        Forcing and observation data
    param1, param2 : str
        Names of parameters to analyze
    n_points : int
        Number of points per dimension
    warmup : int
        Warmup period in days

    Returns
    -------
    dict
        Dictionary with 2D response surface data
    """
    from symfluence.models.hbv.model import simulate

    bounds1 = EXTENDED_BOUNDS[param1]
    bounds2 = EXTENDED_BOUNDS[param2]

    values1 = np.linspace(bounds1[0], bounds1[1], n_points)
    values2 = np.linspace(bounds2[0], bounds2[1], n_points)

    kge_surface = np.zeros((n_points, n_points))

    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            test_params = params_opt.copy()
            test_params[param1] = v1
            test_params[param2] = v2

            try:
                runoff, _ = simulate(precip, temp, pet, test_params,
                                      use_jax=True, timestep_hours=24)
                metrics = calc_metrics(np.array(runoff), obs, warmup)
                kge_surface[j, i] = metrics['kge']  # Note: j,i for proper orientation
            except Exception:
                kge_surface[j, i] = np.nan

    return {
        'param1': param1,
        'param2': param2,
        'values1': values1,
        'values2': values2,
        'kge_surface': kge_surface,
        'optimum1': params_opt.get(param1, DEFAULT_PARAMS.get(param1)),
        'optimum2': params_opt.get(param2, DEFAULT_PARAMS.get(param2))
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, int]:
    """Load forcing and observation data."""
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print('='*60)

    ds = xr.open_dataset(FORCING_FILE)
    times_forcing = pd.to_datetime(ds['time'].values)

    mask = (times_forcing >= CAL_START) & (times_forcing <= VAL_END)
    times = times_forcing[mask]
    precip = ds['pr'].values[mask].astype(np.float64)
    temp = ds['temp'].values[mask].astype(np.float64)
    pet = ds['pet'].values[mask].astype(np.float64)
    ds.close()

    obs_df = pd.read_csv(OBS_FILE, parse_dates=['datetime'])
    obs_df = obs_df.set_index('datetime')
    obs_df = obs_df.reindex(times)
    obs_mm = (obs_df['discharge_cms'].values * 86400 / (CATCHMENT_AREA_KM2 * 1e6) * 1000).astype(np.float64)

    cal_end_idx = int(np.searchsorted(times.values, pd.Timestamp(CAL_END).value)) + 1

    print(f"Study period: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
    print(f"Calibration: {times[0].strftime('%Y-%m-%d')} to {times[cal_end_idx-1].strftime('%Y-%m-%d')} ({cal_end_idx} days)")
    print(f"Validation: {times[cal_end_idx].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')} ({len(times) - cal_end_idx} days)")
    print("\nData statistics:")
    print(f"  Precip: mean={np.mean(precip):.2f} mm/day, annual={np.mean(precip)*365:.0f} mm")
    print(f"  Temp: mean={np.mean(temp):.2f}°C, range=[{np.min(temp):.1f}, {np.max(temp):.1f}]°C")
    print(f"  PET: mean={np.mean(pet):.3f} mm/day")
    print(f"  Obs Q: mean={np.nanmean(obs_mm):.3f} mm/day, annual={np.nanmean(obs_mm)*365:.0f} mm")
    print(f"  Runoff ratio: {np.nanmean(obs_mm)/np.mean(precip):.2f}")

    return precip, temp, pet, obs_mm, times, cal_end_idx


# =============================================================================
# JIT-COMPILED CALIBRATION
# =============================================================================

def create_jit_objective(precip, temp, pet, obs, warmup=365):
    """Create JIT-compiled composite objective function."""
    precip_jax = jnp.array(precip)
    temp_jax = jnp.array(temp)
    pet_jax = jnp.array(pet)
    obs_jax = jnp.array(obs)

    @jax.jit
    def composite_loss(params_array):
        """Composite loss: KGE + log-KGE for full spectrum fit."""
        tt, cfmax, sfcf, cfr, cwh, fc, lp, beta, k0, k1, k2, uzl, perc, maxbas = params_array

        params = HBVParameters(
            tt=tt, cfmax=cfmax, sfcf=sfcf, cfr=cfr, cwh=cwh,
            fc=fc, lp=lp, beta=beta, k0=k0, k1=k1, k2=k2,
            uzl=uzl, perc=perc, maxbas=maxbas,
            smoothing=jnp.array(20.0),
            smoothing_enabled=jnp.array(True)
        )

        runoff, _ = simulate_jax(precip_jax, temp_jax, pet_jax, params,
                                  warmup_days=0, timestep_hours=24)

        sim_eval = runoff[warmup:]
        obs_eval = obs_jax[warmup:]

        # Standard KGE
        sim_mean = jnp.mean(sim_eval)
        obs_mean = jnp.mean(obs_eval)
        sim_std = jnp.std(sim_eval)
        obs_std = jnp.std(obs_eval)

        cov = jnp.mean((sim_eval - sim_mean) * (obs_eval - obs_mean))
        r = cov / (sim_std * obs_std + 1e-10)
        alpha = sim_std / (obs_std + 1e-10)
        beta_kge = sim_mean / (obs_mean + 1e-10)
        kge = 1.0 - jnp.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta_kge - 1)**2)

        # Log-transformed KGE for low flow emphasis
        sim_log = jnp.log(sim_eval + 0.01)
        obs_log = jnp.log(obs_eval + 0.01)

        sim_log_mean = jnp.mean(sim_log)
        obs_log_mean = jnp.mean(obs_log)
        sim_log_std = jnp.std(sim_log)
        obs_log_std = jnp.std(obs_log)

        cov_log = jnp.mean((sim_log - sim_log_mean) * (obs_log - obs_log_mean))
        r_log = cov_log / (sim_log_std * obs_log_std + 1e-10)
        alpha_log = sim_log_std / (obs_log_std + 1e-10)
        beta_log = sim_log_mean / (obs_log_mean + 1e-10)
        kge_log = 1.0 - jnp.sqrt((r_log - 1)**2 + (alpha_log - 1)**2 + (beta_log - 1)**2)

        # Weighted combination
        return -(0.7 * kge + 0.3 * kge_log)

    # Warm up JIT
    print("  Warming up JIT compilation...")
    x0 = jnp.array([DEFAULT_PARAMS[p] for p in PARAM_NAMES])
    _ = composite_loss(x0)
    print("  JIT compilation complete.")

    def objective(x):
        result = float(composite_loss(jnp.array(x)))
        if np.isnan(result) or result > 10:
            return 10.0
        return result

    return objective


def calibrate(precip, temp, pet, obs, warmup=365, verbose=True,
              track_convergence=False) -> Tuple[Dict[str, float], float, Optional[Dict[str, ConvergenceHistory]]]:
    """
    Calibrate HBV using two-stage optimization.

    Parameters
    ----------
    precip, temp, pet, obs : np.ndarray
        Forcing and observation data
    warmup : int
        Warmup period in days
    verbose : bool
        Print progress
    track_convergence : bool
        If True, track and return convergence history

    Returns
    -------
    params : dict
        Calibrated parameters
    best_kge : float
        Best KGE achieved
    convergence : dict or None
        Convergence history for DE and local optimization (if track_convergence=True)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CALIBRATION (JIT-accelerated)")
        print('='*60)
        print(f"Parameters: {PARAM_NAMES}")
        print("Objective: Composite KGE (0.7×KGE + 0.3×log-KGE)")

    objective = create_jit_objective(precip, temp, pet, obs, warmup)
    bounds = [(EXTENDED_BOUNDS[p][0], EXTENDED_BOUNDS[p][1]) for p in PARAM_NAMES]

    # Initialize convergence tracking
    convergence_histories = None
    de_callback = None
    if track_convergence:
        convergence_histories = {
            'de': ConvergenceHistory(method='differential_evolution'),
            'local': ConvergenceHistory(method='L-BFGS-B')
        }
        de_callback = ConvergenceCallback(objective, convergence_histories['de'])

    # Stage 1: Global search
    if verbose:
        print(f"\nStage 1: Global search (DE, maxiter={DE_MAXITER}, popsize={DE_POPSIZE})")
    t0 = time.time()
    result_de = differential_evolution(
        objective, bounds,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=DE_TOL,
        seed=DE_SEED,
        disp=verbose,
        workers=1,
        polish=False,
        callback=de_callback
    )
    de_time = time.time() - t0
    if verbose:
        print(f"  DE completed in {de_time:.1f}s, best loss: {result_de.fun:.4f}")

    # Stage 2: Multi-start local refinement
    if verbose:
        print(f"\nStage 2: Multi-start local refinement ({LOCAL_RESTARTS} starts)")
    best_result = result_de
    np.random.seed(DE_SEED + 1)

    # Track local optimization
    local_callback = None
    if track_convergence and convergence_histories is not None:
        local_callback = GradientDescentCallback(objective, convergence_histories['local'])
        # Record starting point from DE
        convergence_histories['local'].record(0, result_de.fun, result_de.x)

    for i in range(LOCAL_RESTARTS):
        x_start = result_de.x + np.random.randn(len(PARAM_NAMES)) * 0.1 * \
                  np.array([b[1]-b[0] for b in bounds])
        x_start = np.clip(x_start, [b[0] for b in bounds], [b[1] for b in bounds])

        result_local = minimize(
            objective, x_start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100},
            callback=local_callback if track_convergence else None
        )
        if result_local.fun < best_result.fun:
            best_result = result_local
            if verbose:
                print(f"    Start {i+1}: Improved to loss={result_local.fun:.4f}")

    # Build parameters
    params = DEFAULT_PARAMS.copy()
    params['smoothing_enabled'] = True
    params['smoothing'] = 20.0
    for i, name in enumerate(PARAM_NAMES):
        params[name] = best_result.x[i]

    # Calculate actual KGE for reporting
    runoff, _ = simulate(precip, temp, pet, params, use_jax=True, timestep_hours=24)
    metrics = calc_metrics(np.array(runoff), obs, warmup)

    if verbose:
        print("\nCalibration results:")
        print(f"  KGE: {metrics['kge']:.4f}")
        print(f"  NSE: {metrics['nse']:.4f}")
        print(f"  r: {metrics['r']:.4f}, alpha: {metrics['alpha']:.4f}, beta: {metrics['beta']:.4f}")
        print("\nCalibrated parameters:")
        for name in PARAM_NAMES:
            print(f"  {name}: {DEFAULT_PARAMS[name]:.4f} -> {params[name]:.4f}")

    return params, metrics['kge'], convergence_histories


# =============================================================================
# SIMULATION AND EVALUATION
# =============================================================================

def run_simulation(precip, temp, pet, params, method='discrete'):
    """Run HBV simulation."""
    if method == 'discrete':
        runoff, _ = simulate(precip, temp, pet, params, use_jax=True, timestep_hours=24)
        return np.array(runoff)
    elif method == 'ode':
        if not HAS_DIFFRAX:
            raise ImportError("diffrax not installed")
        scaled_params = scale_params_for_timestep(params, 24)
        params_obj = create_params_from_dict(scaled_params, use_jax=True)
        runoff, _ = simulate_ode_with_routing(precip, temp, pet, params_obj,
                                               timestep_hours=24, smoothing=params.get('smoothing', 20.0))
        return np.array(runoff)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(precip, temp, pet, obs, params, cal_end_idx, method='discrete', name="Model"):
    """Evaluate model on calibration and validation periods."""
    runoff = run_simulation(precip, temp, pet, params, method=method)
    cal_metrics = calc_metrics(runoff[:cal_end_idx], obs[:cal_end_idx], WARMUP_DAYS)
    val_metrics = calc_metrics(runoff[cal_end_idx:], obs[cal_end_idx:], warmup=0)
    return {
        'name': name,
        'method': method,
        'runoff': runoff,
        'cal': cal_metrics,
        'val': val_metrics
    }


# =============================================================================
# PLOTTING
# =============================================================================

def create_plots(results, obs, times, cal_end_idx, params_cal, output_dir):
    """Create all diagnostic plots."""
    times_dt = [datetime.strptime(str(t)[:10], '%Y-%m-%d') for t in times]

    # Main figure
    _fig = plt.figure(figsize=(16, 14))

    # 1. Full timeseries
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(times_dt, obs, 'k-', alpha=0.7, linewidth=1, label='Observed')
    for key, res in results.items():
        if 'Uncal' not in res['name']:
            style = '-' if res['method'] == 'discrete' else '--'
            ax1.plot(times_dt, res['runoff'], style, alpha=0.6, linewidth=0.8, label=res['name'])
    ax1.axvline(times_dt[cal_end_idx], color='gray', linestyle='--', linewidth=2)
    ax1.set_ylabel('Runoff (mm/day)')
    ax1.set_title(f'HBV Simulation - {CATCHMENT_NAME} ({CAL_START[:4]}-{VAL_END[:4]})',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.text(0.02, 0.95, 'Calibration', transform=ax1.transAxes, fontsize=9, va='top', fontweight='bold')
    ax1.text(0.55, 0.95, 'Validation', transform=ax1.transAxes, fontsize=9, va='top', fontweight='bold')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)

    # 2. Calibration detail
    cal_start, cal_end = WARMUP_DAYS, WARMUP_DAYS + 365
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(times_dt[cal_start:cal_end], obs[cal_start:cal_end], 'k-', alpha=0.8, linewidth=1.2, label='Observed')
    for key, res in results.items():
        if 'Uncal' not in res['name']:
            style = '-' if res['method'] == 'discrete' else '--'
            ax2.plot(times_dt[cal_start:cal_end], res['runoff'][cal_start:cal_end],
                    style, alpha=0.7, linewidth=1, label=res['name'])
    ax2.set_ylabel('Runoff (mm/day)')
    discrete_res = next((r for r in results.values() if r['method'] == 'discrete' and 'Uncal' not in r['name']), None)
    if discrete_res:
        ax2.set_title(f'Calibration Detail (Year 2) - KGE={discrete_res["cal"]["kge"]:.2f}',
                     fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.grid(True, alpha=0.3)

    # 3. Validation detail
    val_start = cal_end_idx
    val_end = min(cal_end_idx + 365, len(times_dt))
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(times_dt[val_start:val_end], obs[val_start:val_end], 'k-', alpha=0.8, linewidth=1.2, label='Observed')
    for key, res in results.items():
        if 'Uncal' not in res['name']:
            style = '-' if res['method'] == 'discrete' else '--'
            ax3.plot(times_dt[val_start:val_end], res['runoff'][val_start:val_end],
                    style, alpha=0.7, linewidth=1, label=res['name'])
    ax3.set_ylabel('Runoff (mm/day)')
    if discrete_res:
        ax3.set_title(f'Validation Detail (Year 1) - KGE={discrete_res["val"]["kge"]:.2f}',
                     fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3.grid(True, alpha=0.3)

    # 4. Performance table
    ax4 = plt.subplot(3, 2, 4)
    ax4.axis('off')
    table_data = [['Period', 'Method', 'NSE', 'KGE', 'r', 'α', 'β']]
    for key, res in results.items():
        if 'Uncal' not in res['name']:
            table_data.append(['Cal', res['name'].replace(' (Cal)', ''),
                              f'{res["cal"]["nse"]:.3f}', f'{res["cal"]["kge"]:.3f}',
                              f'{res["cal"]["r"]:.3f}', f'{res["cal"]["alpha"]:.3f}',
                              f'{res["cal"]["beta"]:.3f}'])
            table_data.append(['Val', res['name'].replace(' (Cal)', ''),
                              f'{res["val"]["nse"]:.3f}', f'{res["val"]["kge"]:.3f}',
                              f'{res["val"]["r"]:.3f}', f'{res["val"]["alpha"]:.3f}',
                              f'{res["val"]["beta"]:.3f}'])
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.12, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    for i in range(7):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    # 5. Observed vs simulated scatter
    ax5 = plt.subplot(3, 2, 5)
    if discrete_res:
        mask_cal = ~np.isnan(obs[:cal_end_idx])
        ax5.scatter(obs[:cal_end_idx][mask_cal][WARMUP_DAYS:],
                   discrete_res['runoff'][:cal_end_idx][mask_cal][WARMUP_DAYS:],
                   alpha=0.4, s=10, c='blue', label='Calibration')
        mask_val = ~np.isnan(obs[cal_end_idx:])
        ax5.scatter(obs[cal_end_idx:][mask_val], discrete_res['runoff'][cal_end_idx:][mask_val],
                   alpha=0.4, s=10, c='orange', label='Validation')
    max_val = max(np.nanmax(obs), discrete_res['runoff'].max()) if discrete_res else 10
    ax5.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
    ax5.set_xlabel('Observed (mm/day)')
    ax5.set_ylabel('Simulated (mm/day)')
    ax5.set_title('Observed vs Simulated', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.set_aspect('equal', adjustable='box')

    # 6. Method comparison
    ax6 = plt.subplot(3, 2, 6)
    ode_res = next((r for r in results.values() if r['method'] == 'ode' and 'Uncal' not in r['name']), None)
    if discrete_res and ode_res:
        ax6.scatter(discrete_res['runoff'], ode_res['runoff'], alpha=0.3, s=5, c='green')
        max_val = max(discrete_res['runoff'].max(), ode_res['runoff'].max())
        ax6.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
        corr = np.corrcoef(discrete_res['runoff'], ode_res['runoff'])[0, 1]
        rmse = np.sqrt(np.mean((discrete_res['runoff'] - ode_res['runoff'])**2))
        ax6.text(0.05, 0.95, f'r = {corr:.4f}\nRMSE = {rmse:.3f} mm/day',
                transform=ax6.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax6.set_xlabel('Discrete (mm/day)')
    ax6.set_ylabel('ODE (mm/day)')
    ax6.set_title('Discrete vs ODE Agreement', fontsize=11, fontweight='bold')
    ax6.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_dir / 'hbv_bow_banff_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'hbv_bow_banff_results.png'}")


# =============================================================================
# WRR PUBLICATION FIGURE
# =============================================================================

def create_wrr_figure(
    results: Dict[str, Any],
    obs: np.ndarray,
    times: pd.DatetimeIndex,
    cal_end_idx: int,
    convergence_history: Optional[Dict[str, ConvergenceHistory]],
    sensitivity_data: Optional[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Create WRR-compliant publication figure (6-panel, 3x2 grid).

    Layout:
    (a) Hydrograph: Full Period        | (b) Method Comparison (Discrete vs ODE)
    (c) Calibration Detail (1 year)    | (d) Obs vs Sim Scatter
    (e) Optimization Convergence       | (f) Parameter Sensitivity

    Parameters
    ----------
    results : dict
        Simulation results from evaluate_model
    obs : np.ndarray
        Observed discharge
    times : pd.DatetimeIndex
        Time index
    cal_end_idx : int
        Index marking end of calibration period
    convergence_history : dict or None
        Convergence history from calibration
    sensitivity_data : dict or None
        Parameter sensitivity analysis results
    output_dir : Path
        Output directory for figures
    """
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': WRR_FONT_SIZE,
        'axes.labelsize': WRR_FONT_SIZE,
        'axes.titlesize': WRR_TITLE_SIZE,
        'xtick.labelsize': WRR_FONT_SIZE,
        'ytick.labelsize': WRR_FONT_SIZE,
        'legend.fontsize': WRR_FONT_SIZE - 1,
        'figure.dpi': WRR_DPI,
        'savefig.dpi': WRR_DPI,
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.75,
        'patch.linewidth': 0.5,
    })

    # Convert mm to inches
    fig_width = WRR_FIGURE_WIDTH_MM / 25.4  # 6.7 inches
    fig_height = WRR_FIGURE_HEIGHT_MM / 25.4  # 7.1 inches

    # Create figure with GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.10, right=0.98, top=0.96, bottom=0.06)

    # Convert times to datetime
    times_dt = pd.to_datetime(times)

    # Get results
    discrete_res = next((r for r in results.values()
                         if r['method'] == 'discrete' and 'Uncal' not in r['name']), None)
    ode_res = next((r for r in results.values()
                    if r['method'] == 'ode' and 'Uncal' not in r['name']), None)

    # ==========================================================================
    # Panel (a): Full Period Hydrograph
    # ==========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(times_dt, obs, color=WRR_COLORS['observed'], linewidth=0.5,
              alpha=0.8, label='Observed')
    if discrete_res:
        ax_a.plot(times_dt, discrete_res['runoff'], color=WRR_COLORS['discrete'],
                  linewidth=0.5, alpha=0.7, label='Discrete')
    if ode_res:
        ax_a.plot(times_dt, ode_res['runoff'], color=WRR_COLORS['ode'],
                  linewidth=0.5, alpha=0.7, linestyle='--', label='ODE')

    # Mark calibration/validation split
    split_date = times_dt[cal_end_idx]
    ax_a.axvline(split_date, color='gray', linestyle=':', linewidth=0.75)
    ax_a.text(times_dt[cal_end_idx // 2], ax_a.get_ylim()[1] * 0.95,
              'Cal', ha='center', fontsize=WRR_FONT_SIZE - 1, style='italic')
    ax_a.text(times_dt[cal_end_idx + (len(times_dt) - cal_end_idx) // 2],
              ax_a.get_ylim()[1] * 0.95, 'Val', ha='center',
              fontsize=WRR_FONT_SIZE - 1, style='italic')

    ax_a.set_ylabel('Runoff (mm d$^{-1}$)')
    ax_a.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_a.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax_a.set_xlim(times_dt[0], times_dt[-1])

    # ==========================================================================
    # Panel (b): Method Comparison (Discrete vs ODE scatter)
    # ==========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    if discrete_res and ode_res:
        ax_b.scatter(discrete_res['runoff'], ode_res['runoff'],
                     c=WRR_COLORS['discrete'], s=2, alpha=0.3, edgecolors='none')
        max_val = max(discrete_res['runoff'].max(), ode_res['runoff'].max())
        ax_b.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5)

        # Calculate statistics
        corr = np.corrcoef(discrete_res['runoff'], ode_res['runoff'])[0, 1]
        rmse = np.sqrt(np.mean((discrete_res['runoff'] - ode_res['runoff'])**2))

        ax_b.text(0.05, 0.95, f'r = {corr:.4f}\nRMSE = {rmse:.4f} mm d$^{{-1}}$',
                  transform=ax_b.transAxes, fontsize=WRR_FONT_SIZE - 1,
                  va='top', bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.8, edgecolor='none'))

    ax_b.set_xlabel('Discrete (mm d$^{-1}$)')
    ax_b.set_ylabel('ODE (mm d$^{-1}$)')
    ax_b.set_aspect('equal', adjustable='box')

    # ==========================================================================
    # Panel (c): Calibration Detail (1 year with peak flow)
    # ==========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    # Find year with highest peak in calibration period (excluding warmup)
    cal_start = WARMUP_DAYS
    yearly_peaks = []
    for year_start in range(cal_start, cal_end_idx - 365, 365):
        year_end = year_start + 365
        peak = np.nanmax(obs[year_start:year_end])
        yearly_peaks.append((year_start, peak))
    if yearly_peaks:
        best_year_start = max(yearly_peaks, key=lambda x: x[1])[0]
    else:
        best_year_start = cal_start

    detail_start = best_year_start
    detail_end = min(best_year_start + 365, cal_end_idx)

    ax_c.plot(times_dt[detail_start:detail_end], obs[detail_start:detail_end],
              color=WRR_COLORS['observed'], linewidth=0.75, label='Observed')
    if discrete_res:
        ax_c.plot(times_dt[detail_start:detail_end],
                  discrete_res['runoff'][detail_start:detail_end],
                  color=WRR_COLORS['discrete'], linewidth=0.75, label='Discrete')
    if ode_res:
        ax_c.plot(times_dt[detail_start:detail_end],
                  ode_res['runoff'][detail_start:detail_end],
                  color=WRR_COLORS['ode'], linewidth=0.75, linestyle='--', label='ODE')

    ax_c.set_ylabel('Runoff (mm d$^{-1}$)')
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_c.legend(loc='upper right', framealpha=0.9, edgecolor='none')

    # ==========================================================================
    # Panel (d): Observed vs Simulated Scatter (Cal + Val)
    # ==========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    if discrete_res:
        # Calibration points (after warmup)
        mask_cal = ~np.isnan(obs[WARMUP_DAYS:cal_end_idx])
        ax_d.scatter(obs[WARMUP_DAYS:cal_end_idx][mask_cal],
                     discrete_res['runoff'][WARMUP_DAYS:cal_end_idx][mask_cal],
                     c=WRR_COLORS['cal_period'], s=3, alpha=0.4,
                     edgecolors='none', label='Calibration')

        # Validation points
        mask_val = ~np.isnan(obs[cal_end_idx:])
        ax_d.scatter(obs[cal_end_idx:][mask_val],
                     discrete_res['runoff'][cal_end_idx:][mask_val],
                     c=WRR_COLORS['val_period'], s=3, alpha=0.4,
                     edgecolors='none', label='Validation')

        max_val = max(np.nanmax(obs), discrete_res['runoff'].max())
        ax_d.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5)

    ax_d.set_xlabel('Observed (mm d$^{-1}$)')
    ax_d.set_ylabel('Simulated (mm d$^{-1}$)')
    ax_d.set_aspect('equal', adjustable='box')
    ax_d.legend(loc='upper left', framealpha=0.9, edgecolor='none')

    # ==========================================================================
    # Panel (e): Optimization Convergence
    # ==========================================================================
    ax_e = fig.add_subplot(gs[2, 0])

    if convergence_history:
        de_hist = convergence_history.get('de')
        local_hist = convergence_history.get('local')

        # Plot DE convergence
        if de_hist and de_hist.iterations:
            # Convert loss to KGE (loss = -(0.7*KGE + 0.3*log-KGE))
            de_iterations = np.array(de_hist.iterations)
            de_losses = np.array(de_hist.losses)
            ax_e.plot(de_iterations, de_losses, color=WRR_COLORS['discrete'],
                      linewidth=0.75, label='Global (DE)')

        # Plot local refinement
        if local_hist and local_hist.iterations:
            # Offset local iterations to continue from DE
            offset = de_iterations[-1] + 1 if de_hist and de_hist.iterations else 0
            local_iterations = np.array(local_hist.iterations) + offset
            local_losses = np.array(local_hist.losses)
            ax_e.plot(local_iterations, local_losses, color=WRR_COLORS['ode'],
                      linewidth=0.75, label='Local (L-BFGS-B)')

            # Mark transition
            ax_e.axvline(offset, color='gray', linestyle=':', linewidth=0.5)

    ax_e.set_xlabel('Iteration')
    ax_e.set_ylabel('Loss (negative composite KGE)')
    ax_e.legend(loc='upper right', framealpha=0.9, edgecolor='none')

    # ==========================================================================
    # Panel (f): Parameter Sensitivity
    # ==========================================================================
    ax_f = fig.add_subplot(gs[2, 1])

    if sensitivity_data:
        # Plot sensitivity for key parameters
        key_params = ['fc', 'cfmax', 'k1', 'beta']
        colors = [WRR_COLORS['discrete'], WRR_COLORS['ode'], '#009E73', '#CC79A7']

        for param, color in zip(key_params, colors):
            if param in sensitivity_data:
                data = sensitivity_data[param]
                # Normalize x-axis to [0, 1] for comparison
                x_norm = (data['values'] - data['bounds'][0]) / (data['bounds'][1] - data['bounds'][0])
                ax_f.plot(x_norm, data['kges'], color=color, linewidth=0.75, label=param)

                # Mark optimum
                opt_norm = (data['optimum'] - data['bounds'][0]) / (data['bounds'][1] - data['bounds'][0])
                opt_idx = np.argmin(np.abs(data['values'] - data['optimum']))
                ax_f.scatter([opt_norm], [data['kges'][opt_idx]], color=color,
                            s=15, marker='o', zorder=5)

        ax_f.set_xlabel('Normalized parameter value')
        ax_f.set_ylabel('KGE')
        ax_f.legend(loc='lower right', framealpha=0.9, edgecolor='none', ncol=2)
        ax_f.set_xlim(0, 1)
    else:
        ax_f.text(0.5, 0.5, 'Sensitivity analysis\nnot computed',
                  ha='center', va='center', transform=ax_f.transAxes,
                  fontsize=WRR_FONT_SIZE)

    # ==========================================================================
    # Add panel labels (a-f)
    # ==========================================================================
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    for ax, label in zip(axes, 'abcdef'):
        ax.text(-0.12, 1.05, f'({label})', transform=ax.transAxes,
                fontsize=WRR_TITLE_SIZE, fontweight='bold', va='bottom')

    # ==========================================================================
    # Save figures in multiple formats
    # ==========================================================================
    # PDF (vector, preferred for submission)
    pdf_path = output_dir / 'hbv_wrr_figure1.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {pdf_path}")

    # PNG (raster backup)
    png_path = output_dir / 'hbv_wrr_figure1.png'
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {png_path}")

    # TIFF (high-res, if required)
    tiff_path = output_dir / 'hbv_wrr_figure1.tiff'
    fig.savefig(tiff_path, format='tiff', bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {tiff_path}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main(output_dir=None, wrr_figure=False):
    """
    Run the complete calibration study.

    Parameters
    ----------
    output_dir : str or Path, optional
        Output directory for results
    wrr_figure : bool
        If True, generate WRR publication figure with convergence and sensitivity
    """
    print("\n" + "="*70)
    print(f"HBV CALIBRATION STUDY: {CATCHMENT_NAME}")
    print("="*70)
    print("Discrete solver: JAX lax.scan with backpropagation through time")
    print("ODE solver: diffrax with adjoint gradients")
    print(f"ODE solver available: {HAS_DIFFRAX}")
    if wrr_figure:
        print("WRR figure generation: ENABLED")

    if output_dir is None:
        output_dir = Path("/Users/darrieythorsson/Desktop")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    precip, temp, pet, obs_mm, times, cal_end_idx = load_data()
    precip_cal = precip[:cal_end_idx]
    temp_cal = temp[:cal_end_idx]
    pet_cal = pet[:cal_end_idx]
    obs_cal = obs_mm[:cal_end_idx]

    results = {}

    # Uncalibrated evaluation
    print(f"\n{'='*60}")
    print("UNCALIBRATED EVALUATION")
    print('='*60)
    params_default = DEFAULT_PARAMS.copy()
    params_default['smoothing_enabled'] = True
    params_default['smoothing'] = 20.0
    results['uncal'] = evaluate_model(precip, temp, pet, obs_mm, params_default, cal_end_idx,
                                       method='discrete', name='Discrete (Uncal)')
    print(f"Uncalibrated - Cal KGE: {results['uncal']['cal']['kge']:.3f}, Val KGE: {results['uncal']['val']['kge']:.3f}")

    # Calibration (track convergence if WRR figure requested)
    params_cal, best_kge, convergence_history = calibrate(
        precip_cal, temp_cal, pet_cal, obs_cal,
        track_convergence=wrr_figure
    )

    # Calibrated evaluation
    print(f"\n{'='*60}")
    print("CALIBRATED EVALUATION")
    print('='*60)
    results['discrete'] = evaluate_model(precip, temp, pet, obs_mm, params_cal, cal_end_idx,
                                          method='discrete', name='Discrete (Cal)')
    print(f"Discrete - Cal KGE: {results['discrete']['cal']['kge']:.3f}, Val KGE: {results['discrete']['val']['kge']:.3f}")

    if HAS_DIFFRAX:
        results['ode'] = evaluate_model(precip, temp, pet, obs_mm, params_cal, cal_end_idx,
                                         method='ode', name='ODE (Cal)')
        print(f"ODE - Cal KGE: {results['ode']['cal']['kge']:.3f}, Val KGE: {results['ode']['val']['kge']:.3f}")
        corr = np.corrcoef(results['discrete']['runoff'], results['ode']['runoff'])[0, 1]
        print(f"\nDiscrete vs ODE correlation: {corr:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"\n{'Method':<18} {'Cal NSE':>10} {'Cal KGE':>10} {'Val NSE':>10} {'Val KGE':>10}")
    print("-"*60)
    for key, res in results.items():
        if 'Uncal' not in res['name']:
            print(f"{res['name']:<18} {res['cal']['nse']:>10.3f} {res['cal']['kge']:>10.3f} "
                  f"{res['val']['nse']:>10.3f} {res['val']['kge']:>10.3f}")

    # Compute parameter sensitivity if WRR figure requested
    sensitivity_data = None
    if wrr_figure:
        print(f"\n{'='*60}")
        print("PARAMETER SENSITIVITY ANALYSIS")
        print('='*60)
        print("Computing 1D sensitivity for key parameters...")
        key_params = ['fc', 'cfmax', 'k1', 'beta']
        sensitivity_data = compute_parameter_sensitivity(
            params_cal, precip_cal, temp_cal, pet_cal, obs_cal,
            param_names=key_params, n_points=15, warmup=WARMUP_DAYS
        )
        print(f"  Completed sensitivity analysis for: {key_params}")

    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print('='*60)

    create_plots(results, obs_mm, times, cal_end_idx, params_cal, output_dir)

    # Generate WRR publication figure if requested
    if wrr_figure:
        print("\nGenerating WRR publication figure...")
        create_wrr_figure(
            results=results,
            obs=obs_mm,
            times=times,
            cal_end_idx=cal_end_idx,
            convergence_history=convergence_history,
            sensitivity_data=sensitivity_data,
            output_dir=output_dir
        )

    # Save JSON
    results_summary = {
        'study': {
            'catchment': CATCHMENT_NAME,
            'area_km2': CATCHMENT_AREA_KM2,
            'gauge': WSC_GAUGE,
            'cal_period': f"{CAL_START} to {CAL_END}",
            'val_period': f"{CAL_END} to {VAL_END}",
            'warmup_days': WARMUP_DAYS,
        },
        'parameters': {k: float(v) for k, v in params_cal.items()
                       if isinstance(v, (int, float, np.floating))},
        'metrics': {
            key: {
                'name': res['name'],
                'method': res['method'],
                'calibration': {k: float(v) for k, v in res['cal'].items()},
                'validation': {k: float(v) for k, v in res['val'].items()}
            }
            for key, res in results.items()
        },
        'timestamp': datetime.now().isoformat()
    }
    results_file = output_dir / 'hbv_bow_banff_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved: {results_file}")

    print(f"\n{'='*60}")
    print("STUDY COMPLETE")
    print('='*60)

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HBV Calibration Study: Bow at Banff')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--wrr-figure', action='store_true',
                        help='Generate WRR publication figure with convergence and sensitivity')
    args = parser.parse_args()
    main(output_dir=args.output_dir, wrr_figure=args.wrr_figure)
