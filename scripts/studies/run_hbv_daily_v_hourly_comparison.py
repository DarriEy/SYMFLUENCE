#!/usr/bin/env python3
"""
HBV Timestep Consistency Validation: Hourly vs Daily

Validates that the HBV implementation produces consistent results across
timesteps when using the same parameters. This tests the correctness of
the sub-daily parameter scaling implementation.

Approach:
1. Calibrate model using daily timestep (where we have real observations)
2. Run hourly simulation with the SAME daily-calibrated parameters
3. Aggregate hourly outputs to daily for fair comparison
4. Compare metrics to verify timestep consistency

This is the correct validation approach when sub-daily observations are
not available (hourly obs derived from daily would create circular logic).
"""

import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore', category=UserWarning)

# JAX for optimization
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    HAS_JAX = True
    # Suppress JAX warnings
    jax.config.update("jax_platform_name", "cpu")
except ImportError:
    HAS_JAX = False
    print("JAX not available - calibration will be slow")

from symfluence.models.hbv.model import (
    simulate, PARAM_BOUNDS
)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data")
CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
DOMAIN_NAME = "Bow_at_Banff_lumped_era5"
OUTPUT_DIR = CODE_DIR / "hbv_calibration_comparison"

# Calibration settings
# DDS is robust and works well with ~500-2000 iterations
# With DDS: Daily ~4min, Hourly ~20min for 2000 iterations
CALIBRATION_ITERATIONS = 2000
WARMUP_DAYS = 365

# Parameters to calibrate (most sensitive ones)
PARAMS_TO_CALIBRATE = ['tt', 'cfmax', 'fc', 'lp', 'beta', 'k0', 'k1', 'k2', 'perc', 'maxbas']

# Initial parameter guess (reasonable for snow-dominated basin)
INITIAL_PARAMS = {
    'tt': 0.0,
    'cfmax': 4.0,
    'sfcf': 1.0,
    'cfr': 0.05,
    'cwh': 0.1,
    'fc': 300.0,
    'lp': 0.6,
    'beta': 2.5,
    'k0': 0.3,
    'k1': 0.1,
    'k2': 0.02,
    'uzl': 30.0,
    'perc': 2.0,
    'maxbas': 3.0,
}


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def calculate_metrics(sim: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Calculate KGE, NSE, and other metrics."""
    valid = ~(np.isnan(sim) | np.isnan(obs))
    if np.sum(valid) < 10:
        return {'kge': np.nan, 'nse': np.nan, 'pbias': np.nan, 'rmse': np.nan}

    sim_v, obs_v = sim[valid], obs[valid]

    # KGE
    r = np.corrcoef(sim_v, obs_v)[0, 1]
    alpha = np.std(sim_v) / np.std(obs_v)
    beta = np.mean(sim_v) / np.mean(obs_v)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # NSE
    nse = 1 - np.sum((sim_v - obs_v)**2) / np.sum((obs_v - np.mean(obs_v))**2)

    # PBIAS
    pbias = 100 * np.sum(sim_v - obs_v) / np.sum(obs_v)

    # RMSE
    rmse = np.sqrt(np.mean((sim_v - obs_v)**2))

    return {'kge': kge, 'nse': nse, 'pbias': pbias, 'rmse': rmse, 'r': r}


def load_data(timestep_hours: int, logger) -> Tuple[Dict[str, Any], np.ndarray, float, pd.DatetimeIndex]:
    """Load forcing and observation data for specified timestep."""
    domain_dir = DATA_DIR / f"domain_{DOMAIN_NAME}"
    forcing_dir = domain_dir / "forcing" / "HBV_input"

    # Load forcing
    forcing_file = forcing_dir / f"{DOMAIN_NAME}_hbv_forcing_{timestep_hours}h.nc"
    ds = xr.open_dataset(forcing_file)

    forcing = {
        'precip': ds['pr'].values.flatten().astype(np.float32),
        'temp': ds['temp'].values.flatten().astype(np.float32),
        'pet': ds['pet'].values.flatten().astype(np.float32),
    }
    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    ds.close()

    # Load observations
    obs_file = domain_dir / "observations" / "streamflow" / "preprocessed" / f"{DOMAIN_NAME}_streamflow_processed.csv"
    obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
    obs_cms = obs_df.iloc[:, 0]

    # Get catchment area for unit conversion
    area_km2 = 2209.95  # Bow at Banff

    # Convert m³/s to mm/timestep
    seconds_per_timestep = timestep_hours * 3600
    obs_mm = obs_cms * seconds_per_timestep / (area_km2 * 1e6) * 1000  # m³/s -> mm/timestep

    # Resample observations to match forcing timestep
    if timestep_hours == 24:
        obs_resampled = obs_mm.resample('D').mean()
    else:
        obs_resampled = obs_mm.resample(f'{timestep_hours}h').mean()

    # Align with forcing time
    obs_aligned = obs_resampled.reindex(time_index)

    logger.info(f"Loaded {timestep_hours}h data: {len(forcing['precip'])} timesteps, "
                f"{np.sum(~np.isnan(obs_aligned.values))} valid obs")

    return forcing, np.asarray(obs_aligned.values), area_km2, time_index


def run_simulation(forcing: Dict, params: Dict, timestep_hours: int, warmup_days: int) -> np.ndarray:
    """Run HBV simulation with given parameters."""
    runoff, _ = simulate(
        forcing['precip'],
        forcing['temp'],
        forcing['pet'],
        params=params,
        warmup_days=warmup_days,
        use_jax=HAS_JAX,
        timestep_hours=timestep_hours
    )
    return np.array(runoff)


def calibrate_hbv(
    forcing: Dict,
    obs: np.ndarray,
    timestep_hours: int,
    initial_params: Dict,
    params_to_calibrate: list,
    n_iterations: int,
    logger,
    use_dds: bool = True
) -> Tuple[Dict, list, Dict]:
    """
    Calibrate HBV model.

    Args:
        use_dds: If True, use DDS algorithm (robust, recommended).
                 If False, use JAX gradient descent (faster but may get stuck).

    Returns:
        Tuple of (best_params, kge_history, param_history)
        where param_history is a dict mapping param names to lists of values
    """
    if use_dds or not HAS_JAX:
        return _calibrate_dds(forcing, obs, timestep_hours, initial_params,
                              params_to_calibrate, n_iterations, logger)

    logger.info(f"Calibrating {timestep_hours}h model with JAX ({n_iterations} iterations)")

    # Convert forcing to JAX arrays
    precip = jnp.array(forcing['precip'])
    temp = jnp.array(forcing['temp'])
    pet = jnp.array(forcing['pet'])
    obs_jax = jnp.array(np.nan_to_num(obs, nan=0.0))
    obs_mask = jnp.array(~np.isnan(obs))

    # Get parameter bounds
    bounds = {p: PARAM_BOUNDS.get(p, (-1e6, 1e6)) for p in params_to_calibrate}

    # Initialize parameters
    param_values = jnp.array([initial_params[p] for p in params_to_calibrate])

    # Fixed parameters
    fixed_params = {k: v for k, v in initial_params.items() if k not in params_to_calibrate}

    # Warmup in timesteps
    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = WARMUP_DAYS * timesteps_per_day

    # Import here to avoid issues
    from symfluence.models.hbv.model import (
        simulate_jax, HBVParameters, get_routing_buffer_length
    )
    from symfluence.models.hbv.parameters import FLUX_RATE_PARAMS, RECESSION_PARAMS

    # Pre-compute fixed params as arrays
    fixed_param_values = {k: jnp.array(v) for k, v in fixed_params.items()}

    # Scale factor for sub-daily timesteps
    scale_factor = timestep_hours / 24.0

    def loss_fn(param_array):
        """Compute negative KGE loss."""
        # Build parameter dict with JAX arrays (no Python float conversion)
        params_dict = {}
        for i, name in enumerate(params_to_calibrate):
            params_dict[name] = param_array[i]
        for k, v in fixed_param_values.items():
            params_dict[k] = v

        # Scale parameters for timestep
        # Flux rates: linear scaling
        # Recession coefficients: exact exponential scaling
        scaled_dict = {}
        for k, v in params_dict.items():
            if k in FLUX_RATE_PARAMS:
                scaled_dict[k] = v * scale_factor
            elif k in RECESSION_PARAMS:
                # k_subdaily = 1 - (1 - k_daily)^(dt/24)
                k_clamped = jnp.clip(v, 0.0, 0.9999)
                scaled_dict[k] = 1.0 - jnp.power(1.0 - k_clamped, scale_factor)
            else:
                scaled_dict[k] = v

        # Create HBV parameters
        hbv_params = HBVParameters(
            tt=scaled_dict['tt'],
            cfmax=scaled_dict['cfmax'],
            sfcf=scaled_dict['sfcf'],
            cfr=scaled_dict['cfr'],
            cwh=scaled_dict['cwh'],
            fc=scaled_dict['fc'],
            lp=scaled_dict['lp'],
            beta=scaled_dict['beta'],
            k0=scaled_dict['k0'],
            k1=scaled_dict['k1'],
            k2=scaled_dict['k2'],
            uzl=scaled_dict['uzl'],
            perc=scaled_dict['perc'],
            maxbas=scaled_dict['maxbas'],
            smoothing=jnp.array(15.0),
            smoothing_enabled=jnp.array(False),
        )

        # Create initial state
        buffer_length = get_routing_buffer_length(10, timestep_hours)
        initial_state_tuple = (
            jnp.array(0.0),  # snow
            jnp.array(0.0),  # snow_water
            jnp.array(150.0),  # sm
            jnp.array(10.0),  # suz
            jnp.array(10.0),  # slz
            jnp.zeros(buffer_length),  # routing_buffer
        )
        from symfluence.models.hbv.model import HBVState
        initial_state = HBVState(*initial_state_tuple)

        # Run simulation
        runoff, _ = simulate_jax(precip, temp, pet, hbv_params, initial_state,
                                 WARMUP_DAYS, timestep_hours)

        # Calculate KGE on evaluation period
        sim_eval = runoff[warmup_timesteps:]
        obs_eval = obs_jax[warmup_timesteps:]
        mask_eval = obs_mask[warmup_timesteps:]

        # Masked mean and std
        n_valid = jnp.sum(mask_eval)
        sim_masked = jnp.where(mask_eval, sim_eval, 0.0)
        obs_masked = jnp.where(mask_eval, obs_eval, 0.0)

        sim_mean = jnp.sum(sim_masked) / n_valid
        obs_mean = jnp.sum(obs_masked) / n_valid

        sim_std = jnp.sqrt(jnp.sum(mask_eval * (sim_eval - sim_mean)**2) / n_valid)
        obs_std = jnp.sqrt(jnp.sum(mask_eval * (obs_eval - obs_mean)**2) / n_valid)

        # Correlation
        cov = jnp.sum(mask_eval * (sim_eval - sim_mean) * (obs_eval - obs_mean)) / n_valid
        r = cov / (sim_std * obs_std + 1e-10)

        # KGE components
        alpha = sim_std / (obs_std + 1e-10)
        beta = sim_mean / (obs_mean + 1e-10)

        kge = 1.0 - jnp.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return -kge  # Negative for minimization

    # JIT compile loss and gradient
    loss_and_grad = jit(value_and_grad(loss_fn))

    # Adam optimizer state
    lr = 0.05
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m = jnp.zeros_like(param_values)
    v = jnp.zeros_like(param_values)

    # Bounds as arrays
    lower = jnp.array([bounds[p][0] for p in params_to_calibrate])
    upper = jnp.array([bounds[p][1] for p in params_to_calibrate])

    # Training loop
    history = []
    param_history: Dict[str, List[float]] = {name: [] for name in params_to_calibrate}
    best_loss = float('inf')
    best_params = param_values

    start_time = time.time()

    for i in range(n_iterations):
        loss, grads = loss_and_grad(param_values)

        # Adam update
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads**2
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))

        param_values = param_values - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        # Clip to bounds
        param_values = jnp.clip(param_values, lower, upper)

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = param_values

        history.append(-float(loss))  # Store KGE (positive)

        # Track parameter evolution
        for i, name in enumerate(params_to_calibrate):
            param_history[name].append(float(param_values[i]))

        if (i + 1) % 100 == 0:
            logger.info(f"  Iter {i+1}: KGE = {-loss:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"  Calibration complete in {elapsed:.1f}s, best KGE = {-best_loss:.4f}")

    # Build result dict
    result_params = fixed_params.copy()
    for i, name in enumerate(params_to_calibrate):
        result_params[name] = float(best_params[i])

    return result_params, history, param_history


def _calibrate_dds(forcing, obs, timestep_hours, initial_params, params_to_calibrate,
                   n_iterations, logger):
    """
    DDS (Dynamically Dimensioned Search) calibration.

    DDS is a robust optimization algorithm that progressively focuses search
    from global to local as iterations progress. Well-suited for hydrological
    model calibration.

    Reference: Tolson & Shoemaker (2007), Water Resources Research.

    Returns:
        Tuple of (best_params, kge_history, param_history)
    """
    logger.info(f"Calibrating {timestep_hours}h model with DDS ({n_iterations} iterations)")

    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = WARMUP_DAYS * timesteps_per_day
    fixed_params = {k: v for k, v in initial_params.items() if k not in params_to_calibrate}

    # Get bounds
    bounds = [PARAM_BOUNDS.get(p, (-1e6, 1e6)) for p in params_to_calibrate]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    # DDS perturbation range
    r = 0.2

    def evaluate(x_normalized):
        """Evaluate KGE for normalized parameters [0,1]."""
        x = lower + x_normalized * (upper - lower)
        params = fixed_params.copy()
        for i, name in enumerate(params_to_calibrate):
            params[name] = x[i]

        runoff = run_simulation(forcing, params, timestep_hours, WARMUP_DAYS)
        metrics = calculate_metrics(runoff[warmup_timesteps:], obs[warmup_timesteps:])
        return metrics['kge'] if not np.isnan(metrics['kge']) else -1e6

    # Initialize from initial_params (normalized)
    x_init = np.array([initial_params[p] for p in params_to_calibrate])
    x_best = (x_init - lower) / (upper - lower)
    x_best = np.clip(x_best, 0, 1)
    f_best = evaluate(x_best)

    # Initialize history tracking
    kge_history = [f_best]
    param_history: Dict[str, List[float]] = {name: [] for name in params_to_calibrate}

    # Store initial best parameter values
    x_best_denorm = lower + x_best * (upper - lower)
    for i, name in enumerate(params_to_calibrate):
        param_history[name].append(x_best_denorm[i])

    n_params = len(params_to_calibrate)

    for iteration in range(1, n_iterations + 1):
        # Probability of perturbation (decreases with iterations)
        p = 1.0 - np.log(iteration) / np.log(n_iterations)
        p = max(1.0 / n_params, p)

        # Select parameters to perturb
        perturb_mask = np.random.random(n_params) < p
        if not perturb_mask.any():
            perturb_mask[np.random.randint(n_params)] = True

        # Generate candidate solution
        x_new = x_best.copy()
        for i in range(n_params):
            if perturb_mask[i]:
                perturbation = r * np.random.standard_normal()
                x_new[i] = x_best[i] + perturbation
                # Reflect at boundaries
                if x_new[i] < 0:
                    x_new[i] = -x_new[i]
                if x_new[i] > 1:
                    x_new[i] = 2 - x_new[i]
                x_new[i] = np.clip(x_new[i], 0, 1)

        # Evaluate candidate
        f_new = evaluate(x_new)

        # Update if better (DDS is greedy)
        if f_new > f_best:
            x_best = x_new
            f_best = f_new

        kge_history.append(f_best)

        # Store current best parameter values (denormalized)
        x_best_denorm = lower + x_best * (upper - lower)
        for i, name in enumerate(params_to_calibrate):
            param_history[name].append(x_best_denorm[i])

        if iteration % 100 == 0:
            logger.info(f"  Iter {iteration}: KGE = {f_best:.4f}")

    logger.info(f"  DDS complete, best KGE = {f_best:.4f}")

    # Denormalize best solution
    x_final = lower + x_best * (upper - lower)
    result_params = fixed_params.copy()
    for i, name in enumerate(params_to_calibrate):
        result_params[name] = x_final[i]

    return result_params, kge_history, param_history


def create_comparison_plot(
    results: Dict,
    time_daily: pd.DatetimeIndex,
    time_hourly: pd.DatetimeIndex,
    output_path: Path,
    logger,
    warmup_days: int = 365
):
    """Create comprehensive comparison plot."""
    logger.info("Creating comparison plot...")

    fig = plt.figure(figsize=(16, 14))

    # Define evaluation period for plotting - use actual data range
    # Skip warmup year and use remaining period
    eval_start = pd.Timestamp(time_daily[warmup_days])  # After warmup
    eval_end = pd.Timestamp(time_daily[-1])

    # Panel 1: Daily time series comparison
    ax1 = fig.add_subplot(3, 2, 1)

    # Create DataFrame for easier datetime handling
    df_daily = pd.DataFrame({
        'obs': results['obs_daily'],
        'uncalib': results['sim_daily_uncalib'],
        'calib': results['sim_daily_calib']
    }, index=time_daily)

    df_daily_plot = df_daily.loc[eval_start:eval_end]

    ax1.plot(df_daily_plot.index, df_daily_plot['obs'], 'k-',
             label='Observed', linewidth=0.8, alpha=0.7)
    ax1.plot(df_daily_plot.index, df_daily_plot['uncalib'], 'b--',
             label=f"Uncalib (KGE={results['metrics_daily_uncalib']['kge']:.2f})",
             linewidth=0.7, alpha=0.7)
    ax1.plot(df_daily_plot.index, df_daily_plot['calib'], 'r-',
             label=f"Calibrated (KGE={results['metrics_daily_calib']['kge']:.2f})",
             linewidth=0.8, alpha=0.8)

    ax1.set_ylabel('Runoff (mm/day)')
    ax1.set_title('Daily HBV: Calibrated vs Uncalibrated', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Hourly time series (aggregated to daily for comparison)
    ax2 = fig.add_subplot(3, 2, 2)

    # Aggregate hourly to daily for plotting
    df_hourly = pd.DataFrame({
        'obs': results['obs_hourly'],
        'uncalib': results['sim_hourly_uncalib'],
        'calib': results['sim_hourly_calib']
    }, index=time_hourly)

    df_hourly_daily = df_hourly.resample('D').sum()
    df_hourly_plot = df_hourly_daily.loc[eval_start:eval_end]

    ax2.plot(df_hourly_plot.index, df_hourly_plot['obs'], 'k-',
             label='Observed', linewidth=0.8, alpha=0.7)
    ax2.plot(df_hourly_plot.index, df_hourly_plot['uncalib'], 'b--',
             label=f"Uncalib (KGE={results['metrics_hourly_uncalib']['kge']:.2f})",
             linewidth=0.7, alpha=0.7)
    ax2.plot(df_hourly_plot.index, df_hourly_plot['calib'], 'r-',
             label=f"Calibrated (KGE={results['metrics_hourly_calib']['kge']:.2f})",
             linewidth=0.8, alpha=0.8)

    ax2.set_ylabel('Runoff (mm/day)')
    ax2.set_title('Hourly HBV (daily aggregates): Calibrated vs Uncalibrated', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 3: Single year detail - Daily (use a year in the middle of available data)
    ax3 = fig.add_subplot(3, 2, 3)
    # Find a complete year in the data (typically middle of the dataset)
    detail_year = time_daily[len(time_daily)//2].year
    year_start = pd.Timestamp(f"{detail_year}-01-01")
    year_end = pd.Timestamp(f"{detail_year}-12-31")

    df_daily_year = df_daily.loc[year_start:year_end]

    ax3.plot(df_daily_year.index, df_daily_year['obs'], 'k-', label='Observed', linewidth=1.2)
    ax3.plot(df_daily_year.index, df_daily_year['calib'], 'r-', label='Calibrated', linewidth=1.0)
    ax3.set_ylabel('Runoff (mm/day)')
    ax3.set_title(f'Daily HBV: {detail_year} Detail (Calibrated)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Panel 4: Single year detail - Hourly
    ax4 = fig.add_subplot(3, 2, 4)

    df_hourly_year = df_hourly_daily.loc[year_start:year_end]

    ax4.plot(df_hourly_year.index, df_hourly_year['obs'], 'k-', label='Observed', linewidth=1.2)
    ax4.plot(df_hourly_year.index, df_hourly_year['calib'], 'r-', label='Calibrated', linewidth=1.0)
    ax4.set_ylabel('Runoff (mm/day)')
    ax4.set_title(f'Hourly HBV: {detail_year} Detail (Calibrated, daily aggregates)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Panel 5: Scatter plot - Daily calibrated
    ax5 = fig.add_subplot(3, 2, 5)
    valid = ~np.isnan(results['obs_daily']) & ~np.isnan(results['sim_daily_calib'])
    ax5.scatter(results['obs_daily'][valid], results['sim_daily_calib'][valid],
                alpha=0.3, s=5, c='blue')
    max_val = max(np.nanmax(results['obs_daily']), np.nanmax(results['sim_daily_calib']))
    ax5.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    ax5.set_xlabel('Observed (mm/day)')
    ax5.set_ylabel('Simulated (mm/day)')
    ax5.set_title(f"Daily Calibrated: r={results['metrics_daily_calib']['r']:.3f}")
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')

    # Panel 6: Scatter plot - Hourly calibrated (daily aggregates)
    ax6 = fig.add_subplot(3, 2, 6)
    valid_h = ~np.isnan(df_hourly_daily['obs'].values) & ~np.isnan(df_hourly_daily['calib'].values)
    ax6.scatter(df_hourly_daily['obs'].values[valid_h], df_hourly_daily['calib'].values[valid_h],
                alpha=0.3, s=5, c='green')
    max_val_h = max(np.nanmax(df_hourly_daily['obs']), np.nanmax(df_hourly_daily['calib']))
    ax6.plot([0, max_val_h], [0, max_val_h], 'k--', linewidth=1, label='1:1 line')
    ax6.set_xlabel('Observed (mm/day)')
    ax6.set_ylabel('Simulated (mm/day)')
    ax6.set_title(f"Hourly Calibrated (daily agg): r={results['metrics_hourly_calib']['r']:.3f}")
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to: {output_path}")


def create_calibration_progression_plot(
    history_daily: list,
    history_hourly: list,
    param_history_daily: Dict,
    param_history_hourly: Dict,
    params_to_calibrate: list,
    output_path: Path,
    logger
):
    """Create calibration progression plots showing KGE and parameter evolution."""
    logger.info("Creating calibration progression plot...")

    n_params = len(params_to_calibrate)
    n_rows = (n_params + 2) // 2 + 1  # +1 for KGE plot at top

    fig = plt.figure(figsize=(14, 4 * n_rows))

    # Top panel: KGE progression
    ax_kge = fig.add_subplot(n_rows, 2, (1, 2))

    iterations_daily = np.arange(len(history_daily))
    iterations_hourly = np.arange(len(history_hourly))

    ax_kge.plot(iterations_daily, history_daily, 'b-', linewidth=1.5,
                label=f'Daily (final KGE={history_daily[-1]:.3f})', alpha=0.8)
    ax_kge.plot(iterations_hourly, history_hourly, 'r-', linewidth=1.5,
                label=f'Hourly (final KGE={history_hourly[-1]:.3f})', alpha=0.8)

    ax_kge.set_xlabel('Iteration')
    ax_kge.set_ylabel('KGE')
    ax_kge.set_title('Calibration Progress: KGE Evolution', fontweight='bold', fontsize=12)
    ax_kge.legend(loc='lower right', fontsize=10)
    ax_kge.grid(True, alpha=0.3)
    ax_kge.set_xlim(0, max(len(history_daily), len(history_hourly)))

    # Parameter panels
    for i, param in enumerate(params_to_calibrate):
        ax = fig.add_subplot(n_rows, 2, i + 3)  # Start from position 3 (after KGE spanning 1-2)

        daily_values = param_history_daily[param]
        hourly_values = param_history_hourly[param]

        ax.plot(range(len(daily_values)), daily_values, 'b-', linewidth=1.2,
                label=f'Daily (final={daily_values[-1]:.3f})', alpha=0.8)
        ax.plot(range(len(hourly_values)), hourly_values, 'r-', linewidth=1.2,
                label=f'Hourly (final={hourly_values[-1]:.3f})', alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(param)
        ax.set_title(f'Parameter: {param}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add parameter bounds as horizontal lines
        if param in PARAM_BOUNDS:
            lower, upper = PARAM_BOUNDS[param]
            ax.axhline(y=lower, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
            ax.axhline(y=upper, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Calibration progression plot saved to: {output_path}")


def create_timestep_consistency_plot(
    results: Dict,
    time_daily: pd.DatetimeIndex,
    time_hourly: pd.DatetimeIndex,
    output_path: Path,
    logger,
    warmup_days: int = 365
):
    """
    Create timestep consistency validation plot.

    Shows that hourly simulation (aggregated to daily) matches daily simulation
    when using the same parameters. This validates the sub-daily implementation.
    """
    logger.info("Creating timestep consistency validation plot...")

    fig = plt.figure(figsize=(16, 12))

    # Aggregate hourly calibrated to daily for comparison
    df_hourly = pd.DataFrame({
        'sim': results['sim_hourly_calib'],
    }, index=time_hourly)
    df_hourly_daily = df_hourly.resample('D').sum()

    df_daily = pd.DataFrame({
        'obs': results['obs_daily'],
        'sim': results['sim_daily_calib'],
    }, index=time_daily)

    # Align the two dataframes
    common_index = df_daily.index.intersection(df_hourly_daily.index)
    df_daily_aligned = df_daily.loc[common_index]
    df_hourly_aligned = df_hourly_daily.loc[common_index]

    # Skip warmup
    eval_start = pd.Timestamp(time_daily[warmup_days])
    df_daily_plot = df_daily_aligned.loc[eval_start:]
    df_hourly_plot = df_hourly_aligned.loc[eval_start:]

    # Calculate consistency metrics
    daily_sim = df_daily_plot['sim'].values
    hourly_agg = df_hourly_plot['sim'].values
    valid = ~(np.isnan(daily_sim) | np.isnan(hourly_agg))

    if np.sum(valid) > 0:
        diff = hourly_agg[valid] - daily_sim[valid]
        rmse_consistency = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(diff)
        r_consistency = np.corrcoef(daily_sim[valid], hourly_agg[valid])[0, 1]
    else:
        rmse_consistency = max_diff = mean_diff = r_consistency = np.nan

    # Panel 1: Time series comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df_daily_plot.index, df_daily_plot['obs'], 'k-',
             label='Observed', linewidth=1.0, alpha=0.8)
    ax1.plot(df_daily_plot.index, df_daily_plot['sim'], 'b-',
             label=f"Daily sim (KGE={results['metrics_daily_calib']['kge']:.2f})",
             linewidth=0.8, alpha=0.8)
    ax1.plot(df_hourly_plot.index, df_hourly_plot['sim'], 'r--',
             label=f"Hourly→Daily agg (KGE={results['metrics_hourly_calib']['kge']:.2f})",
             linewidth=0.8, alpha=0.8)
    ax1.set_ylabel('Runoff (mm/day)')
    ax1.set_title('Timestep Consistency: Same Parameters, Different Timesteps', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Difference (hourly_agg - daily)
    ax2 = fig.add_subplot(2, 2, 2)
    diff_series = df_hourly_plot['sim'] - df_daily_plot['sim']
    ax2.plot(df_daily_plot.index, diff_series, 'purple', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(df_daily_plot.index, diff_series, 0, alpha=0.3, color='purple')
    ax2.set_ylabel('Difference (mm/day)')
    ax2.set_title(f'Hourly−Daily Difference (RMSE={rmse_consistency:.3f}, r={r_consistency:.4f})',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 3: Scatter plot - Daily vs Hourly aggregated
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(daily_sim[valid], hourly_agg[valid], alpha=0.3, s=5, c='purple')
    max_val = max(np.nanmax(daily_sim), np.nanmax(hourly_agg))
    ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    ax3.set_xlabel('Daily Simulation (mm/day)')
    ax3.set_ylabel('Hourly→Daily Aggregated (mm/day)')
    ax3.set_title(f'Timestep Consistency: r={r_consistency:.4f}', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Panel 4: Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = f"""
TIMESTEP CONSISTENCY VALIDATION
{'='*45}

Parameters: Daily-calibrated (same for both runs)

Consistency Metrics (Hourly-agg vs Daily):
  • Correlation (r):     {r_consistency:.6f}
  • RMSE:                {rmse_consistency:.4f} mm/day
  • Mean difference:     {mean_diff:.4f} mm/day
  • Max |difference|:    {max_diff:.4f} mm/day

Performance Metrics:
  • Daily KGE:           {results['metrics_daily_calib']['kge']:.4f}
  • Hourly-agg KGE:      {results['metrics_hourly_calib']['kge']:.4f}
  • KGE difference:      {results['metrics_hourly_calib']['kge'] - results['metrics_daily_calib']['kge']:.4f}

{'='*45}
Interpretation:
  • r ≈ 1.0 and low RMSE indicates correct implementation
  • Small differences expected due to:
    - Linear approximation of recession coefficients
    - Routing discretization effects
    - Numerical precision differences
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Timestep consistency plot saved to: {output_path}")
    logger.info(f"  Consistency r={r_consistency:.6f}, RMSE={rmse_consistency:.4f} mm/day")


def create_hourly_vs_daily_comparison(
    results: Dict,
    time_daily: pd.DatetimeIndex,
    time_hourly: pd.DatetimeIndex,
    output_path: Path,
    logger,
    warmup_days: int = 365
):
    """Create a direct comparison of hourly vs daily simulations."""
    logger.info("Creating hourly vs daily comparison plot...")

    fig = plt.figure(figsize=(16, 12))

    # Aggregate hourly to daily for fair comparison
    df_hourly = pd.DataFrame({
        'obs': results['obs_hourly'],
        'uncalib': results['sim_hourly_uncalib'],
        'calib': results['sim_hourly_calib']
    }, index=time_hourly)
    df_hourly_daily = df_hourly.resample('D').sum()

    df_daily = pd.DataFrame({
        'obs': results['obs_daily'],
        'uncalib': results['sim_daily_uncalib'],
        'calib': results['sim_daily_calib']
    }, index=time_daily)

    # Define evaluation period
    eval_start = pd.Timestamp(time_daily[warmup_days])
    eval_end = pd.Timestamp(time_daily[-1])

    df_daily_plot = df_daily.loc[eval_start:eval_end]
    df_hourly_plot = df_hourly_daily.loc[eval_start:eval_end]

    # Get metrics for labels
    kge_daily_uncalib = results['metrics_daily_uncalib']['kge']
    kge_daily_calib = results['metrics_daily_calib']['kge']
    kge_hourly_uncalib = results['metrics_hourly_uncalib']['kge']
    kge_hourly_calib = results['metrics_hourly_calib']['kge']

    # Panel 1: Uncalibrated comparison (Daily vs Hourly)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df_daily_plot.index, df_daily_plot['obs'], 'k-',
             label='Observed', linewidth=1.0, alpha=0.8)
    ax1.plot(df_daily_plot.index, df_daily_plot['uncalib'], 'b-',
             label=f'Daily (KGE={kge_daily_uncalib:.2f})', linewidth=0.8, alpha=0.7)
    ax1.plot(df_hourly_plot.index, df_hourly_plot['uncalib'], 'r-',
             label=f'Hourly (KGE={kge_hourly_uncalib:.2f})', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel('Runoff (mm/day)')
    ax1.set_title('UNCALIBRATED: Daily vs Hourly', fontweight='bold', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Calibrated comparison (Daily vs Hourly - SAME PARAMETERS)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df_daily_plot.index, df_daily_plot['obs'], 'k-',
             label='Observed', linewidth=1.0, alpha=0.8)
    ax2.plot(df_daily_plot.index, df_daily_plot['calib'], 'b-',
             label=f'Daily (KGE={kge_daily_calib:.2f})', linewidth=0.8, alpha=0.7)
    ax2.plot(df_hourly_plot.index, df_hourly_plot['calib'], 'r-',
             label=f'Hourly-agg (KGE={kge_hourly_calib:.2f})', linewidth=0.8, alpha=0.7)
    ax2.set_ylabel('Runoff (mm/day)')
    ax2.set_title('SAME PARAMS: Daily vs Hourly-aggregated', fontweight='bold', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 3: Single year detail - Uncalibrated
    ax3 = fig.add_subplot(2, 2, 3)
    detail_year = time_daily[len(time_daily)//2].year
    year_start = pd.Timestamp(f"{detail_year}-01-01")
    year_end = pd.Timestamp(f"{detail_year}-12-31")

    df_daily_year = df_daily.loc[year_start:year_end]
    df_hourly_year = df_hourly_daily.loc[year_start:year_end]

    ax3.plot(df_daily_year.index, df_daily_year['obs'], 'k-',
             label='Observed', linewidth=1.2)
    ax3.plot(df_daily_year.index, df_daily_year['uncalib'], 'b-',
             label='Daily', linewidth=1.0, alpha=0.8)
    ax3.plot(df_hourly_year.index, df_hourly_year['uncalib'], 'r-',
             label='Hourly', linewidth=1.0, alpha=0.8)
    ax3.set_ylabel('Runoff (mm/day)')
    ax3.set_title(f'UNCALIBRATED: {detail_year} Detail', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Panel 4: Single year detail - Same parameters
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(df_daily_year.index, df_daily_year['obs'], 'k-',
             label='Observed', linewidth=1.2)
    ax4.plot(df_daily_year.index, df_daily_year['calib'], 'b-',
             label='Daily', linewidth=1.0, alpha=0.8)
    ax4.plot(df_hourly_year.index, df_hourly_year['calib'], 'r-',
             label='Hourly-agg', linewidth=1.0, alpha=0.8)
    ax4.set_ylabel('Runoff (mm/day)')
    ax4.set_title(f'SAME PARAMS: {detail_year} Detail', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Hourly vs daily comparison plot saved to: {output_path}")

    # Create additional metrics summary plot
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart comparing metrics
    metrics = ['KGE', 'NSE', 'r']
    x = np.arange(len(metrics))
    width = 0.2

    daily_uncalib_vals = [
        results['metrics_daily_uncalib']['kge'],
        results['metrics_daily_uncalib']['nse'],
        results['metrics_daily_uncalib']['r']
    ]
    daily_calib_vals = [
        results['metrics_daily_calib']['kge'],
        results['metrics_daily_calib']['nse'],
        results['metrics_daily_calib']['r']
    ]
    hourly_uncalib_vals = [
        results['metrics_hourly_uncalib']['kge'],
        results['metrics_hourly_uncalib']['nse'],
        results['metrics_hourly_uncalib']['r']
    ]
    hourly_calib_vals = [
        results['metrics_hourly_calib']['kge'],
        results['metrics_hourly_calib']['nse'],
        results['metrics_hourly_calib']['r']
    ]

    ax_bar = axes[0]
    ax_bar.bar(x - 1.5*width, daily_uncalib_vals, width, label='Daily Uncalib', color='lightblue', edgecolor='blue')
    ax_bar.bar(x - 0.5*width, daily_calib_vals, width, label='Daily Calib', color='blue', edgecolor='darkblue')
    ax_bar.bar(x + 0.5*width, hourly_uncalib_vals, width, label='Hourly Uncalib', color='lightcoral', edgecolor='red')
    ax_bar.bar(x + 1.5*width, hourly_calib_vals, width, label='Hourly (same params)', color='red', edgecolor='darkred')

    ax_bar.set_ylabel('Metric Value')
    ax_bar.set_title('Performance Metrics (Hourly uses daily-calibrated params)', fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics)
    ax_bar.legend(loc='lower right', fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax_bar.set_ylim(-0.5, 1.0)

    # Parameter values (calibrated)
    ax_params = axes[1]
    params_to_plot = ['tt', 'cfmax', 'fc', 'lp', 'beta', 'k0', 'k1', 'k2', 'perc', 'maxbas']

    # Show calibrated values vs initial
    calib_vals = [results['params_daily'].get(p, 0) for p in params_to_plot]
    init_vals = [INITIAL_PARAMS.get(p, 0) for p in params_to_plot]

    # Calculate % change from initial
    pct_change = []
    for init, cal in zip(init_vals, calib_vals):
        if abs(init) > 1e-6:
            pct_change.append(100 * (cal - init) / abs(init))
        else:
            pct_change.append(0)

    colors = ['green' if p > 0 else 'red' for p in pct_change]
    y_pos = np.arange(len(params_to_plot))

    ax_params.barh(y_pos, pct_change, color=colors, alpha=0.7, edgecolor='black')
    ax_params.set_yticks(y_pos)
    ax_params.set_yticklabels(params_to_plot)
    ax_params.set_xlabel('% Change from Initial')
    ax_params.set_title('Calibrated Parameter Changes', fontweight='bold')
    ax_params.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax_params.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path.parent / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Metrics comparison plot saved to: {output_path.parent / 'metrics_comparison.png'}")


def print_results_table(results: Dict, logger):
    """Print formatted results table."""
    logger.info("\n" + "=" * 70)
    logger.info("TIMESTEP CONSISTENCY VALIDATION RESULTS")
    logger.info("=" * 70)

    # Metrics table
    logger.info("\nPerformance Metrics (same parameters for daily and hourly):")
    logger.info("-" * 70)
    logger.info(f"{'Configuration':<25} {'KGE':>10} {'NSE':>10} {'PBIAS':>10} {'RMSE':>10}")
    logger.info("-" * 70)

    for name, metrics in [
        ('Daily Uncalibrated', results['metrics_daily_uncalib']),
        ('Hourly Uncalibrated', results['metrics_hourly_uncalib']),
        ('Daily Calibrated', results['metrics_daily_calib']),
        ('Hourly Calibrated*', results['metrics_hourly_calib']),
    ]:
        logger.info(f"{name:<25} {metrics['kge']:>10.3f} {metrics['nse']:>10.3f} "
                   f"{metrics['pbias']:>9.1f}% {metrics['rmse']:>10.3f}")

    logger.info("-" * 70)
    logger.info("* Hourly uses daily-calibrated parameters (same params, different timestep)")

    # Consistency check
    kge_diff = abs(results['metrics_daily_calib']['kge'] - results['metrics_hourly_calib']['kge'])
    logger.info("\nTimestep Consistency Check:")
    logger.info(f"  KGE difference (Daily vs Hourly-agg): {kge_diff:.4f}")
    if kge_diff < 0.05:
        logger.info("  ✓ Excellent consistency - implementation validated")
    elif kge_diff < 0.10:
        logger.info("  ~ Good consistency - minor numerical differences")
    else:
        logger.info("  ! Notable difference - investigate scaling implementation")

    # Calibrated parameters
    logger.info("\nCalibrated Parameters (Daily):")
    logger.info("-" * 70)
    logger.info(f"{'Parameter':<12} {'Initial':>12} {'Calibrated':>12} {'Change %':>12}")
    logger.info("-" * 70)

    for param in PARAMS_TO_CALIBRATE:
        init_val = INITIAL_PARAMS[param]
        cal_val = results['params_daily'][param]
        if abs(init_val) > 1e-6:
            change_pct = 100 * (cal_val - init_val) / init_val
        else:
            change_pct = 0.0
        logger.info(f"{param:<12} {init_val:>12.4f} {cal_val:>12.4f} {change_pct:>11.1f}%")

    logger.info("-" * 70)
    logger.info("=" * 70)


def main():
    """Main execution function."""
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HBV TIMESTEP CONSISTENCY VALIDATION")
    logger.info("=" * 70)
    logger.info("Validating that hourly simulation with daily-calibrated parameters")
    logger.info("produces consistent results when aggregated to daily.")

    results = {}

    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n--- Loading Data ---")

    forcing_daily, obs_daily, area_km2, time_daily = load_data(24, logger)
    forcing_hourly, obs_hourly, _, time_hourly = load_data(1, logger)

    results['obs_daily'] = obs_daily
    results['obs_hourly'] = obs_hourly

    # Evaluation period (skip warmup)
    timesteps_daily = WARMUP_DAYS
    timesteps_hourly = WARMUP_DAYS * 24

    # =========================================================================
    # Uncalibrated Simulations
    # =========================================================================
    logger.info("\n--- Running Uncalibrated Simulations ---")

    # Daily uncalibrated
    logger.info("Running daily uncalibrated...")
    sim_daily_uncalib = run_simulation(forcing_daily, INITIAL_PARAMS, 24, WARMUP_DAYS)
    metrics_daily_uncalib = calculate_metrics(
        sim_daily_uncalib[timesteps_daily:], obs_daily[timesteps_daily:]
    )
    logger.info(f"  Daily uncalibrated KGE: {metrics_daily_uncalib['kge']:.4f}")

    # Hourly uncalibrated
    logger.info("Running hourly uncalibrated...")
    sim_hourly_uncalib = run_simulation(forcing_hourly, INITIAL_PARAMS, 1, WARMUP_DAYS)
    metrics_hourly_uncalib = calculate_metrics(
        sim_hourly_uncalib[timesteps_hourly:], obs_hourly[timesteps_hourly:]
    )
    logger.info(f"  Hourly uncalibrated KGE: {metrics_hourly_uncalib['kge']:.4f}")

    results['sim_daily_uncalib'] = sim_daily_uncalib
    results['sim_hourly_uncalib'] = sim_hourly_uncalib
    results['metrics_daily_uncalib'] = metrics_daily_uncalib
    results['metrics_hourly_uncalib'] = metrics_hourly_uncalib

    # =========================================================================
    # Calibration (Daily only - we have real daily observations)
    # =========================================================================
    logger.info("\n--- Running Calibration (Daily timestep only) ---")
    logger.info("Note: Hourly observations are derived from daily, so we only")
    logger.info("      calibrate at daily timestep to avoid circular logic.")

    # Daily calibration
    params_daily, history_daily, param_history_daily = calibrate_hbv(
        forcing_daily, obs_daily, 24, INITIAL_PARAMS,
        PARAMS_TO_CALIBRATE, CALIBRATION_ITERATIONS, logger
    )

    # Use SAME parameters for hourly (this tests timestep consistency)
    params_hourly = params_daily.copy()
    logger.info("Using daily-calibrated parameters for hourly simulation")
    logger.info("(Testing timestep scaling consistency, not independent calibration)")

    results['params_daily'] = params_daily
    results['params_hourly'] = params_hourly
    results['param_history_daily'] = param_history_daily
    results['history_daily'] = history_daily

    # =========================================================================
    # Calibrated Simulations
    # =========================================================================
    logger.info("\n--- Running Calibrated Simulations ---")

    # Daily calibrated
    logger.info("Running daily calibrated...")
    sim_daily_calib = run_simulation(forcing_daily, params_daily, 24, WARMUP_DAYS)
    metrics_daily_calib = calculate_metrics(
        sim_daily_calib[timesteps_daily:], obs_daily[timesteps_daily:]
    )
    logger.info(f"  Daily calibrated KGE: {metrics_daily_calib['kge']:.4f}")

    # Hourly calibrated
    logger.info("Running hourly calibrated...")
    sim_hourly_calib = run_simulation(forcing_hourly, params_hourly, 1, WARMUP_DAYS)
    metrics_hourly_calib = calculate_metrics(
        sim_hourly_calib[timesteps_hourly:], obs_hourly[timesteps_hourly:]
    )
    logger.info(f"  Hourly calibrated KGE: {metrics_hourly_calib['kge']:.4f}")

    results['sim_daily_calib'] = sim_daily_calib
    results['sim_hourly_calib'] = sim_hourly_calib
    results['metrics_daily_calib'] = metrics_daily_calib
    results['metrics_hourly_calib'] = metrics_hourly_calib

    # =========================================================================
    # Results and Plots
    # =========================================================================
    print_results_table(results, logger)

    create_comparison_plot(
        results, time_daily, time_hourly,
        OUTPUT_DIR / "hbv_calibration_comparison.png",
        logger,
        warmup_days=WARMUP_DAYS
    )

    # Create timestep consistency plot (new focus)
    create_timestep_consistency_plot(
        results, time_daily, time_hourly,
        OUTPUT_DIR / "timestep_consistency.png",
        logger,
        warmup_days=WARMUP_DAYS
    )

    # Create hourly vs daily comparison plot
    create_hourly_vs_daily_comparison(
        results, time_daily, time_hourly,
        OUTPUT_DIR / "hourly_vs_daily_comparison.png",
        logger,
        warmup_days=WARMUP_DAYS
    )

    # Save calibration history (KGE) - daily only
    history_df = pd.DataFrame({
        'iteration': range(len(history_daily)),
        'kge_daily': history_daily,
    })
    history_df.to_csv(OUTPUT_DIR / "calibration_history.csv", index=False)

    # Save parameter history - daily only
    param_history_df_data = {'iteration': range(len(param_history_daily[PARAMS_TO_CALIBRATE[0]]))}
    for param in PARAMS_TO_CALIBRATE:
        param_history_df_data[f'{param}_daily'] = param_history_daily[param]
    param_history_df = pd.DataFrame(param_history_df_data)
    param_history_df.to_csv(OUTPUT_DIR / "parameter_history.csv", index=False)

    # Save final parameters (same for both timesteps)
    params_df = pd.DataFrame({
        'parameter': list(params_daily.keys()),
        'initial': [INITIAL_PARAMS.get(p, np.nan) for p in params_daily.keys()],
        'calibrated': list(params_daily.values()),
    })
    params_df.to_csv(OUTPUT_DIR / "calibrated_parameters.csv", index=False)

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("\nDONE")

    return results


if __name__ == "__main__":
    results = main()
