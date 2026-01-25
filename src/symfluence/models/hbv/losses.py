"""
HBV-96 Loss Functions and Gradient Utilities.

Provides differentiable loss functions (NSE, KGE) for model calibration
and gradient computation utilities for gradient-based optimization.

All loss functions return negative values for minimization (higher metric = lower loss).
"""

from typing import Any, Callable, Dict, Optional
import warnings

import numpy as np

# Lazy JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from .parameters import (
    scale_params_for_timestep,
    create_params_from_dict,
)


# =============================================================================
# LOSS FUNCTIONS (DIFFERENTIABLE)
# =============================================================================

def nse_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> Any:
    """
    Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Negative because optimization minimizes, and higher NSE is better.

    Args:
        params_dict: Parameter dictionary (in DAILY units - will be scaled for timestep)
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        obs: Observed streamflow timeseries (mm/timestep)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Negative NSE (loss to minimize)
    """
    # Import here to avoid circular dependency
    from .model import simulate_jax, simulate_numpy

    # Scale parameters for sub-daily timesteps
    scaled_params = scale_params_for_timestep(params_dict, timestep_hours)
    params = create_params_from_dict(scaled_params, use_jax=use_jax)

    # Convert warmup from days to timesteps
    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = warmup_days * timesteps_per_day

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days, timestep_hours=timestep_hours)
        # Exclude warmup period (in timesteps)
        sim_eval = sim[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        # NSE = 1 - sum((sim-obs)^2) / sum((obs-mean(obs))^2)
        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse  # Negative for minimization
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days, timestep_hours=timestep_hours)
        sim_eval = sim[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        ss_res = np.sum((sim_eval - obs_eval) ** 2)
        ss_tot = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse


def kge_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> Any:
    """
    Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Parameter dictionary (in DAILY units - will be scaled for timestep)
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        obs: Observed streamflow timeseries (mm/timestep)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Negative KGE (loss to minimize)
    """
    # Import here to avoid circular dependency
    from .model import simulate_jax, simulate_numpy

    # Scale parameters for sub-daily timesteps
    scaled_params = scale_params_for_timestep(params_dict, timestep_hours)
    params = create_params_from_dict(scaled_params, use_jax=use_jax)

    # Convert warmup from days to timesteps
    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = warmup_days * timesteps_per_day

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days, timestep_hours=timestep_hours)
        sim_eval = sim[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        # KGE components
        r = jnp.corrcoef(sim_eval, obs_eval)[0, 1]  # Correlation
        alpha = jnp.std(sim_eval) / (jnp.std(obs_eval) + 1e-10)  # Variability ratio
        beta = jnp.mean(sim_eval) / (jnp.mean(obs_eval) + 1e-10)  # Bias ratio

        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days, timestep_hours=timestep_hours)
        sim_eval = sim[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        r = np.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = np.std(sim_eval) / (np.std(obs_eval) + 1e-10)
        beta = np.mean(sim_eval) / (np.mean(obs_eval) + 1e-10)

        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge


# =============================================================================
# GRADIENT FUNCTIONS
# =============================================================================

def get_nse_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Optional[Callable]:
    """
    Get gradient function for NSE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed)
        temp: Temperature timeseries (fixed)
        pet: PET timeseries (fixed)
        obs: Observed streamflow (fixed)
        warmup_days: Warmup period
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        # Convert array back to dict
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True, timestep_hours=timestep_hours)

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Optional[Callable]:
    """
    Get gradient function for KGE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed)
        temp: Temperature timeseries (fixed)
        pet: PET timeseries (fixed)
        obs: Observed streamflow (fixed)
        warmup_days: Warmup period
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True, timestep_hours=timestep_hours)

    return jax.grad(loss_fn)
