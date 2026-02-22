"""
TOPMODEL Loss Functions and Gradient Utilities.

Provides differentiable loss functions (NSE, KGE) for model calibration
and gradient computation utilities for gradient-based optimization.

All loss functions return negative values for minimization (higher metric = lower loss).

TOPMODEL uses daily timesteps (dt=24h) and takes precip, temp, pet forcing arrays.
No sub-daily timestep scaling is needed (unlike HBV).
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

from .parameters import create_params_from_dict


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
) -> Any:
    """
    Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Negative because optimization minimizes, and higher NSE is better.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (deg C)
        pet: PET timeseries (mm/day)
        obs: Observed streamflow timeseries (mm/day)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative NSE (loss to minimize)
    """
    # Import here to avoid circular dependency
    from .model import simulate_jax, simulate_numpy

    params = create_params_from_dict(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

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
) -> Any:
    """
    Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (deg C)
        pet: PET timeseries (mm/day)
        obs: Observed streamflow timeseries (mm/day)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative KGE (loss to minimize)
    """
    # Import here to avoid circular dependency
    from .model import simulate_jax, simulate_numpy

    params = create_params_from_dict(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        # KGE components
        r = jnp.corrcoef(sim_eval, obs_eval)[0, 1]  # Correlation
        alpha = jnp.std(sim_eval) / (jnp.std(obs_eval) + 1e-10)  # Variability ratio
        beta = jnp.mean(sim_eval) / (jnp.mean(obs_eval) + 1e-10)  # Bias ratio

        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

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

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
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

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)
