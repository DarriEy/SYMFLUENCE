"""
Xinanjiang Loss Functions and Gradient Utilities.

Provides differentiable loss functions (NSE, KGE) for model calibration
and gradient computation utilities for gradient-based optimization.

All loss functions return negative values for minimization (higher metric = lower loss).
"""

import warnings
from typing import Any, Callable, Dict, Optional

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

from .parameters import params_dict_to_namedtuple

# =============================================================================
# LOSS FUNCTIONS (DIFFERENTIABLE)
# =============================================================================

def nse_loss(
    params_dict: Dict[str, float],
    precip: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
) -> Any:
    """Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries (mm/day)
        pet: PET timeseries (mm/day)
        obs: Observed runoff timeseries (mm/day)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative NSE (loss to minimize)
    """
    from .model import simulate_jax, simulate_numpy

    params = params_dict_to_namedtuple(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse
    else:
        sim, _ = simulate_numpy(precip, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = np.sum((sim_eval - obs_eval) ** 2)
        ss_tot = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse


def kge_loss(
    params_dict: Dict[str, float],
    precip: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
) -> Any:
    """Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries (mm/day)
        pet: PET timeseries (mm/day)
        obs: Observed runoff timeseries (mm/day)
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative KGE (loss to minimize)
    """
    from .model import simulate_jax, simulate_numpy

    params = params_dict_to_namedtuple(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        r = jnp.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = jnp.std(sim_eval) / (jnp.std(obs_eval) + 1e-10)
        beta = jnp.mean(sim_eval) / (jnp.mean(obs_eval) + 1e-10)

        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge
    else:
        sim, _ = simulate_numpy(precip, pet, params, warmup_days=warmup_days)
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
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
) -> Optional[Callable]:
    """Get gradient function for NSE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed)
        pet: PET timeseries (fixed)
        obs: Observed runoff (fixed)
        warmup_days: Warmup period

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(params_dict, precip, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
) -> Optional[Callable]:
    """Get gradient function for KGE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed)
        pet: PET timeseries (fixed)
        obs: Observed runoff (fixed)
        warmup_days: Warmup period

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(params_dict, precip, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


# =============================================================================
# COUPLED SNOW-17 + XAJ LOSS FUNCTIONS
# =============================================================================

def kge_loss_coupled(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    day_of_year: Any,
    warmup_days: int = 365,
    latitude: float = 45.0,
    si: float = 100.0,
    use_jax: bool = True,
) -> Any:
    """Compute negative KGE loss for coupled Snow-17 + XAJ.

    Args:
        params_dict: Combined parameter dictionary (XAJ + Snow-17 keys)
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (C)
        pet: PET timeseries (mm/day)
        obs: Observed runoff (mm/day)
        day_of_year: Day of year array
        warmup_days: Days to exclude from loss
        latitude: Catchment latitude
        si: SWE threshold for areal depletion
        use_jax: Whether to use JAX backend

    Returns:
        Negative KGE (loss to minimize)
    """
    from .model import simulate_coupled_jax, simulate_coupled_numpy
    from .parameters import split_params

    xaj_dict, snow17_dict = split_params(params_dict)

    if use_jax and HAS_JAX:
        sim, _ = simulate_coupled_jax(
            precip, temp, pet, day_of_year, xaj_dict, snow17_dict,
            latitude=latitude, si=si,
        )
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        r = jnp.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = jnp.std(sim_eval) / (jnp.std(obs_eval) + 1e-10)
        beta = jnp.mean(sim_eval) / (jnp.mean(obs_eval) + 1e-10)
        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge
    else:
        sim, _ = simulate_coupled_numpy(
            precip, temp, pet, day_of_year, xaj_dict, snow17_dict,
            latitude=latitude, si=si,
        )
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        r = np.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = np.std(sim_eval) / (np.std(obs_eval) + 1e-10)
        beta = np.mean(sim_eval) / (np.mean(obs_eval) + 1e-10)
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge


def nse_loss_coupled(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    day_of_year: Any,
    warmup_days: int = 365,
    latitude: float = 45.0,
    si: float = 100.0,
    use_jax: bool = True,
) -> Any:
    """Compute negative NSE loss for coupled Snow-17 + XAJ.

    Args:
        params_dict: Combined parameter dictionary (XAJ + Snow-17 keys)
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (C)
        pet: PET timeseries (mm/day)
        obs: Observed runoff (mm/day)
        day_of_year: Day of year array
        warmup_days: Days to exclude from loss
        latitude: Catchment latitude
        si: SWE threshold for areal depletion
        use_jax: Whether to use JAX backend

    Returns:
        Negative NSE (loss to minimize)
    """
    from .model import simulate_coupled_jax, simulate_coupled_numpy
    from .parameters import split_params

    xaj_dict, snow17_dict = split_params(params_dict)

    if use_jax and HAS_JAX:
        sim, _ = simulate_coupled_jax(
            precip, temp, pet, day_of_year, xaj_dict, snow17_dict,
            latitude=latitude, si=si,
        )
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse
    else:
        sim, _ = simulate_coupled_numpy(
            precip, temp, pet, day_of_year, xaj_dict, snow17_dict,
            latitude=latitude, si=si,
        )
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = np.sum((sim_eval - obs_eval) ** 2)
        ss_tot = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse
