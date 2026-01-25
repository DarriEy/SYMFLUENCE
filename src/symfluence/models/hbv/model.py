"""
HBV-96 Model Core - JAX Implementation.

Pure JAX functions for the HBV-96 hydrological model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- GPU acceleration when available

The HBV-96 model consists of four main routines:
1. Snow routine - Degree-day accumulation/melt with refreezing
2. Soil routine - Beta-function recharge, ET reduction
3. Response routine - Two-box (upper/lower zone) with percolation
4. Routing routine - Triangular transfer function convolution

Temporal Resolution:
    This implementation supports both daily and sub-daily (e.g., hourly)
    simulation through automatic parameter scaling. Parameters are specified
    in daily units and scaled internally using scale_params_for_timestep().

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple
import warnings

import numpy as np

# Lazy JAX import with numpy fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None
    lax = None
    warnings.warn(
        "JAX not available. HBV model will use NumPy backend with reduced functionality. "
        "Install JAX for autodiff, JIT compilation, and GPU support: pip install jax jaxlib"
    )

# Import parameter utilities and data structures
from .parameters import (
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    RATE_PARAMS,
    FLUX_RATE_PARAMS,
    RECESSION_PARAMS,
    DURATION_PARAMS,
    HBVParameters,
    create_params_from_dict,
    scale_params_for_timestep,
    get_routing_buffer_length,
)

# Re-export for backward compatibility
__all__ = [
    # From parameters module (re-exported)
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'RATE_PARAMS',
    'FLUX_RATE_PARAMS',
    'RECESSION_PARAMS',
    'DURATION_PARAMS',
    'HBVParameters',
    'create_params_from_dict',
    'scale_params_for_timestep',
    'get_routing_buffer_length',
    # Core model exports
    'HAS_JAX',
    'HBVState',
    'create_initial_state',
    'snow_routine_jax',
    'soil_routine_jax',
    'response_routine_jax',
    'routing_routine_jax',
    'triangular_weights',
    'step_jax',
    'simulate_jax',
    'simulate_numpy',
    'simulate',
    'jit_simulate',
    'simulate_ensemble',
    # Loss functions (re-exported for backward compatibility)
    'nse_loss',
    'kge_loss',
    'get_nse_gradient_fn',
    'get_kge_gradient_fn',
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HBVState(NamedTuple):
    """
    HBV-96 model state variables.

    Represents the current state of storages in the model.

    Attributes:
        snow: Snow pack storage (mm water equivalent)
        snow_water: Liquid water content in snow pack (mm)
        sm: Soil moisture storage (mm)
        suz: Upper zone storage (mm)
        slz: Lower zone storage (mm)
        routing_buffer: Buffer for triangular routing (mm), length = max_routing_days
    """
    snow: Any        # float or array
    snow_water: Any
    sm: Any
    suz: Any
    slz: Any
    routing_buffer: Any  # 1D array of length max_routing_days


def create_initial_state(
    initial_snow: float = 0.0,
    initial_sm: float = 150.0,
    initial_suz: float = 10.0,
    initial_slz: float = 10.0,
    max_routing_days: int = 10,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> HBVState:
    """
    Create initial HBV state.

    Args:
        initial_snow: Initial snow storage (mm).
        initial_sm: Initial soil moisture (mm).
        initial_suz: Initial upper zone storage (mm).
        initial_slz: Initial lower zone storage (mm).
        max_routing_days: Maximum routing days (buffer size in days).
        use_jax: Whether to use JAX arrays.
        timestep_hours: Model timestep in hours (affects routing buffer length).

    Returns:
        HBVState namedtuple.
    """
    # Calculate buffer length in timesteps
    buffer_length = get_routing_buffer_length(max_routing_days, timestep_hours)

    if use_jax and HAS_JAX:
        return HBVState(
            snow=jnp.array(initial_snow),
            snow_water=jnp.array(0.0),
            sm=jnp.array(initial_sm),
            suz=jnp.array(initial_suz),
            slz=jnp.array(initial_slz),
            routing_buffer=jnp.zeros(buffer_length),
        )
    else:
        return HBVState(
            snow=np.float64(initial_snow),
            snow_water=np.float64(0.0),
            sm=np.float64(initial_sm),
            suz=np.float64(initial_suz),
            slz=np.float64(initial_slz),
            routing_buffer=np.zeros(buffer_length),
        )


# =============================================================================
# SMOOTH APPROXIMATIONS
# =============================================================================

def _smooth_threshold(val, threshold, smoothing, enabled, use_jax):
    """
    Smooth or hard threshold based on enabled flag.
    Returns ~1 if val > threshold, ~0 if val < threshold.
    """
    if use_jax and HAS_JAX:
        smooth_val = jax.nn.sigmoid(smoothing * (val - threshold))
        hard_val = jnp.where(val > threshold, 1.0, 0.0)
        return jnp.where(enabled, smooth_val, hard_val)
    else:
        if enabled:
            x = smoothing * (val - threshold)
            return np.where(x >= 0,
                            1 / (1 + np.exp(-x)),
                            np.exp(x) / (1 + np.exp(x)))
        else:
            return np.where(val > threshold, 1.0, 0.0)


def _smooth_relu(val, threshold, smoothing, enabled, use_jax):
    """
    Smooth or hard ReLU based on enabled flag.
    Returns max(val - threshold, 0).
    """
    if use_jax and HAS_JAX:
        x = (val - threshold) * smoothing
        smooth_val = jax.nn.softplus(x) / smoothing
        hard_val = jnp.maximum(val - threshold, 0.0)
        return jnp.where(enabled, smooth_val, hard_val)
    else:
        if enabled:
            x = (val - threshold) * smoothing
            out = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
            return out / smoothing
        else:
            return np.maximum(val - threshold, 0.0)


def _smooth_min(val, limit, smoothing, enabled, use_jax):
    """
    Smooth or hard min based on enabled flag.
    Returns min(val, limit).
    """
    return val - _smooth_relu(val, limit, smoothing, enabled, use_jax)


# =============================================================================
# CORE ROUTINES (JAX IMPLEMENTATION)
# =============================================================================

def _get_backend(use_jax: bool = True):
    """Get the appropriate array backend (JAX or NumPy)."""
    if use_jax and HAS_JAX:
        return jnp
    return np


def snow_routine_jax(
    precip: Any,
    temp: Any,
    snow: Any,
    snow_water: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 snow routine (JAX version).

    Partitions precipitation into rain and snow based on temperature threshold.
    Calculates snowmelt using degree-day method with refreezing.
    Uses smooth approximations for differentiability.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (°C)
        snow: Current snow storage (mm SWE)
        snow_water: Liquid water in snow (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_snow, new_snow_water, rainfall_plus_melt)
    """
    # Partition precipitation
    rain_frac = _smooth_threshold(
        temp, params.tt, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    rainfall = precip * rain_frac
    snowfall = precip * params.sfcf * (1.0 - rain_frac)

    # Add snowfall to pack
    snow = snow + snowfall

    # Potential melt (degree-day)
    pot_melt = params.cfmax * _smooth_relu(
        temp, params.tt, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # Actual melt limited by available snow
    melt = _smooth_min(
        pot_melt, snow, params.smoothing, params.smoothing_enabled, use_jax=True
    )
    snow = snow - melt

    # Add melt to liquid water in snow
    snow_water = snow_water + melt + rainfall

    # Refreezing of liquid water when temp < tt
    pot_refreeze = params.cfr * params.cfmax * _smooth_relu(
        params.tt, temp, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    refreeze = _smooth_min(
        pot_refreeze, snow_water, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    snow = snow + refreeze
    snow_water = snow_water - refreeze

    # Water holding capacity
    max_water = params.cwh * snow

    outflow = _smooth_relu(
        snow_water, max_water, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    snow_water = snow_water - outflow

    return snow, snow_water, outflow


def soil_routine_jax(
    rainfall_plus_melt: Any,
    pet: Any,
    sm: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 soil moisture routine (JAX version).

    Calculates recharge to groundwater and actual evapotranspiration.
    Uses beta-function for non-linear recharge relationship.

    Args:
        rainfall_plus_melt: Water input from snow routine (mm/day)
        pet: Potential evapotranspiration (mm/day)
        sm: Current soil moisture (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_sm, recharge, actual_et)
    """
    # Relative soil moisture
    rel_sm = sm / params.fc

    # Recharge using beta function (non-linear)
    smoothed_rel_sm = _smooth_min(
        rel_sm, 1.0, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    recharge = rainfall_plus_melt * jnp.power(
        smoothed_rel_sm,
        params.beta
    )

    # Soil moisture update
    sm = sm + rainfall_plus_melt - recharge

    # Evapotranspiration reduction below LP threshold
    lp_threshold = params.lp * params.fc

    effective_sm_for_et = _smooth_min(
        sm, lp_threshold, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    et_factor = effective_sm_for_et / lp_threshold

    actual_et = pet * et_factor

    # Limit ET to available soil moisture (smooth min)
    actual_et = _smooth_min(
        actual_et, sm, params.smoothing, params.smoothing_enabled, use_jax=True
    )
    sm = sm - actual_et

    # Ensure non-negative
    sm = jnp.maximum(sm, 0.0)

    return sm, recharge, actual_et


def response_routine_jax(
    recharge: Any,
    suz: Any,
    slz: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 response routine (JAX version).

    Two-box groundwater model with percolation from upper to lower zone.
    Upper zone produces fast and intermediate flow, lower zone produces baseflow.

    Args:
        recharge: Recharge from soil moisture (mm/day)
        suz: Upper zone storage (mm)
        slz: Lower zone storage (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_suz, new_slz, total_runoff)
    """
    # Add recharge to upper zone
    suz = suz + recharge

    # Percolation from upper to lower zone
    perc = _smooth_min(
        params.perc, suz, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    suz = suz - perc
    slz = slz + perc

    # Upper zone outflow
    # Q0 = k0 * max(SUZ - UZL, 0)  (fast surface runoff)
    q0 = params.k0 * _smooth_relu(
        suz, params.uzl, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # Q1 = k1 * SUZ (interflow)
    q1 = params.k1 * suz

    # Lower zone outflow (baseflow)
    q2 = params.k2 * slz

    # Update storages
    suz = suz - q0 - q1
    slz = slz - q2

    # Ensure non-negative
    suz = jnp.maximum(suz, 0.0)
    slz = jnp.maximum(slz, 0.0)

    # Total runoff before routing
    total_runoff = q0 + q1 + q2

    return suz, slz, total_runoff


def triangular_weights(
    maxbas: float,
    buffer_length: int = 10,
    timestep_hours: int = 24
) -> Any:
    """
    Calculate triangular weighting function for routing.

    The triangular transfer function distributes runoff over time according
    to a triangular unit hydrograph (Lindström et al., 1997, Eq. 8).

    Args:
        maxbas: Base of triangle in DAYS (parameter units)
        buffer_length: Maximum buffer length in TIMESTEPS
        timestep_hours: Model timestep in hours

    Returns:
        Array of weights (sums to 1.0)
    """
    # Convert maxbas from days to timesteps
    timesteps_per_day = 24.0 / timestep_hours
    maxbas_timesteps = maxbas * timesteps_per_day

    if HAS_JAX:
        timesteps = jnp.arange(1, buffer_length + 1, dtype=jnp.float32)
        rising = jnp.where(
            timesteps <= maxbas_timesteps / 2,
            timesteps / (maxbas_timesteps / 2),
            0.0
        )
        falling = jnp.where(
            (timesteps > maxbas_timesteps / 2) & (timesteps <= maxbas_timesteps),
            (maxbas_timesteps - timesteps) / (maxbas_timesteps / 2),
            0.0
        )
        weights = rising + falling
        weights = weights / jnp.sum(weights + 1e-10)
        return weights
    else:
        timesteps = np.arange(1, buffer_length + 1, dtype=np.float64)
        rising = np.where(
            timesteps <= maxbas_timesteps / 2,
            timesteps / (maxbas_timesteps / 2),
            0.0
        )
        falling = np.where(
            (timesteps > maxbas_timesteps / 2) & (timesteps <= maxbas_timesteps),
            (maxbas_timesteps - timesteps) / (maxbas_timesteps / 2),
            0.0
        )
        weights = rising + falling
        weights = weights / np.sum(weights + 1e-10)
        return weights


def routing_routine_jax(
    runoff: Any,
    routing_buffer: Any,
    params: HBVParameters,
    timestep_hours: int = 24
) -> Tuple[Any, Any]:
    """
    HBV-96 triangular routing routine (JAX version).

    Applies triangular transfer function to smooth runoff response.

    Args:
        runoff: Total runoff before routing (mm/timestep)
        routing_buffer: Previous routing buffer state
        params: HBV parameters (maxbas in DAYS)
        timestep_hours: Model timestep in hours

    Returns:
        Tuple of (routed_runoff, new_routing_buffer)
    """
    buffer_length = routing_buffer.shape[0]

    # Get triangular weights
    weights = triangular_weights(params.maxbas, buffer_length, timestep_hours)

    # Distribute this timestep's runoff across future timesteps
    new_buffer = routing_buffer + runoff * weights

    # Output is the first element
    routed_runoff = new_buffer[0]

    # Shift buffer (advance by one timestep)
    if HAS_JAX:
        new_buffer = jnp.concatenate([new_buffer[1:], jnp.array([0.0])])
    else:
        new_buffer = np.concatenate([new_buffer[1:], np.array([0.0])])

    return routed_runoff, new_buffer


# =============================================================================
# SINGLE TIMESTEP
# =============================================================================

def step_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    state: HBVState,
    params: HBVParameters,
    timestep_hours: int = 24
) -> Tuple[HBVState, Any]:
    """
    Execute one timestep of HBV-96 model (JAX version).

    Runs all four routines in sequence: snow, soil, response, routing.

    Args:
        precip: Precipitation (mm/timestep)
        temp: Air temperature (°C)
        pet: Potential evapotranspiration (mm/timestep)
        state: Current model state
        params: Model parameters (already scaled for timestep)
        timestep_hours: Model timestep in hours

    Returns:
        Tuple of (new_state, routed_runoff)
    """
    # Snow routine
    snow, snow_water, rainfall_plus_melt = snow_routine_jax(
        precip, temp, state.snow, state.snow_water, params
    )

    # Soil moisture routine
    sm, recharge, actual_et = soil_routine_jax(
        rainfall_plus_melt, pet, state.sm, params
    )

    # Response routine (groundwater)
    suz, slz, total_runoff = response_routine_jax(
        recharge, state.suz, state.slz, params
    )

    # Routing routine
    routed_runoff, routing_buffer = routing_routine_jax(
        total_runoff, state.routing_buffer, params, timestep_hours
    )

    # Create new state
    new_state = HBVState(
        snow=snow,
        snow_water=snow_water,
        sm=sm,
        suz=suz,
        slz=slz,
        routing_buffer=routing_buffer,
    )

    return new_state, routed_runoff


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    params: HBVParameters,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Tuple[Any, HBVState]:
    """
    Run full HBV-96 simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/timestep), shape (n_timesteps,)
        temp: Temperature timeseries (°C), shape (n_timesteps,)
        pet: PET timeseries (mm/timestep), shape (n_timesteps,)
        params: HBV parameters (should be pre-scaled for timestep)
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days (included in output)
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if not HAS_JAX:
        return simulate_numpy(precip, temp, pet, params, initial_state, warmup_days, timestep_hours)

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=True, timestep_hours=timestep_hours)

    # Stack forcing for scan
    forcing = jnp.stack([precip, temp, pet], axis=1)

    def scan_fn(state, forcing_step):
        p, t, e = forcing_step
        new_state, runoff = step_jax(p, t, e, state, params, timestep_hours)
        return new_state, runoff

    # Run simulation using scan (efficient and differentiable)
    final_state, runoff = lax.scan(scan_fn, initial_state, forcing)

    return runoff, final_state


def simulate_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: HBVParameters,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Tuple[np.ndarray, HBVState]:
    """
    Run full HBV-96 simulation using NumPy (fallback when JAX not available).

    Args:
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        params: HBV parameters (should be pre-scaled for timestep)
        initial_state: Initial model state
        warmup_days: Number of warmup days
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_timesteps = len(precip)

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=False, timestep_hours=timestep_hours)

    # Storage for results
    runoff = np.zeros(n_timesteps)
    state = initial_state

    for i in range(n_timesteps):
        # Snow routine (numpy version)
        snow, snow_water, rainfall_plus_melt = _snow_routine_numpy(
            precip[i], temp[i], state.snow, state.snow_water, params
        )

        # Soil routine (numpy version)
        sm, recharge, actual_et = _soil_routine_numpy(
            rainfall_plus_melt, pet[i], state.sm, params
        )

        # Response routine (numpy version)
        suz, slz, total_runoff = _response_routine_numpy(
            recharge, state.suz, state.slz, params
        )

        # Routing routine (numpy version)
        routed_runoff, routing_buffer = _routing_routine_numpy(
            total_runoff, state.routing_buffer, params, timestep_hours
        )

        # Update state
        state = HBVState(
            snow=snow,
            snow_water=snow_water,
            sm=sm,
            suz=suz,
            slz=slz,
            routing_buffer=routing_buffer,
        )

        runoff[i] = routed_runoff

    return runoff, state


# NumPy versions of routines (for fallback)
def _snow_routine_numpy(precip, temp, snow, snow_water, params):
    """NumPy version of snow routine."""
    rainfall = precip if temp > params.tt else 0.0
    snowfall = precip * params.sfcf if temp <= params.tt else 0.0

    snow = snow + snowfall
    pot_melt = params.cfmax * max(temp - params.tt, 0.0)
    melt = min(pot_melt, snow)
    snow = snow - melt

    snow_water = snow_water + melt + rainfall
    pot_refreeze = params.cfr * params.cfmax * max(params.tt - temp, 0.0)
    refreeze = min(pot_refreeze, snow_water)
    snow = snow + refreeze
    snow_water = snow_water - refreeze

    max_water = params.cwh * snow
    outflow = max(snow_water - max_water, 0.0)
    snow_water = min(snow_water, max_water)

    return snow, snow_water, outflow


def _soil_routine_numpy(rainfall_plus_melt, pet, sm, params):
    """NumPy version of soil routine."""
    rel_sm = sm / params.fc
    recharge = rainfall_plus_melt * (min(rel_sm, 1.0) ** params.beta)
    sm = sm + rainfall_plus_melt - recharge

    lp_threshold = params.lp * params.fc
    et_factor = sm / lp_threshold if sm < lp_threshold else 1.0
    actual_et = pet * et_factor
    actual_et = min(actual_et, sm)
    sm = sm - actual_et
    sm = max(sm, 0.0)

    return sm, recharge, actual_et


def _response_routine_numpy(recharge, suz, slz, params):
    """NumPy version of response routine."""
    suz = suz + recharge
    perc = min(params.perc, suz)
    suz = suz - perc
    slz = slz + perc

    q0 = params.k0 * max(suz - params.uzl, 0.0)
    q1 = params.k1 * suz
    q2 = params.k2 * slz

    suz = max(suz - q0 - q1, 0.0)
    slz = max(slz - q2, 0.0)

    return suz, slz, q0 + q1 + q2


def _routing_routine_numpy(runoff, routing_buffer, params, timestep_hours=24):
    """NumPy version of routing routine."""
    buffer_length = len(routing_buffer)
    weights = triangular_weights(params.maxbas, buffer_length, timestep_hours)

    new_buffer = routing_buffer.copy() + runoff * weights
    routed_runoff = new_buffer[0]
    new_buffer = np.concatenate([new_buffer[1:], np.array([0.0])])

    return routed_runoff, new_buffer


# =============================================================================
# ENSEMBLE SIMULATION (VMAP)
# =============================================================================

def simulate_ensemble(
    precip: Any,
    temp: Any,
    pet: Any,
    params_batch: Dict[str, Any],
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365
) -> Any:
    """
    Run ensemble of HBV simulations using JAX vmap.

    Efficiently runs multiple parameter sets in parallel.

    Args:
        precip: Precipitation timeseries, shape (n_days,)
        temp: Temperature timeseries, shape (n_days,)
        pet: PET timeseries, shape (n_days,)
        params_batch: Dictionary with parameter arrays, each shape (n_ensemble,)
        initial_state: Initial state (shared across ensemble)
        warmup_days: Warmup period

    Returns:
        Runoff array, shape (n_ensemble, n_days)
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Running sequential ensemble.")
        return _simulate_ensemble_numpy(precip, temp, pet, params_batch, initial_state, warmup_days)

    n_ensemble = len(params_batch[list(params_batch.keys())[0]])

    # Create batched parameters
    def create_params_for_idx(idx):
        return HBVParameters(**{k: v[idx] for k, v in params_batch.items()})

    # Vectorized simulation
    def sim_single(idx):
        params = create_params_for_idx(idx)
        runoff, _ = simulate_jax(precip, temp, pet, params, initial_state, warmup_days)
        return runoff

    # Use vmap for efficient batching
    batch_sim = jax.vmap(sim_single)
    return batch_sim(jnp.arange(n_ensemble))


def _simulate_ensemble_numpy(precip, temp, pet, params_batch, initial_state, warmup_days):
    """NumPy fallback for ensemble simulation."""
    n_ensemble = len(params_batch[list(params_batch.keys())[0]])
    n_days = len(precip)
    results = np.zeros((n_ensemble, n_days))

    for i in range(n_ensemble):
        params_dict = {k: float(v[i]) for k, v in params_batch.items()}
        params = create_params_from_dict(params_dict, use_jax=False)
        results[i], _ = simulate_numpy(precip, temp, pet, params, initial_state, warmup_days)

    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def simulate(
    precip: Any,
    temp: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> Tuple[Any, HBVState]:
    """
    High-level simulation function with automatic backend selection.

    This function automatically handles parameter scaling for sub-daily timesteps.
    Parameters are specified in their standard daily units and scaled internally.

    Args:
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        params: Parameter dictionary in DAILY units (uses defaults if None).
        initial_state: Initial model state
        warmup_days: Warmup period (in days, converted to timesteps internally)
        use_jax: Whether to prefer JAX backend
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Tuple of (runoff_timeseries, final_state)

    Example:
        # Daily simulation (default)
        runoff, state = simulate(precip_daily, temp_daily, pet_daily)

        # Hourly simulation
        runoff, state = simulate(precip_hourly, temp_hourly, pet_hourly, timestep_hours=1)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Scale parameters for sub-daily timesteps
    scaled_params = scale_params_for_timestep(params, timestep_hours)
    hbv_params = create_params_from_dict(scaled_params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, temp, pet, hbv_params, initial_state, warmup_days, timestep_hours)
    else:
        return simulate_numpy(precip, temp, pet, hbv_params, initial_state, warmup_days, timestep_hours)


def jit_simulate(use_gpu: bool = False):
    """
    Get JIT-compiled simulation function.

    Args:
        use_gpu: Whether to use GPU (if available).

    Returns:
        JIT-compiled simulation function if JAX available.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Returning non-JIT function.")
        return simulate

    @jax.jit
    def _jit_simulate(precip, temp, pet, params, initial_state):
        return simulate_jax(precip, temp, pet, params, initial_state)

    return _jit_simulate


# =============================================================================
# BACKWARD COMPATIBILITY RE-EXPORTS
# =============================================================================

def nse_loss(*args, **kwargs):
    """NSE loss function. Re-exported from losses module for backward compatibility."""
    from .losses import nse_loss as _nse_loss
    return _nse_loss(*args, **kwargs)


def kge_loss(*args, **kwargs):
    """KGE loss function. Re-exported from losses module for backward compatibility."""
    from .losses import kge_loss as _kge_loss
    return _kge_loss(*args, **kwargs)


def get_nse_gradient_fn(*args, **kwargs):
    """Get NSE gradient function. Re-exported from losses module for backward compatibility."""
    from .losses import get_nse_gradient_fn as _get_nse_gradient_fn
    return _get_nse_gradient_fn(*args, **kwargs)


def get_kge_gradient_fn(*args, **kwargs):
    """Get KGE gradient function. Re-exported from losses module for backward compatibility."""
    from .losses import get_kge_gradient_fn as _get_kge_gradient_fn
    return _get_kge_gradient_fn(*args, **kwargs)
