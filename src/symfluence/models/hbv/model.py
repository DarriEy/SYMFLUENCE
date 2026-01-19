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

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

from typing import NamedTuple, Optional, Tuple, Dict, Any
import warnings

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

import numpy as np


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'tt': (-3.0, 3.0),        # Threshold temperature for snow/rain (°C)
    'cfmax': (1.0, 10.0),     # Degree-day factor (mm/°C/day)
    'sfcf': (0.5, 1.5),       # Snowfall correction factor
    'cfr': (0.0, 0.1),        # Refreezing coefficient
    'cwh': (0.0, 0.2),        # Water holding capacity of snow (fraction)
    'fc': (50.0, 700.0),      # Maximum soil moisture storage / field capacity (mm)
    'lp': (0.3, 1.0),         # Soil moisture threshold for ET reduction (fraction of FC)
    'beta': (1.0, 6.0),       # Shape coefficient for soil moisture routine
    'k0': (0.05, 0.99),       # Recession coefficient for fast flow (1/day)
    'k1': (0.01, 0.5),        # Recession coefficient for slow flow (1/day)
    'k2': (0.0001, 0.1),      # Recession coefficient for baseflow (1/day)
    'uzl': (0.0, 100.0),      # Threshold for fast flow (mm)
    'perc': (0.0, 10.0),      # Maximum percolation rate (mm/day)
    'maxbas': (1.0, 7.0),     # Length of triangular routing function (days)
}

# Default parameter values (midpoint of bounds, tuned for temperate catchments)
DEFAULT_PARAMS: Dict[str, float] = {
    'tt': 0.0,
    'cfmax': 3.5,
    'sfcf': 0.9,
    'cfr': 0.05,
    'cwh': 0.1,
    'fc': 250.0,
    'lp': 0.7,
    'beta': 2.5,
    'k0': 0.3,
    'k1': 0.1,
    'k2': 0.01,
    'uzl': 30.0,
    'perc': 2.5,
    'maxbas': 2.5,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HBVParameters(NamedTuple):
    """
    HBV-96 model parameters.

    All parameters are stored as JAX-compatible arrays for differentiation.

    Attributes:
        tt: Threshold temperature for snow/rain partitioning (°C)
        cfmax: Degree-day factor for snowmelt (mm/°C/day)
        sfcf: Snowfall correction factor (-)
        cfr: Refreezing coefficient (-)
        cwh: Water holding capacity of snow (fraction)
        fc: Maximum soil moisture storage / field capacity (mm)
        lp: Soil moisture threshold for ET reduction (fraction of FC)
        beta: Shape coefficient for soil moisture recharge
        k0: Recession coefficient for surface runoff (1/day)
        k1: Recession coefficient for interflow (1/day)
        k2: Recession coefficient for baseflow (1/day)
        uzl: Threshold for surface runoff generation (mm)
        perc: Maximum percolation rate from upper to lower zone (mm/day)
        maxbas: Length of triangular routing function (days)
    """
    tt: Any      # float or array
    cfmax: Any
    sfcf: Any
    cfr: Any
    cwh: Any
    fc: Any
    lp: Any
    beta: Any
    k0: Any
    k1: Any
    k2: Any
    uzl: Any
    perc: Any
    maxbas: Any


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


def create_params_from_dict(
    params_dict: Dict[str, float],
    use_jax: bool = True
) -> HBVParameters:
    """
    Create HBVParameters from a dictionary.

    Args:
        params_dict: Dictionary mapping parameter names to values.
            Missing parameters use defaults.
        use_jax: Whether to convert to JAX arrays (requires JAX).

    Returns:
        HBVParameters namedtuple.
    """
    # Merge with defaults
    full_params = {**DEFAULT_PARAMS, **params_dict}

    if use_jax and HAS_JAX:
        return HBVParameters(
            tt=jnp.array(full_params['tt']),
            cfmax=jnp.array(full_params['cfmax']),
            sfcf=jnp.array(full_params['sfcf']),
            cfr=jnp.array(full_params['cfr']),
            cwh=jnp.array(full_params['cwh']),
            fc=jnp.array(full_params['fc']),
            lp=jnp.array(full_params['lp']),
            beta=jnp.array(full_params['beta']),
            k0=jnp.array(full_params['k0']),
            k1=jnp.array(full_params['k1']),
            k2=jnp.array(full_params['k2']),
            uzl=jnp.array(full_params['uzl']),
            perc=jnp.array(full_params['perc']),
            maxbas=jnp.array(full_params['maxbas']),
        )
    else:
        return HBVParameters(
            tt=np.float64(full_params['tt']),
            cfmax=np.float64(full_params['cfmax']),
            sfcf=np.float64(full_params['sfcf']),
            cfr=np.float64(full_params['cfr']),
            cwh=np.float64(full_params['cwh']),
            fc=np.float64(full_params['fc']),
            lp=np.float64(full_params['lp']),
            beta=np.float64(full_params['beta']),
            k0=np.float64(full_params['k0']),
            k1=np.float64(full_params['k1']),
            k2=np.float64(full_params['k2']),
            uzl=np.float64(full_params['uzl']),
            perc=np.float64(full_params['perc']),
            maxbas=np.float64(full_params['maxbas']),
        )


def create_initial_state(
    initial_snow: float = 0.0,
    initial_sm: float = 150.0,
    initial_suz: float = 10.0,
    initial_slz: float = 10.0,
    max_routing_days: int = 10,
    use_jax: bool = True
) -> HBVState:
    """
    Create initial HBV state.

    Args:
        initial_snow: Initial snow storage (mm).
        initial_sm: Initial soil moisture (mm).
        initial_suz: Initial upper zone storage (mm).
        initial_slz: Initial lower zone storage (mm).
        max_routing_days: Maximum routing days (buffer size).
        use_jax: Whether to use JAX arrays.

    Returns:
        HBVState namedtuple.
    """
    if use_jax and HAS_JAX:
        return HBVState(
            snow=jnp.array(initial_snow),
            snow_water=jnp.array(0.0),
            sm=jnp.array(initial_sm),
            suz=jnp.array(initial_suz),
            slz=jnp.array(initial_slz),
            routing_buffer=jnp.zeros(max_routing_days),
        )
    else:
        return HBVState(
            snow=np.float64(initial_snow),
            snow_water=np.float64(0.0),
            sm=np.float64(initial_sm),
            suz=np.float64(initial_suz),
            slz=np.float64(initial_slz),
            routing_buffer=np.zeros(max_routing_days),
        )


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
    rainfall = jnp.where(temp > params.tt, precip, 0.0)
    snowfall = jnp.where(temp <= params.tt, precip * params.sfcf, 0.0)

    # Add snowfall to pack
    snow = snow + snowfall

    # Potential melt (degree-day)
    pot_melt = params.cfmax * jnp.maximum(temp - params.tt, 0.0)

    # Actual melt limited by available snow
    melt = jnp.minimum(pot_melt, snow)
    snow = snow - melt

    # Add melt to liquid water in snow
    snow_water = snow_water + melt + rainfall

    # Refreezing of liquid water when temp < tt
    pot_refreeze = params.cfr * params.cfmax * jnp.maximum(params.tt - temp, 0.0)
    refreeze = jnp.minimum(pot_refreeze, snow_water)
    snow = snow + refreeze
    snow_water = snow_water - refreeze

    # Water holding capacity
    max_water = params.cwh * snow
    outflow = jnp.maximum(snow_water - max_water, 0.0)
    snow_water = jnp.minimum(snow_water, max_water)

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
    # dQ/dP = (SM/FC)^beta
    recharge = rainfall_plus_melt * jnp.power(
        jnp.minimum(rel_sm, 1.0),
        params.beta
    )

    # Soil moisture update
    sm = sm + rainfall_plus_melt - recharge

    # Evapotranspiration reduction below LP threshold
    # AET/PET = SM / (LP * FC) when SM < LP * FC
    lp_threshold = params.lp * params.fc
    et_factor = jnp.where(
        sm < lp_threshold,
        sm / lp_threshold,
        1.0
    )
    actual_et = pet * et_factor

    # Limit ET to available soil moisture
    actual_et = jnp.minimum(actual_et, sm)
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
    perc = jnp.minimum(params.perc, suz)
    suz = suz - perc
    slz = slz + perc

    # Upper zone outflow
    # Q0 = k0 * max(SUZ - UZL, 0)  (fast surface runoff)
    # Q1 = k1 * SUZ                 (interflow)
    q0 = params.k0 * jnp.maximum(suz - params.uzl, 0.0)
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


def triangular_weights(maxbas: float, max_days: int = 10) -> Any:
    """
    Calculate triangular weighting function for routing.

    Args:
        maxbas: Base of triangle (days)
        max_days: Maximum buffer length

    Returns:
        Array of weights (sums to 1.0)
    """
    if HAS_JAX:
        days = jnp.arange(1, max_days + 1, dtype=jnp.float32)
        # Rising limb (0 to maxbas/2)
        rising = jnp.where(days <= maxbas / 2, days / (maxbas / 2), 0.0)
        # Falling limb (maxbas/2 to maxbas)
        falling = jnp.where(
            (days > maxbas / 2) & (days <= maxbas),
            (maxbas - days) / (maxbas / 2),
            0.0
        )
        weights = rising + falling
        # Normalize to sum to 1
        weights = weights / jnp.sum(weights + 1e-10)
        return weights
    else:
        days = np.arange(1, max_days + 1, dtype=np.float64)
        rising = np.where(days <= maxbas / 2, days / (maxbas / 2), 0.0)
        falling = np.where(
            (days > maxbas / 2) & (days <= maxbas),
            (maxbas - days) / (maxbas / 2),
            0.0
        )
        weights = rising + falling
        weights = weights / np.sum(weights + 1e-10)
        return weights


def routing_routine_jax(
    runoff: Any,
    routing_buffer: Any,
    params: HBVParameters
) -> Tuple[Any, Any]:
    """
    HBV-96 triangular routing routine (JAX version).

    Applies triangular transfer function to smooth runoff response.

    Args:
        runoff: Total runoff before routing (mm/day)
        routing_buffer: Previous routing buffer state
        params: HBV parameters

    Returns:
        Tuple of (routed_runoff, new_routing_buffer)
    """
    max_days = routing_buffer.shape[0]

    # Get triangular weights
    weights = triangular_weights(params.maxbas, max_days)

    # Distribute today's runoff across future days
    new_buffer = routing_buffer + runoff * weights

    # Output is the first element
    routed_runoff = new_buffer[0]

    # Shift buffer (advance by one day)
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
    params: HBVParameters
) -> Tuple[HBVState, Any]:
    """
    Execute one timestep of HBV-96 model (JAX version).

    Runs all four routines in sequence: snow, soil, response, routing.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (°C)
        pet: Potential evapotranspiration (mm/day)
        state: Current model state
        params: Model parameters

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
        total_runoff, state.routing_buffer, params
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
    warmup_days: int = 365
) -> Tuple[Any, HBVState]:
    """
    Run full HBV-96 simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/day), shape (n_days,)
        temp: Temperature timeseries (°C), shape (n_days,)
        pet: PET timeseries (mm/day), shape (n_days,)
        params: HBV parameters
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days (included in output but typically ignored)

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if not HAS_JAX:
        return simulate_numpy(precip, temp, pet, params, initial_state, warmup_days)

    n_days = precip.shape[0]

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=True)

    # Stack forcing for scan
    forcing = jnp.stack([precip, temp, pet], axis=1)

    def scan_fn(state, forcing_day):
        p, t, e = forcing_day
        new_state, runoff = step_jax(p, t, e, state, params)
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
    warmup_days: int = 365
) -> Tuple[np.ndarray, HBVState]:
    """
    Run full HBV-96 simulation using NumPy (fallback when JAX not available).

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/day)
        params: HBV parameters
        initial_state: Initial model state
        warmup_days: Number of warmup days

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_days = len(precip)

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=False)

    # Storage for results
    runoff = np.zeros(n_days)
    state = initial_state

    for i in range(n_days):
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
            total_runoff, state.routing_buffer, params
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


def _routing_routine_numpy(runoff, routing_buffer, params):
    """NumPy version of routing routine."""
    max_days = len(routing_buffer)
    weights = triangular_weights(params.maxbas, max_days)

    new_buffer = routing_buffer.copy() + runoff * weights
    routed_runoff = new_buffer[0]
    new_buffer = np.concatenate([new_buffer[1:], np.array([0.0])])

    return routed_runoff, new_buffer


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
    use_jax: bool = True
) -> float:
    """
    Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Negative because optimization minimizes, and higher NSE is better.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries
        temp: Temperature timeseries
        pet: PET timeseries
        obs: Observed streamflow timeseries
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative NSE (loss to minimize)
    """
    params = create_params_from_dict(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days)
        # Exclude warmup period
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        # NSE = 1 - sum((sim-obs)^2) / sum((obs-mean(obs))^2)
        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse  # Negative for minimization
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
    use_jax: bool = True
) -> float:
    """
    Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries
        temp: Temperature timeseries
        pet: PET timeseries
        obs: Observed streamflow timeseries
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative KGE (loss to minimize)
    """
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
    warmup_days: int = 365
):
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
        # Convert array back to dict
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365
):
    """
    Get gradient function for KGE loss.

    Returns a function that computes gradients w.r.t. parameters.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


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
    use_jax: bool = True
) -> Tuple[Any, HBVState]:
    """
    High-level simulation function with automatic backend selection.

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/day)
        params: Parameter dictionary (uses defaults if None)
        initial_state: Initial model state
        warmup_days: Warmup period
        use_jax: Whether to prefer JAX backend

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if params is None:
        params = DEFAULT_PARAMS

    hbv_params = create_params_from_dict(params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, temp, pet, hbv_params, initial_state, warmup_days)
    else:
        return simulate_numpy(precip, temp, pet, hbv_params, initial_state, warmup_days)


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

    # Configure device
    if use_gpu and len(jax.devices('gpu')) > 0:
        device = jax.devices('gpu')[0]
    else:
        device = jax.devices('cpu')[0]

    @jax.jit
    def _jit_simulate(precip, temp, pet, params, initial_state):
        return simulate_jax(precip, temp, pet, params, initial_state)

    return _jit_simulate
