"""
HEC-HMS Model Core - JAX Implementation.

Pure JAX/NumPy functions for native HEC-HMS hydrological model algorithms:
1. Snow routine - ATI-based Temperature Index snow model
2. Loss routine - SCS Curve Number Continuous method
3. Transform routine - Clark Unit Hydrograph (linear reservoir)
4. Baseflow routine - Linear Reservoir groundwater

Enables:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- DDS and other evolutionary calibration

References:
    US Army Corps of Engineers (2000). Hydrologic Modeling System HEC-HMS
    Technical Reference Manual. Hydrologic Engineering Center, Davis, CA.

    Feldman, A.D. (2000). Hydrologic Modeling System HEC-HMS Technical
    Reference Manual. US Army Corps of Engineers, Davis, CA.
"""

from typing import Any, Dict, Optional, Tuple
import warnings
import math

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
        "JAX not available. HEC-HMS model will use NumPy backend. "
        "Install JAX for autodiff, JIT compilation, and GPU support: pip install jax jaxlib"
    )

from .parameters import (
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    HecHmsParameters,
    HecHmsState,
    create_params_from_dict,
    create_initial_state,
)

__all__ = [
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'HecHmsParameters',
    'HecHmsState',
    'create_params_from_dict',
    'create_initial_state',
    'HAS_JAX',
    'snow_step',
    'loss_step',
    'transform_step',
    'baseflow_step',
    'step',
    'simulate_jax',
    'simulate_numpy',
    'simulate',
]


# =============================================================================
# CORE ROUTINES (Dual JAX/NumPy via xp.where)
# =============================================================================

def _get_backend(use_jax: bool = True):
    """Get the appropriate array backend (JAX or NumPy)."""
    if use_jax and HAS_JAX:
        return jnp
    return np


def snow_step(
    precip: Any,
    temp: Any,
    state_swe: Any,
    state_liquid: Any,
    state_ati: Any,
    state_cold_content: Any,
    params: HecHmsParameters,
    day_of_year: Any = 1,
    use_jax: bool = True
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    HEC-HMS ATI-based Temperature Index snow routine.

    Rain/snow partition at px_temp; ATI tracks smoothed temperature;
    seasonal melt rate interpolated between meltrate_min (Dec 21) and
    meltrate_max (Jun 21); cold content deficit must be overcome before
    melt; liquid water holding capacity in snowpack.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (deg C)
        state_swe: Current snow water equivalent (mm)
        state_liquid: Liquid water in snowpack (mm)
        state_ati: Antecedent temperature index (deg C)
        state_cold_content: Cold content deficit (mm)
        params: Model parameters
        day_of_year: Day of year (1-366) for seasonal melt rate
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_swe, new_liquid, new_ati, new_cold_content, rain_plus_melt)
    """
    xp = _get_backend(use_jax)

    # --- Rain/snow partition ---
    rain = xp.where(temp > params.px_temp, precip, 0.0)
    snow = xp.where(temp <= params.px_temp, precip, 0.0)

    # Add snowfall to SWE
    swe = state_swe + snow

    # --- Update ATI (exponential smoothing of temperature) ---
    ati = xp.where(
        temp > 0.0,
        params.ati_meltrate_coeff * state_ati + temp,
        state_ati
    )

    # --- Seasonal melt rate (sinusoidal interpolation) ---
    # meltrate_min on winter solstice (day 355), meltrate_max on summer solstice (day 172)
    seasonal_factor = 0.5 * (1.0 - xp.cos(2.0 * math.pi * (day_of_year - 81.0) / 365.0))
    meltrate = params.meltrate_min + seasonal_factor * (params.meltrate_max - params.meltrate_min)

    # --- Cold content (energy deficit) ---
    cold_update = params.ati_cold_rate_coeff * (params.base_temp - temp)
    cold_content = xp.where(
        temp < params.base_temp,
        xp.minimum(state_cold_content + cold_update, params.cold_limit),
        xp.maximum(state_cold_content + cold_update, 0.0)
    )

    # --- Potential melt ---
    pot_melt = meltrate * xp.maximum(temp - params.base_temp, 0.0)

    # Cold content must be overcome before melt occurs
    net_melt_energy = xp.maximum(pot_melt - cold_content, 0.0)
    cold_content = xp.maximum(cold_content - pot_melt, 0.0)

    # Actual melt limited by available SWE
    actual_melt = xp.minimum(net_melt_energy, swe)
    swe = swe - actual_melt

    # --- Liquid water in snowpack ---
    liquid = state_liquid + actual_melt + rain * xp.where(swe > 0.0, 1.0, 0.0)

    # Water holding capacity
    max_liquid = params.water_capacity * swe
    outflow = xp.maximum(liquid - max_liquid, 0.0)
    liquid = xp.minimum(liquid, max_liquid)

    # Rain that falls on bare ground passes through directly
    direct_rain = rain * xp.where(swe <= 0.0, 1.0, 0.0)
    rain_plus_melt = outflow + direct_rain

    # Ensure non-negative
    swe = xp.maximum(swe, 0.0)
    liquid = xp.maximum(liquid, 0.0)

    return swe, liquid, ati, cold_content, rain_plus_melt


def loss_step(
    rain_plus_melt: Any,
    pet: Any,
    soil_deficit: Any,
    params: HecHmsParameters,
    use_jax: bool = True
) -> Tuple[Any, Any, Any]:
    """
    HEC-HMS SCS Curve Number Continuous loss method.

    S = 25400/CN - 254; Ia = initial_abstraction_ratio * S;
    excess = (P-Ia)^2 / (P-Ia+S) when P > Ia.
    Soil deficit is tracked continuously and recovered by ET.

    Args:
        rain_plus_melt: Water input (mm/day)
        pet: Potential evapotranspiration (mm/day)
        soil_deficit: Current soil moisture deficit (mm)
        params: Model parameters
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_soil_deficit, excess_precip, actual_et)
    """
    xp = _get_backend(use_jax)

    # Maximum retention
    s_max = 25400.0 / xp.maximum(params.cn, 1.0) - 254.0

    # Initial abstraction
    ia = params.initial_abstraction_ratio * s_max

    # Current available retention is the soil deficit
    s_current = xp.maximum(soil_deficit, 0.0)

    # Effective precipitation after initial abstraction
    p_eff = xp.maximum(rain_plus_melt - ia * (s_current / xp.maximum(s_max, 1e-6)), 0.0)

    # SCS excess: Q = P_eff^2 / (P_eff + s_current) when P_eff > 0
    excess = xp.where(
        p_eff > 0.0,
        p_eff * p_eff / xp.maximum(p_eff + s_current, 1e-6),
        0.0
    )

    # Infiltration = input - excess
    infiltration = rain_plus_melt - excess

    # Update soil deficit (reduced by infiltration)
    soil_deficit = xp.maximum(s_current - infiltration, 0.0)

    # ET recovery of soil deficit (proportional to available moisture)
    moisture_fraction = 1.0 - soil_deficit / xp.maximum(s_max, 1e-6)
    actual_et = pet * xp.maximum(xp.minimum(moisture_fraction, 1.0), 0.0)

    # ET increases the deficit (soil dries)
    soil_deficit = xp.minimum(soil_deficit + actual_et, s_max)

    return soil_deficit, excess, actual_et


def transform_step(
    excess: Any,
    clark_storage: Any,
    params: HecHmsParameters,
    use_jax: bool = True
) -> Tuple[Any, Any]:
    """
    HEC-HMS Clark Unit Hydrograph transform (linear reservoir attenuation).

    Translation portion is simplified (instantaneous for lumped mode).
    Attenuation: C2 = exp(-dt/R), C1 = 1 - C2;
    new_storage = C2 * storage + C1 * inflow * R;
    outflow = new_storage / R.

    Args:
        excess: Excess precipitation (mm/day)
        clark_storage: Current Clark reservoir storage (mm)
        params: Model parameters
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_clark_storage, direct_runoff)
    """
    xp = _get_backend(use_jax)

    # Translation fraction: proportion entering reservoir this timestep
    # For lumped daily, tc controls how much excess is delayed
    dt = 1.0  # daily timestep

    # Attenuation through linear reservoir
    r = xp.maximum(params.r_coeff, 1e-6)
    c2 = xp.exp(-dt / r)
    c1 = 1.0 - c2

    # Inflow: excess precip (translation is instantaneous in lumped mode)
    # For tc > 1 day, spread excess over tc days via simple fraction
    translation_frac = xp.minimum(dt / xp.maximum(params.tc, 1e-6), 1.0)
    inflow = excess * translation_frac

    # Linear reservoir routing
    new_storage = c2 * clark_storage + c1 * inflow * r
    outflow = new_storage / r

    # Ensure non-negative
    new_storage = xp.maximum(new_storage, 0.0)
    outflow = xp.maximum(outflow, 0.0)

    return new_storage, outflow


def baseflow_step(
    infiltration_surplus: Any,
    gw_storage_1: Any,
    gw_storage_2: Any,
    params: HecHmsParameters,
    use_jax: bool = True
) -> Tuple[Any, Any, Any]:
    """
    HEC-HMS Linear Reservoir baseflow method.

    k = dt / gw_storage_coeff; baseflow = gw_storage * k.
    deep_perc_fraction partitions infiltration between GW recharge and deep loss.

    Args:
        infiltration_surplus: Water available for GW recharge (mm/day)
        gw_storage_1: Shallow groundwater storage (mm)
        gw_storage_2: Deep groundwater storage (mm)
        params: Model parameters
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_gw_storage_1, new_gw_storage_2, baseflow)
    """
    xp = _get_backend(use_jax)

    dt = 1.0  # daily

    # Partition infiltration surplus
    deep_perc = infiltration_surplus * params.deep_perc_fraction
    gw_recharge = infiltration_surplus * (1.0 - params.deep_perc_fraction)

    # Shallow GW reservoir
    gw_1 = gw_storage_1 + gw_recharge
    k1 = dt / xp.maximum(params.gw_storage_coeff, 1e-6)
    k1 = xp.minimum(k1, 1.0)
    baseflow_1 = gw_1 * k1
    gw_1 = gw_1 - baseflow_1

    # Deep GW reservoir (slower, 3x storage coeff)
    gw_2 = gw_storage_2 + deep_perc
    k2 = dt / xp.maximum(params.gw_storage_coeff * 3.0, 1e-6)
    k2 = xp.minimum(k2, 1.0)
    baseflow_2 = gw_2 * k2
    gw_2 = gw_2 - baseflow_2

    # Ensure non-negative
    gw_1 = xp.maximum(gw_1, 0.0)
    gw_2 = xp.maximum(gw_2, 0.0)

    return gw_1, gw_2, baseflow_1 + baseflow_2


# =============================================================================
# SINGLE TIMESTEP
# =============================================================================

def step(
    precip: Any,
    temp: Any,
    pet: Any,
    state: HecHmsState,
    params: HecHmsParameters,
    day_of_year: Any = 1,
    use_jax: bool = True
) -> Tuple[HecHmsState, Any]:
    """
    Execute one timestep of the HEC-HMS model.

    Runs all four routines in sequence: snow, loss, transform, baseflow.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (deg C)
        pet: Potential evapotranspiration (mm/day)
        state: Current model state
        params: Model parameters
        day_of_year: Day of year for seasonal melt rate
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_state, total_streamflow)
    """
    # Snow routine
    swe, liquid, ati, cold_content, rain_plus_melt = snow_step(
        precip, temp,
        state.snow_swe, state.snow_liquid, state.snow_ati, state.snow_cold_content,
        params, day_of_year, use_jax
    )

    # Loss routine
    soil_deficit, excess, actual_et = loss_step(
        rain_plus_melt, pet, state.soil_deficit, params, use_jax
    )

    # Infiltration surplus for baseflow: input - excess - ET portion already handled
    # The water that infiltrates but isn't retained goes to GW
    xp = _get_backend(use_jax)
    infiltration_surplus = xp.maximum(rain_plus_melt - excess - actual_et, 0.0) * 0.5

    # Transform routine (direct runoff)
    clark_storage, direct_runoff = transform_step(
        excess, state.clark_storage, params, use_jax
    )

    # Baseflow routine
    gw_1, gw_2, baseflow = baseflow_step(
        infiltration_surplus, state.gw_storage_1, state.gw_storage_2, params, use_jax
    )

    # Total streamflow
    total_flow = direct_runoff + baseflow

    # Create new state
    new_state = HecHmsState(
        snow_swe=swe,
        snow_liquid=liquid,
        snow_ati=ati,
        snow_cold_content=cold_content,
        soil_deficit=soil_deficit,
        clark_storage=clark_storage,
        gw_storage_1=gw_1,
        gw_storage_2=gw_2,
    )

    return new_state, total_flow


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    params: HecHmsParameters,
    initial_state: Optional[HecHmsState] = None,
    warmup_days: int = 365,
    day_of_year_start: int = 1
) -> Tuple[Any, HecHmsState]:
    """
    Run full HEC-HMS simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/day), shape (n_days,)
        temp: Temperature timeseries (deg C), shape (n_days,)
        pet: PET timeseries (mm/day), shape (n_days,)
        params: HEC-HMS parameters
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days
        day_of_year_start: Day of year for first timestep

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if not HAS_JAX:
        return simulate_numpy(precip, temp, pet, params, initial_state,
                              warmup_days, day_of_year_start)

    if initial_state is None:
        initial_state = create_initial_state(cn=float(params.cn), use_jax=True)

    n_steps = precip.shape[0]

    # Create day-of-year array
    doy = jnp.mod(jnp.arange(n_steps) + day_of_year_start - 1, 365) + 1

    # Stack forcing for scan
    forcing = jnp.stack([precip, temp, pet, doy.astype(jnp.float64)], axis=1)

    def scan_fn(state, forcing_step):
        p, t, e, d = forcing_step
        new_state, runoff = step(p, t, e, state, params, d, use_jax=True)
        return new_state, runoff

    final_state, runoff = lax.scan(scan_fn, initial_state, forcing)

    return runoff, final_state


def simulate_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: HecHmsParameters,
    initial_state: Optional[HecHmsState] = None,
    warmup_days: int = 365,
    day_of_year_start: int = 1
) -> Tuple[np.ndarray, HecHmsState]:
    """
    Run full HEC-HMS simulation using NumPy (fallback when JAX not available).

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (deg C)
        pet: PET timeseries (mm/day)
        params: HEC-HMS parameters
        initial_state: Initial model state
        warmup_days: Number of warmup days
        day_of_year_start: Day of year for first timestep

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_timesteps = len(precip)

    if initial_state is None:
        initial_state = create_initial_state(cn=float(params.cn), use_jax=False)

    runoff = np.zeros(n_timesteps)
    state = initial_state

    for i in range(n_timesteps):
        doy = (i + day_of_year_start - 1) % 365 + 1
        state, runoff[i] = step(
            precip[i], temp[i], pet[i], state, params, doy, use_jax=False
        )

    return runoff, state


def simulate(
    precip: Any,
    temp: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    initial_state: Optional[HecHmsState] = None,
    warmup_days: int = 365,
    use_jax: bool = True,
    day_of_year_start: int = 1,
    **kwargs
) -> Tuple[Any, HecHmsState]:
    """
    High-level simulation function with automatic backend selection.

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (deg C)
        pet: PET timeseries (mm/day)
        params: Parameter dictionary (uses defaults if None)
        initial_state: Initial model state
        warmup_days: Warmup period in days
        use_jax: Whether to prefer JAX backend
        day_of_year_start: Day of year for first timestep

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    hms_params = create_params_from_dict(params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, temp, pet, hms_params, initial_state,
                            warmup_days, day_of_year_start)
    else:
        return simulate_numpy(precip, temp, pet, hms_params, initial_state,
                              warmup_days, day_of_year_start)
