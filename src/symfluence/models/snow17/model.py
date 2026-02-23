"""
Snow-17 Temperature Index Snow Model — Dual-Backend (JAX + NumPy).

Anderson (1973, 2006) temperature-index model for snow accumulation and ablation.
All branching logic uses ``xp.where()`` for JAX differentiability.

Key physics:
1. Rain/snow partition via PXTEMP with 2C mixed-phase transition
2. Snowfall correction (SCF)
3. Seasonal melt factor sinusoid (hemisphere-aware)
4. Non-rain and rain-on-snow melt
5. Heat deficit tracking (TIPM-weighted antecedent temperature)
6. Liquid water routing with PLWHC capacity and refreezing
7. Areal depletion curve
8. Daily ground melt (DAYGM)

Two interfaces:
- Functional API: ``snow17_step()``, ``snow17_simulate()`` with ``xp`` parameter
- ``lax.scan``-based: ``snow17_simulate_jax()``

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.
"""

from typing import Any, Optional, Tuple

import numpy as np

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

from .parameters import (
    DEFAULT_ADC,
    SNOW17_DEFAULTS,
    Snow17Params,
    Snow17State,
    create_initial_state,
    params_dict_to_namedtuple,
)

__all__ = [
    'HAS_JAX',
    'seasonal_melt_factor',
    'snow17_step',
    'snow17_simulate_jax',
    'snow17_simulate_numpy',
    'snow17_simulate',
    'create_initial_state',
    'Snow17State',
]


# =============================================================================
# PHYSICS FUNCTIONS (BACKEND-AGNOSTIC)
# =============================================================================

def seasonal_melt_factor(
    doy: Any,
    mfmax: Any,
    mfmin: Any,
    lat: float = 45.0,
    xp: Any = np,
) -> Any:
    """Compute seasonal melt factor using sinusoidal variation.

    Melt factor varies sinusoidally between MFMIN (Dec 21) and MFMAX (Jun 21)
    in the Northern Hemisphere (reversed for Southern).

    Args:
        doy: Day of year (1-366), scalar or array
        mfmax: Maximum melt factor (mm/C/6hr)
        mfmin: Minimum melt factor (mm/C/6hr)
        lat: Latitude in degrees (positive=North)
        xp: Array backend (jnp or np)

    Returns:
        Melt factor (mm/C/6hr)
    """
    phase_shift = xp.where(
        xp.asarray(lat) >= 0.0,
        xp.asarray(0.0),
        xp.asarray(183.0),
    )
    mf = (mfmax + mfmin) / 2.0 + (mfmax - mfmin) / 2.0 * xp.sin(
        (xp.asarray(doy, dtype=float) - 81.0 + phase_shift) * 2.0 * xp.pi / 365.0
    )
    return mf


def areal_depletion(
    swe: Any,
    si: float,
    adc: Any,
    xp: Any = np,
) -> Any:
    """Compute areal snow cover fraction from depletion curve.

    Args:
        swe: Current snow water equivalent (mm)
        si: SWE threshold for 100% areal coverage (mm)
        adc: 11-point areal depletion curve array
        xp: Array backend (jnp or np)

    Returns:
        Fraction of area covered by snow (0-1)
    """
    ratio = xp.clip(swe / xp.maximum(xp.asarray(si, dtype=float), 1e-10), 0.0, 1.0)
    # Linear interpolation using xp.interp
    adc_x = xp.linspace(0.0, 1.0, 11)
    cover = xp.interp(ratio, adc_x, adc)
    return cover


def snow17_step(
    precip: Any,
    temp: Any,
    dt: float,
    state: Snow17State,
    params: Snow17Params,
    doy: Any,
    lat: float = 45.0,
    si: float = 100.0,
    adc: Any = None,
    xp: Any = np,
) -> Tuple[Snow17State, Any]:
    """Execute one timestep of the Snow-17 model (branch-free).

    All branching uses ``xp.where()`` for JAX differentiability.

    Args:
        precip: Total precipitation (mm/dt)
        temp: Air temperature (C)
        dt: Timestep in days (1.0 for daily)
        state: Current Snow-17 state
        params: Snow-17 parameters
        doy: Day of year (1-366)
        lat: Latitude (degrees, positive=North)
        si: SWE threshold for 100% areal coverage (mm)
        adc: Areal depletion curve (11-point). Uses DEFAULT_ADC if None.
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (new_state, rain_plus_melt in mm/dt)
    """
    if adc is None:
        adc = xp.asarray(DEFAULT_ADC, dtype=float)

    w_i = state.w_i
    w_q = state.w_q
    w_qx = state.w_qx
    deficit = state.deficit
    ati = state.ati
    swe = state.swe

    dt_f = xp.asarray(dt, dtype=float)

    # Melt factor (6hr units in params, scale to timestep)
    mf = seasonal_melt_factor(doy, params.MFMAX, params.MFMIN, lat, xp)
    mf_dt = mf * (dt_f * 4.0)
    nmf_dt = params.NMF * (dt_f * 4.0)
    daygm_dt = params.DAYGM * dt_f

    # --- Rain/snow partition with 2C mixed-phase transition ---
    t_low = params.PXTEMP - 1.0
    t_high = params.PXTEMP + 1.0
    frac_rain = xp.clip((temp - t_low) / xp.maximum(t_high - t_low, 1e-10), 0.0, 1.0)

    rain = precip * frac_rain
    snow = precip * (1.0 - frac_rain) * params.SCF

    # Update SWE tracker
    swe = xp.maximum(swe, w_i + w_q)

    # --- Accumulation ---
    w_i = w_i + snow

    # --- Temperature index update ---
    # When snow exists: update ATI towards current temp
    ati_snow = ati + params.TIPM * (temp - ati)
    # Constrain ATI to non-positive when cold
    ati_snow = xp.where(temp < 0.0, xp.minimum(ati_snow, 0.0), ati_snow)
    # When no snow: ATI = temp
    ati = xp.where(w_i > 0.0, ati_snow, temp)

    # --- Areal depletion ---
    total_pack = w_i + w_q
    areal_cover = xp.where(
        xp.asarray(si, dtype=float) > 0.0,
        areal_depletion(total_pack, si, adc, xp),
        xp.where(total_pack > 0.0, 1.0, 0.0),
    )

    # --- Melt computation (all branches computed, selected with xp.where) ---
    non_rain_melt = mf_dt * xp.maximum(temp - params.MBASE, 0.0) * areal_cover

    # Rain-on-snow melt
    ros_melt = xp.maximum(0.0, params.UADJ * dt_f * 4.0 * (temp - params.MBASE))
    rain_heat = 0.0125 * rain * xp.maximum(temp, 0.0)
    ros_total = ros_melt + rain_heat

    # Select: if rain > 0.3*dt → max(non_rain, ros), else non_rain
    melt_warm = xp.where(rain > 0.3 * dt_f, xp.maximum(non_rain_melt, ros_total), non_rain_melt)

    # If temp > mbase → warm melt + ground melt, else just ground melt
    melt = xp.where(
        temp > params.MBASE,
        melt_warm + daygm_dt * areal_cover,
        daygm_dt * areal_cover,
    )

    # No melt if no ice
    melt = xp.where(w_i > 0.0, melt, 0.0)

    # Limit melt to available ice
    melt = xp.minimum(melt, w_i)

    # --- Heat deficit update ---
    # Cold: increase deficit; warm: decrease deficit
    deficit_cold = deficit + nmf_dt * xp.maximum(params.MBASE - temp, 0.0) * areal_cover
    deficit_warm = xp.maximum(0.0, deficit - melt * 0.33)
    deficit = xp.where(w_i > 0.0, xp.where(temp < 0.0, deficit_cold, deficit_warm), deficit)

    # Apply deficit: melt must overcome deficit before liquid water released
    melt_after_deficit = xp.maximum(melt - deficit, 0.0)
    deficit_after_melt = xp.maximum(deficit - melt, 0.0)

    # Only apply deficit logic when both deficit > 0 and melt > 0
    has_both = (deficit > 0.0) & (melt > 0.0)
    melt = xp.where(has_both & (w_i > 0.0), melt_after_deficit, melt)
    deficit = xp.where(has_both & (w_i > 0.0), deficit_after_melt, deficit)

    # Update ice storage
    w_i = w_i - melt

    # --- Liquid water routing ---
    w_q = w_q + melt + rain

    # Liquid water holding capacity
    w_qx = params.PLWHC * w_i

    # Excess liquid water becomes outflow
    outflow = xp.maximum(w_q - w_qx, 0.0)
    w_q = xp.minimum(w_q, w_qx)

    # Refreezing when cold
    refreeze = xp.where(
        (temp < params.MBASE) & (w_q > 0.0),
        xp.minimum(w_q, nmf_dt * xp.maximum(params.MBASE - temp, 0.0)),
        0.0,
    )
    w_i = w_i + refreeze
    w_q = w_q - refreeze

    # Clean up when no snow remains
    no_snow = (w_i <= 0.0) & (w_q <= 0.0)
    w_i = xp.where(no_snow, 0.0, xp.maximum(w_i, 0.0))
    w_q = xp.where(no_snow, 0.0, xp.maximum(w_q, 0.0))
    w_qx = xp.where(no_snow, 0.0, xp.maximum(w_qx, 0.0))
    deficit = xp.where(no_snow, 0.0, xp.maximum(deficit, 0.0))
    swe_current = w_i + w_q
    swe = xp.where(no_snow, 0.0, xp.maximum(swe, swe_current))

    new_state = Snow17State(
        w_i=w_i, w_q=w_q, w_qx=w_qx,
        deficit=deficit, ati=ati, swe=swe,
    )

    return new_state, xp.maximum(outflow, 0.0)


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def snow17_simulate_jax(
    precip: Any,
    temp: Any,
    doy: Any,
    params: Snow17Params,
    state: Optional[Snow17State] = None,
    lat: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
    adc: Any = None,
) -> Tuple[Any, Snow17State]:
    """Run Snow-17 simulation using JAX lax.scan.

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (C)
        doy: Day of year array (1-366)
        params: Snow-17 parameters
        state: Initial state (default: no snow)
        lat: Latitude (degrees)
        si: SWE threshold for areal depletion
        dt: Timestep in days
        adc: Areal depletion curve

    Returns:
        Tuple of (rain_plus_melt array, final state)
    """
    if not HAS_JAX:
        return snow17_simulate_numpy(precip, temp, doy, params, state, lat, si, dt, adc)

    if state is None:
        state = create_initial_state(use_jax=True)

    if adc is None:
        adc = jnp.asarray(DEFAULT_ADC, dtype=float)

    forcing = jnp.stack([precip, temp, jnp.asarray(doy, dtype=float)], axis=1)

    def scan_fn(carry, forcing_step):
        p, t, d = forcing_step
        new_state, rpm = snow17_step(p, t, dt, carry, params, d, lat, si, adc, xp=jnp)
        return new_state, rpm

    final_state, rain_plus_melt = lax.scan(scan_fn, state, forcing)
    return rain_plus_melt, final_state


def snow17_simulate_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    doy: np.ndarray,
    params: Snow17Params,
    state: Optional[Snow17State] = None,
    lat: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
    adc: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Snow17State]:
    """Run Snow-17 simulation using NumPy (Python loop fallback).

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (C)
        doy: Day of year array (1-366)
        params: Snow-17 parameters
        state: Initial state (default: no snow)
        lat: Latitude (degrees)
        si: SWE threshold for areal depletion
        dt: Timestep in days
        adc: Areal depletion curve

    Returns:
        Tuple of (rain_plus_melt array, final state)
    """
    n = len(precip)
    rain_plus_melt = np.zeros(n)

    if state is None:
        state = create_initial_state(use_jax=False)

    if adc is None:
        adc = DEFAULT_ADC.copy()

    for i in range(n):
        state, rpm = snow17_step(
            np.float64(precip[i]), np.float64(temp[i]), dt, state, params,
            np.float64(doy[i]), lat, si, adc, xp=np,
        )
        rain_plus_melt[i] = float(rpm)

    return rain_plus_melt, state


def snow17_simulate(
    precip: Any,
    temp: Any,
    doy: Any,
    params=None,
    state: Optional[Snow17State] = None,
    lat: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
    use_jax: bool = True,
    adc: Any = None,
) -> Tuple[Any, Snow17State]:
    """High-level Snow-17 simulation with automatic backend selection.

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (C)
        doy: Day of year array (1-366)
        params: Snow17Params namedtuple or dict (uses defaults if None)
        state: Initial state (default: no snow)
        lat: Latitude (degrees)
        si: SWE threshold for areal depletion
        dt: Timestep in days
        use_jax: Whether to prefer JAX backend
        adc: Areal depletion curve

    Returns:
        Tuple of (rain_plus_melt array, final state)
    """
    if params is None:
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=(use_jax and HAS_JAX))
    elif isinstance(params, dict):
        params = params_dict_to_namedtuple(params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return snow17_simulate_jax(precip, temp, doy, params, state, lat, si, dt, adc)
    else:
        return snow17_simulate_numpy(
            np.asarray(precip), np.asarray(temp), np.asarray(doy),
            params, state, lat, si, dt,
            np.asarray(adc) if adc is not None else None,
        )
