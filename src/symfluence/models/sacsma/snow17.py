"""
Snow-17 Temperature Index Snow Model.

Anderson (1973, 2006) temperature-index model for snow accumulation and ablation.
Pure NumPy implementation.

Key physics:
1. Rain/snow partition via PXTEMP with 2°C mixed-phase transition
2. Snowfall correction (SCF)
3. Seasonal melt factor sinusoid
4. Non-rain and rain-on-snow melt
5. Heat deficit tracking (TIPM-weighted antecedent temperature)
6. Liquid water routing with PLWHC capacity and refreezing
7. Areal depletion curve
8. Daily ground melt (DAYGM)

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from .parameters import Snow17Parameters


class Snow17State(NamedTuple):
    """Snow-17 model state variables."""
    w_i: float       # Ice portion of SWE (mm)
    w_q: float       # Liquid water in snowpack (mm)
    w_qx: float      # Liquid water capacity (mm)
    deficit: float   # Heat deficit (mm, energy equiv.)
    ati: float       # Antecedent temperature index (°C)
    swe: float       # Total SWE for areal depletion (mm)


# Default 11-point areal depletion curve (fraction of area covered by snow)
# Index corresponds to SWE/SI ratio from 0.0 to 1.0 in 0.1 increments
DEFAULT_ADC = np.array([
    0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
])


def _seasonal_melt_factor(day_of_year: int, mfmax: float, mfmin: float,
                          latitude: float = 45.0) -> float:
    """Compute seasonal melt factor using sinusoidal variation.

    Melt factor varies sinusoidally between MFMIN (Dec 21) and MFMAX (Jun 21)
    in the Northern Hemisphere (reversed for Southern).

    Args:
        day_of_year: Julian day (1-366)
        mfmax: Maximum melt factor (mm/°C/6hr)
        mfmin: Minimum melt factor (mm/°C/6hr)
        latitude: Latitude in degrees (positive=North)

    Returns:
        Melt factor (mm/°C/6hr)
    """
    # Day 81 = March 22 (equinox), sin peaks at day 172 (Jun 21)
    if latitude >= 0:
        # Northern hemisphere
        mf = (mfmax + mfmin) / 2.0 + (mfmax - mfmin) / 2.0 * np.sin(
            (day_of_year - 81) * 2.0 * np.pi / 365.0
        )
    else:
        # Southern hemisphere: shift by 6 months
        mf = (mfmax + mfmin) / 2.0 + (mfmax - mfmin) / 2.0 * np.sin(
            (day_of_year - 81 + 183) * 2.0 * np.pi / 365.0
        )
    return float(mf)


def _areal_depletion(swe: float, si: float, adc: np.ndarray = DEFAULT_ADC) -> float:
    """Compute areal snow cover fraction from depletion curve.

    Args:
        swe: Current snow water equivalent (mm)
        si: SWE threshold for 100% areal coverage (mm)
        adc: 11-point areal depletion curve

    Returns:
        Fraction of area covered by snow (0-1)
    """
    if si <= 0 or swe <= 0:
        return 0.0
    ratio = min(swe / si, 1.0)
    idx = ratio * 10.0
    idx_lo = int(idx)
    idx_hi = min(idx_lo + 1, 10)
    frac = idx - idx_lo
    return float(adc[idx_lo] * (1.0 - frac) + adc[idx_hi] * frac)


def snow17_step(
    precip: float,
    temp: float,
    dt: float,
    state: Snow17State,
    params: Snow17Parameters,
    day_of_year: int,
    latitude: float = 45.0,
    si: float = 100.0,
    adc: np.ndarray = DEFAULT_ADC,
) -> Tuple[Snow17State, float]:
    """Execute one timestep of the Snow-17 model.

    Args:
        precip: Total precipitation (mm/dt)
        temp: Air temperature (°C)
        dt: Timestep in days (1.0 for daily)
        state: Current Snow-17 state
        params: Snow-17 parameters
        day_of_year: Julian day (1-366)
        latitude: Latitude (degrees, positive=North)
        si: SWE threshold for 100% areal coverage (mm)
        adc: Areal depletion curve (11-point array)

    Returns:
        Tuple of (new_state, rain_plus_melt in mm/dt)
    """
    w_i = state.w_i
    w_q = state.w_q
    w_qx = state.w_qx
    deficit = state.deficit
    ati = state.ati
    swe = state.swe

    # Melt factor (6hr units in params, scale to daily: ×4 for dt=1)
    mf = _seasonal_melt_factor(day_of_year, params.MFMAX, params.MFMIN, latitude)
    # Scale from 6hr to timestep
    mf_dt = mf * (dt * 4.0)
    nmf_dt = params.NMF * (dt * 4.0)
    daygm_dt = params.DAYGM * dt

    # --- Rain/snow partition with 2°C mixed-phase transition ---
    t_low = params.PXTEMP - 1.0   # All snow below this
    t_high = params.PXTEMP + 1.0  # All rain above this

    if temp <= t_low:
        frac_rain = 0.0
    elif temp >= t_high:
        frac_rain = 1.0
    else:
        frac_rain = (temp - t_low) / (t_high - t_low)

    rain = precip * frac_rain
    snow = precip * (1.0 - frac_rain) * params.SCF

    # Update SWE tracker (for areal depletion)
    swe = max(swe, w_i + w_q)  # Track maximum pack

    # --- Accumulation ---
    w_i += snow

    # --- Temperature index update ---
    if w_i > 0:
        # When snow exists, update ATI towards current temp
        ati = ati + params.TIPM * (temp - ati)
        # Constrain ATI to non-positive when used for deficit
        ati = min(ati, 0.0) if temp < 0 else ati
    else:
        ati = temp

    # --- Areal depletion ---
    if si > 0:
        areal_cover = _areal_depletion(w_i + w_q, si, adc)
    else:
        areal_cover = 1.0 if (w_i + w_q) > 0 else 0.0

    # --- Melt computation ---
    melt = 0.0

    if w_i > 0:
        if temp > params.MBASE:
            # Non-rain melt
            non_rain_melt = mf_dt * (temp - params.MBASE) * areal_cover

            # Rain-on-snow melt (Stefan-Boltzmann + condensation)
            if rain > 0.3 * dt:  # Threshold for rain-on-snow
                # Simplified rain-on-snow: energy from warm rain
                ros_melt = max(0.0, params.UADJ * dt * 4.0 * (temp - params.MBASE))
                # Heat from rain itself
                rain_heat = 0.0125 * rain * max(temp, 0.0)
                ros_total = ros_melt + rain_heat
                melt = max(non_rain_melt, ros_total)
            else:
                melt = non_rain_melt

            # Ground melt
            melt += daygm_dt * areal_cover
        else:
            # Below melt temperature: only ground melt
            melt = daygm_dt * areal_cover

        # Limit melt to available ice
        melt = min(melt, w_i)

        # --- Heat deficit update ---
        if temp < 0:
            # Increase deficit (cold content builds)
            deficit += nmf_dt * (params.MBASE - temp) * areal_cover
        else:
            # Decrease deficit (warming reduces cold content)
            deficit = max(0.0, deficit - melt * 0.33)  # 0.33 = heat of fusion approx

        # Apply deficit: melt must overcome deficit before liquid water released
        if deficit > 0 and melt > 0:
            if melt > deficit:
                melt -= deficit
                deficit = 0.0
            else:
                deficit -= melt
                melt = 0.0

        # Update ice storage
        w_i -= melt

    # --- Liquid water routing ---
    # Add rain and melt to liquid water
    w_q += melt + rain

    # Liquid water holding capacity
    w_qx = params.PLWHC * w_i

    # Excess liquid water becomes outflow
    if w_q > w_qx:
        outflow = w_q - w_qx
        w_q = w_qx
    else:
        outflow = 0.0

    # Refreezing when cold
    if temp < params.MBASE and w_q > 0:
        refreeze = min(w_q, nmf_dt * (params.MBASE - temp))
        w_i += refreeze
        w_q -= refreeze
        deficit = max(0.0, deficit)  # Refreezing doesn't change deficit sign

    # If no snow remains, clean up state
    if w_i <= 0 and w_q <= 0:
        w_i = 0.0
        w_q = 0.0
        w_qx = 0.0
        deficit = 0.0
        swe = 0.0

    # Update SWE tracker
    current_swe = w_i + w_q
    swe = max(swe, current_swe)

    new_state = Snow17State(
        w_i=max(w_i, 0.0),
        w_q=max(w_q, 0.0),
        w_qx=max(w_qx, 0.0),
        deficit=max(deficit, 0.0),
        ati=ati,
        swe=swe,
    )

    return new_state, max(outflow, 0.0)


def snow17_simulate(
    precip: np.ndarray,
    temp: np.ndarray,
    day_of_year: np.ndarray,
    params: Snow17Parameters,
    initial_state: Optional[Snow17State] = None,
    dt: float = 1.0,
    latitude: float = 45.0,
    si: float = 100.0,
    adc: np.ndarray = DEFAULT_ADC,
) -> Tuple[np.ndarray, Snow17State]:
    """Run Snow-17 simulation over a time series.

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (°C)
        day_of_year: Julian day array (1-366)
        params: Snow-17 parameters
        initial_state: Initial state (default: no snow)
        dt: Timestep in days
        latitude: Latitude (degrees)
        si: SWE threshold for areal depletion
        adc: Areal depletion curve

    Returns:
        Tuple of (rain_plus_melt array in mm/dt, final state)
    """
    n = len(precip)
    rain_plus_melt = np.zeros(n)

    if initial_state is None:
        initial_state = Snow17State(
            w_i=0.0, w_q=0.0, w_qx=0.0,
            deficit=0.0, ati=0.0, swe=0.0,
        )

    state = initial_state

    for i in range(n):
        state, rpm = snow17_step(
            precip[i], temp[i], dt, state, params,
            int(day_of_year[i]), latitude, si, adc,
        )
        rain_plus_melt[i] = rpm

    return rain_plus_melt, state
