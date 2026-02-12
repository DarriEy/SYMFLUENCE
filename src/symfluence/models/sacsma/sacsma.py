"""
Sacramento Soil Moisture Accounting (SAC-SMA) Model.

Burnash (1995) dual-layer tension/free water conceptual model.
Pure NumPy implementation.

Key physics:
1. ET demand sequence: UZTWC -> UZFWC -> LZTWC -> ADIMC
2. Percolation with ZPERC/REXP demand curve
3. Surface runoff: direct (PCTIM), saturation-excess (ADIMP), UZ overflow
4. Interflow from upper zone free water
5. Primary + supplemental baseflow from lower zone
6. Deep recharge (SIDE fraction lost)

References:
    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
    Computer Models of Watershed Hydrology, 311-366.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from .parameters import SacSmaParameters


class SacSmaState(NamedTuple):
    """SAC-SMA model state variables (all in mm)."""
    uztwc: float   # Upper zone tension water content
    uzfwc: float   # Upper zone free water content
    lztwc: float   # Lower zone tension water content
    lzfpc: float   # Lower zone primary free water content
    lzfsc: float   # Lower zone supplemental free water content
    adimc: float   # Additional impervious area content


def _create_default_state(params: SacSmaParameters) -> SacSmaState:
    """Create initial state at 50% capacity."""
    return SacSmaState(
        uztwc=params.UZTWM * 0.5,
        uzfwc=params.UZFWM * 0.5,
        lztwc=params.LZTWM * 0.5,
        lzfpc=params.LZFPM * 0.5,
        lzfsc=params.LZFSM * 0.5,
        adimc=(params.UZTWM + params.LZTWM) * 0.5,
    )


def sacsma_step(
    pxv: float,
    pet: float,
    dt: float,
    state: SacSmaState,
    params: SacSmaParameters,
) -> Tuple[SacSmaState, float, float, float, float]:
    """Execute one timestep of the SAC-SMA model.

    Args:
        pxv: Effective precipitation (rain + snowmelt, mm/dt)
        pet: Potential evapotranspiration (mm/dt)
        dt: Timestep in days (1.0 for daily)
        state: Current model state
        params: SAC-SMA parameters

    Returns:
        Tuple of (new_state, surface_runoff, interflow, baseflow, actual_et)
        All fluxes in mm/dt.
    """
    uztwc = state.uztwc
    uzfwc = state.uzfwc
    lztwc = state.lztwc
    lzfpc = state.lzfpc
    lzfsc = state.lzfsc
    adimc = state.adimc

    # Capacity parameters
    uztwm = params.UZTWM
    uzfwm = params.UZFWM
    lztwm = params.LZTWM
    lzfpm = params.LZFPM
    lzfsm = params.LZFSM

    # Ensure states are non-negative and within bounds
    uztwc = max(0.0, min(uztwc, uztwm))
    uzfwc = max(0.0, min(uzfwc, uzfwm))
    lztwc = max(0.0, min(lztwc, lztwm))
    lzfpc = max(0.0, min(lzfpc, lzfpm))
    lzfsc = max(0.0, min(lzfsc, lzfsm))
    adimc = max(0.0, min(adimc, uztwm + lztwm))

    total_et = 0.0
    total_surface = 0.0
    total_interflow = 0.0
    total_baseflow = 0.0

    # =========================================================================
    # 1. EVAPOTRANSPIRATION
    # =========================================================================
    remaining_et = pet

    # ET from upper zone tension water
    e1 = min(remaining_et, uztwc)
    uztwc -= e1
    remaining_et -= e1
    total_et += e1

    # ET from upper zone free water (proportional to remaining demand)
    if remaining_et > 0 and uzfwc > 0:
        # Reduce free water proportionally
        e2_ratio = remaining_et / uztwm if uztwm > 0 else 0.0
        e2 = min(uzfwc, e2_ratio * uzfwc)
        uzfwc -= e2
        remaining_et -= e2
        total_et += e2

    # ET from lower zone tension water
    if remaining_et > 0 and lztwc > 0:
        e3_demand = remaining_et * (lztwc / (uztwm + lztwm)) if (uztwm + lztwm) > 0 else 0.0
        e3 = min(lztwc, e3_demand)
        lztwc -= e3
        remaining_et -= e3
        total_et += e3

    # ET from ADIMP area (additional impervious)
    if remaining_et > 0 and adimc > 0:
        e5 = min(adimc, remaining_et * params.ADIMP)
        adimc -= e5
        total_et += e5

    # Riparian vegetation loss
    if params.RIVA > 0:
        e_riva = pet * params.RIVA
        total_et += e_riva

    # =========================================================================
    # 2. PERCOLATION (upper zone -> lower zone)
    # =========================================================================
    # Lower zone deficiency ratio
    lz_capacity = lztwm + lzfpm + lzfsm
    lz_content = lztwc + lzfpc + lzfsc
    lz_deficiency = max(0.0, lz_capacity - lz_content)

    if lz_capacity > 0 and uzfwc > 0:
        lz_def_ratio = lz_deficiency / lz_capacity
        # Percolation demand
        pbase = lzfpm * params.LZPK + lzfsm * params.LZSK
        perc_demand = pbase * (1.0 + params.ZPERC * (lz_def_ratio ** params.REXP))
        perc_demand *= dt

        # Actual percolation limited by available UZ free water
        perc = min(uzfwc, perc_demand)
        uzfwc -= perc

        # Distribute percolation to lower zone
        # Split between tension and free water
        # Tension water gets filled first, remainder goes to free water
        lztwc_deficit = max(0.0, lztwm - lztwc)

        # Fraction going to free water directly
        perc_to_free = perc * params.PFREE
        perc_to_tension = perc * (1.0 - params.PFREE)

        # Fill tension water
        to_lztw = min(perc_to_tension, lztwc_deficit)
        lztwc += to_lztw
        perc_remaining = perc_to_tension - to_lztw

        # Remaining tension allocation goes to free water
        perc_to_free += perc_remaining

        # Split free water between primary and supplemental
        if (lzfpm + lzfsm) > 0:
            frac_primary = lzfpm / (lzfpm + lzfsm)
        else:
            frac_primary = 0.5
        lzfpc += perc_to_free * frac_primary
        lzfsc += perc_to_free * (1.0 - frac_primary)
    else:
        perc = 0.0

    # =========================================================================
    # 3. SURFACE RUNOFF
    # =========================================================================
    if pxv > 0:
        # Direct runoff from permanent impervious area
        direct_runoff = pxv * params.PCTIM

        # Check for upper zone tension water overflow
        twx = max(0.0, pxv - (uztwm - uztwc))

        # Fill tension water
        uztwc = min(uztwc + pxv - twx, uztwm)

        # Excess water after tension water is full
        if twx > 0:
            # Try to fill free water
            fwx = max(0.0, twx - (uzfwm - uzfwc))
            uzfwc = min(uzfwc + twx - fwx, uzfwm)

            # Overflow becomes surface runoff
            if fwx > 0:
                direct_runoff += fwx

        # Additional impervious area runoff
        # ADIMP area generates runoff when saturated
        if params.ADIMP > 0:
            adimc += pxv
            adimc_max = uztwm + lztwm
            if adimc > adimc_max:
                adimp_ro = (adimc - adimc_max) * params.ADIMP
                adimc = adimc_max
            else:
                # Fraction of ADIMP area generating runoff
                adimp_ratio = adimc / adimc_max if adimc_max > 0 else 0.0
                adimp_ro = pxv * adimp_ratio * params.ADIMP
            direct_runoff += adimp_ro

        total_surface += direct_runoff
    else:
        # No precip: ADIMP tracks tension water
        if params.ADIMP > 0:
            adimc = max(adimc, uztwc)

    # =========================================================================
    # 4. INTERFLOW (upper zone free water depletion)
    # =========================================================================
    if uzfwc > 0:
        q_interflow = uzfwc * (1.0 - (1.0 - params.UZK) ** dt)
        q_interflow = min(q_interflow, uzfwc)
        uzfwc -= q_interflow
        total_interflow += q_interflow

    # =========================================================================
    # 5. BASEFLOW (lower zone free water depletion)
    # =========================================================================
    # Primary baseflow (slow)
    if lzfpc > 0:
        q_primary = lzfpc * (1.0 - (1.0 - params.LZPK) ** dt)
        q_primary = min(q_primary, lzfpc)
        lzfpc -= q_primary
    else:
        q_primary = 0.0

    # Supplemental baseflow (fast)
    if lzfsc > 0:
        q_supplemental = lzfsc * (1.0 - (1.0 - params.LZSK) ** dt)
        q_supplemental = min(q_supplemental, lzfsc)
        lzfsc -= q_supplemental
    else:
        q_supplemental = 0.0

    total_baseflow = q_primary + q_supplemental

    # Deep recharge loss (SIDE fraction)
    deep_loss = total_baseflow * params.SIDE

    # Reserve constraint: ensure RSERV fraction of LZ free water is maintained
    lzfp_reserve = lzfpm * params.RSERV
    lzfs_reserve = lzfsm * params.RSERV
    if lzfpc < lzfp_reserve and lzfsc > lzfs_reserve:
        transfer = min(lzfsc - lzfs_reserve, lzfp_reserve - lzfpc)
        lzfpc += transfer
        lzfsc -= transfer

    # Effective baseflow (subtract deep loss)
    effective_baseflow = max(0.0, total_baseflow - deep_loss)

    # =========================================================================
    # 6. ASSEMBLE STATE AND OUTPUTS
    # =========================================================================
    # Ensure non-negative
    uztwc = max(0.0, uztwc)
    uzfwc = max(0.0, uzfwc)
    lztwc = max(0.0, lztwc)
    lzfpc = max(0.0, lzfpc)
    lzfsc = max(0.0, lzfsc)
    adimc = max(0.0, adimc)

    new_state = SacSmaState(
        uztwc=uztwc,
        uzfwc=uzfwc,
        lztwc=lztwc,
        lzfpc=lzfpc,
        lzfsc=lzfsc,
        adimc=adimc,
    )

    return new_state, total_surface, total_interflow, effective_baseflow, total_et


def sacsma_simulate(
    pxv: np.ndarray,
    pet: np.ndarray,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, SacSmaState]:
    """Run SAC-SMA simulation over a time series.

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    n = len(pxv)
    total_flow = np.zeros(n)

    if initial_state is None:
        initial_state = _create_default_state(params)

    state = initial_state

    for i in range(n):
        state, surface, interflow, baseflow, _ = sacsma_step(
            pxv[i], pet[i], dt, state, params,
        )
        total_flow[i] = surface + interflow + baseflow

    return total_flow, state
