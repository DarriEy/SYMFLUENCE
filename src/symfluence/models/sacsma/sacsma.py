# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sacramento Soil Moisture Accounting (SAC-SMA) Model — Dual-Backend (JAX + NumPy).

Burnash (1995) dual-layer tension/free water conceptual model.
All branching uses ``xp.where()`` for JAX differentiability.

Key physics:
1. ET demand sequence: UZTWC -> UZFWC -> LZTWC -> ADIMC
2. Percolation with ZPERC/REXP demand curve
3. Surface runoff: direct (PCTIM), saturation-excess (ADIMP), UZ overflow
4. Interflow from upper zone free water
5. Primary + supplemental baseflow from lower zone
6. Deep recharge (SIDE fraction lost)

Two interfaces:
- Functional API: ``sacsma_step()`` with ``xp`` parameter
- ``lax.scan``-based: ``sacsma_simulate_jax()``

References:
    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
    Computer Models of Watershed Hydrology, 311-366.
"""

from typing import Any, NamedTuple, Optional, Tuple

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

from .parameters import SacSmaParameters

__all__ = [
    'HAS_JAX',
    'SacSmaState',
    'sacsma_step',
    'sacsma_simulate_jax',
    'sacsma_simulate_numpy',
    'sacsma_simulate',
    '_create_default_state',
]


class SacSmaState(NamedTuple):
    """SAC-SMA model state variables (all in mm). Fields use Any for JAX."""
    uztwc: Any   # Upper zone tension water content
    uzfwc: Any   # Upper zone free water content
    lztwc: Any   # Lower zone tension water content
    lzfpc: Any   # Lower zone primary free water content
    lzfsc: Any   # Lower zone supplemental free water content
    adimc: Any   # Additional impervious area content


def _create_default_state(
    params: SacSmaParameters,
    use_jax: bool = False,
) -> SacSmaState:
    """Create initial state at 50% capacity.

    Args:
        params: SAC-SMA parameters
        use_jax: Whether to use JAX arrays

    Returns:
        SacSmaState at 50% of storage capacities
    """
    uztwc = params.UZTWM * 0.5
    uzfwc = params.UZFWM * 0.5
    lztwc = params.LZTWM * 0.5
    lzfpc = params.LZFPM * 0.5
    lzfsc = params.LZFSM * 0.5
    adimc = (params.UZTWM + params.LZTWM) * 0.5

    if use_jax and HAS_JAX:
        return SacSmaState(
            uztwc=jnp.asarray(uztwc, dtype=float),
            uzfwc=jnp.asarray(uzfwc, dtype=float),
            lztwc=jnp.asarray(lztwc, dtype=float),
            lzfpc=jnp.asarray(lzfpc, dtype=float),
            lzfsc=jnp.asarray(lzfsc, dtype=float),
            adimc=jnp.asarray(adimc, dtype=float),
        )
    else:
        return SacSmaState(
            uztwc=np.float64(uztwc),
            uzfwc=np.float64(uzfwc),
            lztwc=np.float64(lztwc),
            lzfpc=np.float64(lzfpc),
            lzfsc=np.float64(lzfsc),
            adimc=np.float64(adimc),
        )


def sacsma_step(
    pxv: Any,
    pet: Any,
    dt: float,
    state: SacSmaState,
    params: SacSmaParameters,
    xp: Any = np,
) -> Tuple[SacSmaState, Any, Any, Any, Any]:
    """Execute one timestep of the SAC-SMA model (branch-free).

    All branching uses ``xp.where()`` for JAX differentiability.

    Args:
        pxv: Effective precipitation (rain + snowmelt, mm/dt)
        pet: Potential evapotranspiration (mm/dt)
        dt: Timestep in days (1.0 for daily)
        state: Current model state
        params: SAC-SMA parameters
        xp: Array backend (jnp or np)

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
    uztwc = xp.clip(uztwc, 0.0, uztwm)
    uzfwc = xp.clip(uzfwc, 0.0, uzfwm)
    lztwc = xp.clip(lztwc, 0.0, lztwm)
    lzfpc = xp.clip(lzfpc, 0.0, lzfpm)
    lzfsc = xp.clip(lzfsc, 0.0, lzfsm)
    adimc = xp.clip(adimc, 0.0, uztwm + lztwm)

    zero = xp.asarray(0.0, dtype=float)
    total_et = zero
    total_surface = zero
    total_interflow = zero

    # =========================================================================
    # 1. EVAPOTRANSPIRATION
    # =========================================================================
    remaining_et = pet

    # ET from upper zone tension water
    e1 = xp.minimum(remaining_et, uztwc)
    uztwc = uztwc - e1
    remaining_et = remaining_et - e1
    total_et = total_et + e1

    # ET from upper zone free water (proportional to remaining demand)
    e2_ratio = xp.where(uztwm > 0.0, remaining_et / xp.maximum(uztwm, 1e-10), zero)
    e2 = xp.minimum(uzfwc, e2_ratio * uzfwc)
    e2 = xp.where((remaining_et > 0.0) & (uzfwc > 0.0), e2, zero)
    uzfwc = uzfwc - e2
    remaining_et = remaining_et - e2
    total_et = total_et + e2

    # ET from lower zone tension water
    total_tw = uztwm + lztwm
    e3_demand = xp.where(total_tw > 0.0, remaining_et * (lztwc / xp.maximum(total_tw, 1e-10)), zero)
    e3 = xp.minimum(lztwc, e3_demand)
    e3 = xp.where((remaining_et > 0.0) & (lztwc > 0.0), e3, zero)
    lztwc = lztwc - e3
    remaining_et = remaining_et - e3
    total_et = total_et + e3

    # ET from ADIMP area
    e5 = xp.minimum(adimc, remaining_et * params.ADIMP)
    e5 = xp.where((remaining_et > 0.0) & (adimc > 0.0), e5, zero)
    adimc = adimc - e5
    total_et = total_et + e5

    # Riparian vegetation loss
    e_riva = pet * params.RIVA
    total_et = total_et + e_riva

    # =========================================================================
    # 2. PERCOLATION (upper zone -> lower zone)
    # =========================================================================
    lz_capacity = lztwm + lzfpm + lzfsm
    lz_content = lztwc + lzfpc + lzfsc
    lz_deficiency = xp.maximum(zero, lz_capacity - lz_content)

    # Safe deficiency ratio for power function
    lz_def_ratio = lz_deficiency / xp.maximum(lz_capacity, 1e-10)
    safe_def_ratio = xp.maximum(lz_def_ratio, 1e-10)

    # Percolation demand
    pbase = lzfpm * params.LZPK + lzfsm * params.LZSK
    perc_demand = pbase * (1.0 + params.ZPERC * xp.power(safe_def_ratio, params.REXP))
    perc_demand = perc_demand * dt

    # Actual percolation limited by available UZ free water
    perc = xp.minimum(uzfwc, perc_demand)
    perc = xp.where((lz_capacity > 0.0) & (uzfwc > 0.0), perc, zero)
    uzfwc = uzfwc - perc

    # Distribute percolation to lower zone
    lztwc_deficit = xp.maximum(zero, lztwm - lztwc)
    perc_to_free = perc * params.PFREE
    perc_to_tension = perc * (1.0 - params.PFREE)

    to_lztw = xp.minimum(perc_to_tension, lztwc_deficit)
    lztwc = lztwc + to_lztw
    perc_remaining = perc_to_tension - to_lztw
    perc_to_free = perc_to_free + perc_remaining

    # Split free water between primary and supplemental
    frac_primary = xp.where(
        (lzfpm + lzfsm) > 0.0,
        lzfpm / xp.maximum(lzfpm + lzfsm, 1e-10),
        xp.asarray(0.5, dtype=float),
    )
    lzfpc = lzfpc + perc_to_free * frac_primary
    lzfsc = lzfsc + perc_to_free * (1.0 - frac_primary)

    # =========================================================================
    # 3. SURFACE RUNOFF
    # =========================================================================
    # Direct runoff from permanent impervious area
    direct_runoff = pxv * params.PCTIM

    # Upper zone tension water excess
    twx = xp.maximum(zero, pxv - (uztwm - uztwc))
    uztwc_new = xp.minimum(uztwc + pxv - twx, uztwm)

    # Free water excess (only when twx > 0)
    fwx = xp.maximum(zero, twx - (uzfwm - uzfwc))
    uzfwc_new = xp.minimum(uzfwc + twx - fwx, uzfwm)

    # When twx > 0: use new values and add overflow
    uztwc = xp.where(pxv > 0.0, uztwc_new, uztwc)
    overflow_ro = xp.where(twx > 0.0, fwx, zero)
    uzfwc = xp.where((pxv > 0.0) & (twx > 0.0), uzfwc_new, uzfwc)
    direct_runoff = direct_runoff + overflow_ro

    # ADIMP area runoff
    adimc_max = uztwm + lztwm
    # When pxv > 0: add precip to adimc, compute ADIMP runoff
    adimc_wet = adimc + pxv
    # Overflow path: adimc > max
    adimp_overflow = (adimc_wet - adimc_max) * params.ADIMP
    # Saturation-fraction path: adimc <= max
    adimp_ratio = adimc_wet / xp.maximum(adimc_max, 1e-10)
    adimp_partial = pxv * adimp_ratio * params.ADIMP
    adimp_ro = xp.where(adimc_wet > adimc_max, adimp_overflow, adimp_partial)
    adimc_after_wet = xp.where(adimc_wet > adimc_max, adimc_max, adimc_wet)

    # When no precip: ADIMP tracks tension water
    adimc_dry = xp.maximum(adimc, uztwc)

    # Select based on pxv > 0 and ADIMP > 0
    has_adimp = params.ADIMP > 0.0
    adimc = xp.where(
        pxv > 0.0,
        xp.where(has_adimp, adimc_after_wet, adimc),
        xp.where(has_adimp, adimc_dry, adimc),
    )
    direct_runoff = xp.where((pxv > 0.0) & has_adimp, direct_runoff + adimp_ro, direct_runoff)

    # Only count direct runoff when pxv > 0
    total_surface = xp.where(pxv > 0.0, direct_runoff, zero)

    # =========================================================================
    # 4. INTERFLOW (upper zone free water depletion)
    # =========================================================================
    q_interflow = uzfwc * (1.0 - xp.power(1.0 - params.UZK, dt))
    q_interflow = xp.minimum(q_interflow, uzfwc)
    q_interflow = xp.where(uzfwc > 0.0, q_interflow, zero)
    uzfwc = uzfwc - q_interflow
    total_interflow = q_interflow

    # =========================================================================
    # 5. BASEFLOW (lower zone free water depletion)
    # =========================================================================
    # Primary baseflow (slow)
    q_primary = lzfpc * (1.0 - xp.power(1.0 - params.LZPK, dt))
    q_primary = xp.minimum(q_primary, lzfpc)
    q_primary = xp.where(lzfpc > 0.0, q_primary, zero)
    lzfpc = lzfpc - q_primary

    # Supplemental baseflow (fast)
    q_supplemental = lzfsc * (1.0 - xp.power(1.0 - params.LZSK, dt))
    q_supplemental = xp.minimum(q_supplemental, lzfsc)
    q_supplemental = xp.where(lzfsc > 0.0, q_supplemental, zero)
    lzfsc = lzfsc - q_supplemental

    total_baseflow = q_primary + q_supplemental

    # Deep recharge loss (SIDE fraction)
    deep_loss = total_baseflow * params.SIDE

    # Reserve constraint: ensure RSERV fraction of LZ free water is maintained
    lzfp_reserve = lzfpm * params.RSERV
    lzfs_reserve = lzfsm * params.RSERV
    transfer = xp.minimum(lzfsc - lzfs_reserve, lzfp_reserve - lzfpc)
    transfer = xp.maximum(transfer, zero)
    do_transfer = (lzfpc < lzfp_reserve) & (lzfsc > lzfs_reserve)
    lzfpc = xp.where(do_transfer, lzfpc + transfer, lzfpc)
    lzfsc = xp.where(do_transfer, lzfsc - transfer, lzfsc)

    # Effective baseflow (subtract deep loss)
    effective_baseflow = xp.maximum(zero, total_baseflow - deep_loss)

    # =========================================================================
    # 6. ASSEMBLE STATE AND OUTPUTS
    # =========================================================================
    new_state = SacSmaState(
        uztwc=xp.maximum(uztwc, zero),
        uzfwc=xp.maximum(uzfwc, zero),
        lztwc=xp.maximum(lztwc, zero),
        lzfpc=xp.maximum(lzfpc, zero),
        lzfsc=xp.maximum(lzfsc, zero),
        adimc=xp.maximum(adimc, zero),
    )

    return new_state, total_surface, total_interflow, effective_baseflow, total_et


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def sacsma_simulate_jax(
    pxv: Any,
    pet: Any,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
) -> Tuple[Any, SacSmaState]:
    """Run SAC-SMA simulation using JAX lax.scan.

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    if not HAS_JAX:
        return sacsma_simulate_numpy(
            np.asarray(pxv), np.asarray(pet), params, initial_state, dt,
        )

    if initial_state is None:
        initial_state = _create_default_state(params, use_jax=True)

    forcing = jnp.stack([pxv, pet], axis=1)

    def scan_fn(carry, forcing_step):
        p, e = forcing_step
        new_state, surface, interflow, baseflow, _ = sacsma_step(
            p, e, dt, carry, params, xp=jnp,
        )
        return new_state, surface + interflow + baseflow

    final_state, total_flow = lax.scan(scan_fn, initial_state, forcing)
    return total_flow, final_state


def sacsma_simulate_numpy(
    pxv: np.ndarray,
    pet: np.ndarray,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, SacSmaState]:
    """Run SAC-SMA simulation using NumPy (Python loop fallback).

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
        initial_state = _create_default_state(params, use_jax=False)

    state = initial_state

    for i in range(n):
        state, surface, interflow, baseflow, _ = sacsma_step(
            np.float64(pxv[i]), np.float64(pet[i]), dt, state, params, xp=np,
        )
        total_flow[i] = float(surface + interflow + baseflow)

    return total_flow, state


def sacsma_simulate(
    pxv: Any,
    pet: Any,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
    use_jax: bool = False,
) -> Tuple[Any, SacSmaState]:
    """Run SAC-SMA simulation with automatic backend selection.

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days
        use_jax: Whether to prefer JAX backend

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    if use_jax and HAS_JAX:
        return sacsma_simulate_jax(pxv, pet, params, initial_state, dt)
    else:
        return sacsma_simulate_numpy(
            np.asarray(pxv), np.asarray(pet), params, initial_state, dt,
        )
