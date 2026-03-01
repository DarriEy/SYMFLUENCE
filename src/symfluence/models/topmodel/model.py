# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TOPMODEL Core - JAX Implementation.

Pure JAX/NumPy functions for the TOPMODEL (Beven & Kirkby 1979) algorithm:
1. Snow routine - Degree-day with rain/snow partition
2. TOPMODEL routine - Exponential transmissivity baseflow, saturation-excess
   overland flow, root zone / unsaturated zone accounting
3. Routing routine - Linear reservoir channel routing

Uses a parametric topographic index distribution (discretized normal) to
avoid DEM preprocessing. 11 calibration parameters, 5 logical state variables
(s_bar scalar, srz/suz arrays of N_TI_BINS, swe scalar, q_routed scalar).

Enables:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- DDS and other evolutionary calibration

References:
    Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable
    contributing area model of basin hydrology. Hydrological Sciences
    Bulletin, 24(1), 43-69.

    Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed.
    Wiley-Blackwell.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

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
        "JAX not available. TOPMODEL will use NumPy backend. "
        "Install JAX for autodiff, JIT compilation, and GPU support: pip install jax jaxlib"
    )

from .parameters import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    TopmodelParameters,
    TopmodelState,
    create_initial_state,
    create_params_from_dict,
    generate_ti_distribution,
)

__all__ = [
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'TopmodelParameters',
    'TopmodelState',
    'create_params_from_dict',
    'create_initial_state',
    'generate_ti_distribution',
    'HAS_JAX',
    'snow_step',
    'topmodel_step',
    'route_step',
    'step',
    'simulate_jax',
    'simulate_numpy',
    'simulate',
    'nse_loss',
    'kge_loss',
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
    swe: Any,
    params: TopmodelParameters,
    use_jax: bool = True
) -> Tuple[Any, Any, Any]:
    """
    Degree-day snow routine.

    Partitions precipitation at T_snow, melts at DDF * max(T - T_melt, 0).

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (deg C)
        swe: Current snow water equivalent (mm)
        params: Model parameters
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_swe, p_eff_mm, actual_melt_mm)
            p_eff_mm: Effective precipitation reaching soil (mm/day)
    """
    xp = _get_backend(use_jax)

    # Rain/snow partition
    rain = xp.where(temp > params.T_snow, precip, 0.0)
    snow = xp.where(temp <= params.T_snow, precip, 0.0)

    # Add snowfall to SWE
    new_swe = swe + snow

    # Potential melt
    pot_melt = params.DDF * xp.maximum(temp - params.T_melt, 0.0)

    # Actual melt limited by available SWE
    actual_melt = xp.minimum(pot_melt, new_swe)
    new_swe = new_swe - actual_melt

    # Effective precipitation = rain + melt
    p_eff = rain + actual_melt

    # Ensure non-negative
    new_swe = xp.maximum(new_swe, 0.0)

    return new_swe, p_eff, actual_melt


def topmodel_step(
    p_eff: Any,
    pet: Any,
    s_bar: Any,
    srz: Any,
    suz: Any,
    lnaotb: Any,
    dist_area: Any,
    params: TopmodelParameters,
    dt: float = 24.0,
    use_jax: bool = True
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    TOPMODEL subsurface routine for one timestep.

    Steps:
    1. Baseflow from exponential transmissivity profile
    2. Local deficit per TI class
    3. Root zone accounting (infiltration, ET)
    4. Unsaturated zone drainage
    5. Overland flow (saturation excess)
    6. Mean deficit update

    Args:
        p_eff: Effective precipitation (mm/day) - converted to m internally
        pet: Potential evapotranspiration (mm/day) - converted to m internally
        s_bar: Mean saturation deficit (m)
        srz: Root zone deficit per TI class (m), shape (n_bins,)
        suz: Unsaturated zone storage per TI class (m), shape (n_bins,)
        lnaotb: TI bin centers, shape (n_bins,)
        dist_area: Fractional area per bin, shape (n_bins,)
        params: Model parameters
        dt: Timestep in hours (24 for daily)
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_s_bar, new_srz, new_suz, q_baseflow, q_overland, actual_et)
            All flows in mm/day, storages in m.
    """
    xp = _get_backend(use_jax)

    # Convert mm/day inputs to m for internal computation
    p_eff_m = p_eff / 1000.0
    pet_m = pet / 1000.0

    # --- 1. Baseflow ---
    # szq = exp(lnTe + ln(dt) - lambda_bar)
    # Since lambda_bar is absorbed (TI distribution centered at 0), lambda_bar = 0
    # dt is in hours; for daily: dt=24
    ln_dt = xp.log(xp.maximum(dt, 1e-6))
    szq = xp.exp(params.lnTe + ln_dt)
    q_b_m = szq * xp.exp(-s_bar / xp.maximum(params.m, 1e-6))

    # --- 2. Local deficit per TI class ---
    # S_local_i = max(S_bar + m * (lambda_bar - lnaotb_i), 0)
    # With lambda_bar = 0:
    s_local = xp.maximum(s_bar + params.m * (0.0 - lnaotb), 0.0)

    # --- 3. Root zone accounting ---
    # Infiltration reduces root zone deficit
    new_srz = xp.maximum(srz - p_eff_m, 0.0)

    # Excess water that the root zone cannot hold → unsaturated zone
    excess_to_uz = xp.maximum(-(srz - p_eff_m), 0.0)  # = max(p_eff_m - srz, 0)
    new_suz = suz + excess_to_uz

    # --- 4. Evapotranspiration ---
    # Ea = Ep * (1 - Srz / Srmax), capped at available moisture
    et_fraction = xp.maximum(1.0 - new_srz / xp.maximum(params.Srmax, 1e-6), 0.0)
    actual_et_m = xp.minimum(pet_m * et_fraction, xp.maximum(params.Srmax - new_srz, 0.0))
    new_srz = new_srz + actual_et_m  # ET increases deficit

    # --- 5. Unsaturated zone drainage ---
    # quz = min(Suz / (S_local * td), Suz) where S_local > 0
    quz = xp.where(
        s_local > 0.0,
        xp.minimum(
            new_suz / xp.maximum(s_local * params.td, 1e-10),
            new_suz
        ),
        0.0
    )
    new_suz = new_suz - quz

    # --- 6. Overland flow (saturation excess) ---
    # Where S_local = 0, all UZ storage becomes overland flow
    # Plus any UZ storage exceeding local deficit
    ex = xp.maximum(new_suz - s_local, 0.0)
    new_suz = new_suz - ex

    # --- 7. Mean deficit update ---
    # S_bar += -sum(dist * quz) + Q_b
    recharge = xp.sum(dist_area * quz)
    new_s_bar = s_bar - recharge + q_b_m
    new_s_bar = xp.maximum(new_s_bar, 0.0)

    # --- Convert outputs to mm/day ---
    q_baseflow = q_b_m * 1000.0          # m -> mm
    q_overland = xp.sum(dist_area * ex) * 1000.0  # m -> mm
    actual_et = xp.sum(dist_area * actual_et_m) * 1000.0  # m -> mm

    return new_s_bar, new_srz, new_suz, q_baseflow, q_overland, actual_et


def route_step(
    q_in: Any,
    q_routed: Any,
    params: TopmodelParameters,
    dt: float = 24.0,
    use_jax: bool = True
) -> Any:
    """
    Linear reservoir channel routing.

    Q_routed += (Q_in - Q_routed) * dt / k_route

    Args:
        q_in: Total inflow (mm/day)
        q_routed: Current routed flow (mm/day)
        params: Model parameters
        dt: Timestep in hours
        use_jax: Whether to use JAX backend

    Returns:
        New routed flow (mm/day)
    """
    xp = _get_backend(use_jax)

    # Linear reservoir: fraction = dt / k_route, capped at 1
    frac = xp.minimum(dt / xp.maximum(params.k_route, 1e-6), 1.0)
    new_q_routed = q_routed + (q_in - q_routed) * frac

    return xp.maximum(new_q_routed, 0.0)


# =============================================================================
# SINGLE TIMESTEP
# =============================================================================

def step(
    precip: Any,
    temp: Any,
    pet: Any,
    state: TopmodelState,
    params: TopmodelParameters,
    lnaotb: Any,
    dist_area: Any,
    dt: float = 24.0,
    use_jax: bool = True
) -> Tuple[TopmodelState, Any]:
    """
    Execute one timestep of TOPMODEL.

    Runs snow -> TOPMODEL subsurface -> routing in sequence.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (deg C)
        pet: Potential evapotranspiration (mm/day)
        state: Current model state
        params: Model parameters
        lnaotb: TI bin centers
        dist_area: Fractional area per bin
        dt: Timestep in hours (24 for daily)
        use_jax: Whether to use JAX backend

    Returns:
        Tuple of (new_state, total_streamflow_mm_day)
    """
    # Snow routine
    new_swe, p_eff, _ = snow_step(
        precip, temp, state.swe, params, use_jax
    )

    # TOPMODEL subsurface routine
    new_s_bar, new_srz, new_suz, q_baseflow, q_overland, _ = topmodel_step(
        p_eff, pet, state.s_bar, state.srz, state.suz,
        lnaotb, dist_area, params, dt, use_jax
    )

    # Total unrouted flow
    q_total = q_baseflow + q_overland

    # Route through linear reservoir
    new_q_routed = route_step(q_total, state.q_routed, params, dt, use_jax)

    # Create new state
    new_state = TopmodelState(
        s_bar=new_s_bar,
        srz=new_srz,
        suz=new_suz,
        swe=new_swe,
        q_routed=new_q_routed,
    )

    return new_state, new_q_routed


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    params: TopmodelParameters,
    initial_state: Optional[TopmodelState] = None,
    warmup_days: int = 365,
    dt: float = 24.0
) -> Tuple[Any, TopmodelState]:
    """
    Run full TOPMODEL simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/day), shape (n_days,)
        temp: Temperature timeseries (deg C), shape (n_days,)
        pet: PET timeseries (mm/day), shape (n_days,)
        params: TOPMODEL parameters
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days
        dt: Timestep in hours

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if not HAS_JAX:
        return simulate_numpy(precip, temp, pet, params, initial_state,
                              warmup_days, dt)

    if initial_state is None:
        initial_state = create_initial_state(params=params, use_jax=True)

    # Generate TI distribution (pass JAX tracer directly, no float() cast)
    lnaotb, dist_area = generate_ti_distribution(
        ti_std=params.ti_std, use_jax=True
    )

    # Stack forcing for scan
    forcing = jnp.stack([precip, temp, pet], axis=1)

    def scan_fn(state, forcing_step):
        p, t, e = forcing_step
        new_state, runoff = step(p, t, e, state, params, lnaotb, dist_area, dt, use_jax=True)
        return new_state, runoff

    final_state, runoff = lax.scan(scan_fn, initial_state, forcing)

    return runoff, final_state


def simulate_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: TopmodelParameters,
    initial_state: Optional[TopmodelState] = None,
    warmup_days: int = 365,
    dt: float = 24.0
) -> Tuple[np.ndarray, TopmodelState]:
    """
    Run full TOPMODEL simulation using NumPy (fallback when JAX not available).

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (deg C)
        pet: PET timeseries (mm/day)
        params: TOPMODEL parameters
        initial_state: Initial model state
        warmup_days: Number of warmup days
        dt: Timestep in hours

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_timesteps = len(precip)

    if initial_state is None:
        initial_state = create_initial_state(params=params, use_jax=False)

    # Generate TI distribution
    lnaotb, dist_area = generate_ti_distribution(
        ti_std=float(params.ti_std), use_jax=False
    )

    runoff = np.zeros(n_timesteps)
    state = initial_state

    for i in range(n_timesteps):
        state, runoff[i] = step(
            precip[i], temp[i], pet[i], state, params,
            lnaotb, dist_area, dt, use_jax=False
        )

    return runoff, state


def simulate(
    precip: Any,
    temp: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    initial_state: Optional[TopmodelState] = None,
    warmup_days: int = 365,
    use_jax: bool = True,
    dt: float = 24.0,
    **kwargs
) -> Tuple[Any, TopmodelState]:
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
        dt: Timestep in hours (24 for daily)

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    tm_params = create_params_from_dict(params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, temp, pet, tm_params, initial_state,
                            warmup_days, dt)
    else:
        return simulate_numpy(precip, temp, pet, tm_params, initial_state,
                              warmup_days, dt)


# =============================================================================
# LOSS FUNCTION RE-EXPORTS (from losses.py)
# =============================================================================

def nse_loss(*args, **kwargs):
    """Compute negative NSE loss. See losses.py for details."""
    from .losses import nse_loss as _nse_loss
    return _nse_loss(*args, **kwargs)


def kge_loss(*args, **kwargs):
    """Compute negative KGE loss. See losses.py for details."""
    from .losses import kge_loss as _kge_loss
    return _kge_loss(*args, **kwargs)
