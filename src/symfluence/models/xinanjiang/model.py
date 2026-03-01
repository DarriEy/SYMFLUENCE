# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Xinanjiang (XAJ) Model Core - JAX Implementation.

Pure JAX functions for the Xinanjiang rainfall-runoff model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- GPU acceleration when available

The Xinanjiang model is a saturation-excess model with three main components:
1. Evapotranspiration - 3-layer (upper/lower/deep) with PET correction
2. Runoff generation - Saturation excess with parabolic storage capacity curve
3. Source separation - Free water storage partitioning into surface/interflow/groundwater
4. Routing - Linear reservoirs for interflow and groundwater

Clean-room implementation from published equations. No vendored code.

References:
    Zhao, R.-J. (1992). The Xinanjiang model applied in China.
    Journal of Hydrology, 135(1-4), 371-381.

    Zhao, R.-J., Zhuang, Y., Fang, L., Liu, X., & Zhang, Q. (1980).
    The Xinanjiang model. Hydrological Forecasting, IAHS Publ. 129, 351-356.
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
        "JAX not available. Xinanjiang model will use NumPy backend with reduced functionality. "
        "Install JAX for autodiff, JIT compilation, and GPU support: pip install jax jaxlib"
    )

from .parameters import (
    DEFAULT_PARAMS,
    XinanjiangParams,
    XinanjiangState,
    params_dict_to_namedtuple,
)

__all__ = [
    'HAS_JAX',
    'XinanjiangState',
    'create_initial_state',
    'step_jax',
    'simulate_jax',
    'simulate_numpy',
    'simulate',
    'jit_simulate',
    'simulate_coupled_jax',
    'simulate_coupled_numpy',
]


# =============================================================================
# STATE INITIALIZATION
# =============================================================================

def create_initial_state(
    initial_wu: float = 5.0,
    initial_wl: float = 40.0,
    initial_wd: float = 40.0,
    initial_s: float = 5.0,
    initial_fr: float = 0.1,
    initial_qi: float = 0.0,
    initial_qg: float = 0.0,
    use_jax: bool = True,
) -> XinanjiangState:
    """Create initial Xinanjiang model state.

    Args:
        initial_wu: Initial upper layer tension water (mm)
        initial_wl: Initial lower layer tension water (mm)
        initial_wd: Initial deep layer tension water (mm)
        initial_s: Initial free water storage (mm)
        initial_fr: Initial runoff contributing area fraction
        initial_qi: Initial interflow reservoir (mm)
        initial_qg: Initial groundwater reservoir (mm)
        use_jax: Whether to use JAX arrays

    Returns:
        XinanjiangState namedtuple
    """
    if use_jax and HAS_JAX:
        return XinanjiangState(
            wu=jnp.array(initial_wu),
            wl=jnp.array(initial_wl),
            wd=jnp.array(initial_wd),
            s=jnp.array(initial_s),
            fr=jnp.array(initial_fr),
            qi=jnp.array(initial_qi),
            qg=jnp.array(initial_qg),
        )
    else:
        return XinanjiangState(
            wu=np.float64(initial_wu),
            wl=np.float64(initial_wl),
            wd=np.float64(initial_wd),
            s=np.float64(initial_s),
            fr=np.float64(initial_fr),
            qi=np.float64(initial_qi),
            qg=np.float64(initial_qg),
        )


# =============================================================================
# CORE PHYSICS (JAX-COMPATIBLE)
# =============================================================================

def calculate_evap(
    wu: Any, wl: Any, wd: Any,
    prcp: Any, pet: Any,
    um: Any, lm: Any, dm: Any, c: Any, k: Any,
    xp: Any,
) -> Tuple[Any, Any, Any]:
    """Calculate 3-layer evapotranspiration (Zhao 1992, Section 3.1).

    Three layers draw ET in order: upper first, then lower proportionally,
    then deep layer via the C coefficient.

    Args:
        wu, wl, wd: Current layer storages (mm)
        prcp: Precipitation (mm/day)
        pet: Potential evapotranspiration (mm/day)
        um, lm, dm: Layer capacities (mm)
        c: Deep layer ET coefficient
        k: PET correction factor
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (eu, el, ed) - evaporation from each layer (mm/day)
    """
    ep = k * pet  # Corrected PET

    # Upper layer evaporation
    eu = xp.where(wu + prcp >= ep, ep, wu + prcp)

    # Remaining PET after upper layer
    remaining = ep - eu

    # Lower layer evaporation (proportional to storage)
    # When upper layer provides enough, el = 0
    # When WL >= C*LM: el = remaining * WL/LM
    # When WL < C*LM: el = C * remaining (deep layer assists)
    el_normal = remaining * wl / xp.maximum(lm, 1e-10)
    el_limited = remaining * c
    el = xp.where(
        remaining <= 1e-10,
        0.0,
        xp.where(wl >= c * lm, el_normal, el_limited)
    )
    el = xp.minimum(el, wl)

    # Deep layer evaporation
    # Only when lower layer storage is below C*LM threshold
    ed_val = remaining * c - el
    ed = xp.where(
        (remaining > 1e-10) & (wl < c * lm),
        xp.maximum(ed_val, 0.0),
        0.0
    )
    ed = xp.minimum(ed, wd)

    return eu, el, ed


def calculate_prcp_runoff(
    b: Any, im: Any, wm: Any, w0: Any, pe: Any,
    xp: Any,
) -> Tuple[Any, Any]:
    """Calculate runoff from net rainfall using saturation excess (Zhao 1992, Eq. 3).

    The parabolic distribution of storage capacity across the catchment:
        f(W') = 1 - (1 - W'/WMM)^B

    where WMM = WM * (1 + B) is the maximum point storage capacity.

    Args:
        b: Storage capacity curve exponent
        im: Impervious area fraction
        wm: Total tension water capacity WM = UM + LM + DM (mm)
        w0: Current total tension water W0 = WU + WL + WD (mm)
        pe: Net rainfall = P - E (mm/day)
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (r, r_im) - pervious area runoff and impervious area runoff (mm/day)
    """
    # Maximum point storage capacity
    wmm = wm * (1.0 + b)

    # Current storage capacity ordinate
    # A = WMM * (1 - (1 - W0/WM)^(1/(1+B)))
    w_ratio = xp.clip(w0 / xp.maximum(wm, 1e-10), 0.0, 1.0)
    a = wmm * (1.0 - xp.power(1.0 - w_ratio, 1.0 / (1.0 + b)))
    a = xp.clip(a, 0.0, wmm)

    # Impervious area direct runoff
    r_im = pe * im

    # Pervious area runoff (saturation excess)
    # Three cases based on PE + A relative to WMM
    pe_pervious = pe * (1.0 - im)

    # Case: PE + A >= WMM (entire catchment saturated)
    r_saturated = pe_pervious - (wm - w0)

    # Case: PE + A < WMM (partial saturation)
    pa = pe_pervious + a
    r_partial = pe_pervious - (wm - w0) + wm * xp.power(
        xp.clip(1.0 - pa / xp.maximum(wmm, 1e-10), 0.0, 1.0),
        1.0 + b
    )

    r_pervious = xp.where(
        pe_pervious + a >= wmm,
        r_saturated,
        r_partial
    )

    # Runoff cannot be negative
    r_pervious = xp.maximum(r_pervious, 0.0)

    # No runoff when PE <= 0
    r = xp.where(pe > 0.0, r_pervious + r_im, 0.0)
    r_im = xp.where(pe > 0.0, r_im, 0.0)

    return r, r_im


def update_tension_water(
    wu: Any, wl: Any, wd: Any,
    eu: Any, el: Any, ed: Any,
    pe: Any, r: Any,
    um: Any, lm: Any, dm: Any,
    xp: Any,
) -> Tuple[Any, Any, Any]:
    """Update 3-layer tension water storage after runoff and ET.

    Water balance: layer storage += net input - losses
    Excess from upper flows to lower, excess from lower flows to deep.

    Args:
        wu, wl, wd: Current layer storages (mm)
        eu, el, ed: Evaporation from each layer (mm/day)
        pe: Net rainfall (mm/day)
        r: Total runoff (mm/day)
        um, lm, dm: Layer capacities (mm)
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (wu_new, wl_new, wd_new)
    """
    # Net input to tension water = PE - R (what doesn't run off)
    net_input = pe - r

    # Update upper layer first
    wu_new = wu - eu + xp.where(pe > 0.0, net_input, 0.0)

    # Overflow from upper to lower
    excess_u = xp.maximum(wu_new - um, 0.0)
    wu_new = xp.minimum(wu_new, um)

    # Update lower layer
    wl_new = wl - el + excess_u

    # Overflow from lower to deep
    excess_l = xp.maximum(wl_new - lm, 0.0)
    wl_new = xp.minimum(wl_new, lm)

    # Update deep layer
    wd_new = wd - ed + excess_l
    wd_new = xp.clip(wd_new, 0.0, dm)

    # Ensure non-negative
    wu_new = xp.maximum(wu_new, 0.0)
    wl_new = xp.maximum(wl_new, 0.0)

    return wu_new, wl_new, wd_new


def calculate_sources(
    pe: Any, r: Any, r_im: Any,
    sm: Any, ex: Any, ki: Any, kg: Any,
    s: Any, fr: Any,
    xp: Any,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Separate total runoff into 3 sources using free water storage (Zhao 1992, Section 3.3).

    Free water storage has its own parabolic capacity distribution, similar
    to the tension water. Water is partitioned into:
    - RS: Surface runoff (excess beyond free water capacity)
    - RI: Interflow (KI fraction of free water)
    - RG: Groundwater (KG fraction of free water)

    Args:
        pe: Net rainfall (mm/day)
        r: Total runoff (mm/day)
        r_im: Impervious area runoff (mm/day)
        sm: Free water capacity (mm)
        ex: Free water capacity curve exponent
        ki: Interflow outflow coefficient
        kg: Groundwater outflow coefficient
        s: Current free water storage (mm)
        fr: Current runoff contributing area fraction
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (rs, ri, rg, s_new, fr_new)
    """
    # Runoff contributing area fraction
    fr_new = xp.where(pe > 1e-10, r / xp.maximum(pe, 1e-10), fr)
    fr_new = xp.clip(fr_new, 0.01, 1.0)

    # Net inflow to free water storage per unit area
    # R - R_IM is the pervious runoff; divide by FR to get per-unit-area
    net_r = (r - r_im) / xp.maximum(fr_new, 1e-10)

    # Free water maximum point capacity
    smm = sm * (1.0 + ex)

    # Current free water storage capacity ordinate
    s_ratio = xp.clip(s / xp.maximum(sm, 1e-10), 0.0, 1.0)
    au = smm * (1.0 - xp.power(1.0 - s_ratio, 1.0 / (1.0 + ex)))
    au = xp.clip(au, 0.0, smm)

    # Surface runoff (excess beyond capacity)
    rs_saturated = (net_r - (sm - s)) * fr_new
    rs_partial = (net_r - (sm - s) + sm * xp.power(
        xp.clip(1.0 - (net_r + au) / xp.maximum(smm, 1e-10), 0.0, 1.0),
        1.0 + ex
    )) * fr_new

    rs = xp.where(
        net_r + au >= smm,
        rs_saturated,
        rs_partial
    )
    rs = xp.maximum(rs, 0.0) + r_im

    # Update free water storage
    s_new = s + (net_r - rs / xp.maximum(fr_new, 1e-10))

    # Interflow and groundwater from free water
    ri = ki * s_new * fr_new
    rg = kg * s_new * fr_new

    # Deplete free water storage
    s_new = s_new * (1.0 - ki - kg)
    s_new = xp.clip(s_new, 0.0, sm)

    # No source separation when PE <= 0
    rs = xp.where(pe > 1e-10, rs, 0.0)
    ri = xp.where(pe > 1e-10, ri, 0.0)
    rg = xp.where(pe > 1e-10, rg, 0.0)
    s_new = xp.where(pe > 1e-10, s_new, s * (1.0 - ki - kg))
    fr_new = xp.where(pe > 1e-10, fr_new, fr)

    return rs, ri, rg, s_new, fr_new


def linear_reservoir(inflow: Any, recession: Any, last_q: Any, xp: Any) -> Any:
    """Linear reservoir routing.

    Q(t) = recession * Q(t-1) + (1 - recession) * inflow

    Args:
        inflow: Current inflow (mm/day)
        recession: Recession constant (0 < C < 1)
        last_q: Previous outflow (mm/day)
        xp: Array backend (jnp or np)

    Returns:
        Current outflow (mm/day)
    """
    return recession * last_q + (1.0 - recession) * inflow


# =============================================================================
# SINGLE TIMESTEP
# =============================================================================

def step_jax(
    precip: Any,
    pet: Any,
    state: XinanjiangState,
    params: XinanjiangParams,
) -> Tuple[XinanjiangState, Any]:
    """Execute one timestep of Xinanjiang model (JAX version).

    Runs generation, source separation, and linear reservoir routing.

    Args:
        precip: Precipitation (mm/day)
        pet: Potential evapotranspiration (mm/day)
        state: Current model state
        params: Model parameters

    Returns:
        Tuple of (new_state, total_outflow_mm)
    """
    xp = jnp  # JAX backend

    wu, wl, wd = state.wu, state.wl, state.wd

    # Total tension water capacity and storage
    wm = params.UM + params.LM + params.DM
    w0 = wu + wl + wd

    # 1. Evapotranspiration
    eu, el, ed = calculate_evap(
        wu, wl, wd, precip, pet,
        params.UM, params.LM, params.DM, params.C, params.K, xp
    )

    # Net rainfall: PE = P - E, where E = EU + EL + ED
    pe = precip - (eu + el + ed)

    # 2. Runoff generation (saturation excess)
    r, r_im = calculate_prcp_runoff(
        params.B, params.IM, wm, w0, pe, xp
    )

    # 3. Update tension water storage
    wu_new, wl_new, wd_new = update_tension_water(
        wu, wl, wd, eu, el, ed, pe, r,
        params.UM, params.LM, params.DM, xp
    )

    # 4. Source separation (3-source split)
    rs, ri, rg, s_new, fr_new = calculate_sources(
        pe, r, r_im, params.SM, params.EX, params.KI, params.KG,
        state.s, state.fr, xp
    )

    # 5. Linear reservoir routing for interflow and groundwater
    qi = linear_reservoir(ri, params.CI, state.qi, xp)
    qg = linear_reservoir(rg, params.CG, state.qg, xp)

    # Total outflow
    q_total = rs + qi + qg
    q_total = xp.maximum(q_total, 0.0)

    new_state = XinanjiangState(
        wu=wu_new, wl=wl_new, wd=wd_new,
        s=s_new, fr=fr_new,
        qi=qi, qg=qg,
    )

    return new_state, q_total


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_jax(
    precip: Any,
    pet: Any,
    params: XinanjiangParams,
    initial_state: Optional[XinanjiangState] = None,
    warmup_days: int = 365,
) -> Tuple[Any, XinanjiangState]:
    """Run full Xinanjiang simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/day), shape (n_timesteps,)
        pet: PET timeseries (mm/day), shape (n_timesteps,)
        params: Xinanjiang parameters (XinanjiangParams namedtuple)
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days (included in output)

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if not HAS_JAX:
        return simulate_numpy(precip, pet, params, initial_state, warmup_days)

    if initial_state is None:
        initial_state = create_initial_state(use_jax=True)

    # Stack forcing for scan
    forcing = jnp.stack([precip, pet], axis=1)

    def scan_fn(state, forcing_step):
        p, e = forcing_step
        new_state, q = step_jax(p, e, state, params)
        return new_state, q

    # Run simulation
    final_state, runoff = lax.scan(scan_fn, initial_state, forcing)

    return runoff, final_state


def simulate_numpy(
    precip: np.ndarray,
    pet: np.ndarray,
    params: XinanjiangParams,
    initial_state: Optional[XinanjiangState] = None,
    warmup_days: int = 365,
) -> Tuple[np.ndarray, XinanjiangState]:
    """Run full Xinanjiang simulation using NumPy (fallback).

    Args:
        precip: Precipitation timeseries (mm/day)
        pet: PET timeseries (mm/day)
        params: Xinanjiang parameters
        initial_state: Initial model state
        warmup_days: Number of warmup days

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_timesteps = len(precip)
    xp = np

    if initial_state is None:
        initial_state = create_initial_state(use_jax=False)

    runoff = np.zeros(n_timesteps)
    state = initial_state

    wm = float(params.UM + params.LM + params.DM)

    for i in range(n_timesteps):
        p = precip[i]
        e = pet[i]
        wu, wl, wd = state.wu, state.wl, state.wd
        w0 = wu + wl + wd

        # Evapotranspiration
        eu, el, ed = calculate_evap(
            wu, wl, wd, p, e,
            params.UM, params.LM, params.DM, params.C, params.K, xp
        )

        # Net rainfall
        pe = p - (eu + el + ed)

        # Runoff generation
        r, r_im = calculate_prcp_runoff(
            params.B, params.IM, wm, w0, pe, xp
        )

        # Update tension water
        wu_new, wl_new, wd_new = update_tension_water(
            wu, wl, wd, eu, el, ed, pe, r,
            params.UM, params.LM, params.DM, xp
        )

        # Source separation
        rs, ri, rg, s_new, fr_new = calculate_sources(
            pe, r, r_im, params.SM, params.EX, params.KI, params.KG,
            state.s, state.fr, xp
        )

        # Linear reservoir routing
        qi = linear_reservoir(ri, params.CI, state.qi, xp)
        qg = linear_reservoir(rg, params.CG, state.qg, xp)

        q_total = max(float(rs + qi + qg), 0.0)

        state = XinanjiangState(
            wu=wu_new, wl=wl_new, wd=wd_new,
            s=s_new, fr=fr_new,
            qi=qi, qg=qg,
        )

        runoff[i] = q_total

    return runoff, state


def _try_dcoupler_coupled(
    precip: Any,
    temp: Any,
    pet: Any,
    day_of_year: Any,
    xaj_params_dict: Dict[str, float],
    snow17_params_dict: Dict[str, float],
    n_timesteps: int,
    dt: float = 1.0,
) -> Optional[Tuple[Any, XinanjiangState]]:
    """Attempt coupled Snow-17 + XAJ simulation via dCoupler graph.

    Returns None if dCoupler is unavailable or execution fails.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from symfluence.coupling import INSTALL_SUGGESTION, is_dcoupler_available
        if not is_dcoupler_available():
            logger.debug(INSTALL_SUGGESTION)
            return None

        import torch

        from symfluence.coupling import CouplingGraphBuilder

        graph_config = {
            'HYDROLOGICAL_MODEL': 'XAJ',
            'SNOW_MODULE': 'SNOW17',
        }
        builder = CouplingGraphBuilder()
        graph = builder.build(graph_config)

        outputs = graph.forward(
            external_inputs={
                'snow': {
                    'precip': torch.as_tensor(np.asarray(precip), dtype=torch.float64),
                    'temp': torch.as_tensor(np.asarray(temp), dtype=torch.float64),
                },
                'land': {
                    'pet': torch.as_tensor(np.asarray(pet), dtype=torch.float64),
                },
            },
            n_timesteps=n_timesteps,
            dt=dt * 86400.0,
        )

        runoff_tensor = outputs['land']['runoff']
        runoff = runoff_tensor.detach().numpy()
        logger.info("Snow-17 + XAJ coupled simulation completed via dCoupler graph")
        return runoff, create_initial_state(use_jax=False)

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.debug(f"dCoupler coupled path failed: {e}. Using native implementation.")
        return None


def simulate(
    precip: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    initial_state: Optional[XinanjiangState] = None,
    warmup_days: int = 365,
    use_jax: bool = True,
    temp: Any = None,
    day_of_year: Any = None,
    snow17_params: Optional[Dict[str, float]] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
    coupling_mode: str = 'auto',
) -> Tuple[Any, XinanjiangState]:
    """High-level simulation function with automatic backend selection.

    When ``temp`` and ``snow17_params`` are provided, runs the coupled
    Snow-17 + XAJ pipeline. Otherwise, runs standalone XAJ.

    Args:
        precip: Precipitation timeseries (mm/day)
        pet: PET timeseries (mm/day)
        params: XAJ parameter dictionary (uses defaults if None)
        initial_state: Initial model state
        warmup_days: Warmup period in days
        use_jax: Whether to prefer JAX backend
        temp: Temperature timeseries (C) — triggers coupled mode
        day_of_year: Day of year array (1-366) — required for coupled mode
        snow17_params: Snow-17 parameter dictionary — triggers coupled mode
        latitude: Catchment latitude for Snow-17
        si: SWE threshold for areal depletion
        dt: Timestep in days
        coupling_mode: 'auto' tries dCoupler first for coupled mode,
            'dcoupler' forces dCoupler, 'native' skips dCoupler.

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Coupled Snow-17 + XAJ mode
    if temp is not None and snow17_params is not None:
        if day_of_year is None:
            raise ValueError("day_of_year required for coupled Snow-17 + XAJ mode")

        # The dCoupler graph path is available via explicit opt-in only.
        # The native lax.scan coupling is faster and correctly forwards
        # user params/initial state, so 'auto' uses native.
        if coupling_mode == 'dcoupler':
            result = _try_dcoupler_coupled(
                precip, temp, pet, day_of_year, params, snow17_params,
                n_timesteps=len(precip), dt=dt,
            )
            if result is not None:
                return result
            raise RuntimeError(
                "COUPLING_MODE='dcoupler' but dCoupler execution failed. "
                "Install dCoupler with: pip install dcoupler"
            )

        # Native coupled path (lax.scan or Python loop)
        if use_jax and HAS_JAX:
            return simulate_coupled_jax(
                precip, temp, pet, day_of_year, params, snow17_params,
                initial_state, latitude, si, dt,
            )
        else:
            return simulate_coupled_numpy(
                precip, temp, pet, day_of_year, params, snow17_params,
                initial_state, latitude, si, dt,
            )

    # Standalone XAJ mode
    xaj_params = params_dict_to_namedtuple(params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, pet, xaj_params, initial_state, warmup_days)
    else:
        return simulate_numpy(precip, pet, xaj_params, initial_state, warmup_days)


def jit_simulate(use_gpu: bool = False):
    """Get JIT-compiled simulation function.

    Args:
        use_gpu: Whether to use GPU (if available).

    Returns:
        JIT-compiled simulation function if JAX available.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Returning non-JIT function.")
        return simulate

    @jax.jit
    def _jit_simulate(precip, pet, params, initial_state):
        return simulate_jax(precip, pet, params, initial_state)

    return _jit_simulate


# =============================================================================
# COUPLED SNOW-17 + XAJ SIMULATION
# =============================================================================

def simulate_coupled_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    day_of_year: Any,
    xaj_params_dict: Dict[str, float],
    snow17_params_dict: Dict[str, float],
    initial_state: Optional[XinanjiangState] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
) -> Tuple[Any, XinanjiangState]:
    """Run coupled Snow-17 + XAJ simulation using JAX lax.scan.

    Snow-17 processes precip/temp into rain_plus_melt, which feeds into XAJ.

    Args:
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (C)
        pet: PET timeseries (mm/day)
        day_of_year: Day of year array (1-366)
        xaj_params_dict: XAJ parameter dictionary
        snow17_params_dict: Snow-17 parameter dictionary
        initial_state: Initial XAJ state
        latitude: Catchment latitude
        si: SWE threshold for areal depletion
        dt: Timestep in days

    Returns:
        Tuple of (runoff_timeseries, final_xaj_state)
    """
    from symfluence.models.snow17.model import snow17_step as s17_step
    from symfluence.models.snow17.parameters import (
        DEFAULT_ADC,
    )
    from symfluence.models.snow17.parameters import (
        create_initial_state as s17_init,
    )
    from symfluence.models.snow17.parameters import (
        params_dict_to_namedtuple as s17_params_to_nt,
    )

    if not HAS_JAX:
        return simulate_coupled_numpy(
            precip, temp, pet, day_of_year, xaj_params_dict,
            snow17_params_dict, initial_state, latitude, si, dt,
        )

    xaj_params = params_dict_to_namedtuple(xaj_params_dict, use_jax=True)
    s17_params = s17_params_to_nt(snow17_params_dict, use_jax=True)

    if initial_state is None:
        initial_state = create_initial_state(use_jax=True)

    s17_state = s17_init(use_jax=True)
    adc = jnp.asarray(DEFAULT_ADC, dtype=float)

    # Stack forcing
    forcing = jnp.stack([
        precip, temp, pet, jnp.asarray(day_of_year, dtype=float),
    ], axis=1)

    def scan_fn(carry, forcing_step):
        xaj_state, snow_state = carry
        p, t, e, doy = forcing_step

        # Snow-17 step
        new_snow_state, rain_plus_melt = s17_step(
            p, t, dt, snow_state, s17_params, doy, latitude, si, adc, xp=jnp,
        )

        # XAJ step with rain_plus_melt as effective precipitation
        new_xaj_state, q = step_jax(rain_plus_melt, e, xaj_state, xaj_params)

        return (new_xaj_state, new_snow_state), q

    (final_xaj, _final_snow), runoff = lax.scan(
        scan_fn, (initial_state, s17_state), forcing,
    )

    return runoff, final_xaj


def simulate_coupled_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    day_of_year: np.ndarray,
    xaj_params_dict: Dict[str, float],
    snow17_params_dict: Dict[str, float],
    initial_state: Optional[XinanjiangState] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    dt: float = 1.0,
) -> Tuple[np.ndarray, XinanjiangState]:
    """Run coupled Snow-17 + XAJ simulation using NumPy (Python loop).

    Args:
        precip: Precipitation array (mm/day)
        temp: Temperature array (C)
        pet: PET array (mm/day)
        day_of_year: Day of year array (1-366)
        xaj_params_dict: XAJ parameter dictionary
        snow17_params_dict: Snow-17 parameter dictionary
        initial_state: Initial XAJ state
        latitude: Catchment latitude
        si: SWE threshold for areal depletion
        dt: Timestep in days

    Returns:
        Tuple of (runoff_timeseries, final_xaj_state)
    """
    from symfluence.models.snow17.model import snow17_step as s17_step
    from symfluence.models.snow17.parameters import (
        DEFAULT_ADC,
    )
    from symfluence.models.snow17.parameters import (
        create_initial_state as s17_init,
    )
    from symfluence.models.snow17.parameters import (
        params_dict_to_namedtuple as s17_params_to_nt,
    )

    xaj_params = params_dict_to_namedtuple(xaj_params_dict, use_jax=False)
    s17_params = s17_params_to_nt(snow17_params_dict, use_jax=False)

    if initial_state is None:
        initial_state = create_initial_state(use_jax=False)

    s17_state = s17_init(use_jax=False)
    adc = DEFAULT_ADC.copy()

    n = len(precip)
    runoff = np.zeros(n)
    xaj_state = initial_state

    xp = np
    wm = float(xaj_params.UM + xaj_params.LM + xaj_params.DM)

    for i in range(n):
        # Snow-17 step
        s17_state, rain_plus_melt = s17_step(
            np.float64(precip[i]), np.float64(temp[i]), dt,
            s17_state, s17_params, np.float64(day_of_year[i]),
            latitude, si, adc, xp=np,
        )

        p = float(rain_plus_melt)
        e = float(pet[i])
        wu, wl, wd = xaj_state.wu, xaj_state.wl, xaj_state.wd
        w0 = wu + wl + wd

        # Evapotranspiration
        eu, el, ed = calculate_evap(
            wu, wl, wd, p, e,
            xaj_params.UM, xaj_params.LM, xaj_params.DM, xaj_params.C, xaj_params.K, xp,
        )

        pe = p - (eu + el + ed)

        # Runoff generation
        r, r_im = calculate_prcp_runoff(
            xaj_params.B, xaj_params.IM, wm, w0, pe, xp,
        )

        # Update tension water
        wu_new, wl_new, wd_new = update_tension_water(
            wu, wl, wd, eu, el, ed, pe, r,
            xaj_params.UM, xaj_params.LM, xaj_params.DM, xp,
        )

        # Source separation
        rs, ri, rg, s_new, fr_new = calculate_sources(
            pe, r, r_im, xaj_params.SM, xaj_params.EX, xaj_params.KI, xaj_params.KG,
            xaj_state.s, xaj_state.fr, xp,
        )

        # Linear reservoir routing
        qi = linear_reservoir(ri, xaj_params.CI, xaj_state.qi, xp)
        qg = linear_reservoir(rg, xaj_params.CG, xaj_state.qg, xp)

        q_total = max(float(rs + qi + qg), 0.0)

        xaj_state = XinanjiangState(
            wu=wu_new, wl=wl_new, wd=wd_new,
            s=s_new, fr=fr_new,
            qi=qi, qg=qg,
        )

        runoff[i] = q_total

    return runoff, xaj_state
