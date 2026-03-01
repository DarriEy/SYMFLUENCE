# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Snow-17 + SAC-SMA Model Orchestrator — Dual-Backend (JAX + NumPy).

Runs Snow-17 first to partition precipitation into rain+melt,
then feeds output into SAC-SMA as effective precipitation.

Supports:
- ``snow_module='snow17'``: coupled mode (default)
- ``snow_module='none'``: standalone SAC-SMA (no snow processing)
- ``use_jax=True``: JAX lax.scan backend for autodiff/JIT
- ``use_jax=False``: NumPy Python-loop fallback (default)

Usage:
    from symfluence.models.sacsma.model import simulate
    flow, final_state = simulate(precip, temp, pet, params)
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple

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

from symfluence.models.snow17.model import (
    snow17_simulate_numpy,
    snow17_step,
)
from symfluence.models.snow17.parameters import (
    DEFAULT_ADC,
    Snow17Params,
    Snow17State,
)
from symfluence.models.snow17.parameters import (
    create_initial_state as snow17_create_initial_state,
)
from symfluence.models.snow17.parameters import (
    params_dict_to_namedtuple as snow17_params_dict_to_namedtuple,
)

from .parameters import (
    DEFAULT_PARAMS,
    split_params,
)
from .parameters import (
    params_dict_to_namedtuple as sacsma_params_dict_to_namedtuple,
)
from .sacsma import (
    SacSmaState,
    _create_default_state,
    sacsma_step,
)
from .sacsma import (
    sacsma_simulate_jax as _sacsma_sim_jax,
)
from .sacsma import (
    sacsma_simulate_numpy as _sacsma_sim_numpy,
)


class SacSmaSnow17State(NamedTuple):
    """Combined state for coupled Snow-17 + SAC-SMA model. Fields use Any for JAX."""
    snow17: Any  # Snow17State
    sacsma: Any  # SacSmaState


def _simulate_coupled_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    day_of_year: Any,
    snow17_params: Snow17Params,
    sacsma_params: Any,
    snow17_state: Snow17State,
    sacsma_state: SacSmaState,
    latitude: float = 45.0,
    dt: float = 1.0,
    si: float = 100.0,
) -> Tuple[Any, SacSmaSnow17State]:
    """Coupled Snow-17 + SAC-SMA simulation via JAX lax.scan.

    Interleaves snow17_step and sacsma_step per timestep for correct
    coupled dynamics and end-to-end differentiability.
    """
    if not HAS_JAX:
        return _simulate_coupled_numpy(
            np.asarray(precip), np.asarray(temp), np.asarray(pet),
            np.asarray(day_of_year), snow17_params, sacsma_params,
            snow17_state, sacsma_state, latitude, dt, si,
        )

    adc = jnp.asarray(DEFAULT_ADC, dtype=float)
    forcing = jnp.stack([
        precip, temp, pet, jnp.asarray(day_of_year, dtype=float),
    ], axis=1)

    def scan_fn(carry, forcing_step):
        s17_state, sac_state = carry
        p, t, e, doy = forcing_step

        # Snow-17 step
        new_s17, rpm = snow17_step(p, t, dt, s17_state, snow17_params, doy,
                                   latitude, si, adc, xp=jnp)

        # SAC-SMA step with rain+melt as effective precipitation
        new_sac, surf, interf, base, _ = sacsma_step(rpm, e, dt, sac_state,
                                                      sacsma_params, xp=jnp)

        return (new_s17, new_sac), surf + interf + base

    init = (snow17_state, sacsma_state)
    (final_s17, final_sac), total_flow = lax.scan(scan_fn, init, forcing)

    return total_flow, SacSmaSnow17State(snow17=final_s17, sacsma=final_sac)


def _simulate_coupled_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    day_of_year: np.ndarray,
    snow17_params: Snow17Params,
    sacsma_params: Any,
    snow17_state: Snow17State,
    sacsma_state: SacSmaState,
    latitude: float = 45.0,
    dt: float = 1.0,
    si: float = 100.0,
) -> Tuple[np.ndarray, SacSmaSnow17State]:
    """Coupled Snow-17 + SAC-SMA simulation via NumPy sequential mode.

    Runs Snow-17 as a full time series first, then SAC-SMA. Equivalent to
    the original implementation.
    """
    adc = DEFAULT_ADC.copy()

    # Step 1: Run Snow-17 to get rain+melt
    rain_plus_melt, snow17_final = snow17_simulate_numpy(
        precip, temp, day_of_year, snow17_params,
        state=snow17_state, lat=latitude, si=si, dt=dt, adc=adc,
    )

    # Step 2: Run SAC-SMA with rain+melt
    total_flow, sacsma_final = _sacsma_sim_numpy(
        rain_plus_melt, pet, sacsma_params,
        initial_state=sacsma_state, dt=dt,
    )

    return total_flow, SacSmaSnow17State(snow17=snow17_final, sacsma=sacsma_final)


def _try_dcoupler_coupled(
    precip: Any,
    temp: Any,
    pet: Any,
    day_of_year: Any,
    n_timesteps: int,
    dt: float = 1.0,
) -> Optional[Tuple[Any, SacSmaSnow17State]]:
    """Attempt coupled Snow-17 + SAC-SMA simulation via dCoupler graph.

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
            'HYDROLOGICAL_MODEL': 'SACSMA',
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
        logger.info("Snow-17 + SAC-SMA coupled simulation completed via dCoupler graph")

        dummy_s17 = Snow17State(
            w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0,
        )
        dummy_sac = _create_default_state(
            sacsma_params_dict_to_namedtuple({}, use_jax=False),
            use_jax=False,
        )
        return runoff, SacSmaSnow17State(snow17=dummy_s17, sacsma=dummy_sac)

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.debug(f"dCoupler coupled path failed: {e}. Using native implementation.")
        return None


def simulate(
    precip: Any,
    temp: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    day_of_year: Optional[Any] = None,
    initial_state: Optional[SacSmaSnow17State] = None,
    warmup_days: int = 365,
    latitude: float = 45.0,
    dt: float = 1.0,
    si: float = 100.0,
    use_jax: bool = False,
    snow_module: str = 'snow17',
    start_date: Optional[str] = None,
    coupling_mode: str = 'auto',
) -> Tuple[Any, SacSmaSnow17State]:
    """Run coupled or standalone SAC-SMA simulation.

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (C)
        pet: PET array (mm/dt)
        params: Combined parameter dictionary (Snow-17 + SAC-SMA).
                Uses defaults for any missing parameters.
        day_of_year: Julian day array (1-366). If None, generated from
                     start_date (if provided) or raises ValueError in
                     coupled mode.
        initial_state: Initial coupled state. If None, Snow-17 starts
                       with no snow, SAC-SMA at 50% capacity.
        warmup_days: Warmup period in days (output includes warmup).
        latitude: Latitude for melt factor seasonality.
        dt: Timestep in days (1.0 = daily).
        si: SWE threshold for areal depletion in Snow-17.
        use_jax: Whether to use JAX backend.
        snow_module: 'snow17' for coupled mode, 'none' for standalone SAC-SMA.
        start_date: Start date string (e.g. '2004-01-01') for generating
                    calendar-correct day_of_year when day_of_year is None.
        coupling_mode: 'auto' tries dCoupler first for coupled mode,
            'dcoupler' forces dCoupler, 'native' skips dCoupler.

    Returns:
        Tuple of (total_flow mm/dt, final_state)
    """
    import pandas as pd

    n = len(precip)
    actual_jax = use_jax and HAS_JAX

    # Parameters
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        params = full_params

    snow17_dict, sacsma_dict = split_params(params)
    sacsma_p = sacsma_params_dict_to_namedtuple(sacsma_dict, use_jax=actual_jax)

    # Day of year
    if day_of_year is None:
        if start_date is not None:
            dates = pd.date_range(start=start_date, periods=n, freq='D')
            day_of_year = dates.dayofyear.values
        elif snow_module == 'snow17':
            raise ValueError(
                "day_of_year or start_date required for coupled Snow-17 mode. "
                "Pass day_of_year array or start_date='YYYY-MM-DD'."
            )
        else:
            # Standalone SAC-SMA doesn't use day_of_year
            day_of_year = np.ones(n, dtype=int)

    if snow_module == 'none':
        # Standalone SAC-SMA: precip goes directly as effective precipitation
        if initial_state is None:
            sac_state = _create_default_state(sacsma_p, use_jax=actual_jax)
            s17_state = Snow17State(
                w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0,
            )
        else:
            sac_state = initial_state.sacsma
            s17_state = initial_state.snow17

        if actual_jax:
            total_flow, sac_final = _sacsma_sim_jax(
                jnp.asarray(precip), jnp.asarray(pet), sacsma_p,
                initial_state=sac_state, dt=dt,
            )
        else:
            total_flow, sac_final = _sacsma_sim_numpy(
                np.asarray(precip), np.asarray(pet), sacsma_p,
                initial_state=sac_state, dt=dt,
            )

        final_state = SacSmaSnow17State(snow17=s17_state, sacsma=sac_final)
        return total_flow, final_state

    # Coupled Snow-17 + SAC-SMA
    # The dCoupler graph path is available via explicit opt-in only.
    # The native lax.scan coupling is faster and correctly forwards
    # user params/initial state, so 'auto' uses native.
    if coupling_mode == 'dcoupler':
        result = _try_dcoupler_coupled(
            precip, temp, pet, day_of_year,
            n_timesteps=n, dt=dt,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "COUPLING_MODE='dcoupler' but dCoupler execution failed. "
            "Install dCoupler with: pip install dcoupler"
        )

    # Native coupled path
    snow17_p = snow17_params_dict_to_namedtuple(snow17_dict, use_jax=actual_jax)

    # Initial states
    if initial_state is None:
        if actual_jax:
            s17_state = snow17_create_initial_state(use_jax=True)
        else:
            s17_state = Snow17State(
                w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0,
            )
        sac_state = _create_default_state(sacsma_p, use_jax=actual_jax)
    else:
        s17_state = initial_state.snow17
        sac_state = initial_state.sacsma

    if actual_jax:
        return _simulate_coupled_jax(
            jnp.asarray(precip), jnp.asarray(temp), jnp.asarray(pet),
            jnp.asarray(day_of_year, dtype=float),
            snow17_p, sacsma_p, s17_state, sac_state,
            latitude, dt, si,
        )
    else:
        return _simulate_coupled_numpy(
            np.asarray(precip), np.asarray(temp), np.asarray(pet),
            np.asarray(day_of_year),
            snow17_p, sacsma_p, s17_state, sac_state,
            latitude, dt, si,
        )


def jit_simulate(
    snow17_params: Snow17Params,
    sacsma_params: Any,
    latitude: float = 45.0,
    dt: float = 1.0,
    si: float = 100.0,
):
    """Create a JIT-compiled coupled simulation function.

    Args:
        snow17_params: Snow-17 parameters (frozen into closure)
        sacsma_params: SAC-SMA parameters (frozen into closure)
        latitude: Latitude
        dt: Timestep in days
        si: SWE threshold

    Returns:
        JIT-compiled function: (precip, temp, pet, doy) -> (flow, state)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for jit_simulate")

    @jax.jit
    def _jit_fn(precip, temp, pet, day_of_year):
        s17_state = snow17_create_initial_state(use_jax=True)
        sac_state = _create_default_state(sacsma_params, use_jax=True)
        return _simulate_coupled_jax(
            precip, temp, pet, day_of_year,
            snow17_params, sacsma_params, s17_state, sac_state,
            latitude, dt, si,
        )

    return _jit_fn
