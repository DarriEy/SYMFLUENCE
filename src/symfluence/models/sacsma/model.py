"""
Coupled Snow-17 + SAC-SMA Model Orchestrator.

Runs Snow-17 first to partition precipitation into rain+melt,
then feeds output into SAC-SMA as effective precipitation.

Usage:
    from symfluence.models.sacsma.model import simulate
    flow, final_state = simulate(precip, temp, pet, params)
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from .parameters import (
    DEFAULT_PARAMS,
    split_params,
)
from .snow17 import Snow17State, snow17_simulate
from .sacsma import SacSmaState, sacsma_simulate


class SacSmaSnow17State(NamedTuple):
    """Combined state for coupled Snow-17 + SAC-SMA model."""
    snow17: Snow17State
    sacsma: SacSmaState


def simulate(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    day_of_year: Optional[np.ndarray] = None,
    initial_state: Optional[SacSmaSnow17State] = None,
    warmup_days: int = 365,
    latitude: float = 45.0,
    dt: float = 1.0,
    si: float = 100.0,
) -> Tuple[np.ndarray, SacSmaSnow17State]:
    """Run coupled Snow-17 + SAC-SMA simulation.

    Args:
        precip: Precipitation array (mm/dt)
        temp: Temperature array (°C)
        pet: PET array (mm/dt)
        params: Combined parameter dictionary (Snow-17 + SAC-SMA).
                Uses defaults for any missing parameters.
        day_of_year: Julian day array (1-366). If None, assumes
                     starting from day 1 with daily timestep.
        initial_state: Initial coupled state. If None, Snow-17 starts
                       with no snow, SAC-SMA at 50% capacity.
        warmup_days: Warmup period in days (output includes warmup).
        latitude: Latitude for melt factor seasonality.
        dt: Timestep in days (1.0 = daily).
        si: SWE threshold for areal depletion in Snow-17.

    Returns:
        Tuple of (total_flow mm/dt, final_state)
    """
    n = len(precip)

    # Parameters
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        params = full_params

    snow17_params, sacsma_params = split_params(params)

    # Day of year
    if day_of_year is None:
        day_of_year = np.arange(1, n + 1) % 365 + 1

    # Initial states
    if initial_state is None:
        snow17_state = Snow17State(
            w_i=0.0, w_q=0.0, w_qx=0.0,
            deficit=0.0, ati=0.0, swe=0.0,
        )
        sacsma_state = SacSmaState(
            uztwc=sacsma_params.UZTWM * 0.5,
            uzfwc=sacsma_params.UZFWM * 0.5,
            lztwc=sacsma_params.LZTWM * 0.5,
            lzfpc=sacsma_params.LZFPM * 0.5,
            lzfsc=sacsma_params.LZFSM * 0.5,
            adimc=(sacsma_params.UZTWM + sacsma_params.LZTWM) * 0.5,
        )
    else:
        snow17_state = initial_state.snow17
        sacsma_state = initial_state.sacsma

    # Step 1: Run Snow-17 to get rain+melt
    rain_plus_melt, snow17_final = snow17_simulate(
        precip, temp, day_of_year, snow17_params,
        initial_state=snow17_state,
        dt=dt,
        latitude=latitude,
        si=si,
    )

    # Step 2: Run SAC-SMA with rain+melt as effective precipitation
    total_flow, sacsma_final = sacsma_simulate(
        rain_plus_melt, pet, sacsma_params,
        initial_state=sacsma_state,
        dt=dt,
    )

    final_state = SacSmaSnow17State(
        snow17=snow17_final,
        sacsma=sacsma_final,
    )

    return total_flow, final_state
