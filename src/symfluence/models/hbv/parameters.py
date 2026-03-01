# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HBV-96 Parameter Definitions and Utilities.

This module provides parameter bounds, defaults, data structures, and
transformation utilities for the HBV-96 model.

Parameter Units (Daily Convention):
    All parameters are defined in DAILY units per HBV-96 convention
    (Lindström et al., 1997). For sub-daily simulation, rate parameters
    are scaled using scale_params_for_timestep().

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

# Lazy JAX import with numpy fallback
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


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
    'k0': (0.05, 0.5),        # Recession coefficient for fast flow (1/day)
    'k1': (0.01, 0.3),        # Recession coefficient for slow flow (1/day)
    'k2': (0.0001, 0.1),      # Recession coefficient for baseflow (1/day)
    'uzl': (0.0, 100.0),      # Threshold for fast flow (mm)
    'perc': (0.0, 20.0),      # Maximum percolation rate (mm/day)
    'maxbas': (1.0, 7.0),     # Length of triangular routing function (days)
    'smoothing': (1.0, 50.0), # Smoothing factor for thresholds (dimensionless)
}

# Default parameter values (midpoint of bounds, tuned for temperate catchments)
# NOTE: All parameters are defined in DAILY units per HBV-96 convention
DEFAULT_PARAMS: Dict[str, Any] = {
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
    'smoothing': 15.0,
    'smoothing_enabled': False,
}

# Parameters that require temporal scaling for sub-daily timesteps
# Flux rates (mm/day) - use linear scaling
FLUX_RATE_PARAMS = {'cfmax', 'perc'}

# Recession coefficients (1/day) - use exact exponential scaling
# These represent exponential decay: S(t+dt) = S(t) * (1-k)^dt
# Linear scaling is only accurate for small k; exact formula is always correct
RECESSION_PARAMS = {'k0', 'k1', 'k2'}

# Combined for backward compatibility
RATE_PARAMS = FLUX_RATE_PARAMS | RECESSION_PARAMS

# Parameters that represent durations in days
DURATION_PARAMS = {'maxbas'}


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
        smoothing: Smoothing factor for threshold approximations
        smoothing_enabled: Whether to use smooth approximations (bool/int)
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
    smoothing: Any
    smoothing_enabled: Any


# =============================================================================
# PARAMETER UTILITIES
# =============================================================================

def create_params_from_dict(
    params_dict: Dict[str, Any],
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
            smoothing=jnp.array(full_params.get('smoothing', 15.0)),
            smoothing_enabled=jnp.array(full_params.get('smoothing_enabled', False), dtype=bool),
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
            smoothing=np.float64(full_params.get('smoothing', 15.0)),
            smoothing_enabled=bool(full_params.get('smoothing_enabled', False)),
        )


def scale_params_for_timestep(
    params_dict: Dict[str, float],
    timestep_hours: int = 24
) -> Dict[str, float]:
    """
    Scale HBV parameters from daily to sub-daily timestep.

    The HBV-96 model parameters are conventionally defined in daily units
    (Lindström et al., 1997). For sub-daily simulation, rate parameters must
    be scaled appropriately for the timestep duration.

    Scaling approach:
    - Flux rate parameters (cfmax, perc): Linear scaling by (timestep_hours / 24)
      These have units of mm/day and represent fluxes per time unit.

    - Recession coefficients (k0, k1, k2): Exact exponential scaling
      These represent exponential reservoir decay: S(t+dt) = S(t) * (1-k)^dt
      The exact relationship is: k_subdaily = 1 - (1 - k_daily)^(dt/24)
      This is mathematically exact and avoids the ~5% error of linear scaling.

    - Duration parameters (maxbas): remain in original units (days)
      The routing buffer length is adjusted separately based on timestep.

    - Dimensionless parameters (sfcf, cfr, cwh, lp, beta): unchanged
    - Threshold parameters (tt, fc, uzl): unchanged (represent states, not rates)
    - Smoothing parameters: unchanged

    Args:
        params_dict: Dictionary of HBV parameters in daily units.
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Dictionary with parameters scaled for the specified timestep.

    References:
        Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
        Development and test of the distributed HBV-96 hydrological model.
        Journal of Hydrology, 201(1-4), 272-288.
    """
    if timestep_hours == 24:
        return params_dict.copy()

    if timestep_hours < 1 or timestep_hours > 24:
        raise ValueError(f"timestep_hours must be between 1 and 24, got {timestep_hours}")

    scale_factor = timestep_hours / 24.0
    scaled = params_dict.copy()

    # Scale flux rate parameters (linear scaling)
    for param in FLUX_RATE_PARAMS:
        if param in scaled:
            scaled[param] = scaled[param] * scale_factor

    # Scale recession coefficients (exact exponential scaling)
    # k_subdaily = 1 - (1 - k_daily)^(dt/24)
    for param in RECESSION_PARAMS:
        if param in scaled:
            k_daily = scaled[param]
            # Ensure k_daily is in valid range [0, 1) for the formula
            k_daily = min(max(k_daily, 0.0), 0.9999)
            scaled[param] = 1.0 - (1.0 - k_daily) ** scale_factor

    return scaled


def get_routing_buffer_length(maxbas_days: float, timestep_hours: int = 24) -> int:
    """
    Calculate routing buffer length in timesteps.

    Args:
        maxbas_days: MAXBAS parameter value in days.
        timestep_hours: Model timestep in hours.

    Returns:
        Buffer length in number of timesteps.
    """
    timesteps_per_day = 24 / timestep_hours
    buffer_length = int(np.ceil(maxbas_days * timesteps_per_day)) + 2
    return max(buffer_length, 10)  # Minimum buffer of 10 timesteps
