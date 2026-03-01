# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HEC-HMS Parameter Definitions and Utilities.

This module provides parameter bounds, defaults, data structures, and
utilities for the native Python/JAX implementation of HEC-HMS algorithms.

Algorithms implemented:
    - Temperature-Index (ATI) Snow Model
    - SCS Curve Number Continuous Loss
    - Clark Unit Hydrograph Transform
    - Linear Reservoir Baseflow

Parameter Units:
    All parameters are defined in DAILY units. For sub-daily simulation,
    rate parameters are scaled using scale_params_for_timestep().

References:
    US Army Corps of Engineers (2000). Hydrologic Modeling System HEC-HMS
    Technical Reference Manual. Hydrologic Engineering Center, Davis, CA.
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
    # Snow (ATI Temperature Index)
    'px_temp': (-2.0, 4.0),               # Rain/snow partition temperature (deg C)
    'base_temp': (-3.0, 3.0),             # Base temperature for melt (deg C)
    'ati_meltrate_coeff': (0.5, 1.5),     # ATI meltrate coefficient (-)
    'meltrate_max': (2.0, 10.0),          # Maximum melt rate (mm/deg C/day)
    'meltrate_min': (0.0, 3.0),           # Minimum melt rate (mm/deg C/day)
    'cold_limit': (0.0, 50.0),            # Cold limit for cold content (mm)
    'ati_cold_rate_coeff': (0.0, 0.3),    # ATI cold rate coefficient (-)
    'water_capacity': (0.0, 0.3),         # Liquid water holding capacity (fraction of SWE)

    # Loss (SCS Curve Number Continuous)
    'cn': (30.0, 98.0),                   # Curve number (-)
    'initial_abstraction_ratio': (0.05, 0.3),  # Ia/S ratio (-)

    # Transform (Clark Unit Hydrograph)
    'tc': (0.5, 20.0),                    # Time of concentration (days)
    'r_coeff': (0.5, 20.0),              # Storage coefficient R (days)

    # Baseflow (Linear Reservoir)
    'gw_storage_coeff': (1.0, 100.0),     # GW storage coefficient (days)
    'deep_perc_fraction': (0.0, 0.5),     # Deep percolation fraction (-)
}

# Default parameter values (tuned for temperate catchments)
DEFAULT_PARAMS: Dict[str, Any] = {
    'px_temp': 1.0,
    'base_temp': 0.0,
    'ati_meltrate_coeff': 0.98,
    'meltrate_max': 5.0,
    'meltrate_min': 1.0,
    'cold_limit': 10.0,
    'ati_cold_rate_coeff': 0.1,
    'water_capacity': 0.05,
    'cn': 65.0,
    'initial_abstraction_ratio': 0.2,
    'tc': 3.0,
    'r_coeff': 5.0,
    'gw_storage_coeff': 30.0,
    'deep_perc_fraction': 0.1,
}

# Parameters that require temporal scaling for sub-daily timesteps
FLUX_RATE_PARAMS = {'meltrate_max', 'meltrate_min'}
DURATION_PARAMS = {'tc', 'r_coeff', 'gw_storage_coeff'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HecHmsParameters(NamedTuple):
    """
    HEC-HMS model parameters.

    Attributes:
        px_temp: Rain/snow partition temperature (deg C)
        base_temp: Base temperature for snowmelt (deg C)
        ati_meltrate_coeff: ATI meltrate coefficient (-)
        meltrate_max: Maximum melt rate (mm/deg C/day)
        meltrate_min: Minimum melt rate (mm/deg C/day)
        cold_limit: Cold limit for cold content deficit (mm)
        ati_cold_rate_coeff: ATI cold rate coefficient (-)
        water_capacity: Liquid water holding capacity (fraction of SWE)
        cn: SCS Curve Number (-)
        initial_abstraction_ratio: Initial abstraction ratio Ia/S (-)
        tc: Time of concentration (days)
        r_coeff: Clark storage coefficient R (days)
        gw_storage_coeff: Groundwater storage coefficient (days)
        deep_perc_fraction: Deep percolation fraction (-)
    """
    px_temp: Any
    base_temp: Any
    ati_meltrate_coeff: Any
    meltrate_max: Any
    meltrate_min: Any
    cold_limit: Any
    ati_cold_rate_coeff: Any
    water_capacity: Any
    cn: Any
    initial_abstraction_ratio: Any
    tc: Any
    r_coeff: Any
    gw_storage_coeff: Any
    deep_perc_fraction: Any


class HecHmsState(NamedTuple):
    """
    HEC-HMS model state variables.

    Attributes:
        snow_swe: Snow water equivalent (mm)
        snow_liquid: Liquid water in snowpack (mm)
        snow_ati: Antecedent temperature index (deg C)
        snow_cold_content: Cold content deficit (mm)
        soil_deficit: Soil moisture deficit (mm); starts at S = 25400/CN - 254
        clark_storage: Clark linear reservoir storage (mm)
        gw_storage_1: Groundwater storage (mm)
        gw_storage_2: Deep groundwater storage (mm)
    """
    snow_swe: Any
    snow_liquid: Any
    snow_ati: Any
    snow_cold_content: Any
    soil_deficit: Any
    clark_storage: Any
    gw_storage_1: Any
    gw_storage_2: Any


# =============================================================================
# PARAMETER UTILITIES
# =============================================================================

def create_params_from_dict(
    params_dict: Dict[str, Any],
    use_jax: bool = True
) -> HecHmsParameters:
    """
    Create HecHmsParameters from a dictionary.

    Args:
        params_dict: Dictionary mapping parameter names to values.
            Missing parameters use defaults.
        use_jax: Whether to convert to JAX arrays (requires JAX).

    Returns:
        HecHmsParameters namedtuple.
    """
    full_params = {**DEFAULT_PARAMS, **params_dict}

    if use_jax and HAS_JAX:
        return HecHmsParameters(
            px_temp=jnp.array(full_params['px_temp']),
            base_temp=jnp.array(full_params['base_temp']),
            ati_meltrate_coeff=jnp.array(full_params['ati_meltrate_coeff']),
            meltrate_max=jnp.array(full_params['meltrate_max']),
            meltrate_min=jnp.array(full_params['meltrate_min']),
            cold_limit=jnp.array(full_params['cold_limit']),
            ati_cold_rate_coeff=jnp.array(full_params['ati_cold_rate_coeff']),
            water_capacity=jnp.array(full_params['water_capacity']),
            cn=jnp.array(full_params['cn']),
            initial_abstraction_ratio=jnp.array(full_params['initial_abstraction_ratio']),
            tc=jnp.array(full_params['tc']),
            r_coeff=jnp.array(full_params['r_coeff']),
            gw_storage_coeff=jnp.array(full_params['gw_storage_coeff']),
            deep_perc_fraction=jnp.array(full_params['deep_perc_fraction']),
        )
    else:
        return HecHmsParameters(
            px_temp=np.float64(full_params['px_temp']),
            base_temp=np.float64(full_params['base_temp']),
            ati_meltrate_coeff=np.float64(full_params['ati_meltrate_coeff']),
            meltrate_max=np.float64(full_params['meltrate_max']),
            meltrate_min=np.float64(full_params['meltrate_min']),
            cold_limit=np.float64(full_params['cold_limit']),
            ati_cold_rate_coeff=np.float64(full_params['ati_cold_rate_coeff']),
            water_capacity=np.float64(full_params['water_capacity']),
            cn=np.float64(full_params['cn']),
            initial_abstraction_ratio=np.float64(full_params['initial_abstraction_ratio']),
            tc=np.float64(full_params['tc']),
            r_coeff=np.float64(full_params['r_coeff']),
            gw_storage_coeff=np.float64(full_params['gw_storage_coeff']),
            deep_perc_fraction=np.float64(full_params['deep_perc_fraction']),
        )


def create_initial_state(
    cn=65.0,
    use_jax: bool = True
) -> HecHmsState:
    """
    Create initial HEC-HMS model state.

    Args:
        cn: Curve Number for computing initial soil deficit.
            Can be a JAX tracer when use_jax=True (no float() cast).
        use_jax: Whether to use JAX arrays.

    Returns:
        HecHmsState namedtuple.
    """
    if use_jax and HAS_JAX:
        # Use jnp.maximum instead of Python max() to preserve JAX tracers
        s_max = 25400.0 / jnp.maximum(cn, 1.0) - 254.0
        return HecHmsState(
            snow_swe=jnp.array(0.0),
            snow_liquid=jnp.array(0.0),
            snow_ati=jnp.array(0.0),
            snow_cold_content=jnp.array(0.0),
            soil_deficit=s_max,
            clark_storage=jnp.array(0.0),
            gw_storage_1=jnp.array(10.0),
            gw_storage_2=jnp.array(5.0),
        )
    else:
        # NumPy path: safe to use Python max()
        s_max = 25400.0 / max(float(cn), 1.0) - 254.0
        return HecHmsState(
            snow_swe=np.float64(0.0),
            snow_liquid=np.float64(0.0),
            snow_ati=np.float64(0.0),
            snow_cold_content=np.float64(0.0),
            soil_deficit=np.float64(s_max),
            clark_storage=np.float64(0.0),
            gw_storage_1=np.float64(10.0),
            gw_storage_2=np.float64(5.0),
        )
