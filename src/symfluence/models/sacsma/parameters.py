# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SAC-SMA + Snow-17 Parameter Definitions.

Defines all 26 parameters (10 Snow-17 + 16 SAC-SMA) with bounds, defaults,
and NamedTuple structures for the coupled model.

Snow-17 definitions are imported from the shared ``symfluence.models.snow17``
module. SAC-SMA NamedTuple fields use ``Any`` types for JAX tracer compatibility.

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.

    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
    Computer Models of Watershed Hydrology, 311-366.
"""

from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

# Import Snow-17 definitions from the shared module
from symfluence.models.snow17.parameters import (
    SNOW17_DEFAULTS,
    SNOW17_PARAM_BOUNDS,
    SNOW17_PARAM_NAMES,
    Snow17Params,
)
from symfluence.models.snow17.parameters import (
    params_dict_to_namedtuple as snow17_params_dict_to_namedtuple,
)

# Backward-compatibility alias
Snow17Parameters = Snow17Params


# =============================================================================
# SAC-SMA PARAMETERS
# =============================================================================

SACSMA_PARAM_NAMES: List[str] = [
    'UZTWM', 'UZFWM', 'UZK', 'LZTWM', 'LZFPM', 'LZFSM',
    'LZPK', 'LZSK', 'ZPERC', 'REXP', 'PFREE', 'PCTIM',
    'ADIMP', 'RIVA', 'SIDE', 'RSERV',
]

SACSMA_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'UZTWM': (10.0, 150.0),   # Upper zone tension water max (mm)
    'UZFWM': (1.0, 150.0),    # Upper zone free water max (mm)
    'UZK': (0.15, 0.75),      # Upper zone lateral depletion rate (1/day)
    'LZTWM': (1.0, 500.0),    # Lower zone tension water max (mm)
    'LZFPM': (1.0, 1000.0),   # Lower zone primary free water max (mm) - LOG
    'LZFSM': (1.0, 1000.0),   # Lower zone supplemental free water max (mm) - LOG
    'LZPK': (0.001, 0.05),    # Primary baseflow depletion rate (1/day) - LOG
    'LZSK': (0.01, 0.25),     # Supplemental baseflow depletion rate (1/day) - LOG
    'ZPERC': (1.0, 350.0),    # Max percolation rate scaling (-) - LOG
    'REXP': (1.0, 5.0),       # Percolation curve exponent (-)
    'PFREE': (0.0, 0.8),      # Fraction percolation to free water (-)
    'PCTIM': (0.0, 0.1),      # Permanent impervious area fraction (-)
    'ADIMP': (0.0, 0.4),      # Additional impervious area fraction (-)
    'RIVA': (0.0, 0.2),       # Riparian vegetation ET fraction (-)
    'SIDE': (0.0, 0.5),       # Deep recharge fraction (-)
    'RSERV': (0.0, 0.4),      # Lower zone free water reserve fraction (-)
}

SACSMA_DEFAULTS: Dict[str, float] = {
    'UZTWM': 50.0,
    'UZFWM': 40.0,
    'UZK': 0.3,
    'LZTWM': 130.0,
    'LZFPM': 60.0,
    'LZFSM': 25.0,
    'LZPK': 0.01,
    'LZSK': 0.05,
    'ZPERC': 40.0,
    'REXP': 2.0,
    'PFREE': 0.3,
    'PCTIM': 0.01,
    'ADIMP': 0.05,
    'RIVA': 0.0,
    'SIDE': 0.0,
    'RSERV': 0.3,
}

# Parameters requiring log transform for calibration (span 1.5-3 orders of magnitude)
LOG_TRANSFORM_PARAMS = {'ZPERC', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK'}


class SacSmaParameters(NamedTuple):
    """SAC-SMA model parameters. Fields use Any for JAX tracer compatibility."""
    UZTWM: Any   # Upper zone tension water maximum (mm)
    UZFWM: Any   # Upper zone free water maximum (mm)
    UZK: Any     # Upper zone lateral depletion rate (1/day)
    LZTWM: Any   # Lower zone tension water maximum (mm)
    LZFPM: Any   # Lower zone primary free water maximum (mm)
    LZFSM: Any   # Lower zone supplemental free water maximum (mm)
    LZPK: Any    # Primary baseflow depletion rate (1/day)
    LZSK: Any    # Supplemental baseflow depletion rate (1/day)
    ZPERC: Any   # Maximum percolation rate scaling (-)
    REXP: Any    # Percolation curve exponent (-)
    PFREE: Any   # Fraction percolation to free water (-)
    PCTIM: Any   # Permanent impervious area fraction (-)
    ADIMP: Any   # Additional impervious area fraction (-)
    RIVA: Any    # Riparian vegetation ET fraction (-)
    SIDE: Any    # Deep recharge fraction (-)
    RSERV: Any   # Lower zone free water reserve fraction (-)


# =============================================================================
# COMBINED PARAMETERS
# =============================================================================

# All parameter bounds
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    **SNOW17_PARAM_BOUNDS,
    **SACSMA_PARAM_BOUNDS,
}

# All default values
DEFAULT_PARAMS: Dict[str, float] = {
    **SNOW17_DEFAULTS,
    **SACSMA_DEFAULTS,
}


# =============================================================================
# PARAMETER UTILITIES
# =============================================================================

def params_dict_to_namedtuple(
    params_dict: Dict[str, float],
    use_jax: bool = False,
) -> SacSmaParameters:
    """Convert parameter dictionary to SacSmaParameters NamedTuple.

    Preserves JAX tracers when ``use_jax=True``.

    Args:
        params_dict: Dictionary of parameter name -> value
        use_jax: Whether to preserve JAX tracers (True) or cast to np.float64

    Returns:
        SacSmaParameters namedtuple
    """
    try:
        import jax.numpy as jnp
        _has_jax = True
    except ImportError:
        _has_jax = False

    values = {}
    for name in SACSMA_PARAM_NAMES:
        val = params_dict.get(name, SACSMA_DEFAULTS[name])
        if use_jax and _has_jax:
            values[name] = val if hasattr(val, 'shape') else jnp.asarray(val)
        else:
            values[name] = np.float64(val)

    return SacSmaParameters(**values)


def create_snow17_params(params_dict: Dict[str, float]) -> Snow17Params:
    """Create Snow17Params from a dictionary, filling missing with defaults."""
    return snow17_params_dict_to_namedtuple(params_dict, use_jax=False)


def create_sacsma_params(params_dict: Dict[str, float]) -> SacSmaParameters:
    """Create SacSmaParameters from a dictionary, filling missing with defaults."""
    return params_dict_to_namedtuple(params_dict, use_jax=False)


def split_params(params_dict: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split combined parameter dict into Snow-17 and SAC-SMA sub-dicts.

    Returns plain dicts so callers can convert to NamedTuples with the
    appropriate backend (JAX or NumPy).

    Args:
        params_dict: Combined parameter dictionary

    Returns:
        Tuple of (snow17_dict, sacsma_dict)
    """
    snow17_dict = {}
    sacsma_dict = {}
    for key, val in params_dict.items():
        if key in SNOW17_PARAM_NAMES:
            snow17_dict[key] = val
        elif key in SACSMA_PARAM_NAMES:
            sacsma_dict[key] = val

    # Fill missing with defaults
    for name in SNOW17_PARAM_NAMES:
        if name not in snow17_dict:
            snow17_dict[name] = SNOW17_DEFAULTS[name]
    for name in SACSMA_PARAM_NAMES:
        if name not in sacsma_dict:
            sacsma_dict[name] = SACSMA_DEFAULTS[name]

    return snow17_dict, sacsma_dict


def get_param_transform(param_name: str) -> str:
    """Get the transform type for a parameter ('linear' or 'log')."""
    return 'log' if param_name in LOG_TRANSFORM_PARAMS else 'linear'
