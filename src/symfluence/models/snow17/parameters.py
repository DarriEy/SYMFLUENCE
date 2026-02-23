"""
Snow-17 Parameters and State Definitions.

Parameter bounds, defaults, state/param NamedTuples, and conversion utilities
for the standalone Snow-17 temperature index snow model.

All NamedTuple fields use ``Any`` type for JAX tracer compatibility.

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.
"""

from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

# =============================================================================
# STATE AND PARAMETER STRUCTURES
# =============================================================================

class Snow17State(NamedTuple):
    """Snow-17 model state variables."""
    w_i: Any     # Ice portion of SWE (mm)
    w_q: Any     # Liquid water in snowpack (mm)
    w_qx: Any    # Liquid water capacity (mm)
    deficit: Any  # Heat deficit (mm, energy equiv.)
    ati: Any     # Antecedent temperature index (C)
    swe: Any     # Total SWE for areal depletion (mm)


class Snow17Params(NamedTuple):
    """Snow-17 model parameters."""
    SCF: Any     # Snowfall correction factor
    PXTEMP: Any  # Rain/snow threshold temperature (C)
    MFMAX: Any   # Max melt factor Jun 21 (mm/C/6hr)
    MFMIN: Any   # Min melt factor Dec 21 (mm/C/6hr)
    NMF: Any     # Negative melt factor (mm/C/6hr)
    MBASE: Any   # Base melt temperature (C)
    TIPM: Any    # Antecedent temperature index weight
    UADJ: Any    # Rain-on-snow wind function (mm/mb/6hr)
    PLWHC: Any   # Liquid water holding capacity
    DAYGM: Any   # Daily ground melt (mm/day)


# =============================================================================
# PARAMETER NAMES AND BOUNDS
# =============================================================================

SNOW17_PARAM_NAMES: List[str] = [
    'SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF',
    'MBASE', 'TIPM', 'UADJ', 'PLWHC', 'DAYGM',
]

SNOW17_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'SCF':    (0.7, 1.4),
    'PXTEMP': (-2.0, 2.0),
    'MFMAX':  (0.5, 2.0),
    'MFMIN':  (0.05, 0.6),
    'NMF':    (0.05, 0.5),
    'MBASE':  (0.0, 1.0),
    'TIPM':   (0.01, 1.0),
    'UADJ':   (0.01, 0.2),
    'PLWHC':  (0.01, 0.3),
    'DAYGM':  (0.0, 0.3),
}

SNOW17_DEFAULTS: Dict[str, float] = {
    'SCF':    1.0,
    'PXTEMP': 0.0,
    'MFMAX':  1.0,
    'MFMIN':  0.3,
    'NMF':    0.15,
    'MBASE':  0.0,
    'TIPM':   0.1,
    'UADJ':   0.04,
    'PLWHC':  0.04,
    'DAYGM':  0.0,
}

# 11-point areal depletion curve (fraction of area covered by snow)
# Index corresponds to SWE/SI ratio from 0.0 to 1.0 in 0.1 increments
DEFAULT_ADC = np.array([
    0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
])


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def params_dict_to_namedtuple(
    params_dict: Dict[str, float],
    use_jax: bool = True,
) -> Snow17Params:
    """Convert parameter dictionary to Snow17Params NamedTuple.

    Args:
        params_dict: Dictionary of parameter name -> value
        use_jax: Whether to preserve JAX tracers (True) or cast to np.float64

    Returns:
        Snow17Params namedtuple
    """
    try:
        import jax.numpy as jnp
        _has_jax = True
    except ImportError:
        _has_jax = False

    values = {}
    for name in SNOW17_PARAM_NAMES:
        val = params_dict.get(name, SNOW17_DEFAULTS[name])
        if use_jax and _has_jax:
            values[name] = val if hasattr(val, 'shape') else jnp.array(float(val))
        else:
            values[name] = np.float64(val)

    return Snow17Params(**values)


def create_initial_state(use_jax: bool = True) -> Snow17State:
    """Create initial Snow-17 state (no snow).

    Args:
        use_jax: Whether to use JAX arrays

    Returns:
        Snow17State with all zeros
    """
    try:
        import jax.numpy as jnp
        _has_jax = True
    except ImportError:
        _has_jax = False

    if use_jax and _has_jax:
        return Snow17State(
            w_i=jnp.array(0.0), w_q=jnp.array(0.0), w_qx=jnp.array(0.0),
            deficit=jnp.array(0.0), ati=jnp.array(0.0), swe=jnp.array(0.0),
        )
    else:
        return Snow17State(
            w_i=np.float64(0.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(0.0),
        )
