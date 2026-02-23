"""
Xinanjiang Model Parameters.

Parameter definitions, bounds, defaults, and conversion utilities
for the Xinanjiang (XAJ) rainfall-runoff model.

The 15 parameters follow Zhao (1992) with standard naming conventions.
Parameters are organized into three groups:
- Generation: K, B, IM, UM, LM, DM, C (evapotranspiration + runoff)
- Source separation: SM, EX, KI, KG (free water storage)
- Routing: CS, L, CI, CG (channel and reservoir routing)

References:
    Zhao, R.-J. (1992). The Xinanjiang model applied in China.
    Journal of Hydrology, 135(1-4), 371-381.
"""

from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAM_NAMES: List[str] = [
    'K', 'B', 'IM', 'UM', 'LM', 'DM', 'C',
    'SM', 'EX', 'KI', 'KG', 'CS', 'L', 'CI', 'CG',
]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Generation parameters
    'K':  (0.1, 1.5),      # PET correction factor (>1 allows sublimation compensation)
    'B':  (0.1, 2.0),      # Tension water capacity curve exponent (Zhao 1992 allows up to ~2)
    'IM': (0.01, 0.1),     # Impervious area fraction
    'UM': (5.0, 50.0),     # Upper layer tension water capacity (mm)
    'LM': (50.0, 120.0),   # Lower layer tension water capacity (mm)
    'DM': (50.0, 200.0),   # Deep layer tension water capacity (mm)
    'C':  (0.0, 0.2),      # Deep layer ET coefficient

    # Source separation parameters
    'SM': (1.0, 200.0),    # Free water capacity (mm)
    'EX': (0.5, 2.0),      # Free water capacity curve exponent
    'KI': (0.0, 0.7),      # Interflow outflow coefficient
    'KG': (0.0, 0.7),      # Groundwater outflow coefficient

    # Routing parameters
    # Note: CS (channel recession) and L (lag time) are excluded from
    # calibration bounds because they are not used in the current lumped
    # model formulation. They remain in PARAM_NAMES and DEFAULT_PARAMS
    # for structural completeness (Zhao 1992).
    'CI': (0.0, 0.9),      # Interflow recession constant
    'CG': (0.98, 0.998),   # Groundwater recession constant
}

DEFAULT_PARAMS: Dict[str, float] = {
    'K':  0.5,
    'B':  0.3,
    'IM': 0.05,
    'UM': 10.0,
    'LM': 80.0,
    'DM': 80.0,
    'C':  0.1,
    'SM': 20.0,
    'EX': 1.2,
    'KI': 0.3,
    'KG': 0.3,
    'CS': 0.5,
    'L':  1.0,
    'CI': 0.5,
    'CG': 0.99,
}

LOG_TRANSFORM_PARAMS = {'SM'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class XinanjiangParams(NamedTuple):
    """Xinanjiang model parameters as a NamedTuple for JAX compatibility."""
    K: Any     # PET correction factor
    B: Any     # Tension water capacity curve exponent
    IM: Any    # Impervious area fraction
    UM: Any    # Upper layer tension water capacity (mm)
    LM: Any    # Lower layer tension water capacity (mm)
    DM: Any    # Deep layer tension water capacity (mm)
    C: Any     # Deep layer ET coefficient
    SM: Any    # Free water capacity (mm)
    EX: Any    # Free water capacity curve exponent
    KI: Any    # Interflow outflow coefficient
    KG: Any    # Groundwater outflow coefficient
    CS: Any    # Channel recession constant
    L: Any     # Lag time (timesteps)
    CI: Any    # Interflow recession constant
    CG: Any    # Groundwater recession constant


class XinanjiangState(NamedTuple):
    """Xinanjiang model state variables.

    Attributes:
        wu: Upper layer tension water storage (mm)
        wl: Lower layer tension water storage (mm)
        wd: Deep layer tension water storage (mm)
        s: Free water storage (mm)
        fr: Runoff contributing area fraction (0-1)
        qi: Interflow reservoir storage (mm)
        qg: Groundwater reservoir storage (mm)
    """
    wu: Any   # Upper layer tension water (mm)
    wl: Any   # Lower layer tension water (mm)
    wd: Any   # Deep layer tension water (mm)
    s: Any    # Free water storage (mm)
    fr: Any   # Runoff contributing area fraction
    qi: Any   # Interflow reservoir (mm)
    qg: Any   # Groundwater reservoir (mm)


# =============================================================================
# CONSTRAINT ENFORCEMENT
# =============================================================================

def enforce_ki_kg_constraint(ki: float, kg: float, max_sum: float = 0.99) -> Tuple[float, float]:
    """Enforce KI + KG < max_sum by proportional scaling.

    Args:
        ki: Interflow coefficient
        kg: Groundwater coefficient
        max_sum: Maximum allowed sum (default 0.99)

    Returns:
        Tuple of (adjusted_ki, adjusted_kg)
    """
    total = ki + kg
    if total >= max_sum:
        scale = max_sum / (total + 1e-10)
        return ki * scale, kg * scale
    return ki, kg


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def params_dict_to_namedtuple(params_dict: Dict[str, float], use_jax: bool = True) -> XinanjiangParams:
    """Convert parameter dictionary to XinanjiangParams NamedTuple.

    Args:
        params_dict: Dictionary of parameter name -> value
        use_jax: Whether to convert to JAX arrays

    Returns:
        XinanjiangParams namedtuple

    Note:
        When use_jax=True, values are passed through as-is (no float() cast)
        to preserve JAX tracer compatibility for autodiff.
    """
    try:
        import jax.numpy as jnp
        HAS_JAX = True
    except ImportError:
        HAS_JAX = False

    # Enforce KI+KG constraint (only for concrete values, not JAX tracers)
    ki = params_dict.get('KI', DEFAULT_PARAMS['KI'])
    kg = params_dict.get('KG', DEFAULT_PARAMS['KG'])
    if use_jax and HAS_JAX:
        # Use JAX-compatible constraint: scale if sum >= 0.99
        total = ki + kg
        scale = jnp.where(total >= 0.99, 0.99 / (total + 1e-10), 1.0)
        ki = ki * scale
        kg = kg * scale
    else:
        ki, kg = enforce_ki_kg_constraint(ki, kg)

    values = {}
    for name in PARAM_NAMES:
        val = params_dict.get(name, DEFAULT_PARAMS[name])
        if name == 'KI':
            val = ki
        elif name == 'KG':
            val = kg
        if use_jax and HAS_JAX:
            # Don't cast to float() — preserves JAX tracers for autodiff
            values[name] = val if hasattr(val, 'shape') else jnp.array(val)
        else:
            values[name] = np.float64(val)

    return XinanjiangParams(**values)


# =============================================================================
# SNOW-17 COUPLING UTILITIES
# =============================================================================

def split_params(params_dict: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split a combined parameter dictionary into XAJ and Snow-17 subsets.

    Args:
        params_dict: Combined parameter dictionary with both XAJ and Snow-17 keys

    Returns:
        Tuple of (xaj_dict, snow17_dict)
    """
    from symfluence.models.snow17.parameters import SNOW17_DEFAULTS, SNOW17_PARAM_NAMES

    snow17_dict = {}
    xaj_dict = {}
    for key, val in params_dict.items():
        if key in SNOW17_PARAM_NAMES:
            snow17_dict[key] = val
        else:
            xaj_dict[key] = val

    # Fill missing Snow-17 defaults
    for name in SNOW17_PARAM_NAMES:
        if name not in snow17_dict:
            snow17_dict[name] = SNOW17_DEFAULTS[name]

    # Fill missing XAJ defaults
    for name in PARAM_NAMES:
        if name not in xaj_dict:
            xaj_dict[name] = DEFAULT_PARAMS[name]

    return xaj_dict, snow17_dict


def get_combined_param_names() -> List[str]:
    """Get combined XAJ + Snow-17 parameter names."""
    from symfluence.models.snow17.parameters import SNOW17_PARAM_NAMES
    return PARAM_NAMES + SNOW17_PARAM_NAMES


def get_combined_defaults() -> Dict[str, float]:
    """Get combined default parameters for XAJ + Snow-17."""
    from symfluence.models.snow17.parameters import SNOW17_DEFAULTS
    combined = DEFAULT_PARAMS.copy()
    combined.update(SNOW17_DEFAULTS)
    return combined


def params_dict_to_array(params_dict: Dict[str, float]) -> np.ndarray:
    """Convert parameter dictionary to array (ordered by PARAM_NAMES).

    Args:
        params_dict: Dictionary of parameter name -> value

    Returns:
        1D numpy array of parameter values
    """
    return np.array([params_dict.get(name, DEFAULT_PARAMS[name]) for name in PARAM_NAMES])


def params_array_to_dict(params_array: np.ndarray) -> Dict[str, float]:
    """Convert parameter array to dictionary.

    Args:
        params_array: 1D array of parameter values (ordered by PARAM_NAMES)

    Returns:
        Dictionary of parameter name -> value
    """
    return dict(zip(PARAM_NAMES, params_array))
