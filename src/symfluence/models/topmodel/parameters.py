# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TOPMODEL Parameter Definitions and Utilities.

This module provides parameter bounds, defaults, data structures, and
utilities for the native Python/JAX implementation of TOPMODEL (Beven & Kirkby 1979).

Algorithm components:
    - Degree-day snow module
    - Exponential transmissivity baseflow
    - Saturation-excess overland flow with topographic index distribution
    - Linear reservoir channel routing

Parameter Units:
    All parameters are defined in DAILY units. The topographic index
    distribution is generated internally as a discretized normal distribution.

References:
    Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable
    contributing area model of basin hydrology. Hydrological Sciences
    Bulletin, 24(1), 43-69.
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
# CONSTANTS
# =============================================================================

N_TI_BINS = 30  # Number of topographic index bins


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Subsurface / transmissivity
    'm': (0.001, 0.3),             # Transmissivity decay parameter (m)
    'lnTe': (-7.0, 10.0),         # Effective log transmissivity ln(m^2/h)
    'Srmax': (0.005, 0.5),        # Max root zone storage (m)
    'Sr0': (0.0, 0.1),            # Initial root zone deficit (m)
    'td': (0.1, 50.0),            # Unsaturated zone time delay (h/m)

    # Routing
    'k_route': (1.0, 200.0),      # Routing reservoir coefficient (h)

    # Snow (degree-day)
    'DDF': (0.5, 10.0),           # Degree-day melt factor (mm/degC/day)
    'T_melt': (-2.0, 3.0),        # Melt threshold temperature (degC)
    'T_snow': (-2.0, 3.0),        # Snow/rain threshold temperature (degC)

    # Topographic index distribution
    'ti_std': (1.0, 10.0),        # TI distribution spread (-)

    # Initial conditions
    'S0': (0.0, 2.0),             # Initial mean deficit (m)
}

# Default parameter values
DEFAULT_PARAMS: Dict[str, Any] = {
    'm': 0.05,
    'lnTe': 1.0,
    'Srmax': 0.05,
    'Sr0': 0.01,
    'td': 5.0,
    'k_route': 48.0,
    'DDF': 3.5,
    'T_melt': 0.0,
    'T_snow': 1.0,
    'ti_std': 4.0,
    'S0': 0.5,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TopmodelParameters(NamedTuple):
    """
    TOPMODEL parameters.

    Attributes:
        m: Transmissivity decay parameter (m)
        lnTe: Effective log transmissivity ln(m^2/h)
        Srmax: Max root zone storage (m)
        Sr0: Initial root zone deficit (m)
        td: Unsaturated zone time delay (h/m)
        k_route: Routing reservoir coefficient (h)
        DDF: Degree-day melt factor (mm/degC/day)
        T_melt: Melt threshold temperature (degC)
        T_snow: Snow/rain threshold temperature (degC)
        ti_std: TI distribution spread (-)
        S0: Initial mean deficit (m)
    """
    m: Any
    lnTe: Any
    Srmax: Any
    Sr0: Any
    td: Any
    k_route: Any
    DDF: Any
    T_melt: Any
    T_snow: Any
    ti_std: Any
    S0: Any


class TopmodelState(NamedTuple):
    """
    TOPMODEL state variables.

    Attributes:
        s_bar: Mean saturation deficit (m)
        srz: Root zone deficit per TI class (m), shape (N_TI_BINS,)
        suz: Unsaturated zone storage per TI class (m), shape (N_TI_BINS,)
        swe: Snow water equivalent (mm)
        q_routed: Routed streamflow (mm/day)
    """
    s_bar: Any
    srz: Any
    suz: Any
    swe: Any
    q_routed: Any


# =============================================================================
# PARAMETER UTILITIES
# =============================================================================

def create_params_from_dict(
    params_dict: Dict[str, Any],
    use_jax: bool = True
) -> TopmodelParameters:
    """
    Create TopmodelParameters from a dictionary.

    Args:
        params_dict: Dictionary mapping parameter names to values.
            Missing parameters use defaults.
        use_jax: Whether to convert to JAX arrays (requires JAX).

    Returns:
        TopmodelParameters namedtuple.
    """
    full_params = {**DEFAULT_PARAMS, **params_dict}

    if use_jax and HAS_JAX:
        return TopmodelParameters(
            m=jnp.array(full_params['m']),
            lnTe=jnp.array(full_params['lnTe']),
            Srmax=jnp.array(full_params['Srmax']),
            Sr0=jnp.array(full_params['Sr0']),
            td=jnp.array(full_params['td']),
            k_route=jnp.array(full_params['k_route']),
            DDF=jnp.array(full_params['DDF']),
            T_melt=jnp.array(full_params['T_melt']),
            T_snow=jnp.array(full_params['T_snow']),
            ti_std=jnp.array(full_params['ti_std']),
            S0=jnp.array(full_params['S0']),
        )
    else:
        return TopmodelParameters(
            m=np.float64(full_params['m']),
            lnTe=np.float64(full_params['lnTe']),
            Srmax=np.float64(full_params['Srmax']),
            Sr0=np.float64(full_params['Sr0']),
            td=np.float64(full_params['td']),
            k_route=np.float64(full_params['k_route']),
            DDF=np.float64(full_params['DDF']),
            T_melt=np.float64(full_params['T_melt']),
            T_snow=np.float64(full_params['T_snow']),
            ti_std=np.float64(full_params['ti_std']),
            S0=np.float64(full_params['S0']),
        )


def generate_ti_distribution(
    ti_std=4.0,
    n_bins: int = N_TI_BINS,
    use_jax: bool = True
) -> Tuple[Any, Any]:
    """
    Generate a discretized topographic index distribution.

    Creates n_bins equally-spaced bins of a normal distribution
    centered at 0 with standard deviation ti_std. The absolute
    TI values are absorbed into lnTe.

    Args:
        ti_std: Standard deviation of the TI distribution.
            Can be a JAX tracer when use_jax=True (no float() cast).
        n_bins: Number of bins.
        use_jax: Whether to return JAX arrays.

    Returns:
        Tuple of (lnaotb, dist_area) where:
            lnaotb: TI bin centers, shape (n_bins,)
            dist_area: Fractional area per bin, shape (n_bins,), sums to 1.0
    """
    xp = jnp if (use_jax and HAS_JAX) else np

    # Create bin centers spanning +/- 3 sigma
    lnaotb = xp.linspace(-3.0 * ti_std, 3.0 * ti_std, n_bins)

    # Normal distribution PDF at bin centers
    # Use xp.maximum instead of Python max() to preserve JAX tracers
    pdf = xp.exp(-0.5 * (lnaotb / xp.maximum(ti_std, 1e-6)) ** 2)

    # Normalize to sum to 1
    dist_area = pdf / xp.sum(pdf)

    return lnaotb, dist_area


def create_initial_state(
    params: TopmodelParameters = None,
    use_jax: bool = True
) -> TopmodelState:
    """
    Create initial TOPMODEL state.

    Args:
        params: Model parameters (for initial deficit S0 and Sr0).
            Parameter values can be JAX tracers when use_jax=True.
        use_jax: Whether to use JAX arrays.

    Returns:
        TopmodelState namedtuple.
    """
    xp = jnp if (use_jax and HAS_JAX) else np

    # Avoid float() cast to preserve JAX tracers
    if params is not None:
        s0 = params.S0
        sr0 = params.Sr0
    else:
        s0 = 0.5
        sr0 = 0.01

    return TopmodelState(
        s_bar=xp.asarray(s0),
        srz=xp.ones(N_TI_BINS) * sr0,
        suz=xp.zeros(N_TI_BINS),
        swe=xp.array(0.0),
        q_routed=xp.array(0.0),
    )
