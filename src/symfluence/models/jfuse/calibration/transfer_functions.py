# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
JAX-Differentiable Transfer Functions for Distributed jFUSE Calibration.

Maps catchment attributes to spatially varying FUSE parameters via linear
transfer functions: param_i(gru) = a_i + b_i * attr_norm_i(gru).

Instead of calibrating 14 uniform parameters, we calibrate 28 transfer
function coefficients (14 pairs of a, b) that produce per-GRU parameters
through differentiable operations. This is fully compatible with JAX autodiff.

For `smooth_frac` (numerical smoothing), no spatial variation is applied:
the coefficient is a single `a` value broadcast uniformly.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    from jax import Array
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from jfuse.fuse.state import NUM_PARAMETERS, PARAM_BOUNDS, PARAM_NAMES

# ============================================================================
# Parameter-to-attribute mapping
# ============================================================================

# Maps each calibrated jFUSE parameter to its driving catchment attribute.
# Attributes are normalized to [0, 1] before use.
PARAM_ATTR_MAP: Dict[str, str] = {
    'S1_max':      'precip_mm_yr',   # Upper storage scales with wetness
    'S2_max':      'precip_mm_yr',   # Lower storage scales with wetness
    'ku':          'aridity',        # Drainage rate varies with water availability
    'ki':          'aridity',        # Interflow varies with aridity
    'ks':          'aridity',        # Baseflow varies with aridity
    'n':           'aridity',        # TOPMODEL decay varies with climate
    'Ac_max':      'precip_mm_yr',   # Saturated area fraction scales with wetness
    'b':           'precip_mm_yr',   # VIC parameter scales with precipitation
    'f_rchr':      'aridity',        # Recharge fraction relates to climate
    'T_rain':      'elev_m',         # Rain/snow threshold varies with elevation
    'T_melt':      'elev_m',         # Melt threshold varies with elevation
    'MFMAX':       'temp_C',         # Max melt factor varies with temperature
    'MFMIN':       'snow_frac',      # Min melt factor varies with snow prevalence
    'smooth_frac': 'constant',       # Numerical smoothing — no spatial variation
}

# Default calibrated parameter list (matches existing JFUSE_PARAMS_TO_CALIBRATE)
DEFAULT_CALIBRATED_PARAMS = list(PARAM_ATTR_MAP.keys())

# Default bounds for the `b` (slope) coefficient
DEFAULT_B_BOUNDS = (-5.0, 5.0)


# ============================================================================
# Configuration class (setup-time, NumPy)
# ============================================================================

class JaxTransferFunctionConfig:
    """Setup-time configuration for JAX transfer functions.

    Loads and normalizes GRU attributes, builds the attribute matrix,
    coefficient names, and coefficient bounds. All heavy NumPy work
    happens here so that the JAX function only sees pre-built arrays.
    """

    def __init__(
        self,
        attributes_path: str,
        calibrated_params: Optional[List[str]] = None,
        non_coastal_indices: Optional[np.ndarray] = None,
        b_bounds: Tuple[float, float] = DEFAULT_B_BOUNDS,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            attributes_path: Path to subcatchment_attributes.csv (7618 rows).
            calibrated_params: Parameter names to calibrate. Defaults to
                DEFAULT_CALIBRATED_PARAMS (14 params).
            non_coastal_indices: Row indices for non-coastal GRUs. If None,
                rows with is_coastal==0 are auto-detected.
            b_bounds: Bounds for all `b` coefficients.
            logger: Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.b_bounds = b_bounds
        self.calibrated_params = calibrated_params or DEFAULT_CALIBRATED_PARAMS

        # Load and filter attributes
        import pandas as pd
        attrs_df = pd.read_csv(attributes_path)
        self.logger.info(f"Loaded {len(attrs_df)} GRU attributes from {attributes_path}")

        if non_coastal_indices is not None:
            attrs_df = attrs_df.iloc[non_coastal_indices].reset_index(drop=True)
        elif 'is_coastal' in attrs_df.columns:
            attrs_df = attrs_df[attrs_df['is_coastal'] == 0].reset_index(drop=True)

        self.n_grus = len(attrs_df)
        self.logger.info(f"Using {self.n_grus} non-coastal GRUs")

        # Normalize each attribute column to [0, 1]
        self._attr_stats: Dict[str, Tuple[float, float]] = {}
        attr_cols = ['elev_m', 'precip_mm_yr', 'temp_C', 'aridity', 'snow_frac']
        for col in attr_cols:
            if col in attrs_df.columns:
                mn = float(attrs_df[col].min())
                mx = float(attrs_df[col].max())
                rng = mx - mn
                self._attr_stats[col] = (mn, mx)
                if rng > 0:
                    attrs_df[f'{col}_norm'] = (attrs_df[col] - mn) / rng
                else:
                    attrs_df[f'{col}_norm'] = 0.5

        # Build attribute matrix (n_grus, n_calibrated_params)
        # Each column is the normalized attribute for the corresponding param.
        n_params = len(self.calibrated_params)
        attr_matrix = np.zeros((self.n_grus, n_params), dtype=np.float32)
        for i, pname in enumerate(self.calibrated_params):
            attr_name = PARAM_ATTR_MAP.get(pname, 'constant')
            if attr_name == 'constant':
                # No spatial variation — column of zeros so b*0 = 0
                attr_matrix[:, i] = 0.0
            else:
                norm_col = f'{attr_name}_norm'
                if norm_col in attrs_df.columns:
                    attr_matrix[:, i] = attrs_df[norm_col].values.astype(np.float32)
                else:
                    self.logger.warning(
                        f"Attribute '{attr_name}' not found for param '{pname}', "
                        f"using zeros (constant)"
                    )
                    attr_matrix[:, i] = 0.0

        self._attr_matrix = attr_matrix

        # Build coefficient names and bounds
        self._coeff_names: List[str] = []
        self._coeff_bounds: List[Tuple[float, float]] = []
        for pname in self.calibrated_params:
            a_lo, a_hi = PARAM_BOUNDS[pname]
            self._coeff_names.append(f'{pname}_a')
            self._coeff_bounds.append((a_lo, a_hi))
            self._coeff_names.append(f'{pname}_b')
            self._coeff_bounds.append(self.b_bounds)

        # Build param_indices: position of each calibrated param in PARAM_NAMES
        self._param_indices = np.array(
            [PARAM_NAMES.index(p) for p in self.calibrated_params],
            dtype=np.int32,
        )

        # Build default full parameter array (30,) from Parameters.default()
        from jfuse.fuse.state import Parameters
        default_p = Parameters.default(n_hrus=1)
        self._default_full_params = np.array(
            [float(getattr(default_p, name)) for name in PARAM_NAMES],
            dtype=np.float32,
        )

        # Full parameter bounds as arrays (30,)
        self._lower_bounds = np.array(
            [PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=np.float32
        )
        self._upper_bounds = np.array(
            [PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=np.float32
        )

        self.logger.info(
            f"Transfer function config: {n_params} params, "
            f"{self.n_coefficients} coefficients, {self.n_grus} GRUs"
        )

    # -- Properties ----------------------------------------------------------

    @property
    def n_calibrated_params(self) -> int:
        return len(self.calibrated_params)

    @property
    def n_coefficients(self) -> int:
        return len(self._coeff_names)

    @property
    def coefficient_names(self) -> List[str]:
        return list(self._coeff_names)

    @property
    def coefficient_bounds(self) -> List[Tuple[float, float]]:
        return list(self._coeff_bounds)

    @property
    def param_indices(self) -> np.ndarray:
        return self._param_indices

    # -- JAX array accessors -------------------------------------------------

    def get_attr_matrix_jax(self) -> "Array":
        return jnp.array(self._attr_matrix)

    def get_default_full_params_jax(self) -> "Array":
        return jnp.array(self._default_full_params)

    def get_param_indices_jax(self) -> "Array":
        return jnp.array(self._param_indices, dtype=jnp.int32)

    def get_lower_bounds_jax(self) -> "Array":
        return jnp.array(self._lower_bounds)

    def get_upper_bounds_jax(self) -> "Array":
        return jnp.array(self._upper_bounds)

    def get_default_coefficients(self) -> np.ndarray:
        """Return a reasonable starting point for coefficients.

        Sets each `a` to the actual parameter default (from Parameters.default())
        and each `b` to 0 (equivalent to uniform parameters).
        """
        coeffs = np.zeros(self.n_coefficients, dtype=np.float32)
        for i, pname in enumerate(self.calibrated_params):
            coeffs[2 * i] = self._default_full_params[PARAM_NAMES.index(pname)]
            coeffs[2 * i + 1] = 0.0            # b = 0 (uniform)
        return coeffs


# ============================================================================
# Pure JAX transfer function
# ============================================================================

def apply_transfer_functions(
    coeff_array: "Array",      # (2*n_calib,) — pairs of (a, b)
    attr_matrix: "Array",      # (n_grus, n_calib) — normalized attributes
    default_params: "Array",   # (30,) — defaults for non-calibrated params
    param_indices: "Array",    # (n_calib,) — index into PARAM_NAMES
    lower_bounds: "Array",     # (30,)
    upper_bounds: "Array",     # (30,)
    n_grus: int,
) -> "Array":
    """Apply linear transfer functions to produce per-GRU parameters.

    For each calibrated parameter i:
        values = a_i + b_i * attr_norm_i(gru)

    The result is a (n_grus, 30) array of full parameter sets, with
    non-calibrated parameters filled from defaults and all values
    clipped to valid bounds.

    This function is pure JAX and fully differentiable via jax.grad.
    The Python for-loop (14 iterations) is unrolled at JIT trace time.

    Args:
        coeff_array: Flat array of transfer function coefficients.
            Layout: [a_0, b_0, a_1, b_1, ..., a_{n-1}, b_{n-1}]
        attr_matrix: Normalized attribute values per GRU per param.
        default_params: Default parameter values for the full 30-param set.
        param_indices: Which index in PARAM_NAMES each calibrated param maps to.
        lower_bounds: Lower bounds for all 30 parameters.
        upper_bounds: Upper bounds for all 30 parameters.
        n_grus: Number of GRUs (used for broadcasting).

    Returns:
        Array of shape (n_grus, 30) with per-GRU parameter values.
    """
    n_calib = param_indices.shape[0]

    # Start with default params broadcast to (n_grus, 30)
    full_params = jnp.broadcast_to(default_params[None, :], (n_grus, NUM_PARAMETERS))
    # Make a mutable copy via identity add
    full_params = full_params + jnp.zeros((n_grus, NUM_PARAMETERS))

    # Apply transfer function for each calibrated parameter
    for i in range(n_calib):
        a_i = coeff_array[2 * i]
        b_i = coeff_array[2 * i + 1]
        # values: (n_grus,)
        values = a_i + b_i * attr_matrix[:, i]
        idx = param_indices[i]
        full_params = full_params.at[:, idx].set(values)

    # Clip to valid bounds
    full_params = jnp.clip(full_params, lower_bounds[None, :], upper_bounds[None, :])

    return full_params
