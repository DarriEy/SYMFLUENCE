"""
HBV-96 Model Core - JAX Implementation.

Pure JAX functions for the HBV-96 hydrological model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- GPU acceleration when available

The HBV-96 model consists of four main routines:
1. Snow routine - Degree-day accumulation/melt with refreezing
2. Soil routine - Beta-function recharge, ET reduction
3. Response routine - Two-box (upper/lower zone) with percolation
4. Routing routine - Triangular transfer function convolution

Temporal Resolution: Daily vs Sub-Daily Operation
=================================================

The HBV-96 model was originally developed and parameterized for DAILY timesteps
(Lindström et al., 1997). All parameter values in the literature are defined
in daily units. This implementation supports both daily and sub-daily (e.g., hourly)
simulation through automatic parameter scaling.

**Parameter Units (Daily Convention):**

From Lindström et al. (1997), Table 1:
- CFMAX: Degree-day factor [mm °C⁻¹ day⁻¹] (Eq. 2)
- K0, K1, K2: Recession coefficients [day⁻¹] (Eq. 6-7)
- PERC: Maximum percolation rate [mm day⁻¹] (Eq. 5)
- MAXBAS: Routing parameter [days] (Eq. 8)
- FC, LP, BETA, TT, etc.: Dimensionless or state-based (no temporal scaling needed)

**Sub-Daily Scaling Approach:**

For timestep Δt (in hours), rate parameters are scaled as:
    param_subdaily = param_daily × (Δt / 24)

This applies to: CFMAX, K0, K1, K2, PERC

For example, with hourly timestep (Δt=1):
    CFMAX_hourly = CFMAX_daily / 24
    K1_hourly = K1_daily / 24

The MAXBAS routing parameter remains in days; the triangular weighting function
is distributed across more timesteps accordingly.

**Important Considerations for Sub-Daily Simulation:**

1. **Forcing Data**: Precipitation and PET must be provided at the model timestep
   resolution (e.g., mm/hour for hourly simulation). The preprocessor handles
   this conversion when HBV_TIMESTEP_HOURS < 24.

2. **Recession Coefficient Accuracy**: Linear scaling of K0, K1, K2 is an
   approximation. The exact relationship is:
       k_subdaily = 1 - (1 - k_daily)^(Δt/24)
   For typical HBV k values (0.01 - 0.5), linear scaling is accurate within ~5%.

3. **Snow Routine**: Degree-day melt (CFMAX) scales linearly with timestep.
   The temperature threshold (TT) and refreezing coefficient (CFR) are unchanged.

4. **Calibration**: Parameters should still be calibrated in DAILY units.
   The simulate() function automatically applies scaling when timestep_hours < 24.

**Configuration:**

Set HBV_TIMESTEP_HOURS in your configuration:
- HBV_TIMESTEP_HOURS = 24  (default, daily simulation)
- HBV_TIMESTEP_HOURS = 1   (hourly simulation)

**References:**

Primary Reference:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
    https://doi.org/10.1016/S0022-1694(96)02128-8

Key Equations from Lindström et al. (1997):
- Eq. 1: Snow/rain partitioning based on threshold temperature TT
- Eq. 2: Snowmelt M = CFMAX × (T - TT) when T > TT [mm/day]
- Eq. 3: Soil moisture recharge dQ/dP = (SM/FC)^BETA
- Eq. 4: Evapotranspiration reduction below LP threshold
- Eq. 5: Percolation from upper to lower zone (max = PERC mm/day)
- Eq. 6: Upper zone outflow Q0 = K0 × max(SUZ - UZL, 0) + K1 × SUZ
- Eq. 7: Lower zone outflow Q2 = K2 × SLZ
- Eq. 8: Triangular weighting function for routing over MAXBAS days

Additional References:
    Bergström, S. (1976). Development and application of a conceptual runoff
    model for Scandinavian catchments. SMHI Reports RHO No. 7, Norrköping.

    Seibert, J. (1999). Regionalisation of parameters for a conceptual
    rainfall-runoff model. Agricultural and Forest Meteorology, 98-99, 279-293.
"""

from typing import NamedTuple, Optional, Tuple, Dict, Any
import warnings

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
        "JAX not available. HBV model will use NumPy backend with reduced functionality. "
        "Install JAX for autodiff, JIT compilation, and GPU support: pip install jax jaxlib"
    )

import numpy as np


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
    'k0': (0.05, 0.99),       # Recession coefficient for fast flow (1/day)
    'k1': (0.01, 0.5),        # Recession coefficient for slow flow (1/day)
    'k2': (0.0001, 0.1),      # Recession coefficient for baseflow (1/day)
    'uzl': (0.0, 100.0),      # Threshold for fast flow (mm)
    'perc': (0.0, 10.0),      # Maximum percolation rate (mm/day)
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
# These are rate parameters with units of /day or mm/day
RATE_PARAMS = {'cfmax', 'k0', 'k1', 'k2', 'perc'}
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


class HBVState(NamedTuple):
    """
    HBV-96 model state variables.

    Represents the current state of storages in the model.

    Attributes:
        snow: Snow pack storage (mm water equivalent)
        snow_water: Liquid water content in snow pack (mm)
        sm: Soil moisture storage (mm)
        suz: Upper zone storage (mm)
        slz: Lower zone storage (mm)
        routing_buffer: Buffer for triangular routing (mm), length = max_routing_days
    """
    snow: Any        # float or array
    snow_water: Any
    sm: Any
    suz: Any
    slz: Any
    routing_buffer: Any  # 1D array of length max_routing_days


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
    be scaled proportionally to the timestep duration.

    Scaling approach (following Lindström et al., 1997, Eq. 1-8):
    - Rate parameters (cfmax, k0, k1, k2, perc): scaled by (timestep_hours / 24)
      These have units of /day or mm/day and represent fluxes per time unit.
    - Duration parameters (maxbas): remain in original units (days)
      The routing buffer length is adjusted separately based on timestep.
    - Dimensionless parameters (sfcf, cfr, cwh, lp, beta): unchanged
    - Threshold parameters (tt, fc, uzl): unchanged (represent states, not rates)
    - Smoothing: unchanged
    - Smoothing Enabled: unchanged

    For recession coefficients (k0, k1, k2), linear scaling is an approximation.
    The exact relationship is: k_subdaily = 1 - (1 - k_daily)^(dt/24)
    For small k values typical in HBV, linear scaling is accurate within ~5%.

    Args:
        params_dict: Dictionary of HBV parameters in daily units.
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Dictionary with parameters scaled for the specified timestep.

    References:
        Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
        Development and test of the distributed HBV-96 hydrological model.
        Journal of Hydrology, 201(1-4), 272-288.
        https://doi.org/10.1016/S0022-1694(96)02128-8
    """
    if timestep_hours == 24:
        return params_dict.copy()

    if timestep_hours < 1 or timestep_hours > 24:
        raise ValueError(f"timestep_hours must be between 1 and 24, got {timestep_hours}")

    scale_factor = timestep_hours / 24.0
    scaled = params_dict.copy()

    # Scale rate parameters
    for param in RATE_PARAMS:
        if param in scaled:
            scaled[param] = scaled[param] * scale_factor

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


def create_initial_state(
    initial_snow: float = 0.0,
    initial_sm: float = 150.0,
    initial_suz: float = 10.0,
    initial_slz: float = 10.0,
    max_routing_days: int = 10,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> HBVState:
    """
    Create initial HBV state.

    Args:
        initial_snow: Initial snow storage (mm).
        initial_sm: Initial soil moisture (mm).
        initial_suz: Initial upper zone storage (mm).
        initial_slz: Initial lower zone storage (mm).
        max_routing_days: Maximum routing days (buffer size in days).
        use_jax: Whether to use JAX arrays.
        timestep_hours: Model timestep in hours (affects routing buffer length).

    Returns:
        HBVState namedtuple.
    """
    # Calculate buffer length in timesteps
    buffer_length = get_routing_buffer_length(max_routing_days, timestep_hours)

    if use_jax and HAS_JAX:
        return HBVState(
            snow=jnp.array(initial_snow),
            snow_water=jnp.array(0.0),
            sm=jnp.array(initial_sm),
            suz=jnp.array(initial_suz),
            slz=jnp.array(initial_slz),
            routing_buffer=jnp.zeros(buffer_length),
        )
    else:
        return HBVState(
            snow=np.float64(initial_snow),
            snow_water=np.float64(0.0),
            sm=np.float64(initial_sm),
            suz=np.float64(initial_suz),
            slz=np.float64(initial_slz),
            routing_buffer=np.zeros(buffer_length),
        )


# =============================================================================
# SMOOTH APPROXIMATIONS
# =============================================================================

def _smooth_threshold(val, threshold, smoothing, enabled, use_jax):
    """
    Smooth or hard threshold based on enabled flag.
    Returns ~1 if val > threshold, ~0 if val < threshold.
    """
    if use_jax and HAS_JAX:
        # If enabled is a JAX array (tracer), this uses jnp.where for both branches
        smooth_val = jax.nn.sigmoid(smoothing * (val - threshold))
        hard_val = jnp.where(val > threshold, 1.0, 0.0)
        return jnp.where(enabled, smooth_val, hard_val)
    else:
        if enabled:
            x = smoothing * (val - threshold)
            return np.where(x >= 0,
                            1 / (1 + np.exp(-x)),
                            np.exp(x) / (1 + np.exp(x)))
        else:
            return np.where(val > threshold, 1.0, 0.0)

def _smooth_relu(val, threshold, smoothing, enabled, use_jax):
    """
    Smooth or hard ReLU based on enabled flag.
    Returns max(val - threshold, 0).
    """
    if use_jax and HAS_JAX:
        x = (val - threshold) * smoothing
        smooth_val = jax.nn.softplus(x) / smoothing
        hard_val = jnp.maximum(val - threshold, 0.0)
        return jnp.where(enabled, smooth_val, hard_val)
    else:
        if enabled:
            x = (val - threshold) * smoothing
            out = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
            return out / smoothing
        else:
            return np.maximum(val - threshold, 0.0)

def _smooth_min(val, limit, smoothing, enabled, use_jax):
    """
    Smooth or hard min based on enabled flag.
    Returns min(val, limit).
    """
    # min(a, b) = a - max(a - b, 0)
    return val - _smooth_relu(val, limit, smoothing, enabled, use_jax)


# =============================================================================
# CORE ROUTINES (JAX IMPLEMENTATION)
# =============================================================================

def _get_backend(use_jax: bool = True):
    """Get the appropriate array backend (JAX or NumPy)."""
    if use_jax and HAS_JAX:
        return jnp
    return np


def snow_routine_jax(
    precip: Any,
    temp: Any,
    snow: Any,
    snow_water: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 snow routine (JAX version).

    Partitions precipitation into rain and snow based on temperature threshold.
    Calculates snowmelt using degree-day method with refreezing.
    Uses smooth approximations for differentiability.

    Args:
        precip: Precipitation (mm/day)
        temp: Air temperature (°C)
        snow: Current snow storage (mm SWE)
        snow_water: Liquid water in snow (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_snow, new_snow_water, rainfall_plus_melt)
    """
    # Partition precipitation
    rain_frac = _smooth_threshold(
        temp, params.tt, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    rainfall = precip * rain_frac
    snowfall = precip * params.sfcf * (1.0 - rain_frac)

    # Add snowfall to pack
    snow = snow + snowfall

    # Potential melt (degree-day)
    # pot_melt = cfmax * max(temp - tt, 0)
    pot_melt = params.cfmax * _smooth_relu(
        temp, params.tt, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # Actual melt limited by available snow
    # melt = min(pot_melt, snow)
    melt = _smooth_min(
        pot_melt, snow, params.smoothing, params.smoothing_enabled, use_jax=True
    )
    snow = snow - melt

    # Add melt to liquid water in snow
    snow_water = snow_water + melt + rainfall

    # Refreezing of liquid water when temp < tt
    # pot_refreeze = cfr * cfmax * max(tt - temp, 0)
    # Note reversed args for smooth_relu to get max(tt - temp, 0)
    pot_refreeze = params.cfr * params.cfmax * _smooth_relu(
        params.tt, temp, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # refreeze = min(pot_refreeze, snow_water)
    refreeze = _smooth_min(
        pot_refreeze, snow_water, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    snow = snow + refreeze
    snow_water = snow_water - refreeze

    # Water holding capacity
    max_water = params.cwh * snow

    # outflow = max(snow_water - max_water, 0)
    outflow = _smooth_relu(
        snow_water, max_water, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # snow_water = min(snow_water, max_water)
    snow_water = snow_water - outflow

    return snow, snow_water, outflow


def soil_routine_jax(
    rainfall_plus_melt: Any,
    pet: Any,
    sm: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 soil moisture routine (JAX version).

    Calculates recharge to groundwater and actual evapotranspiration.
    Uses beta-function for non-linear recharge relationship.

    Args:
        rainfall_plus_melt: Water input from snow routine (mm/day)
        pet: Potential evapotranspiration (mm/day)
        sm: Current soil moisture (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_sm, recharge, actual_et)
    """
    # Relative soil moisture
    rel_sm = sm / params.fc

    # Recharge using beta function (non-linear)
    # dQ/dP = (SM/FC)^beta
    # We smooth the min(rel_sm, 1.0) to avoid kink at saturation
    smoothed_rel_sm = _smooth_min(
        rel_sm, 1.0, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    recharge = rainfall_plus_melt * jnp.power(
        smoothed_rel_sm,
        params.beta
    )

    # Soil moisture update
    sm = sm + rainfall_plus_melt - recharge

    # Evapotranspiration reduction below LP threshold
    # AET/PET = SM / (LP * FC) when SM < LP * FC
    lp_threshold = params.lp * params.fc

    # Smooth min(sm, lp_threshold)
    effective_sm_for_et = _smooth_min(
        sm, lp_threshold, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # et_factor = effective_sm / lp_threshold
    et_factor = effective_sm_for_et / lp_threshold

    actual_et = pet * et_factor

    # Limit ET to available soil moisture (smooth min)
    actual_et = _smooth_min(
        actual_et, sm, params.smoothing, params.smoothing_enabled, use_jax=True
    )
    sm = sm - actual_et

    # Ensure non-negative
    sm = jnp.maximum(sm, 0.0)

    return sm, recharge, actual_et


def response_routine_jax(
    recharge: Any,
    suz: Any,
    slz: Any,
    params: HBVParameters
) -> Tuple[Any, Any, Any]:
    """
    HBV-96 response routine (JAX version).

    Two-box groundwater model with percolation from upper to lower zone.
    Upper zone produces fast and intermediate flow, lower zone produces baseflow.

    Args:
        recharge: Recharge from soil moisture (mm/day)
        suz: Upper zone storage (mm)
        slz: Lower zone storage (mm)
        params: HBV parameters

    Returns:
        Tuple of (new_suz, new_slz, total_runoff)
    """
    # Add recharge to upper zone
    suz = suz + recharge

    # Percolation from upper to lower zone
    # perc = min(params.perc, suz)
    perc = _smooth_min(
        params.perc, suz, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    suz = suz - perc
    slz = slz + perc

    # Upper zone outflow
    # Q0 = k0 * max(SUZ - UZL, 0)  (fast surface runoff)
    # This is the critical threshold we want to smooth for UZL calibration
    q0 = params.k0 * _smooth_relu(
        suz, params.uzl, params.smoothing, params.smoothing_enabled, use_jax=True
    )

    # Q1 = k1 * SUZ                 (interflow)
    q1 = params.k1 * suz

    # Lower zone outflow (baseflow)
    q2 = params.k2 * slz

    # Update storages
    suz = suz - q0 - q1
    slz = slz - q2

    # Ensure non-negative
    suz = jnp.maximum(suz, 0.0)
    slz = jnp.maximum(slz, 0.0)

    # Total runoff before routing
    total_runoff = q0 + q1 + q2

    return suz, slz, total_runoff



def triangular_weights(
    maxbas: float,
    buffer_length: int = 10,
    timestep_hours: int = 24
) -> Any:
    """
    Calculate triangular weighting function for routing.

    The triangular transfer function distributes runoff over time according
    to a triangular unit hydrograph (Lindström et al., 1997, Eq. 8).
    For sub-daily timesteps, the triangle is stretched across more timesteps.

    Args:
        maxbas: Base of triangle in DAYS (parameter units)
        buffer_length: Maximum buffer length in TIMESTEPS
        timestep_hours: Model timestep in hours (for converting maxbas to timesteps)

    Returns:
        Array of weights (sums to 1.0)

    References:
        Lindström et al. (1997), Eq. 8: Triangular weighting function
    """
    # Convert maxbas from days to timesteps
    timesteps_per_day = 24.0 / timestep_hours
    maxbas_timesteps = maxbas * timesteps_per_day

    if HAS_JAX:
        timesteps = jnp.arange(1, buffer_length + 1, dtype=jnp.float32)
        # Rising limb (0 to maxbas/2)
        rising = jnp.where(
            timesteps <= maxbas_timesteps / 2,
            timesteps / (maxbas_timesteps / 2),
            0.0
        )
        # Falling limb (maxbas/2 to maxbas)
        falling = jnp.where(
            (timesteps > maxbas_timesteps / 2) & (timesteps <= maxbas_timesteps),
            (maxbas_timesteps - timesteps) / (maxbas_timesteps / 2),
            0.0
        )
        weights = rising + falling
        # Normalize to sum to 1
        weights = weights / jnp.sum(weights + 1e-10)
        return weights
    else:
        timesteps = np.arange(1, buffer_length + 1, dtype=np.float64)
        rising = np.where(
            timesteps <= maxbas_timesteps / 2,
            timesteps / (maxbas_timesteps / 2),
            0.0
        )
        falling = np.where(
            (timesteps > maxbas_timesteps / 2) & (timesteps <= maxbas_timesteps),
            (maxbas_timesteps - timesteps) / (maxbas_timesteps / 2),
            0.0
        )
        weights = rising + falling
        weights = weights / np.sum(weights + 1e-10)
        return weights


def routing_routine_jax(
    runoff: Any,
    routing_buffer: Any,
    params: HBVParameters,
    timestep_hours: int = 24
) -> Tuple[Any, Any]:
    """
    HBV-96 triangular routing routine (JAX version).

    Applies triangular transfer function to smooth runoff response.
    For sub-daily timesteps, the routing function is applied per-timestep.

    Args:
        runoff: Total runoff before routing (mm/timestep)
        routing_buffer: Previous routing buffer state
        params: HBV parameters (maxbas in DAYS)
        timestep_hours: Model timestep in hours (for weight calculation)

    Returns:
        Tuple of (routed_runoff, new_routing_buffer)
    """
    buffer_length = routing_buffer.shape[0]

    # Get triangular weights (maxbas is in days, converted internally)
    weights = triangular_weights(params.maxbas, buffer_length, timestep_hours)

    # Distribute this timestep's runoff across future timesteps
    new_buffer = routing_buffer + runoff * weights

    # Output is the first element
    routed_runoff = new_buffer[0]

    # Shift buffer (advance by one timestep)
    if HAS_JAX:
        new_buffer = jnp.concatenate([new_buffer[1:], jnp.array([0.0])])
    else:
        new_buffer = np.concatenate([new_buffer[1:], np.array([0.0])])

    return routed_runoff, new_buffer


# =============================================================================
# SINGLE TIMESTEP
# =============================================================================

def step_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    state: HBVState,
    params: HBVParameters,
    timestep_hours: int = 24
) -> Tuple[HBVState, Any]:
    """
    Execute one timestep of HBV-96 model (JAX version).

    Runs all four routines in sequence: snow, soil, response, routing.
    Parameters should already be scaled for the timestep using scale_params_for_timestep().

    Args:
        precip: Precipitation (mm/timestep)
        temp: Air temperature (°C)
        pet: Potential evapotranspiration (mm/timestep)
        state: Current model state
        params: Model parameters (already scaled for timestep)
        timestep_hours: Model timestep in hours (for routing weights)

    Returns:
        Tuple of (new_state, routed_runoff)
    """
    # Snow routine
    snow, snow_water, rainfall_plus_melt = snow_routine_jax(
        precip, temp, state.snow, state.snow_water, params
    )

    # Soil moisture routine
    sm, recharge, actual_et = soil_routine_jax(
        rainfall_plus_melt, pet, state.sm, params
    )

    # Response routine (groundwater)
    suz, slz, total_runoff = response_routine_jax(
        recharge, state.suz, state.slz, params
    )

    # Routing routine (needs timestep_hours for weight calculation)
    routed_runoff, routing_buffer = routing_routine_jax(
        total_runoff, state.routing_buffer, params, timestep_hours
    )

    # Create new state
    new_state = HBVState(
        snow=snow,
        snow_water=snow_water,
        sm=sm,
        suz=suz,
        slz=slz,
        routing_buffer=routing_buffer,
    )

    return new_state, routed_runoff


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_jax(
    precip: Any,
    temp: Any,
    pet: Any,
    params: HBVParameters,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Tuple[Any, HBVState]:
    """
    Run full HBV-96 simulation using JAX lax.scan (JIT-compatible).

    Args:
        precip: Precipitation timeseries (mm/timestep), shape (n_timesteps,)
        temp: Temperature timeseries (°C), shape (n_timesteps,)
        pet: PET timeseries (mm/timestep), shape (n_timesteps,)
        params: HBV parameters (should be pre-scaled for timestep)
        initial_state: Initial model state (uses defaults if None)
        warmup_days: Number of warmup days (included in output but typically ignored)
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Tuple of (runoff_timeseries, final_state)

    Note:
        For sub-daily simulation, parameters should be scaled using
        scale_params_for_timestep() before creating HBVParameters.
    """
    if not HAS_JAX:
        return simulate_numpy(precip, temp, pet, params, initial_state, warmup_days, timestep_hours)

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=True, timestep_hours=timestep_hours)

    # Stack forcing for scan
    forcing = jnp.stack([precip, temp, pet], axis=1)

    def scan_fn(state, forcing_step):
        p, t, e = forcing_step
        new_state, runoff = step_jax(p, t, e, state, params, timestep_hours)
        return new_state, runoff

    # Run simulation using scan (efficient and differentiable)
    final_state, runoff = lax.scan(scan_fn, initial_state, forcing)

    return runoff, final_state


def simulate_numpy(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: HBVParameters,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    timestep_hours: int = 24
) -> Tuple[np.ndarray, HBVState]:
    """
    Run full HBV-96 simulation using NumPy (fallback when JAX not available).

    Args:
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        params: HBV parameters (should be pre-scaled for timestep)
        initial_state: Initial model state
        warmup_days: Number of warmup days
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).

    Returns:
        Tuple of (runoff_timeseries, final_state)
    """
    n_timesteps = len(precip)

    # Initialize state if not provided
    if initial_state is None:
        initial_state = create_initial_state(use_jax=False, timestep_hours=timestep_hours)

    # Storage for results
    runoff = np.zeros(n_timesteps)
    state = initial_state

    for i in range(n_timesteps):
        # Snow routine (numpy version)
        snow, snow_water, rainfall_plus_melt = _snow_routine_numpy(
            precip[i], temp[i], state.snow, state.snow_water, params
        )

        # Soil routine (numpy version)
        sm, recharge, actual_et = _soil_routine_numpy(
            rainfall_plus_melt, pet[i], state.sm, params
        )

        # Response routine (numpy version)
        suz, slz, total_runoff = _response_routine_numpy(
            recharge, state.suz, state.slz, params
        )

        # Routing routine (numpy version)
        routed_runoff, routing_buffer = _routing_routine_numpy(
            total_runoff, state.routing_buffer, params, timestep_hours
        )

        # Update state
        state = HBVState(
            snow=snow,
            snow_water=snow_water,
            sm=sm,
            suz=suz,
            slz=slz,
            routing_buffer=routing_buffer,
        )

        runoff[i] = routed_runoff

    return runoff, state


# NumPy versions of routines (for fallback)
def _snow_routine_numpy(precip, temp, snow, snow_water, params):
    """NumPy version of snow routine."""
    rainfall = precip if temp > params.tt else 0.0
    snowfall = precip * params.sfcf if temp <= params.tt else 0.0

    snow = snow + snowfall
    pot_melt = params.cfmax * max(temp - params.tt, 0.0)
    melt = min(pot_melt, snow)
    snow = snow - melt

    snow_water = snow_water + melt + rainfall
    pot_refreeze = params.cfr * params.cfmax * max(params.tt - temp, 0.0)
    refreeze = min(pot_refreeze, snow_water)
    snow = snow + refreeze
    snow_water = snow_water - refreeze

    max_water = params.cwh * snow
    outflow = max(snow_water - max_water, 0.0)
    snow_water = min(snow_water, max_water)

    return snow, snow_water, outflow


def _soil_routine_numpy(rainfall_plus_melt, pet, sm, params):
    """NumPy version of soil routine."""
    rel_sm = sm / params.fc
    recharge = rainfall_plus_melt * (min(rel_sm, 1.0) ** params.beta)
    sm = sm + rainfall_plus_melt - recharge

    lp_threshold = params.lp * params.fc
    et_factor = sm / lp_threshold if sm < lp_threshold else 1.0
    actual_et = pet * et_factor
    actual_et = min(actual_et, sm)
    sm = sm - actual_et
    sm = max(sm, 0.0)

    return sm, recharge, actual_et


def _response_routine_numpy(recharge, suz, slz, params):
    """NumPy version of response routine."""
    suz = suz + recharge
    perc = min(params.perc, suz)
    suz = suz - perc
    slz = slz + perc

    q0 = params.k0 * max(suz - params.uzl, 0.0)
    q1 = params.k1 * suz
    q2 = params.k2 * slz

    suz = max(suz - q0 - q1, 0.0)
    slz = max(slz - q2, 0.0)

    return suz, slz, q0 + q1 + q2


def _routing_routine_numpy(runoff, routing_buffer, params, timestep_hours=24):
    """NumPy version of routing routine."""
    buffer_length = len(routing_buffer)
    weights = triangular_weights(params.maxbas, buffer_length, timestep_hours)

    new_buffer = routing_buffer.copy() + runoff * weights
    routed_runoff = new_buffer[0]
    new_buffer = np.concatenate([new_buffer[1:], np.array([0.0])])

    return routed_runoff, new_buffer


# =============================================================================
# LOSS FUNCTIONS (DIFFERENTIABLE)
# =============================================================================

def nse_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True
) -> float:
    """
    Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Negative because optimization minimizes, and higher NSE is better.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries
        temp: Temperature timeseries
        pet: PET timeseries
        obs: Observed streamflow timeseries
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative NSE (loss to minimize)
    """
    params = create_params_from_dict(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days)
        # Exclude warmup period
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        # NSE = 1 - sum((sim-obs)^2) / sum((obs-mean(obs))^2)
        ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
        ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse  # Negative for minimization
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        ss_res = np.sum((sim_eval - obs_eval) ** 2)
        ss_tot = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse


def kge_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True
) -> float:
    """
    Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Parameter dictionary
        precip: Precipitation timeseries
        temp: Temperature timeseries
        pet: PET timeseries
        obs: Observed streamflow timeseries
        warmup_days: Days to exclude from loss calculation
        use_jax: Whether to use JAX backend

    Returns:
        Negative KGE (loss to minimize)
    """
    params = create_params_from_dict(params_dict, use_jax=use_jax)

    if use_jax and HAS_JAX:
        sim, _ = simulate_jax(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        # KGE components
        r = jnp.corrcoef(sim_eval, obs_eval)[0, 1]  # Correlation
        alpha = jnp.std(sim_eval) / (jnp.std(obs_eval) + 1e-10)  # Variability ratio
        beta = jnp.mean(sim_eval) / (jnp.mean(obs_eval) + 1e-10)  # Bias ratio

        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge
    else:
        sim, _ = simulate_numpy(precip, temp, pet, params, warmup_days=warmup_days)
        sim_eval = sim[warmup_days:]
        obs_eval = obs[warmup_days:]

        r = np.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = np.std(sim_eval) / (np.std(obs_eval) + 1e-10)
        beta = np.mean(sim_eval) / (np.mean(obs_eval) + 1e-10)

        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge


# =============================================================================
# GRADIENT FUNCTIONS
# =============================================================================

def get_nse_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365
):
    """
    Get gradient function for NSE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed)
        temp: Temperature timeseries (fixed)
        pet: PET timeseries (fixed)
        obs: Observed streamflow (fixed)
        warmup_days: Warmup period

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        # Convert array back to dict
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365
):
    """
    Get gradient function for KGE loss.

    Returns a function that computes gradients w.r.t. parameters.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(params_dict, precip, temp, pet, obs, warmup_days, use_jax=True)

    return jax.grad(loss_fn)


# =============================================================================
# ENSEMBLE SIMULATION (VMAP)
# =============================================================================

def simulate_ensemble(
    precip: Any,
    temp: Any,
    pet: Any,
    params_batch: Dict[str, Any],
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365
) -> Any:
    """
    Run ensemble of HBV simulations using JAX vmap.

    Efficiently runs multiple parameter sets in parallel.

    Args:
        precip: Precipitation timeseries, shape (n_days,)
        temp: Temperature timeseries, shape (n_days,)
        pet: PET timeseries, shape (n_days,)
        params_batch: Dictionary with parameter arrays, each shape (n_ensemble,)
        initial_state: Initial state (shared across ensemble)
        warmup_days: Warmup period

    Returns:
        Runoff array, shape (n_ensemble, n_days)
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Running sequential ensemble.")
        return _simulate_ensemble_numpy(precip, temp, pet, params_batch, initial_state, warmup_days)

    n_ensemble = len(params_batch[list(params_batch.keys())[0]])

    # Create batched parameters
    def create_params_for_idx(idx):
        return HBVParameters(**{k: v[idx] for k, v in params_batch.items()})

    # Vectorized simulation
    def sim_single(idx):
        params = create_params_for_idx(idx)
        runoff, _ = simulate_jax(precip, temp, pet, params, initial_state, warmup_days)
        return runoff

    # Use vmap for efficient batching
    batch_sim = jax.vmap(sim_single)
    return batch_sim(jnp.arange(n_ensemble))


def _simulate_ensemble_numpy(precip, temp, pet, params_batch, initial_state, warmup_days):
    """NumPy fallback for ensemble simulation."""
    n_ensemble = len(params_batch[list(params_batch.keys())[0]])
    n_days = len(precip)
    results = np.zeros((n_ensemble, n_days))

    for i in range(n_ensemble):
        params_dict = {k: float(v[i]) for k, v in params_batch.items()}
        params = create_params_from_dict(params_dict, use_jax=False)
        results[i], _ = simulate_numpy(precip, temp, pet, params, initial_state, warmup_days)

    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def simulate(
    precip: Any,
    temp: Any,
    pet: Any,
    params: Optional[Dict[str, float]] = None,
    initial_state: Optional[HBVState] = None,
    warmup_days: int = 365,
    use_jax: bool = True,
    timestep_hours: int = 24
) -> Tuple[Any, HBVState]:
    """
    High-level simulation function with automatic backend selection.

    This function automatically handles parameter scaling for sub-daily timesteps.
    Parameters are specified in their standard daily units and scaled internally.

    Args:
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        params: Parameter dictionary in DAILY units (uses defaults if None).
            Parameters will be automatically scaled for sub-daily timesteps.
        initial_state: Initial model state
        warmup_days: Warmup period (in days, converted to timesteps internally)
        use_jax: Whether to prefer JAX backend
        timestep_hours: Model timestep in hours (1-24). Default 24 (daily).
            For hourly simulation, set timestep_hours=1. Forcing data (precip, pet)
            should be provided at the same resolution.

    Returns:
        Tuple of (runoff_timeseries, final_state)

    Example:
        # Daily simulation (default)
        runoff, state = simulate(precip_daily, temp_daily, pet_daily)

        # Hourly simulation
        runoff, state = simulate(precip_hourly, temp_hourly, pet_hourly, timestep_hours=1)

    Note:
        For sub-daily timesteps, the following parameters are scaled:
        - cfmax, k0, k1, k2, perc: multiplied by (timestep_hours / 24)
        - maxbas: routing weights are distributed across more timesteps
        See scale_params_for_timestep() for details.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Scale parameters for sub-daily timesteps
    scaled_params = scale_params_for_timestep(params, timestep_hours)
    hbv_params = create_params_from_dict(scaled_params, use_jax=(use_jax and HAS_JAX))

    if use_jax and HAS_JAX:
        return simulate_jax(precip, temp, pet, hbv_params, initial_state, warmup_days, timestep_hours)
    else:
        return simulate_numpy(precip, temp, pet, hbv_params, initial_state, warmup_days, timestep_hours)


def jit_simulate(use_gpu: bool = False):
    """
    Get JIT-compiled simulation function.

    Args:
        use_gpu: Whether to use GPU (if available).

    Returns:
        JIT-compiled simulation function if JAX available.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Returning non-JIT function.")
        return simulate

    @jax.jit
    def _jit_simulate(precip, temp, pet, params, initial_state):
        return simulate_jax(precip, temp, pet, params, initial_state)

    return _jit_simulate
