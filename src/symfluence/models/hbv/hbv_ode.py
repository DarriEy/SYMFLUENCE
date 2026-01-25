"""
HBV-96 Model - ODE Formulation with Adjoint Gradients.

This module provides an alternative implementation of the HBV-96 model
using continuous-time ODE formulation solved with diffrax. This enables:

1. Separation of physics (dynamics) from numerics (solver)
2. Adjoint-based gradient computation via Implicit Function Theorem (IFT)
3. O(1) memory gradient computation for long simulations
4. Adaptive time-stepping for stiff dynamics

The physics are defined as dx/dt = f(x, u(t), θ) where:
- x = [snow, snow_water, sm, suz, slz] is the state vector
- u(t) = [precip(t), temp(t), pet(t)] is the interpolated forcing
- θ = HBV parameters

References:
    Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations.
    NeurIPS 2018.

    Lindström, G., et al. (1997). Development and test of the distributed
    HBV-96 hydrological model. Journal of Hydrology.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    lax = None

try:
    import diffrax
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False
    diffrax = None


from .parameters import (
    DEFAULT_PARAMS,
    HBVParameters,
    create_params_from_dict,
    scale_params_for_timestep,
)


__all__ = [
    'HAS_DIFFRAX',
    'HBVODEState',
    'hbv_dynamics',
    'create_forcing_interpolant',
    'simulate_ode',
    'simulate_ode_with_routing',
    'nse_loss_ode',
    'get_nse_gradient_fn_ode',
    'compare_gradients',
    'AdjointMethod',
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HBVODEState(NamedTuple):
    """
    HBV ODE state vector (5D continuous state).

    Note: Routing is handled separately as it's inherently discrete.
    The ODE integrates the hydrological storages, and routing is applied
    as a post-processing convolution.

    Attributes:
        snow: Snow pack storage (mm SWE)
        snow_water: Liquid water in snow pack (mm)
        sm: Soil moisture storage (mm)
        suz: Upper zone storage (mm)
        slz: Lower zone storage (mm)
    """
    snow: Any
    snow_water: Any
    sm: Any
    suz: Any
    slz: Any


class AdjointMethod:
    """Enum-like class for adjoint method selection."""
    RECURSIVE_CHECKPOINT = "recursive_checkpoint"  # Default, good memory/compute tradeoff
    BACKSOLVE = "backsolve"  # O(1) memory but can be numerically unstable
    DIRECT = "direct"  # Standard backprop through solver (high memory)


# =============================================================================
# SMOOTH APPROXIMATIONS (same as discrete model for consistency)
# =============================================================================

def _smooth_sigmoid(x: Any, scale: float = 1.0) -> Any:
    """Smooth step function: ~0 for x << 0, ~1 for x >> 0."""
    return jax.nn.sigmoid(scale * x)


def _smooth_relu(x: Any, scale: float = 10.0) -> Any:
    """Smooth ReLU: softplus approximation of max(x, 0)."""
    return jax.nn.softplus(scale * x) / scale


def _smooth_min(a: Any, b: Any, scale: float = 10.0) -> Any:
    """Smooth minimum: a - softplus(a - b)."""
    return a - _smooth_relu(a - b, scale)


def _smooth_max(a: Any, b: Any, scale: float = 10.0) -> Any:
    """Smooth maximum: b + softplus(a - b)."""
    return b + _smooth_relu(a - b, scale)


# =============================================================================
# FORCING INTERPOLATION
# =============================================================================

def create_forcing_interpolant(
    times: Any,
    precip: Any,
    temp: Any,
    pet: Any,
    method: str = "linear"
) -> Callable:
    """
    Create interpolation function for forcing data.

    Args:
        times: Time array (e.g., 0, 1, 2, ... for daily)
        precip: Precipitation timeseries (mm/day)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/day)
        method: Interpolation method ("linear" or "cubic")

    Returns:
        Function f(t) -> (precip, temp, pet) at continuous time t
    """
    if not HAS_DIFFRAX:
        raise ImportError("diffrax is required for ODE-based simulation")

    # Stack forcing for vectorized interpolation
    forcing_stack = jnp.stack([precip, temp, pet], axis=-1)  # (n_times, 3)

    # Create diffrax interpolation
    if method == "linear":
        interp = diffrax.LinearInterpolation(ts=times, ys=forcing_stack)
    elif method == "cubic":
        interp = diffrax.CubicInterpolation(ts=times, ys=forcing_stack)  # type: ignore[call-arg]
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return interp  # type: ignore[return-value]


# =============================================================================
# HBV DYNAMICS (PURE PHYSICS - THE HEART OF THE ODE FORMULATION)
# =============================================================================

def hbv_dynamics(
    t: float,
    state: Any,
    args: Tuple[Any, HBVParameters, float]
) -> Any:
    """
    HBV-96 dynamics function: dx/dt = f(x, u(t), θ).

    This formulation closely matches the discrete HBV-96 model by using
    the same smooth approximations and ensuring water balance consistency.

    The key insight is that the discrete model's sequential operations
    (snow → soil → response) create a "one-timestep delay" between processes.
    In continuous time, this is approximated by using rate-limited transfers.

    Args:
        t: Current time (continuous, in days)
        state: State vector [snow, snow_water, sm, suz, slz]
        args: Tuple of (forcing_interpolant, params, smoothing_scale)

    Returns:
        State derivatives [d_snow/dt, d_snow_water/dt, d_sm/dt, d_suz/dt, d_slz/dt]
    """
    forcing_interp, params, smoothing = args

    # Unpack state
    snow, snow_water, sm, suz, slz = state

    # Soft positivity constraints (same as discrete model)
    snow = jnp.maximum(snow, 0.0)
    snow_water = jnp.maximum(snow_water, 0.0)
    sm = jnp.maximum(sm, 0.0)
    suz = jnp.maximum(suz, 0.0)
    slz = jnp.maximum(slz, 0.0)

    # Get forcing at time t
    forcing_t = forcing_interp.evaluate(t)
    precip = jnp.maximum(forcing_t[0], 0.0)
    temp = forcing_t[1]
    pet = jnp.maximum(forcing_t[2], 0.0)

    # =========================================================================
    # SNOW ROUTINE - Matching discrete model's smooth approximations
    # =========================================================================

    # Rain/snow partitioning (same sigmoid as discrete)
    rain_frac = jax.nn.sigmoid(smoothing * (temp - params.tt))
    snow_frac = 1.0 - rain_frac

    rainfall = precip * rain_frac
    snowfall = precip * params.sfcf * snow_frac

    # Degree-day melt: cfmax * softplus(T - tt)
    pot_melt = params.cfmax * jax.nn.softplus(smoothing * (temp - params.tt)) / smoothing

    # Actual melt limited by snow (use smooth min)
    melt = pot_melt - jax.nn.softplus(smoothing * (pot_melt - snow)) / smoothing

    # Refreezing: cfr * cfmax * softplus(tt - T)
    pot_refreeze = params.cfr * params.cfmax * jax.nn.softplus(smoothing * (params.tt - temp)) / smoothing
    refreeze = pot_refreeze - jax.nn.softplus(smoothing * (pot_refreeze - snow_water)) / smoothing

    # Water holding capacity
    max_water = params.cwh * snow
    # Outflow when snow_water > max_water
    outflow = jax.nn.softplus(smoothing * (snow_water - max_water)) / smoothing

    # Snow derivatives
    d_snow = snowfall - melt + refreeze
    d_snow_water = melt + rainfall - refreeze - outflow

    # Water input to soil = outflow from snow
    water_input = outflow

    # =========================================================================
    # SOIL ROUTINE - Beta-function recharge
    # =========================================================================

    # Relative soil moisture
    rel_sm = sm / (params.fc + 1e-6)
    rel_sm_capped = rel_sm - jax.nn.softplus(smoothing * (rel_sm - 1.0)) / smoothing

    # Recharge (beta function)
    recharge = water_input * jnp.power(rel_sm_capped + 1e-6, params.beta)

    # ET reduction below LP threshold
    lp_threshold = params.lp * params.fc
    et_factor = sm / (lp_threshold + 1e-6)
    et_factor = et_factor - jax.nn.softplus(smoothing * (et_factor - 1.0)) / smoothing

    actual_et = pet * et_factor
    # Limit ET to available SM
    actual_et = actual_et - jax.nn.softplus(smoothing * (actual_et - sm)) / smoothing

    # Soil moisture derivative
    d_sm = water_input - recharge - actual_et

    # =========================================================================
    # RESPONSE ROUTINE - Two-box groundwater
    # =========================================================================

    # Percolation (limited by both perc rate and available storage)
    perc = params.perc - jax.nn.softplus(smoothing * (params.perc - suz)) / smoothing

    # Q0: fast flow from upper zone (threshold at uzl)
    suz_above_uzl = jax.nn.softplus(smoothing * (suz - params.uzl)) / smoothing
    q0 = params.k0 * suz_above_uzl

    # Q1: interflow (linear)
    q1 = params.k1 * suz

    # Q2: baseflow (linear)
    q2 = params.k2 * slz

    # Groundwater derivatives
    d_suz = recharge - perc - q0 - q1
    d_slz = perc - q2

    return jnp.array([d_snow, d_snow_water, d_sm, d_suz, d_slz])


def hbv_runoff_rate(
    t: float,
    state: Any,
    args: Tuple[Any, HBVParameters, float]
) -> Any:
    """
    Compute instantaneous runoff rate (for output, not part of ODE state).

    Args:
        t: Current time
        state: State vector
        args: (forcing_interpolant, params, smoothing)

    Returns:
        Total runoff rate (mm/day) = Q0 + Q1 + Q2
    """
    forcing_interp, params, smoothing = args
    snow, snow_water, sm, suz, slz = state

    # Ensure non-negative (matching dynamics)
    suz = jnp.maximum(suz, 0.0)
    slz = jnp.maximum(slz, 0.0)

    # Runoff components (matching dynamics formulation)
    suz_above_uzl = jax.nn.softplus(smoothing * (suz - params.uzl)) / smoothing
    q0 = params.k0 * suz_above_uzl
    q1 = params.k1 * suz
    q2 = params.k2 * slz

    return q0 + q1 + q2


# =============================================================================
# ODE SOLVER
# =============================================================================

def simulate_ode(
    precip: Any,
    temp: Any,
    pet: Any,
    params: HBVParameters,
    t_span: Tuple[float, float] = None,
    dt: float = 1.0,
    initial_state: Optional[HBVODEState] = None,
    adjoint_method: str = AdjointMethod.RECURSIVE_CHECKPOINT,
    solver: str = "tsit5",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_steps: int = 16384,
    smoothing: float = 15.0,
) -> Tuple[Any, Any, Any]:
    """
    Simulate HBV model using ODE solver with adjoint gradients.

    This function solves the HBV dynamics as a continuous ODE and returns
    state trajectories and runoff. Gradients are computed using the adjoint
    method rather than backpropagation through the solver.

    Args:
        precip: Precipitation timeseries (mm/day), shape (n_days,)
        temp: Temperature timeseries (°C), shape (n_days,)
        pet: PET timeseries (mm/day), shape (n_days,)
        params: HBV parameters (HBVParameters namedtuple)
        t_span: Time span (t0, t1). Defaults to (0, n_days).
        dt: Output time step (days). Default 1.0 for daily output.
        initial_state: Initial state. Defaults to standard values.
        adjoint_method: Method for gradient computation.
        solver: ODE solver ("tsit5", "dopri5", "euler", "heun").
        rtol: Relative tolerance for adaptive solvers.
        atol: Absolute tolerance for adaptive solvers.
        max_steps: Maximum solver steps.
        smoothing: Smoothing scale for threshold functions.

    Returns:
        Tuple of (times, states, runoff):
            - times: Output time points, shape (n_output,)
            - states: State trajectories, shape (n_output, 5)
            - runoff: Runoff at each output time, shape (n_output,)
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for ODE-based simulation")
    if not HAS_DIFFRAX:
        raise ImportError("diffrax is required for ODE-based simulation. Install with: pip install diffrax")

    n_days = len(precip)

    # Default time span
    if t_span is None:
        t_span = (0.0, float(n_days - 1))

    # Create time array for forcing interpolation
    forcing_times = jnp.arange(n_days, dtype=jnp.float32)

    # Create forcing interpolant
    forcing_interp = create_forcing_interpolant(
        forcing_times,
        jnp.array(precip),
        jnp.array(temp),
        jnp.array(pet),
        method="linear"
    )

    # Initial state
    if initial_state is None:
        y0 = jnp.array([0.0, 0.0, 150.0, 10.0, 10.0])  # [snow, snow_water, sm, suz, slz]
    else:
        y0 = jnp.array([
            initial_state.snow,
            initial_state.snow_water,
            initial_state.sm,
            initial_state.suz,
            initial_state.slz
        ])

    # Pack arguments
    args = (forcing_interp, params, smoothing)

    # Create ODE term
    term = diffrax.ODETerm(hbv_dynamics)

    # Select solver
    solver_map = {
        "tsit5": diffrax.Tsit5(),
        "dopri5": diffrax.Dopri5(),
        "euler": diffrax.Euler(),
        "heun": diffrax.Heun(),
        "bosh3": diffrax.Bosh3(),
    }

    if solver not in solver_map:
        raise ValueError(f"Unknown solver: {solver}. Available: {list(solver_map.keys())}")

    ode_solver = solver_map[solver]

    # Select adjoint method
    if adjoint_method == AdjointMethod.RECURSIVE_CHECKPOINT:
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif adjoint_method == AdjointMethod.BACKSOLVE:
        adjoint = diffrax.BacksolveAdjoint()
    elif adjoint_method == AdjointMethod.DIRECT:
        adjoint = diffrax.DirectAdjoint()
    else:
        raise ValueError(f"Unknown adjoint method: {adjoint_method}")

    # Step size controller
    if solver in ["euler", "heun"]:
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = dt / 10.0  # Finer steps for fixed-step solvers
    else:
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        dt0 = dt / 2.0  # Initial step size

    # Output times
    n_output = int((t_span[1] - t_span[0]) / dt) + 1
    saveat = diffrax.SaveAt(ts=jnp.linspace(t_span[0], t_span[1], n_output))

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        ode_solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    times = solution.ts
    states = solution.ys  # Shape: (n_output, 5)

    # Compute runoff at each output time
    def compute_runoff_at_t(t_state):
        t, state = t_state
        return hbv_runoff_rate(t, state, args)

    runoff = jax.vmap(lambda ts: compute_runoff_at_t(ts))(
        (times, states)
    )

    return times, states, runoff


def simulate_ode_with_routing(
    precip: Any,
    temp: Any,
    pet: Any,
    params: HBVParameters,
    warmup_days: int = 365,
    timestep_hours: int = 24,
    adjoint_method: str = AdjointMethod.RECURSIVE_CHECKPOINT,
    solver: str = "tsit5",
    smoothing: float = 15.0,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Simulate HBV with ODE solver and apply triangular routing.

    This is the high-level function that matches the interface of the
    discrete simulate() function, including triangular routing.

    Args:
        precip: Precipitation timeseries (mm/timestep)
        temp: Temperature timeseries (°C)
        pet: PET timeseries (mm/timestep)
        params: HBV parameters
        warmup_days: Warmup period (output still includes warmup)
        timestep_hours: Model timestep in hours
        adjoint_method: Adjoint method for gradients
        solver: ODE solver to use
        smoothing: Smoothing scale
        **kwargs: Additional arguments passed to simulate_ode

    Returns:
        Tuple of (routed_runoff, final_state)
    """
    if not HAS_JAX or not HAS_DIFFRAX:
        raise ImportError("JAX and diffrax required for ODE simulation")

    n_timesteps = len(precip)
    dt = timestep_hours / 24.0  # Convert to days

    # Solve ODE
    times, states, runoff = simulate_ode(
        precip=precip,
        temp=temp,
        pet=pet,
        params=params,
        t_span=(0.0, float(n_timesteps - 1) * dt),
        dt=dt,
        adjoint_method=adjoint_method,
        solver=solver,
        smoothing=smoothing,
        **kwargs
    )

    # Apply triangular routing (discrete convolution)
    routed_runoff = apply_triangular_routing(
        runoff, params.maxbas, timestep_hours
    )

    # Extract final state
    final_state = HBVODEState(
        snow=states[-1, 0],
        snow_water=states[-1, 1],
        sm=states[-1, 2],
        suz=states[-1, 3],
        slz=states[-1, 4],
    )

    return routed_runoff, final_state


def apply_triangular_routing(
    runoff: Any,
    maxbas: Any,
    timestep_hours: int = 24,
    max_buffer_length: int = 20
) -> Any:
    """
    Apply triangular routing to runoff timeseries.

    This is a discrete convolution with a triangular kernel,
    applied as post-processing to the ODE-computed runoff.

    Note: This function uses a fixed buffer length to be compatible
    with JAX tracing (maxbas is treated as a soft parameter).

    Args:
        runoff: Unrouted runoff timeseries (mm/timestep)
        maxbas: Base of triangular function (days) - can be traced
        timestep_hours: Timestep in hours
        max_buffer_length: Fixed maximum buffer length for JAX compatibility

    Returns:
        Routed runoff timeseries
    """
    timesteps_per_day = 24.0 / timestep_hours
    maxbas_timesteps = maxbas * timesteps_per_day

    # Use fixed buffer length for JAX compatibility
    buffer_length = max_buffer_length

    timesteps = jnp.arange(1, buffer_length + 1, dtype=jnp.float32)

    # Compute weights (will be near-zero for timesteps > maxbas_timesteps)
    half_maxbas = maxbas_timesteps / 2.0 + 1e-6

    rising = jnp.where(
        timesteps <= half_maxbas,
        timesteps / half_maxbas,
        0.0
    )
    falling = jnp.where(
        (timesteps > half_maxbas) & (timesteps <= maxbas_timesteps),
        (maxbas_timesteps - timesteps) / half_maxbas,
        0.0
    )
    weights = rising + falling
    weights = weights / (jnp.sum(weights) + 1e-10)

    # Reverse weights for convolution (we want causal filter)
    weights_rev = weights[::-1]

    # Apply convolution using lax.conv for JAX compatibility
    # Ensure consistent dtype
    runoff = jnp.asarray(runoff, dtype=jnp.float32)
    weights_rev = jnp.asarray(weights_rev, dtype=jnp.float32)

    # Reshape for 1D convolution: (batch, length, channels)
    runoff_reshaped = runoff[None, :, None]  # (1, T, 1)
    weights_reshaped = weights_rev[:, None, None]  # (K, 1, 1)

    # Use lax.conv_general_dilated for 1D convolution
    routed_reshaped = lax.conv_general_dilated(
        runoff_reshaped,
        weights_reshaped,
        window_strides=(1,),
        padding=[(buffer_length - 1, 0)],  # Causal padding
        dimension_numbers=('NTC', 'TIO', 'NTC'),
    )

    routed = routed_reshaped[0, :, 0]

    return routed


# =============================================================================
# LOSS FUNCTIONS (ODE VERSION)
# =============================================================================

def nse_loss_ode(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24,
    adjoint_method: str = AdjointMethod.RECURSIVE_CHECKPOINT,
    solver: str = "tsit5",
    smoothing: float = 15.0,
) -> float:
    """
    Compute NSE loss using ODE-based simulation.

    This is equivalent to nse_loss() but uses the continuous ODE
    formulation with adjoint gradients.

    Args:
        params_dict: Parameter dictionary (daily units)
        precip, temp, pet: Forcing timeseries
        obs: Observed streamflow
        warmup_days: Warmup period to exclude
        timestep_hours: Model timestep
        adjoint_method: Gradient computation method
        solver: ODE solver
        smoothing: Smoothing scale

    Returns:
        Negative NSE (for minimization)
    """
    # Scale parameters for timestep
    scaled_params = scale_params_for_timestep(params_dict, timestep_hours)
    params = create_params_from_dict(scaled_params, use_jax=True)

    # Simulate
    runoff, _ = simulate_ode_with_routing(
        precip=precip,
        temp=temp,
        pet=pet,
        params=params,
        warmup_days=warmup_days,
        timestep_hours=timestep_hours,
        adjoint_method=adjoint_method,
        solver=solver,
        smoothing=smoothing,
    )

    # Calculate NSE (excluding warmup)
    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = warmup_days * timesteps_per_day

    sim_eval = runoff[warmup_timesteps:]
    obs_eval = obs[warmup_timesteps:]

    ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
    ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
    nse = 1.0 - ss_res / (ss_tot + 1e-10)

    return -nse  # type: ignore[return-value]  # Negative for minimization


def _nse_loss_from_array(
    params_array: Any,
    param_names: list,
    base_params: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int,
    timestep_hours: int,
    adjoint_method: str,
    solver: str,
    smoothing: float,
) -> Any:
    """
    Internal NSE loss function that takes JAX array parameters.

    This version is compatible with JAX autodiff by avoiding float() conversions.
    """
    # Create params dict with JAX-traced values (no float conversion)
    params_dict = base_params.copy()
    for i, name in enumerate(param_names):
        params_dict[name] = params_array[i]

    # Scale parameters for timestep (must work with traced values)
    from .parameters import FLUX_RATE_PARAMS, RECESSION_PARAMS

    scaled_params = params_dict.copy()
    if timestep_hours != 24:
        scale_factor = timestep_hours / 24.0
        for param in FLUX_RATE_PARAMS:
            if param in scaled_params:
                scaled_params[param] = scaled_params[param] * scale_factor
        for param in RECESSION_PARAMS:
            if param in scaled_params:
                k_val = scaled_params[param]
                # Clamp to valid range using smooth approximation
                k_val = jnp.clip(k_val, 1e-6, 0.9999)
                scaled_params[param] = 1.0 - jnp.power(1.0 - k_val, scale_factor)

    # Create HBVParameters (using JAX arrays directly)
    params = HBVParameters(
        tt=jnp.asarray(scaled_params.get('tt', 0.0)),
        cfmax=jnp.asarray(scaled_params.get('cfmax', 3.5)),
        sfcf=jnp.asarray(scaled_params.get('sfcf', 0.9)),
        cfr=jnp.asarray(scaled_params.get('cfr', 0.05)),
        cwh=jnp.asarray(scaled_params.get('cwh', 0.1)),
        fc=jnp.asarray(scaled_params.get('fc', 250.0)),
        lp=jnp.asarray(scaled_params.get('lp', 0.7)),
        beta=jnp.asarray(scaled_params.get('beta', 2.5)),
        k0=jnp.asarray(scaled_params.get('k0', 0.3)),
        k1=jnp.asarray(scaled_params.get('k1', 0.1)),
        k2=jnp.asarray(scaled_params.get('k2', 0.01)),
        uzl=jnp.asarray(scaled_params.get('uzl', 30.0)),
        perc=jnp.asarray(scaled_params.get('perc', 2.5)),
        maxbas=jnp.asarray(scaled_params.get('maxbas', 2.5)),
        smoothing=jnp.asarray(smoothing),
        smoothing_enabled=jnp.asarray(True),
    )

    # Simulate using ODE
    n_timesteps = len(precip)
    dt = timestep_hours / 24.0

    # Create forcing interpolant
    forcing_times = jnp.arange(n_timesteps, dtype=jnp.float32)
    forcing_interp = create_forcing_interpolant(
        forcing_times,
        jnp.asarray(precip),
        jnp.asarray(temp),
        jnp.asarray(pet),
        method="linear"
    )

    # Initial state
    y0 = jnp.array([0.0, 0.0, 150.0, 10.0, 10.0])

    # Pack arguments
    args = (forcing_interp, params, smoothing)

    # Create ODE term
    term = diffrax.ODETerm(hbv_dynamics)

    # Solver
    ode_solver = diffrax.Tsit5() if solver == "tsit5" else diffrax.Dopri5()

    # Adjoint
    if adjoint_method == AdjointMethod.RECURSIVE_CHECKPOINT:
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif adjoint_method == AdjointMethod.BACKSOLVE:
        adjoint = diffrax.BacksolveAdjoint()
    else:
        adjoint = diffrax.DirectAdjoint()

    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    t_span = (0.0, float(n_timesteps - 1) * dt)
    n_output = n_timesteps
    saveat = diffrax.SaveAt(ts=jnp.linspace(t_span[0], t_span[1], n_output))

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        ode_solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt / 2.0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
        max_steps=16384,
    )

    states = solution.ys

    # Compute runoff at each time
    def compute_runoff(state):
        suz, slz = state[3], state[4]
        suz_pos = jax.nn.softplus(suz - 1e-6) + 1e-6
        slz_pos = jax.nn.softplus(slz - 1e-6) + 1e-6
        suz_above_uzl = jax.nn.softplus(smoothing * (suz_pos - params.uzl)) / smoothing
        q0 = params.k0 * suz_above_uzl
        q1 = params.k1 * suz_pos
        q2 = params.k2 * slz_pos
        return q0 + q1 + q2

    runoff = jax.vmap(compute_runoff)(states)

    # Apply routing
    routed = apply_triangular_routing(runoff, params.maxbas, timestep_hours)

    # Calculate NSE
    timesteps_per_day = 24 // timestep_hours
    warmup_timesteps = warmup_days * timesteps_per_day

    sim_eval = routed[warmup_timesteps:]
    obs_eval = obs[warmup_timesteps:]

    ss_res = jnp.sum((sim_eval - obs_eval) ** 2)
    ss_tot = jnp.sum((obs_eval - jnp.mean(obs_eval)) ** 2)
    nse = 1.0 - ss_res / (ss_tot + 1e-10)

    return -nse  # Negative for minimization


def get_nse_gradient_fn_ode(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24,
    adjoint_method: str = AdjointMethod.RECURSIVE_CHECKPOINT,
    solver: str = "tsit5",
    smoothing: float = 15.0,
) -> Callable:
    """
    Get gradient function for NSE loss (ODE version).

    Returns a function that computes gradients using the adjoint method.

    Args:
        precip, temp, pet: Forcing timeseries (fixed)
        obs: Observed streamflow (fixed)
        warmup_days: Warmup period
        timestep_hours: Model timestep
        adjoint_method: Adjoint method for gradients
        solver: ODE solver
        smoothing: Smoothing scale

    Returns:
        Gradient function: (params_array, param_names) -> gradient_array
    """
    if not HAS_JAX:
        raise ImportError("JAX required for gradient computation")
    if not HAS_DIFFRAX:
        raise ImportError("diffrax required for ODE gradient computation")


    # Convert forcing to JAX arrays
    precip_jax = jnp.asarray(precip)
    temp_jax = jnp.asarray(temp)
    pet_jax = jnp.asarray(pet)
    obs_jax = jnp.asarray(obs)

    base_params = DEFAULT_PARAMS.copy()
    base_params['smoothing'] = smoothing
    base_params['smoothing_enabled'] = True

    def loss_fn(params_array, param_names):
        return _nse_loss_from_array(
            params_array, param_names, base_params,
            precip_jax, temp_jax, pet_jax, obs_jax,
            warmup_days, timestep_hours, adjoint_method, solver, smoothing
        )

    return jax.grad(loss_fn)


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_gradients(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24,
    param_names: Optional[list] = None,
    smoothing: float = 15.0,
) -> Dict[str, Any]:
    """
    Compare gradients between discrete (lax.scan) and ODE (adjoint) methods.

    This function computes gradients using both methods and returns
    a comparison including relative differences and timing.

    Args:
        params_dict: Parameter dictionary
        precip, temp, pet: Forcing timeseries
        obs: Observed streamflow
        warmup_days: Warmup period
        timestep_hours: Model timestep
        param_names: Parameters to compute gradients for (defaults to all)
        smoothing: Smoothing scale

    Returns:
        Dictionary with:
            - 'discrete_grads': Gradients from lax.scan method
            - 'ode_grads': Gradients from ODE adjoint method
            - 'relative_diff': Relative difference between gradients
            - 'discrete_time': Time for discrete gradient computation
            - 'ode_time': Time for ODE gradient computation
            - 'loss_discrete': Loss value from discrete method
            - 'loss_ode': Loss value from ODE method
    """
    import time

    from .losses import nse_loss, get_nse_gradient_fn

    if param_names is None:
        param_names = ['tt', 'cfmax', 'sfcf', 'cfr', 'cwh', 'fc', 'lp',
                       'beta', 'k0', 'k1', 'k2', 'uzl', 'perc', 'maxbas']

    # Ensure smoothing is enabled for fair comparison
    params_dict = params_dict.copy()
    params_dict['smoothing'] = smoothing
    params_dict['smoothing_enabled'] = True

    params_array = jnp.array([params_dict[p] for p in param_names])

    # =========================================================================
    # Discrete method (lax.scan with BPTT)
    # =========================================================================

    grad_fn_discrete = get_nse_gradient_fn(
        precip, temp, pet, obs, warmup_days, timestep_hours
    )
    assert grad_fn_discrete is not None, "JAX required for gradient comparison"

    # Warmup JIT
    _ = grad_fn_discrete(params_array, param_names)

    # Timed run
    start = time.perf_counter()
    grads_discrete = grad_fn_discrete(params_array, param_names)
    discrete_time = time.perf_counter() - start

    loss_discrete = nse_loss(
        params_dict, precip, temp, pet, obs,
        warmup_days, use_jax=True, timestep_hours=timestep_hours
    )

    # =========================================================================
    # ODE method (adjoint)
    # =========================================================================

    grad_fn_ode = get_nse_gradient_fn_ode(
        precip, temp, pet, obs, warmup_days, timestep_hours,
        adjoint_method=AdjointMethod.RECURSIVE_CHECKPOINT,
        smoothing=smoothing
    )

    # Warmup JIT
    _ = grad_fn_ode(params_array, param_names)

    # Timed run
    start = time.perf_counter()
    grads_ode = grad_fn_ode(params_array, param_names)
    ode_time = time.perf_counter() - start

    loss_ode = nse_loss_ode(
        params_dict, precip, temp, pet, obs,
        warmup_days, timestep_hours, smoothing=smoothing
    )

    # =========================================================================
    # Comparison
    # =========================================================================

    grads_discrete_dict = dict(zip(param_names, grads_discrete))
    grads_ode_dict = dict(zip(param_names, grads_ode))

    # Relative differences
    rel_diff = {}
    for p in param_names:
        g_d = float(grads_discrete_dict[p])
        g_o = float(grads_ode_dict[p])
        if abs(g_d) > 1e-10:
            rel_diff[p] = abs(g_o - g_d) / abs(g_d)
        else:
            rel_diff[p] = abs(g_o - g_d)

    return {
        'discrete_grads': grads_discrete_dict,
        'ode_grads': grads_ode_dict,
        'relative_diff': rel_diff,
        'discrete_time': discrete_time,
        'ode_time': ode_time,
        'loss_discrete': float(loss_discrete),
        'loss_ode': float(loss_ode),
    }


def verify_adjoint_gradients(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    timestep_hours: int = 24,
    param_names: Optional[list] = None,
    smoothing: float = 15.0,
) -> Dict[str, Any]:
    """
    Verify adjoint gradients against finite differences.

    This function checks that the adjoint method is computing correct
    gradients by comparing against numerical finite difference approximations.

    Args:
        params_dict: Parameter dictionary
        precip, temp, pet: Forcing timeseries
        obs: Observed streamflow
        warmup_days: Warmup period
        timestep_hours: Model timestep
        param_names: Parameters to check (defaults to subset)
        smoothing: Smoothing scale

    Returns:
        Dictionary with:
            - 'adjoint_grads': Gradients from adjoint method
            - 'fd_grads': Gradients from finite differences
            - 'relative_error': Relative error between methods
            - 'adjoint_correct': Boolean indicating if adjoint is correct
    """
    if param_names is None:
        # Use subset for speed
        param_names = ['cfmax', 'fc', 'k1', 'k2', 'beta']

    params_dict = params_dict.copy()
    params_dict['smoothing'] = smoothing
    params_dict['smoothing_enabled'] = True

    # Compute adjoint gradients
    grad_fn = get_nse_gradient_fn_ode(
        precip, temp, pet, obs, warmup_days, timestep_hours,
        adjoint_method=AdjointMethod.RECURSIVE_CHECKPOINT,
        smoothing=smoothing
    )

    params_array = jnp.array([params_dict[p] for p in param_names])
    adjoint_grads = grad_fn(params_array, param_names)
    adjoint_grads_dict = dict(zip(param_names, [float(g) for g in adjoint_grads]))

    # Compute finite difference gradients with appropriate step sizes
    fd_grads_dict = {}
    # base_loss computed for reference but central difference uses loss_plus/loss_minus
    _ = float(nse_loss_ode(
        params_dict, precip, temp, pet, obs,
        warmup_days, timestep_hours, smoothing=smoothing
    ))

    for param in param_names:
        # Use relative step size (0.1% of parameter value, min 0.001)
        base_val = params_dict[param]
        eps = max(0.001 * abs(base_val), 0.001)

        # Central difference for better accuracy
        params_plus = params_dict.copy()
        params_minus = params_dict.copy()
        params_plus[param] = base_val + eps
        params_minus[param] = base_val - eps

        loss_plus = float(nse_loss_ode(
            params_plus, precip, temp, pet, obs,
            warmup_days, timestep_hours, smoothing=smoothing
        ))
        loss_minus = float(nse_loss_ode(
            params_minus, precip, temp, pet, obs,
            warmup_days, timestep_hours, smoothing=smoothing
        ))

        fd_grads_dict[param] = (loss_plus - loss_minus) / (2 * eps)

    # Compute relative errors
    rel_errors = {}
    for param in param_names:
        adj = adjoint_grads_dict[param]
        fd = fd_grads_dict[param]
        # Use symmetric relative error
        denom = (abs(adj) + abs(fd)) / 2 + 1e-8
        rel_errors[param] = abs(adj - fd) / denom

    # Check if adjoint is correct (within tolerance)
    max_error = max(rel_errors.values())
    adjoint_correct = max_error < 0.15  # 15% tolerance for ODE gradients

    return {
        'adjoint_grads': adjoint_grads_dict,
        'fd_grads': fd_grads_dict,
        'relative_error': rel_errors,
        'max_error': max_error,
        'adjoint_correct': adjoint_correct,
    }


def compare_simulations(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    timestep_hours: int = 24,
    smoothing: float = 15.0,
) -> Dict[str, Any]:
    """
    Compare simulation outputs between discrete and ODE methods.

    Args:
        params_dict: Parameter dictionary
        precip, temp, pet: Forcing timeseries
        timestep_hours: Model timestep
        smoothing: Smoothing scale

    Returns:
        Dictionary with:
            - 'discrete_runoff': Runoff from discrete method
            - 'ode_runoff': Runoff from ODE method
            - 'rmse': Root mean square error between outputs
            - 'correlation': Correlation between outputs
            - 'discrete_time': Discrete simulation time
            - 'ode_time': ODE simulation time
    """
    import time

    from .model import simulate

    # Ensure smoothing is enabled
    params_dict = params_dict.copy()
    params_dict['smoothing'] = smoothing
    params_dict['smoothing_enabled'] = True

    scaled_params = scale_params_for_timestep(params_dict, timestep_hours)
    params = create_params_from_dict(scaled_params, use_jax=True)

    # Discrete simulation
    start = time.perf_counter()
    runoff_discrete, _ = simulate(
        precip, temp, pet, params_dict,
        use_jax=True, timestep_hours=timestep_hours
    )
    discrete_time = time.perf_counter() - start

    # ODE simulation
    start = time.perf_counter()
    runoff_ode, _ = simulate_ode_with_routing(
        precip, temp, pet, params,
        timestep_hours=timestep_hours,
        smoothing=smoothing
    )
    ode_time = time.perf_counter() - start

    # Comparison metrics
    rmse = float(jnp.sqrt(jnp.mean((runoff_discrete - runoff_ode) ** 2)))
    correlation = float(jnp.corrcoef(runoff_discrete, runoff_ode)[0, 1])

    return {
        'discrete_runoff': np.array(runoff_discrete),
        'ode_runoff': np.array(runoff_ode),
        'rmse': rmse,
        'correlation': correlation,
        'discrete_time': discrete_time,
        'ode_time': ode_time,
    }
