"""
Distributed HBV Model with Graph-Based Routing.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.

Implements a semi-distributed/distributed HBV model where:
- Nodes (GRUs) run independent HBV-96 simulations
- Edges (reaches) route flow using Muskingum-Cunge
- The entire system is end-to-end differentiable via JAX

This enables gradient-based calibration of:
- HBV parameters (uniform or per-GRU)
- Routing parameters (K, x or channel properties)

Key Features:
- Parallel GRU simulation with vmap
- Sequential routing through network topology
- Support for uniform, per-GRU, or regionalized parameters
- End-to-end differentiable for outlet-based calibration
- Adaptive sub-timestep routing for stability

Example:
    >>> from symfluence.models.hbv.distributed import DistributedHBV
    >>> from symfluence.models.hbv.network import create_synthetic_network
    >>>
    >>> # Create network and model
    >>> network = create_synthetic_network(n_nodes=5, topology='fishbone')
    >>> model = DistributedHBV(network)
    >>>
    >>> # Run simulation
    >>> outlet_flow, states = model.simulate(precip, temp, pet, params)
    >>>
    >>> # Compute gradients for calibration
    >>> grad_fn = model.get_gradient_function(obs_flow)
    >>> gradients = grad_fn(params)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, NamedTuple, Any, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizers import CalibrationResult
from pathlib import Path
import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax, vmap, grad, jit
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None
    lax = None
    vmap = None
    grad = None
    jit = None

from .model import (
    HBVState, PARAM_BOUNDS, DEFAULT_PARAMS, HBVParameters,
    create_params_from_dict, create_initial_state, scale_params_for_timestep,
    simulate_jax, simulate_numpy
)
from .network import RiverNetwork, NetworkBuilder
from .routing import (
    RoutingParams, RoutingState,
    compute_muskingum_params, create_initial_routing_state,
    route_network_step_jax, route_network_step_numpy,
    runoff_mm_to_cms
)
from .regionalization import forward_transfer_function, TransferLayer

def _slice_hbv_params(params: HBVParameters, idx: int) -> HBVParameters:
    """Slice batched HBVParameters for a single GRU."""
    # Handle both JAX and NumPy arrays
    def get_val(arr):
        val = arr[idx]
        return float(val) if hasattr(val, 'item') else val

    return HBVParameters(
        tt=get_val(params.tt),
        cfmax=get_val(params.cfmax),
        sfcf=get_val(params.sfcf),
        cfr=get_val(params.cfr),
        cwh=get_val(params.cwh),
        fc=get_val(params.fc),
        lp=get_val(params.lp),
        beta=get_val(params.beta),
        k0=get_val(params.k0),
        k1=get_val(params.k1),
        k2=get_val(params.k2),
        uzl=get_val(params.uzl),
        perc=get_val(params.perc),
        maxbas=get_val(params.maxbas),
        smoothing=params.smoothing[idx] if hasattr(params.smoothing, '__len__') and len(np.shape(params.smoothing)) > 0 else params.smoothing,
        smoothing_enabled=params.smoothing_enabled[idx] if hasattr(params.smoothing_enabled, '__len__') and len(np.shape(params.smoothing_enabled)) > 0 else params.smoothing_enabled
    )


class DistributedHBVState(NamedTuple):
    """
    State for distributed HBV model.

    Attributes:
        hbv_states: HBV states for each GRU [n_nodes]
        routing_state: Muskingum routing state
        prev_runoff: Previous timestep runoff for routing [n_nodes]
    """
    hbv_states: List[HBVState]  # Or batched arrays
    routing_state: RoutingState
    prev_runoff: Any  # jnp.ndarray [n_nodes]


class DistributedHBVParams(NamedTuple):
    """
    Parameters for distributed HBV model.

    Supports three parameter modes:
    1. Uniform: Same HBV params for all GRUs (hbv_params is HBVParameters)
    2. Per-GRU: Different params per GRU (hbv_params is list or batched)
    3. Regionalized: Params derived from GRU attributes (via transfer function)

    Attributes:
        hbv_params: HBV parameters (uniform or per-GRU)
        routing_params: Muskingum-Cunge routing parameters
        param_mode: 'uniform', 'per_gru', or 'regionalized'
    """
    hbv_params: Any  # HBVParameters or List[HBVParameters] or batched arrays
    routing_params: RoutingParams
    param_mode: str


class DistributedHBV:
    """
    Distributed HBV model with Muskingum-Cunge routing.

    This class orchestrates:
    1. Running HBV for each GRU (parallel with vmap)
    2. Converting runoff to discharge
    3. Routing through the river network
    4. Computing outlet discharge

    All operations are JAX-compatible for end-to-end differentiation.
    """

    def __init__(
        self,
        network: RiverNetwork,
        param_mode: str = 'uniform',
        timestep_hours: int = 24,
        warmup_days: int = 365,
        use_jax: bool = True,
        reference_Q: Optional[float] = None
    ):
        """
        Initialize distributed HBV model.

        Args:
            network: RiverNetwork defining topology and reach properties
            param_mode: Parameter mode ('uniform', 'per_gru', 'regionalized')
            timestep_hours: Model timestep in hours
            warmup_days: Number of warmup days
            use_jax: Whether to use JAX backend
            reference_Q: Reference discharge for routing parameter estimation (m³/s).
                        If None, estimated from catchment area.
        """
        self.network = network
        self.param_mode = param_mode
        self.timestep_hours = timestep_hours
        self.warmup_days = warmup_days
        self.use_jax = use_jax and HAS_JAX
        self.dt_seconds = timestep_hours * 3600.0

        # Estimate reference discharge if not provided
        if reference_Q is None:
            # Rough estimate: 0.5 mm/day runoff from total area
            total_area = float(np.sum(np.asarray(network.node_areas)))
            reference_Q = 0.5 / 1000.0 * total_area / 86400.0  # m³/s
        self.reference_Q = reference_Q

        # Pre-compute routing parameters from channel properties
        self._default_routing_params = self._compute_routing_params()

    def _compute_routing_params(self, reference_Q: Optional[float] = None) -> RoutingParams:
        """Compute routing parameters from network properties."""
        ref_Q = reference_Q or self.reference_Q

        if self.network.n_edges == 0:
            # Single node network, no routing needed
            if self.use_jax:
                return RoutingParams(
                    K=jnp.array([3600.0]),
                    x=jnp.array([0.2]),
                    n_substeps=jnp.array([1])
                )
            else:
                return RoutingParams(
                    K=np.array([3600.0]),
                    x=np.array([0.2]),
                    n_substeps=np.array([1])
                )

        return compute_muskingum_params(
            length=self.network.edge_lengths,
            slope=self.network.edge_slopes,
            width=self.network.edge_widths,
            mannings_n=self.network.edge_mannings,
            reference_Q=ref_Q,
            dt=self.dt_seconds
        )

    def create_initial_state(
        self,
        initial_snow: float = 0.0,
        initial_sm: float = 150.0,
        initial_suz: float = 10.0,
        initial_slz: float = 10.0,
        initial_Q: float = 0.0
    ) -> DistributedHBVState:
        """
        Create initial state for all GRUs and routing.

        Args:
            initial_snow: Initial snow storage (mm)
            initial_sm: Initial soil moisture (mm)
            initial_suz: Initial upper zone storage (mm)
            initial_slz: Initial lower zone storage (mm)
            initial_Q: Initial discharge in reaches (m³/s)

        Returns:
            DistributedHBVState with initialized values
        """
        n_nodes = self.network.n_nodes
        n_edges = self.network.n_edges

        # Create HBV states for each GRU
        hbv_states = [
            create_initial_state(
                initial_snow=initial_snow,
                initial_sm=initial_sm,
                initial_suz=initial_suz,
                initial_slz=initial_slz,
                use_jax=self.use_jax,
                timestep_hours=self.timestep_hours
            )
            for _ in range(n_nodes)
        ]

        # Create routing state
        routing_state = create_initial_routing_state(
            max(n_edges, 1),
            initial_Q=initial_Q,
            use_jax=self.use_jax
        )

        # Initial runoff
        if self.use_jax:
            prev_runoff = jnp.zeros(n_nodes)
        else:
            prev_runoff = np.zeros(n_nodes)

        return DistributedHBVState(
            hbv_states=hbv_states,
            routing_state=routing_state,
            prev_runoff=prev_runoff
        )

    def create_params(
        self,
        hbv_params: Optional[Dict[str, float]] = None,
        routing_params: Optional[RoutingParams] = None,
        per_gru_params: Optional[List[Dict[str, float]]] = None,
        attributes: Optional[Any] = None,
        transfer_weights: Optional[List[TransferLayer]] = None,
        param_mode: Optional[str] = None
    ) -> DistributedHBVParams:
        """
        Create model parameters.

        Args:
            hbv_params: Uniform HBV parameters (dict). Used if param_mode='uniform'.
            routing_params: Routing parameters. If None, computed from channel properties.
            per_gru_params: Per-GRU HBV parameters. Used if param_mode='per_gru'.
            attributes: Catchment attributes [n_nodes, n_features]. Used if param_mode='regionalized'.
            transfer_weights: Weights for transfer function. Used if param_mode='regionalized'.
            param_mode: Override self.param_mode if provided.

        Returns:
            DistributedHBVParams
        """
        mode = param_mode or self.param_mode

        # HBV parameters
        if mode == 'uniform':
            params_dict = hbv_params or DEFAULT_PARAMS.copy()
            scaled_params = scale_params_for_timestep(params_dict, self.timestep_hours)
            hbv = create_params_from_dict(scaled_params, use_jax=self.use_jax)

        elif mode == 'per_gru':
            if per_gru_params is None:
                # Use defaults for all GRUs
                per_gru_params = [DEFAULT_PARAMS.copy() for _ in range(self.network.n_nodes)]

            hbv = [
                create_params_from_dict(
                    scale_params_for_timestep(p, self.timestep_hours),
                    use_jax=self.use_jax
                )
                for p in per_gru_params
            ]

        elif mode == 'regionalized':
            if attributes is None or transfer_weights is None:
                raise ValueError("attributes and transfer_weights required for regionalized mode")

            # Generate parameters from attributes using transfer function
            # The forward_transfer_function returns HBVParameters where fields are arrays [n_nodes]
            # We assume attributes are already normalized if needed
            hbv = forward_transfer_function(
                transfer_weights,
                attributes,
                PARAM_BOUNDS
            )

            # Scale rate parameters for timestep
            # Since hbv is HBVParameters with arrays, we need to scale the arrays
            if self.timestep_hours != 24:
                scale_factor = self.timestep_hours / 24.0
                hbv = HBVParameters(
                    tt=hbv.tt,
                    cfmax=hbv.cfmax * scale_factor,
                    sfcf=hbv.sfcf,
                    cfr=hbv.cfr,
                    cwh=hbv.cwh,
                    fc=hbv.fc,
                    lp=hbv.lp,
                    beta=hbv.beta,
                    k0=hbv.k0 * scale_factor,
                    k1=hbv.k1 * scale_factor,
                    k2=hbv.k2 * scale_factor,
                    uzl=hbv.uzl,
                    perc=hbv.perc * scale_factor,
                    maxbas=hbv.maxbas,
                    smoothing=hbv.smoothing,
                    smoothing_enabled=hbv.smoothing_enabled
                )

        else:
            raise ValueError(f"Unknown param_mode: {mode}")

        # Routing parameters
        routing = routing_params or self._default_routing_params

        return DistributedHBVParams(
            hbv_params=hbv,
            routing_params=routing,
            param_mode=mode
        )

    def simulate(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        params: Optional[DistributedHBVParams] = None,
        initial_state: Optional[DistributedHBVState] = None,
        return_gru_runoff: bool = False
    ) -> Union[Tuple[Any, DistributedHBVState], Tuple[Any, Any, DistributedHBVState]]:
        """
        Run distributed HBV simulation.

        Args:
            precip: Precipitation [n_timesteps, n_nodes] (mm/timestep)
            temp: Temperature [n_timesteps, n_nodes] (°C)
            pet: PET [n_timesteps, n_nodes] (mm/timestep)
            params: Model parameters. If None, uses defaults.
            initial_state: Initial state. If None, uses defaults.
            return_gru_runoff: If True, also return per-GRU runoff

        Returns:
            If return_gru_runoff=False:
                Tuple of (outlet_flow, final_state)
            If return_gru_runoff=True:
                Tuple of (outlet_flow, gru_runoff, final_state)

            outlet_flow: Discharge at outlet [n_timesteps] (m³/s)
            gru_runoff: Runoff from each GRU [n_timesteps, n_nodes] (mm/timestep)
            final_state: Final model state
        """
        if params is None:
            params = self.create_params()

        if initial_state is None:
            initial_state = self.create_initial_state()

        if self.use_jax:
            result = self._simulate_jax(precip, temp, pet, params, initial_state)
        else:
            result = self._simulate_numpy(precip, temp, pet, params, initial_state)

        outlet_flow, gru_runoff, final_state = result

        if return_gru_runoff:
            return outlet_flow, gru_runoff, final_state
        else:
            return outlet_flow, final_state

    def _simulate_jax(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        params: DistributedHBVParams,
        initial_state: DistributedHBVState
    ) -> Tuple[Any, Any, DistributedHBVState]:
        """JAX implementation of distributed simulation."""
        # Step 1: Run HBV for all GRUs
        gru_runoff = self._run_all_grus_jax(precip, temp, pet, params)

        # Step 2: Convert runoff to discharge (m³/s)
        gru_discharge = runoff_mm_to_cms(
            gru_runoff,
            self.network.node_areas,
            self.dt_seconds
        )

        # Step 3: Route through network
        outlet_flow, final_routing_state = self._route_jax(
            gru_discharge,
            params.routing_params,
            initial_state.routing_state
        )

        # Create final state (HBV states updated in _run_all_grus)
        final_state = DistributedHBVState(
            hbv_states=initial_state.hbv_states,  # Not tracking per-GRU final states for now
            routing_state=final_routing_state,
            prev_runoff=gru_runoff[-1]
        )

        return outlet_flow, gru_runoff, final_state

    def _run_all_grus_jax(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        params: DistributedHBVParams
    ) -> Any:
        """Run HBV for all GRUs using vmap.

        Args:
            precip: Precipitation array [n_timesteps, n_nodes] (mm/timestep)
            temp: Temperature array [n_timesteps, n_nodes] (deg C)
            pet: PET array [n_timesteps, n_nodes] (mm/timestep)
            params: Distributed HBV parameters

        Returns:
            Runoff array [n_timesteps, n_nodes] (mm/timestep)

        Raises:
            ValueError: If input arrays have incorrect shape or dimensions
        """
        n_nodes = self.network.n_nodes

        # Validate input array shapes
        for name, arr in [('precip', precip), ('temp', temp), ('pet', pet)]:
            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D array [n_timesteps, n_nodes], "
                    f"got {arr.ndim}D array with shape {arr.shape}"
                )
            if arr.shape[1] != n_nodes:
                raise ValueError(
                    f"{name} has {arr.shape[1]} nodes but network has {n_nodes} nodes. "
                    f"Expected shape [n_timesteps, {n_nodes}], got {arr.shape}"
                )

        # Transpose forcing to [n_nodes, n_timesteps]
        precip_T = precip.T
        temp_T = temp.T
        pet_T = pet.T

        if params.param_mode == 'uniform':
            # Same parameters for all GRUs - use vmap over spatial dimension
            def run_single_gru(p, t, e):
                runoff, _ = simulate_jax(
                    p, t, e,
                    params.hbv_params,
                    warmup_days=0,  # Handle warmup externally
                    timestep_hours=self.timestep_hours
                )
                return runoff

            # vmap over GRUs (first axis after transpose)
            runoff = vmap(run_single_gru)(precip_T, temp_T, pet_T)  # [n_nodes, n_timesteps]
            return runoff.T  # [n_timesteps, n_nodes]

        elif params.param_mode == 'regionalized':
            # Parameters are batched [n_nodes] in params.hbv_params
            # We vmap over forcing AND parameters

            def run_single_gru(p, t, e, gru_params):
                runoff, _ = simulate_jax(
                    p, t, e,
                    gru_params,
                    warmup_days=0,
                    timestep_hours=self.timestep_hours
                )
                return runoff

            # vmap over forcing and params.hbv_params
            runoff = vmap(run_single_gru)(precip_T, temp_T, pet_T, params.hbv_params)
            return runoff.T

        else:
            # Per-GRU parameters (List) - can't easily vmap
            all_runoff = []
            for gru_idx in range(n_nodes):
                gru_params = params.hbv_params[gru_idx]
                runoff, _ = simulate_jax(
                    precip[:, gru_idx],
                    temp[:, gru_idx],
                    pet[:, gru_idx],
                    gru_params,
                    warmup_days=0,
                    timestep_hours=self.timestep_hours
                )
                all_runoff.append(runoff)

            return jnp.stack(all_runoff, axis=1)

    def _route_jax(
        self,
        gru_discharge: Any,
        routing_params: RoutingParams,
        initial_routing_state: RoutingState
    ) -> Tuple[Any, RoutingState]:
        """Route discharge through network using JAX."""
        n_timesteps = gru_discharge.shape[0]

        def scan_fn(carry, t):
            prev_inflows, routing_state = carry
            curr_inflows = gru_discharge[t]

            outlet_flow, new_state = route_network_step_jax(
                curr_inflows, prev_inflows, routing_state, routing_params,
                self.network.edge_from, self.network.edge_to,
                self.network.downstream_idx, self.network.upstream_indices,
                self.network.upstream_count, self.network.topo_order,
                self.network.node_areas, self.dt_seconds
            )

            return (curr_inflows, new_state), outlet_flow

        init_inflows = jnp.zeros(self.network.n_nodes)
        (_, final_state), outlet_series = lax.scan(
            scan_fn,
            (init_inflows, initial_routing_state),
            jnp.arange(n_timesteps)
        )

        return outlet_series, final_state

    def _simulate_numpy(
        self,
        precip: np.ndarray,
        temp: np.ndarray,
        pet: np.ndarray,
        params: DistributedHBVParams,
        initial_state: DistributedHBVState
    ) -> Tuple[np.ndarray, np.ndarray, DistributedHBVState]:
        """NumPy implementation of distributed simulation."""
        n_timesteps = precip.shape[0]
        n_nodes = self.network.n_nodes

        # Step 1: Run HBV for all GRUs
        gru_runoff = np.zeros((n_timesteps, n_nodes))

        for gru_idx in range(n_nodes):
            if params.param_mode == 'uniform':
                gru_params = params.hbv_params
            elif params.param_mode == 'regionalized':
                gru_params = _slice_hbv_params(params.hbv_params, gru_idx)
            else:
                gru_params = params.hbv_params[gru_idx]

            runoff, _ = simulate_numpy(
                precip[:, gru_idx],
                temp[:, gru_idx],
                pet[:, gru_idx],
                gru_params,
                warmup_days=0,
                timestep_hours=self.timestep_hours
            )
            gru_runoff[:, gru_idx] = runoff

        # Step 2: Convert to discharge
        gru_discharge = runoff_mm_to_cms(
            gru_runoff,
            np.asarray(self.network.node_areas),
            self.dt_seconds
        )

        # Step 3: Route through network
        outlet_flow = np.zeros(n_timesteps)
        routing_state = initial_state.routing_state
        prev_inflows = np.zeros(n_nodes)

        for t in range(n_timesteps):
            curr_inflows = gru_discharge[t]

            outlet, routing_state = route_network_step_numpy(
                curr_inflows, prev_inflows, routing_state, params.routing_params,
                np.asarray(self.network.edge_from),
                np.asarray(self.network.edge_to),
                np.asarray(self.network.downstream_idx),
                np.asarray(self.network.upstream_indices),
                np.asarray(self.network.upstream_count),
                np.asarray(self.network.topo_order),
                np.asarray(self.network.node_areas),
                self.dt_seconds
            )

            outlet_flow[t] = outlet
            prev_inflows = curr_inflows

        final_state = DistributedHBVState(
            hbv_states=initial_state.hbv_states,
            routing_state=routing_state,
            prev_runoff=gru_runoff[-1]
        )

        return outlet_flow, gru_runoff, final_state

    def compute_loss(
        self,
        params_dict: Dict[str, float],
        precip: Any,
        temp: Any,
        pet: Any,
        obs: Any,
        metric: str = 'nse',
        warmup_timesteps: Optional[int] = None
    ) -> float:
        """
        Compute loss for calibration.

        Args:
            params_dict: HBV parameter dictionary
            precip: Precipitation [n_timesteps, n_nodes]
            temp: Temperature [n_timesteps, n_nodes]
            pet: PET [n_timesteps, n_nodes]
            obs: Observed outlet discharge [n_timesteps] (m³/s)
            metric: Loss metric ('nse' or 'kge')
            warmup_timesteps: Timesteps to exclude from loss. If None, uses warmup_days.

        Returns:
            Negative metric value (for minimization)
        """
        if warmup_timesteps is None:
            warmup_timesteps = self.warmup_days * (24 // self.timestep_hours)

        # Create parameters
        params = self.create_params(hbv_params=params_dict)

        # Run simulation
        outlet_flow, _ = self.simulate(precip, temp, pet, params)

        # Compute metric
        xp = jnp if self.use_jax else np

        sim = outlet_flow[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        if metric.lower() == 'nse':
            ss_res = xp.sum((sim - obs_eval) ** 2)
            ss_tot = xp.sum((obs_eval - xp.mean(obs_eval)) ** 2)
            nse = 1.0 - ss_res / (ss_tot + 1e-10)
            return -nse
        elif metric.lower() == 'kge':
            r = xp.corrcoef(sim, obs_eval)[0, 1]
            alpha = xp.std(sim) / (xp.std(obs_eval) + 1e-10)
            beta = xp.mean(sim) / (xp.mean(obs_eval) + 1e-10)
            kge = 1.0 - xp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            return -kge
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_loss_function(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        obs: Any,
        metric: str = 'nse',
        param_names: Optional[List[str]] = None,
        warmup_timesteps: Optional[int] = None
    ) -> Callable:
        """
        Get a loss function for optimization.

        Returns a function that takes a parameter array and returns the loss.
        Suitable for use with scipy.optimize or custom optimizers.

        Args:
            precip: Precipitation forcing
            temp: Temperature forcing
            pet: PET forcing
            obs: Observed outlet discharge
            metric: Loss metric
            param_names: Names of parameters in the array. If None, uses all params.
            warmup_timesteps: Warmup period

        Returns:
            Loss function: params_array -> loss_value
        """
        if param_names is None:
            param_names = list(DEFAULT_PARAMS.keys())

        if warmup_timesteps is None:
            warmup_timesteps = self.warmup_days * (24 // self.timestep_hours)

        def loss_fn(params_array):
            params_dict = dict(zip(param_names, params_array))
            return self.compute_loss(
                params_dict, precip, temp, pet, obs, metric, warmup_timesteps
            )

        return loss_fn

    def get_gradient_function(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        obs: Any,
        metric: str = 'nse',
        param_names: Optional[List[str]] = None,
        warmup_timesteps: Optional[int] = None
    ) -> Optional[Callable]:
        """
        Get gradient function for gradient-based optimization.

        Args:
            precip: Precipitation forcing
            temp: Temperature forcing
            pet: PET forcing
            obs: Observed outlet discharge
            metric: Loss metric
            param_names: Parameter names
            warmup_timesteps: Warmup period

        Returns:
            Gradient function if JAX available, None otherwise.
            The function takes params_array and returns gradients array.
        """
        if not self.use_jax:
            warnings.warn("JAX not available for gradient computation")
            return None

        if param_names is None:
            param_names = list(DEFAULT_PARAMS.keys())

        if warmup_timesteps is None:
            warmup_timesteps = self.warmup_days * (24 // self.timestep_hours)

        def loss_fn(params_array):
            params_dict = dict(zip(param_names, params_array))
            return self.compute_loss(
                params_dict, precip, temp, pet, obs, metric, warmup_timesteps
            )

        return jax.grad(loss_fn)

    def get_value_and_grad_function(
        self,
        precip: Any,
        temp: Any,
        pet: Any,
        obs: Any,
        metric: str = 'nse',
        param_names: Optional[List[str]] = None,
        warmup_timesteps: Optional[int] = None
    ) -> Optional[Callable]:
        """
        Get function that returns both loss value and gradients.

        More efficient than calling loss and grad separately.

        Args:
            Same as get_gradient_function

        Returns:
            Function: params_array -> (loss, gradients)
        """
        if not self.use_jax:
            warnings.warn("JAX not available for gradient computation")
            return None

        if param_names is None:
            param_names = list(DEFAULT_PARAMS.keys())

        if warmup_timesteps is None:
            warmup_timesteps = self.warmup_days * (24 // self.timestep_hours)

        def loss_fn(params_array):
            params_dict = dict(zip(param_names, params_array))
            return self.compute_loss(
                params_dict, precip, temp, pet, obs, metric, warmup_timesteps
            )

        return jax.value_and_grad(loss_fn)


def calibrate_distributed_hbv(
    model: DistributedHBV,
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    param_names: Optional[List[str]] = None,
    metric: str = 'nse',
    method: str = 'L-BFGS-B',
    maxiter: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calibrate distributed HBV model using gradient-based optimization.

    Args:
        model: DistributedHBV instance
        precip: Precipitation [n_timesteps, n_nodes]
        temp: Temperature [n_timesteps, n_nodes]
        pet: PET [n_timesteps, n_nodes]
        obs: Observed outlet discharge [n_timesteps]
        param_names: Parameters to calibrate. If None, uses common subset.
        metric: Objective metric ('nse' or 'kge')
        method: Optimization method (scipy.optimize)
        maxiter: Maximum iterations
        verbose: Print progress

    Returns:
        Dictionary with:
            - 'params': Calibrated parameter dictionary
            - 'loss': Final loss value
            - 'metric_value': Final metric value (positive)
            - 'success': Whether optimization succeeded
            - 'niter': Number of iterations
    """
    from scipy.optimize import minimize

    if param_names is None:
        param_names = ['fc', 'beta', 'k0', 'k1', 'k2', 'perc', 'maxbas']

    # Initial values and bounds
    x0 = np.array([DEFAULT_PARAMS[p] for p in param_names])
    bounds = [(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1]) for p in param_names]

    # Get loss and gradient functions
    if model.use_jax:
        value_and_grad_fn = model.get_value_and_grad_function(
            precip, temp, pet, obs, metric, param_names
        )

        def scipy_fn(x):
            loss, grads = value_and_grad_fn(jnp.array(x))
            return float(loss), np.array(grads)

        result = minimize(
            scipy_fn,
            x0,
            method=method,
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': verbose}
        )
    else:
        loss_fn = model.get_loss_function(
            precip, temp, pet, obs, metric, param_names
        )

        result = minimize(
            loss_fn,
            x0,
            method='Nelder-Mead' if method == 'L-BFGS-B' else method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': verbose}
        )

    # Extract results
    calibrated_params = dict(zip(param_names, result.x))

    return {
        'params': calibrated_params,
        'loss': result.fun,
        'metric_value': -result.fun,
        'success': result.success,
        'niter': result.nit if hasattr(result, 'nit') else None,
        'message': result.message if hasattr(result, 'message') else None
    }


def load_distributed_hbv_from_config(
    config: Dict[str, Any],
    river_network_path: Optional[Path] = None,
    catchment_path: Optional[Path] = None
) -> DistributedHBV:
    """
    Create DistributedHBV from configuration dictionary.

    Args:
        config: Configuration dictionary with HBV_* keys
        river_network_path: Path to river network shapefile
        catchment_path: Path to catchment shapefile

    Returns:
        Configured DistributedHBV instance
    """
    # Build network from shapefiles
    builder = NetworkBuilder(
        river_network_path=river_network_path,
        catchment_path=catchment_path,
        use_jax=config.get('HBV_BACKEND', 'jax') == 'jax'
    )

    network = builder.build_from_shapefiles()

    # Create model
    model = DistributedHBV(
        network=network,
        param_mode=config.get('HBV_DISTRIBUTED_PARAM_MODE', 'uniform'),
        timestep_hours=config.get('HBV_TIMESTEP_HOURS', 24),
        warmup_days=config.get('HBV_WARMUP_DAYS', 365),
        use_jax=config.get('HBV_BACKEND', 'jax') == 'jax'
    )

    return model


def calibrate_distributed_hbv_adam(
    model: 'DistributedHBV',
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    param_names: Optional[List[str]] = None,
    n_iterations: int = 500,
    lr_max: float = 0.1,
    lr_min: float = 1e-5,
    warmup_timesteps: Optional[int] = None,
    grad_clip: float = 5.0,
    patience: int = 100,
    ema_decay: float = 0.99,
    loss_weights: Optional[Dict[str, float]] = None,
    use_extended_bounds: bool = True,
    two_phase: bool = True,
    verbose: bool = True
) -> 'CalibrationResult':
    """
    Calibrate distributed HBV using Adam optimizer with advanced techniques.

    This function provides optimized gradient-based calibration with:
    - AdamW optimizer with weight decay
    - Cosine annealing learning rate schedule with warm restarts
    - Composite loss function (NSE + log-NSE + volume bias)
    - Early stopping with patience
    - Exponential moving average of parameters
    - Optional two-phase optimization (exploration + refinement)

    Args:
        model: DistributedHBV instance
        precip: Precipitation [n_timesteps, n_nodes]
        temp: Temperature [n_timesteps, n_nodes]
        pet: PET [n_timesteps, n_nodes]
        obs: Observed outlet discharge [n_timesteps]
        param_names: Parameters to calibrate. Defaults to common HBV params.
        n_iterations: Total optimization iterations
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        warmup_timesteps: Warmup period to exclude from loss. Defaults to model.warmup_days.
        grad_clip: Gradient clipping threshold
        patience: Early stopping patience (iterations without improvement)
        ema_decay: EMA decay rate for parameter smoothing
        loss_weights: Weights for composite loss {'nse': w1, 'log_nse': w2, 'volume': w3}
        use_extended_bounds: Use extended parameter bounds for better exploration
        two_phase: Use two-phase optimization (exploration + refinement)
        verbose: Print progress

    Returns:
        CalibrationResult with calibrated parameters and metrics

    Example:
        >>> model = DistributedHBV(network)
        >>> result = calibrate_distributed_hbv_adam(
        ...     model, precip, temp, pet, obs,
        ...     n_iterations=500, two_phase=True
        ... )
        >>> print(f"NSE: {result.nse:.4f}")
        >>> calibrated_params = result.params
    """
    from .optimizers import EXTENDED_PARAM_BOUNDS

    if not model.use_jax:
        raise ValueError("Adam calibration requires JAX backend. Set use_jax=True.")

    if param_names is None:
        param_names = ['fc', 'beta', 'k0', 'k1', 'k2', 'perc', 'maxbas', 'cfmax', 'tt', 'lp']

    if warmup_timesteps is None:
        warmup_timesteps = model.warmup_days

    if loss_weights is None:
        loss_weights = {'nse': 0.5, 'log_nse': 0.3, 'volume': 0.2}

    bounds = EXTENDED_PARAM_BOUNDS if use_extended_bounds else PARAM_BOUNDS

    # Convert to JAX arrays
    precip_jax = jnp.array(precip)
    temp_jax = jnp.array(temp)
    pet_jax = jnp.array(pet)
    obs_jax = jnp.array(obs)

    if two_phase:
        result = _calibrate_two_phase(
            model, precip_jax, temp_jax, pet_jax, obs_jax,
            param_names, bounds, warmup_timesteps, loss_weights,
            grad_clip, patience, ema_decay, lr_max, lr_min, verbose
        )
    else:
        result = _calibrate_single_phase(
            model, precip_jax, temp_jax, pet_jax, obs_jax,
            param_names, bounds, n_iterations, warmup_timesteps,
            loss_weights, grad_clip, patience, ema_decay,
            lr_max, lr_min, verbose
        )

    return result


def _calibrate_single_phase(
    model: 'DistributedHBV',
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    param_names: List[str],
    bounds: Dict,
    n_iterations: int,
    warmup_timesteps: int,
    loss_weights: Dict[str, float],
    grad_clip: float,
    patience: int,
    ema_decay: float,
    lr_max: float,
    lr_min: float,
    verbose: bool
) -> 'CalibrationResult':
    """Single-phase calibration with composite loss."""
    import time
    from symfluence.optimization.gradient import AdamW, CosineAnnealingWarmRestarts, EMA
    from .optimizers import CalibrationResult

    start_time = time.time()

    if verbose:
        print("\n" + "="*60)
        print("Adam Calibration (Single Phase)")
        print("="*60)
        print(f"Parameters: {param_names}")
        print(f"Iterations: {n_iterations}")
        print(f"LR: [{lr_min}, {lr_max}]")

    # Initial parameters
    x = jnp.array([DEFAULT_PARAMS[p] for p in param_names])

    # Create composite loss function
    loss_fn = _create_composite_loss(
        model, precip, temp, pet, obs, param_names,
        warmup_timesteps, loss_weights
    )
    val_grad_fn = jit(jax.value_and_grad(loss_fn))

    # NSE-only for tracking
    nse_fn = model.get_value_and_grad_function(
        precip, temp, pet, obs, 'nse', param_names, warmup_timesteps
    )
    nse_fn = jit(nse_fn)

    # Initialize optimizer
    optimizer = AdamW(lr=lr_max, weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(
        lr_max=lr_max, lr_min=lr_min, T_0=50, T_mult=2, warmup_steps=20
    )
    ema = EMA(decay=ema_decay)

    history = {'iteration': [], 'nse': [], 'loss': [], 'lr': [], 'grad_norm': []}
    best_nse = -float('inf')
    best_params = x.copy()
    best_iter = 0
    no_improve = 0

    for i in range(n_iterations):
        lr = scheduler.get_lr(i)
        optimizer.lr = lr

        loss, grads = val_grad_fn(x)
        nse_loss, _ = nse_fn(x)
        nse = -float(nse_loss)

        grad_norm = float(jnp.linalg.norm(grads))
        if grad_norm > grad_clip:
            grads = grads * (grad_clip / grad_norm)

        x = optimizer.step(x, grads)
        x = _clip_to_bounds(x, param_names, bounds)
        ema.update(x)

        if nse > best_nse:
            best_nse = nse
            best_params = x.copy()
            best_iter = i
            no_improve = 0
        else:
            no_improve += 1

        history['iteration'].append(i)
        history['nse'].append(nse)
        history['loss'].append(float(loss))
        history['lr'].append(lr)
        history['grad_norm'].append(grad_norm)

        if verbose and (i % 25 == 0 or i == n_iterations - 1):
            print(f"  Iter {i:4d}: NSE={nse:.4f}, Loss={float(loss):.4f}, "
                  f"lr={lr:.5f}, best={best_nse:.4f}@{best_iter}")

        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at iteration {i}")
            break

    # Check EMA
    ema_params = ema.get()
    ema_loss, _ = nse_fn(ema_params)
    ema_nse = -float(ema_loss)

    if ema_nse > best_nse:
        final_params = ema_params
    else:
        final_params = best_params

    total_time = time.time() - start_time

    # Compute final metrics
    metrics = _compute_metrics(model, precip, temp, pet, obs, final_params, param_names, warmup_timesteps)

    if verbose:
        print(f"\nCalibration complete in {total_time:.1f}s")
        print(f"Final NSE: {metrics['nse']:.4f}, KGE: {metrics['kge']:.4f}")

    calibrated_params = dict(zip(param_names, [float(p) for p in final_params]))

    return CalibrationResult(
        params=calibrated_params,
        nse=metrics['nse'],
        kge=metrics['kge'],
        log_nse=metrics['nse_log'],
        rmse=metrics['rmse'],
        volume_bias_pct=metrics['volume_bias_pct'],
        history=history,
        total_time=total_time,
        best_iter=best_iter,
        param_array=final_params
    )


def _calibrate_two_phase(
    model: 'DistributedHBV',
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    param_names: List[str],
    bounds: Dict,
    warmup_timesteps: int,
    loss_weights: Dict[str, float],
    grad_clip: float,
    patience: int,
    ema_decay: float,
    lr_max: float,
    lr_min: float,
    verbose: bool
) -> 'CalibrationResult':
    """
    Two-phase calibration: exploration + refinement.

    Phase 1: Higher LR with composite loss to find good region
    Phase 2: Lower LR with NSE-only loss to polish result
    """
    import time
    from symfluence.optimization.gradient import AdamW, CosineAnnealingWarmRestarts, EMA
    from .optimizers import CalibrationResult

    total_start = time.time()

    if verbose:
        print("\n" + "="*70)
        print("TWO-PHASE CALIBRATION")
        print("="*70)

    # =========================================================================
    # PHASE 1: EXPLORATION
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("PHASE 1: Exploration (Composite Loss, Higher LR)")
        print("-"*70)

    x = jnp.array([DEFAULT_PARAMS[p] for p in param_names])

    # Composite loss for exploration
    loss_fn = _create_composite_loss(
        model, precip, temp, pet, obs, param_names,
        warmup_timesteps, {'nse': 0.6, 'log_nse': 0.25, 'volume': 0.15}
    )
    val_grad_fn = jit(jax.value_and_grad(loss_fn))

    # NSE tracking
    nse_fn = model.get_value_and_grad_function(
        precip, temp, pet, obs, 'nse', param_names, warmup_timesteps
    )
    nse_fn = jit(nse_fn)

    optimizer = AdamW(lr=lr_max * 0.8, weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(
        lr_max=lr_max * 0.8, lr_min=lr_min * 10, T_0=50, T_mult=2, warmup_steps=10
    )
    ema = EMA(decay=0.995)

    history1 = {'iteration': [], 'nse': [], 'loss': [], 'lr': [], 'grad_norm': []}
    best_nse_p1 = -float('inf')
    best_params_p1 = x.copy()
    best_iter_p1 = 0
    no_improve = 0
    patience_p1 = 80

    n_iter_p1 = 200

    for i in range(n_iter_p1):
        lr = scheduler.get_lr(i)
        optimizer.lr = lr

        loss, grads = val_grad_fn(x)
        nse_loss, _ = nse_fn(x)
        nse = -float(nse_loss)

        grad_norm = float(jnp.linalg.norm(grads))
        if grad_norm > grad_clip:
            grads = grads * (grad_clip / grad_norm)

        x = optimizer.step(x, grads)
        x = _clip_to_bounds(x, param_names, bounds)
        ema.update(x)

        if nse > best_nse_p1:
            best_nse_p1 = nse
            best_params_p1 = x.copy()
            best_iter_p1 = i
            no_improve = 0
        else:
            no_improve += 1

        history1['iteration'].append(i)
        history1['nse'].append(nse)
        history1['loss'].append(float(loss))
        history1['lr'].append(lr)
        history1['grad_norm'].append(grad_norm)

        if verbose and (i % 25 == 0 or i == n_iter_p1 - 1):
            print(f"  Iter {i:4d}: NSE={nse:.4f}, lr={lr:.5f}, best={best_nse_p1:.4f}@{best_iter_p1}")

        if no_improve >= patience_p1:
            if verbose:
                print(f"\n  Phase 1 early stop at iteration {i}")
            break

    if verbose:
        print(f"\n  Phase 1 Best NSE: {best_nse_p1:.4f}")

    # =========================================================================
    # PHASE 2: REFINEMENT
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("PHASE 2: Refinement (NSE-only, Lower LR)")
        print("-"*70)

    # Start from phase 1 best
    x = best_params_p1

    # NSE-only loss for refinement
    val_grad_fn = nse_fn

    optimizer = AdamW(lr=0.02, weight_decay=0.0001)
    ema = EMA(decay=0.998)

    history2 = {'iteration': [], 'nse': [], 'lr': [], 'grad_norm': []}
    best_nse_p2 = -float('inf')
    best_params_p2 = x.copy()
    best_iter_p2 = 0
    no_improve = 0
    patience_p2 = patience

    n_iter_p2 = 300

    for i in range(n_iter_p2):
        # Cosine decay (no restarts)
        lr = 0.001 + 0.019 * 0.5 * (1 + np.cos(np.pi * i / n_iter_p2))
        optimizer.lr = lr

        loss, grads = val_grad_fn(x)
        nse = -float(loss)

        grad_norm = float(jnp.linalg.norm(grads))
        if grad_norm > 3.0:
            grads = grads * (3.0 / grad_norm)

        x = optimizer.step(x, grads)
        x = _clip_to_bounds(x, param_names, bounds)
        ema.update(x)

        if nse > best_nse_p2:
            best_nse_p2 = nse
            best_params_p2 = x.copy()
            best_iter_p2 = i
            no_improve = 0
        else:
            no_improve += 1

        history2['iteration'].append(i)
        history2['nse'].append(nse)
        history2['lr'].append(lr)
        history2['grad_norm'].append(grad_norm)

        if verbose and (i % 30 == 0 or i == n_iter_p2 - 1):
            print(f"  Iter {i:4d}: NSE={nse:.4f}, lr={lr:.5f}, best={best_nse_p2:.4f}@{best_iter_p2}")

        if no_improve >= patience_p2:
            if verbose:
                print(f"\n  Phase 2 early stop at iteration {i}")
            break

    # Check EMA
    ema_params = ema.get()
    ema_loss, _ = val_grad_fn(ema_params)
    ema_nse = -float(ema_loss)

    if ema_nse > best_nse_p2:
        final_params = ema_params
    else:
        final_params = best_params_p2

    total_time = time.time() - total_start

    # Compute final metrics
    metrics = _compute_metrics(model, precip, temp, pet, obs, final_params, param_names, warmup_timesteps)

    if verbose:
        print(f"\nCalibration complete in {total_time:.1f}s")
        print(f"Phase 1 best: {best_nse_p1:.4f}, Phase 2 best: {best_nse_p2:.4f}")
        print(f"Final NSE: {metrics['nse']:.4f}, KGE: {metrics['kge']:.4f}")

    # Combine histories
    combined_history = {
        'iteration': history1['iteration'] + [i + n_iter_p1 for i in history2['iteration']],
        'nse': history1['nse'] + history2['nse'],
        'lr': history1['lr'] + history2['lr'],
        'grad_norm': history1['grad_norm'] + history2['grad_norm'],
    }

    calibrated_params = dict(zip(param_names, [float(p) for p in final_params]))

    return CalibrationResult(
        params=calibrated_params,
        nse=metrics['nse'],
        kge=metrics['kge'],
        log_nse=metrics['nse_log'],
        rmse=metrics['rmse'],
        volume_bias_pct=metrics['volume_bias_pct'],
        history=combined_history,
        total_time=total_time,
        best_iter=best_iter_p2 + n_iter_p1,
        param_array=final_params
    )


def _create_composite_loss(
    model: 'DistributedHBV',
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    param_names: List[str],
    warmup_timesteps: int,
    weights: Dict[str, float]
) -> Callable:
    """Create composite loss function combining NSE, log-NSE, and volume bias."""

    eps = 0.1  # For log transform

    def loss_fn(x: Any) -> Any:
        # Build params
        hbv_params = {**DEFAULT_PARAMS}
        for i, name in enumerate(param_names):
            hbv_params[name] = x[i]

        params = model.create_params(hbv_params=hbv_params)
        outlet_flow, _ = model.simulate(precip, temp, pet, params)

        sim = outlet_flow[warmup_timesteps:]
        obs_eval = obs[warmup_timesteps:]

        valid_mask = ~jnp.isnan(obs_eval)
        n_valid = jnp.sum(valid_mask)

        # NSE
        ss_res = jnp.sum(jnp.where(valid_mask, (sim - obs_eval) ** 2, 0.0))
        obs_mean = jnp.sum(jnp.where(valid_mask, obs_eval, 0.0)) / n_valid
        ss_tot = jnp.sum(jnp.where(valid_mask, (obs_eval - obs_mean) ** 2, 0.0))
        nse = 1.0 - ss_res / (ss_tot + 1e-10)

        # Log-NSE
        sim_log = jnp.log(sim + eps)
        obs_log = jnp.log(obs_eval + eps)
        ss_res_log = jnp.sum(jnp.where(valid_mask, (sim_log - obs_log) ** 2, 0.0))
        obs_log_mean = jnp.sum(jnp.where(valid_mask, obs_log, 0.0)) / n_valid
        ss_tot_log = jnp.sum(jnp.where(valid_mask, (obs_log - obs_log_mean) ** 2, 0.0))
        nse_log = 1.0 - ss_res_log / (ss_tot_log + 1e-10)

        # Volume bias
        sim_sum = jnp.sum(jnp.where(valid_mask, sim, 0.0))
        obs_sum = jnp.sum(jnp.where(valid_mask, obs_eval, 0.0))
        vol_bias = jnp.abs(sim_sum - obs_sum) / (obs_sum + 1e-10)

        # Combined loss
        loss = (
            weights.get('nse', 0.5) * (1.0 - nse) +
            weights.get('log_nse', 0.3) * (1.0 - nse_log) +
            weights.get('volume', 0.2) * vol_bias
        )

        return loss

    return loss_fn


def _clip_to_bounds(params: Any, param_names: List[str], bounds: Dict) -> Any:
    """Clip parameters to their valid bounds."""
    clipped = []
    for i, name in enumerate(param_names):
        low, high = bounds.get(name, PARAM_BOUNDS.get(name, (0, 1)))
        clipped.append(jnp.clip(params[i], low, high))
    return jnp.array(clipped)


def _compute_metrics(
    model: 'DistributedHBV',
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    params_array: Any,
    param_names: List[str],
    warmup_timesteps: int
) -> Dict[str, float]:
    """Compute performance metrics for calibrated parameters."""

    hbv_params = {**DEFAULT_PARAMS}
    for i, name in enumerate(param_names):
        hbv_params[name] = float(params_array[i])

    params = model.create_params(hbv_params=hbv_params)
    outlet_flow, _ = model.simulate(precip, temp, pet, params)

    sim = np.array(outlet_flow)[warmup_timesteps:]
    obs_eval = np.array(obs)[warmup_timesteps:]

    valid = ~(np.isnan(sim) | np.isnan(obs_eval))
    sim = sim[valid]
    obs_np = obs_eval[valid]

    # NSE
    ss_res = np.sum((sim - obs_np) ** 2)
    ss_tot = np.sum((obs_np - np.mean(obs_np)) ** 2)
    nse = 1 - ss_res / ss_tot

    # Log-NSE
    eps = 0.1
    sim_log = np.log(sim + eps)
    obs_log = np.log(obs_np + eps)
    ss_res_log = np.sum((sim_log - obs_log) ** 2)
    ss_tot_log = np.sum((obs_log - np.mean(obs_log)) ** 2)
    nse_log = 1 - ss_res_log / ss_tot_log

    # KGE
    r = np.corrcoef(sim, obs_np)[0, 1]
    alpha = np.std(sim) / np.std(obs_np)
    beta = np.mean(sim) / np.mean(obs_np)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # RMSE
    rmse = np.sqrt(np.mean((sim - obs_np) ** 2))

    # Volume bias
    vol_bias = (np.sum(sim) - np.sum(obs_np)) / np.sum(obs_np) * 100

    return {
        'nse': nse,
        'nse_log': nse_log,
        'kge': kge,
        'rmse': rmse,
        'volume_bias_pct': vol_bias,
        'correlation': r,
    }
