"""
Differentiable Muskingum-Cunge River Routing.

Implements the Muskingum-Cunge method for channel routing in JAX,
enabling end-to-end gradient-based calibration of distributed
hydrological models.

The Muskingum-Cunge method is a physically-based routing scheme that:
1. Uses the continuity equation: dS/dt = I - Q
2. Relates storage to inflow/outflow: S = K[x*I + (1-x)*Q]
3. Derives K and x from channel hydraulics (vs. pure calibration)

Key Features:
- Fully differentiable (JAX autodiff compatible)
- Adaptive sub-stepping for Courant number stability
- Support for variable channel properties per reach
- Efficient vectorized operations

References:
    Cunge, J. A. (1969). On the subject of a flood propagation computation
    method (Muskingum method). Journal of Hydraulic Research, 7(2), 205-230.

    Todini, E. (2007). A mass conservative and water storage consistent
    variable parameter Muskingum-Cunge approach. Hydrology and Earth
    System Sciences, 11(5), 1645-1659.
"""

from typing import Tuple, NamedTuple, Optional, Any

import numpy as np

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


class RoutingParams(NamedTuple):
    """
    Muskingum-Cunge routing parameters per reach.

    Attributes:
        K: Storage constant / wave travel time (seconds) [n_reaches]
        x: Weighting factor (0-0.5, typically 0.1-0.3) [n_reaches]
        n_substeps: Number of substeps for stability [n_reaches]
    """
    K: Any  # jnp.ndarray
    x: Any
    n_substeps: Any


class RoutingState(NamedTuple):
    """
    Routing state for all reaches.

    Attributes:
        Q: Outflow from each reach (m³/s) [n_reaches]
        S: Storage in each reach (m³) [n_reaches]
    """
    Q: Any
    S: Any


def compute_muskingum_params(
    length: Any,
    slope: Any,
    width: Any,
    mannings_n: Any,
    reference_Q: Any,
    dt: float = 3600.0,
    min_K: float = 60.0,
    max_K: float = 86400.0 * 7,
) -> RoutingParams:
    """
    Compute Muskingum-Cunge K and x from channel properties.

    Uses Manning's equation to estimate wave celerity and derives
    the routing parameters from channel geometry.

    Args:
        length: Reach length (m)
        slope: Channel slope (m/m)
        width: Channel width (m)
        mannings_n: Manning's roughness coefficient
        reference_Q: Reference discharge for parameter estimation (m³/s)
        dt: Routing timestep (seconds)
        min_K: Minimum K value (seconds)
        max_K: Maximum K value (seconds)

    Returns:
        RoutingParams with K, x, and n_substeps
    """
    xp = jnp if HAS_JAX else np

    # Avoid division by zero
    slope = xp.maximum(slope, 1e-6)
    width = xp.maximum(width, 1.0)
    reference_Q = xp.maximum(reference_Q, 0.1)

    # Estimate flow depth using Manning's equation
    # Q = (1/n) * A * R^(2/3) * S^(1/2)
    # For wide rectangular channel: A ≈ W*h, R ≈ h
    # Q = (1/n) * W * h * h^(2/3) * S^(1/2) = (1/n) * W * h^(5/3) * S^(1/2)
    # h = (Q * n / (W * S^0.5))^(3/5)
    depth = xp.power(
        reference_Q * mannings_n / (width * xp.sqrt(slope)),
        0.6
    )
    depth = xp.maximum(depth, 0.1)  # Minimum 10cm depth

    # Wave celerity (kinematic wave approximation)
    # c = (5/3) * v where v = Q / (W * h)
    velocity = reference_Q / (width * depth)
    celerity = (5.0 / 3.0) * velocity
    celerity = xp.maximum(celerity, 0.1)  # Minimum celerity

    # Muskingum K (travel time through reach)
    K = length / celerity
    K = xp.clip(K, min_K, max_K)

    # Muskingum x (weighting factor)
    # x = 0.5 * (1 - Q / (B * S0 * c * dx))
    # Constrained to [0, 0.5] for numerical stability
    x = 0.5 * (1.0 - reference_Q / (width * slope * celerity * length + 1e-10))
    x = xp.clip(x, 0.0, 0.5)

    # Compute number of substeps for Courant stability
    # Courant number: C = c * dt / dx
    # For stability: C <= 1, or dt <= dx / c
    courant = celerity * dt / length
    n_substeps = xp.ceil(courant).astype(xp.int32)
    n_substeps = xp.maximum(n_substeps, 1)

    return RoutingParams(K=K, x=x, n_substeps=n_substeps)


def muskingum_coefficients(K: Any, x: Any, dt: float) -> Tuple[Any, Any, Any]:
    """
    Compute Muskingum routing coefficients C0, C1, C2.

    The outflow equation is:
        Q2 = C0 * I2 + C1 * I1 + C2 * Q1

    where:
        C0 = (dt - 2*K*x) / (2*K*(1-x) + dt)
        C1 = (dt + 2*K*x) / (2*K*(1-x) + dt)
        C2 = (2*K*(1-x) - dt) / (2*K*(1-x) + dt)

    Args:
        K: Storage constant (seconds)
        x: Weighting factor
        dt: Timestep (seconds)

    Returns:
        Tuple of (C0, C1, C2) coefficients
    """
    denom = 2.0 * K * (1.0 - x) + dt

    C0 = (dt - 2.0 * K * x) / denom
    C1 = (dt + 2.0 * K * x) / denom
    C2 = (2.0 * K * (1.0 - x) - dt) / denom

    return C0, C1, C2


def route_reach_step(
    I_prev: Any,
    I_curr: Any,
    Q_prev: Any,
    K: Any,
    x: Any,
    dt: float
) -> Any:
    """
    Single Muskingum routing step for one reach.

    Args:
        I_prev: Inflow at previous timestep (m³/s)
        I_curr: Inflow at current timestep (m³/s)
        Q_prev: Outflow at previous timestep (m³/s)
        K: Storage constant (seconds)
        x: Weighting factor
        dt: Timestep (seconds)

    Returns:
        Q_curr: Outflow at current timestep (m³/s)
    """
    xp = jnp if HAS_JAX else np

    C0, C1, C2 = muskingum_coefficients(K, x, dt)

    Q_curr = C0 * I_curr + C1 * I_prev + C2 * Q_prev

    # Ensure non-negative outflow
    Q_curr = xp.maximum(Q_curr, 0.0)

    return Q_curr


def route_reach_adaptive(
    I_prev: Any,
    I_curr: Any,
    Q_prev: Any,
    K: Any,
    x: Any,
    dt: float,
    n_substeps: int = 1
) -> Any:
    """
    Route through reach with adaptive sub-stepping.

    Subdivides the timestep to maintain Courant stability.

    Args:
        I_prev: Inflow at previous timestep (m³/s)
        I_curr: Inflow at current timestep (m³/s)
        Q_prev: Outflow at previous timestep (m³/s)
        K: Storage constant (seconds)
        x: Weighting factor
        dt: Total timestep (seconds)
        n_substeps: Number of substeps

    Returns:
        Q_curr: Outflow at end of timestep (m³/s)
    """
    if n_substeps <= 1:
        return route_reach_step(I_prev, I_curr, Q_prev, K, x, dt)

    sub_dt = dt / n_substeps

    # Linearly interpolate inflow across substeps
    Q = Q_prev
    for i in range(n_substeps):
        alpha = (i + 1) / n_substeps
        I_sub_prev = I_prev + (i / n_substeps) * (I_curr - I_prev)
        I_sub_curr = I_prev + alpha * (I_curr - I_prev)
        Q = route_reach_step(I_sub_prev, I_sub_curr, Q, K, x, sub_dt)

    return Q


def route_reach_adaptive_jax(
    I_prev: Any,
    I_curr: Any,
    Q_prev: Any,
    K: Any,
    x: Any,
    dt: float,
    n_substeps: Any
) -> Any:
    """
    JAX-compatible adaptive routing with lax.fori_loop.

    Args:
        I_prev: Inflow at previous timestep (m³/s)
        I_curr: Inflow at current timestep (m³/s)
        Q_prev: Outflow at previous timestep (m³/s)
        K: Storage constant (seconds)
        x: Weighting factor
        dt: Total timestep (seconds)
        n_substeps: Number of substeps (JAX array, scalar)

    Returns:
        Q_curr: Outflow at end of timestep (m³/s)
    """
    if not HAS_JAX:
        return route_reach_adaptive(I_prev, I_curr, Q_prev, K, x, dt, int(n_substeps))

    # Use fixed number of substeps with early exit for JIT compatibility
    max_substeps = 10
    n_substeps = jnp.minimum(n_substeps, max_substeps)
    sub_dt = dt / jnp.maximum(n_substeps, 1)

    def substep_fn(i, Q):
        # Only update if i < n_substeps
        alpha_prev = i / jnp.maximum(n_substeps, 1)
        alpha_curr = (i + 1) / jnp.maximum(n_substeps, 1)

        I_sub_prev = I_prev + alpha_prev * (I_curr - I_prev)
        I_sub_curr = I_prev + alpha_curr * (I_curr - I_prev)

        Q_new = route_reach_step(I_sub_prev, I_sub_curr, Q, K, x, sub_dt)

        # Only update if this substep is active
        return jnp.where(i < n_substeps, Q_new, Q)

    Q_final = lax.fori_loop(0, max_substeps, substep_fn, Q_prev)

    return Q_final


def create_initial_routing_state(
    n_reaches: int,
    initial_Q: float = 0.0,
    use_jax: bool = True
) -> RoutingState:
    """
    Create initial routing state.

    Args:
        n_reaches: Number of river reaches
        initial_Q: Initial outflow for all reaches (m³/s)
        use_jax: Whether to use JAX arrays

    Returns:
        RoutingState with initial values
    """
    if use_jax and HAS_JAX:
        return RoutingState(
            Q=jnp.full(n_reaches, initial_Q),
            S=jnp.zeros(n_reaches)
        )
    else:
        return RoutingState(
            Q=np.full(n_reaches, initial_Q),
            S=np.zeros(n_reaches)
        )


def route_network_step_jax(
    inflows: Any,
    prev_inflows: Any,
    routing_state: RoutingState,
    params: RoutingParams,
    edge_from: Any,
    edge_to: Any,
    downstream_idx: Any,
    upstream_indices: Any,
    upstream_count: Any,
    topo_order: Any,
    node_areas: Any,
    dt: float = 3600.0
) -> Tuple[Any, RoutingState]:
    """
    Route flow through entire network for one timestep (JAX version).

    Processes nodes in topological order (upstream to downstream),
    accumulating flows at confluences and routing through reaches.

    Args:
        inflows: Local runoff from each GRU (m³/s) [n_nodes]
        prev_inflows: Previous timestep inflows [n_nodes]
        routing_state: Current routing state
        params: Routing parameters for each reach
        edge_from: Source node for each edge [n_edges]
        edge_to: Destination node for each edge [n_edges]
        downstream_idx: Downstream node index (-1 if outlet) [n_nodes]
        upstream_indices: Upstream node indices [n_nodes, max_upstream]
        upstream_count: Number of upstream nodes [n_nodes]
        topo_order: Topological order of nodes [n_nodes]
        node_areas: Contributing area per node [n_nodes]
        dt: Timestep in seconds

    Returns:
        Tuple of (outlet_flow, new_routing_state)
    """
    if not HAS_JAX:
        return route_network_step_numpy(
            inflows, prev_inflows, routing_state, params,
            edge_from, edge_to, downstream_idx, upstream_indices,
            upstream_count, topo_order, node_areas, dt
        )

    n_nodes = len(topo_order)
    n_edges = len(edge_from)

    # Initialize node outflows with local inflows
    node_outflows = inflows.copy()
    prev_node_outflows = prev_inflows.copy()

    # New routing state
    new_Q = routing_state.Q.copy()

    # Create edge index lookup: for node i, which edge starts from it?
    # -1 means no outgoing edge (is outlet)
    node_to_edge = jnp.full(n_nodes, -1, dtype=jnp.int32)
    node_to_edge = node_to_edge.at[edge_from].set(jnp.arange(n_edges, dtype=jnp.int32))

    def process_node(carry, node_idx):
        """Process single node in topological order."""
        node_outflows, prev_node_outflows, new_Q = carry

        node = topo_order[node_idx]

        # Sum upstream contributions
        upstream_sum = jnp.float32(0.0)
        prev_upstream_sum = jnp.float32(0.0)

        # Add contributions from all upstream nodes
        def add_upstream(i, sums):
            curr_sum, prev_sum = sums
            upstream_node = upstream_indices[node, i]
            # Only add if valid upstream (-1 means no upstream)
            mask = (i < upstream_count[node]) & (upstream_node >= 0)
            curr_sum = curr_sum + jnp.where(mask, node_outflows[upstream_node], 0.0)
            prev_sum = prev_sum + jnp.where(mask, prev_node_outflows[upstream_node], 0.0)
            return (curr_sum, prev_sum)

        max_upstream = upstream_indices.shape[1]
        upstream_sum, prev_upstream_sum = lax.fori_loop(
            0, max_upstream, add_upstream, (upstream_sum, prev_upstream_sum)
        )

        # Total inflow to this node = local runoff + routed upstream
        total_inflow = inflows[node] + upstream_sum
        prev_total_inflow = prev_inflows[node] + prev_upstream_sum

        # Route through reach if there's a downstream node
        edge_idx = node_to_edge[node]
        has_downstream = downstream_idx[node] >= 0

        # Get routing parameters for this edge (use index 0 if no edge)
        safe_edge_idx = jnp.maximum(edge_idx, 0)
        K = params.K[safe_edge_idx]
        x = params.x[safe_edge_idx]
        n_sub = params.n_substeps[safe_edge_idx]

        # Route through reach
        routed_Q = route_reach_adaptive_jax(
            prev_total_inflow, total_inflow,
            routing_state.Q[safe_edge_idx],
            K, x, dt, n_sub
        )

        # Update outflow: if has downstream, use routed Q; else use total inflow
        node_out = jnp.where(has_downstream, routed_Q, total_inflow)

        # Update state arrays
        node_outflows = node_outflows.at[node].set(node_out)
        new_Q = jnp.where(
            has_downstream & (edge_idx >= 0),
            new_Q.at[safe_edge_idx].set(routed_Q),
            new_Q
        )

        return (node_outflows, prev_node_outflows, new_Q), node_out

    # Process all nodes in topological order
    (node_outflows, _, new_Q), _ = lax.scan(
        process_node,
        (node_outflows, prev_node_outflows, new_Q),
        jnp.arange(n_nodes, dtype=jnp.int32)
    )

    # Outlet flow is the outflow from outlet nodes
    # For single outlet, this is the last node in topo order
    outlet_flow = node_outflows[topo_order[-1]]

    new_state = RoutingState(Q=new_Q, S=routing_state.S)

    return outlet_flow, new_state


def route_network_step_numpy(
    inflows: np.ndarray,
    prev_inflows: np.ndarray,
    routing_state: RoutingState,
    params: RoutingParams,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    downstream_idx: np.ndarray,
    upstream_indices: np.ndarray,
    upstream_count: np.ndarray,
    topo_order: np.ndarray,
    node_areas: np.ndarray,
    dt: float = 3600.0
) -> Tuple[float, RoutingState]:
    """
    NumPy version of network routing step.

    Args:
        Same as route_network_step_jax

    Returns:
        Tuple of (outlet_flow, new_routing_state)
    """
    n_nodes = len(topo_order)
    n_edges = len(edge_from)

    # Node outflows initialized with local runoff
    node_outflows = inflows.copy()
    prev_node_outflows = prev_inflows.copy()

    # Create edge lookup
    node_to_edge = np.full(n_nodes, -1, dtype=np.int32)
    node_to_edge[edge_from] = np.arange(n_edges)

    new_Q = routing_state.Q.copy()

    # Process in topological order
    for node in topo_order:
        # Sum upstream contributions
        upstream_sum = 0.0
        prev_upstream_sum = 0.0

        for i in range(upstream_count[node]):
            upstream_node = upstream_indices[node, i]
            if upstream_node >= 0:
                upstream_sum += node_outflows[upstream_node]
                prev_upstream_sum += prev_node_outflows[upstream_node]

        # Total inflow
        total_inflow = inflows[node] + upstream_sum
        prev_total_inflow = prev_inflows[node] + prev_upstream_sum

        # Route if has downstream
        edge_idx = node_to_edge[node]
        if downstream_idx[node] >= 0 and edge_idx >= 0:
            routed_Q = route_reach_adaptive(
                prev_total_inflow, total_inflow,
                routing_state.Q[edge_idx],
                params.K[edge_idx], params.x[edge_idx],
                dt, int(params.n_substeps[edge_idx])
            )
            node_outflows[node] = routed_Q
            new_Q[edge_idx] = routed_Q
        else:
            node_outflows[node] = total_inflow

    # Outlet flow
    outlet_flow = node_outflows[topo_order[-1]]

    new_state = RoutingState(Q=new_Q, S=routing_state.S)

    return outlet_flow, new_state


def route_timeseries_jax(
    runoff: Any,
    network,  # RiverNetwork
    routing_params: RoutingParams,
    dt: float = 3600.0,
    initial_state: Optional[RoutingState] = None
) -> Tuple[Any, RoutingState]:
    """
    Route runoff timeseries through network (JAX version).

    Args:
        runoff: Runoff from each GRU [n_timesteps, n_nodes] (m³/s)
        network: RiverNetwork with topology
        routing_params: RoutingParams for each reach
        dt: Timestep in seconds
        initial_state: Initial routing state

    Returns:
        Tuple of (outlet_timeseries, final_state)
    """
    if not HAS_JAX:
        return route_timeseries_numpy(runoff, network, routing_params, dt, initial_state)

    n_timesteps, n_nodes = runoff.shape
    n_edges = network.n_edges

    if initial_state is None:
        initial_state = create_initial_routing_state(n_edges, use_jax=True)

    def scan_fn(carry, t):
        prev_inflows, routing_state = carry

        curr_inflows = runoff[t]

        outlet_flow, new_state = route_network_step_jax(
            curr_inflows, prev_inflows, routing_state, routing_params,
            network.edge_from, network.edge_to, network.downstream_idx,
            network.upstream_indices, network.upstream_count,
            network.topo_order, network.node_areas, dt
        )

        return (curr_inflows, new_state), outlet_flow

    # Initial inflows (zeros for first timestep)
    init_inflows = jnp.zeros(n_nodes)

    (_, final_state), outlet_series = lax.scan(
        scan_fn,
        (init_inflows, initial_state),
        jnp.arange(n_timesteps, dtype=jnp.int32)
    )

    return outlet_series, final_state


def route_timeseries_numpy(
    runoff: np.ndarray,
    network,  # RiverNetwork
    routing_params: RoutingParams,
    dt: float = 3600.0,
    initial_state: Optional[RoutingState] = None
) -> Tuple[np.ndarray, RoutingState]:
    """
    NumPy version of timeseries routing.

    Args:
        runoff: Runoff from each GRU [n_timesteps, n_nodes] (m³/s)
        network: RiverNetwork with topology
        routing_params: RoutingParams for each reach
        dt: Timestep in seconds
        initial_state: Initial routing state

    Returns:
        Tuple of (outlet_timeseries, final_state)
    """
    n_timesteps, n_nodes = runoff.shape
    n_edges = network.n_edges

    if initial_state is None:
        initial_state = create_initial_routing_state(n_edges, use_jax=False)

    outlet_series = np.zeros(n_timesteps)
    routing_state = initial_state
    prev_inflows = np.zeros(n_nodes)

    for t in range(n_timesteps):
        curr_inflows = runoff[t]

        outlet_flow, routing_state = route_network_step_numpy(
            curr_inflows, prev_inflows, routing_state, routing_params,
            network.edge_from, network.edge_to, network.downstream_idx,
            network.upstream_indices, network.upstream_count,
            network.topo_order, network.node_areas, dt
        )

        outlet_series[t] = outlet_flow
        prev_inflows = curr_inflows

    return outlet_series, routing_state


def runoff_mm_to_cms(
    runoff_mm: Any,
    areas_m2: Any,
    dt_seconds: float = 3600.0
) -> Any:
    """
    Convert runoff from mm/timestep to m³/s.

    Args:
        runoff_mm: Runoff depth (mm/timestep) [n_timesteps, n_nodes] or [n_nodes]
        areas_m2: Contributing area for each node (m²) [n_nodes]
        dt_seconds: Timestep duration (seconds)

    Returns:
        Runoff in m³/s
    """
    xp = jnp if HAS_JAX else np

    # mm/timestep to m/timestep
    runoff_m = runoff_mm / 1000.0

    # m/timestep * m² = m³/timestep
    # m³/timestep / seconds = m³/s
    if runoff_mm.ndim == 2:
        # [n_timesteps, n_nodes]
        runoff_cms = runoff_m * areas_m2[xp.newaxis, :] / dt_seconds
    else:
        # [n_nodes]
        runoff_cms = runoff_m * areas_m2 / dt_seconds

    return runoff_cms


def cms_to_runoff_mm(
    runoff_cms: Any,
    areas_m2: Any,
    dt_seconds: float = 3600.0
) -> Any:
    """
    Convert runoff from m³/s to mm/timestep.

    Args:
        runoff_cms: Runoff (m³/s)
        areas_m2: Contributing area (m²)
        dt_seconds: Timestep duration (seconds)

    Returns:
        Runoff depth (mm/timestep)
    """
    xp = jnp if HAS_JAX else np

    # m³/s * seconds = m³/timestep
    runoff_m3 = runoff_cms * dt_seconds

    # m³/timestep / m² = m/timestep
    if runoff_cms.ndim == 2:
        runoff_m = runoff_m3 / areas_m2[xp.newaxis, :]
    else:
        runoff_m = runoff_m3 / areas_m2

    # m to mm
    runoff_mm = runoff_m * 1000.0

    return runoff_mm
