"""
River Network Graph Module.

Provides graph-based representation of river networks for distributed
hydrological modeling. Loads network topology from shapefiles and builds
JAX-compatible data structures for differentiable routing.

The network is represented as a Directed Acyclic Graph (DAG) where:
- Nodes represent GRUs (Grouped Response Units) or subcatchments
- Edges represent river reaches connecting upstream to downstream nodes

Key Features:
- Topological sorting for correct evaluation order
- Sparse adjacency representation for memory efficiency
- JAX-compatible arrays for autodiff through the network
- Support for multiple outlets (disconnected subnetworks)
"""

from typing import List, Optional, Tuple, NamedTuple, Any
from pathlib import Path
import warnings

import numpy as np

try:
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    lax = None


class RiverNetwork(NamedTuple):
    """
    JAX-compatible river network representation.

    All arrays are designed for efficient JAX operations (vmap, scan, grad).

    Attributes:
        n_nodes: Number of nodes (GRUs) in the network
        n_edges: Number of edges (reaches) in the network

        # Node properties
        node_ids: Original GRU/HRU IDs from shapefile [n_nodes]
        node_areas: Contributing area per node in m² [n_nodes]
        topo_order: Topological sort order (upstream first) [n_nodes]
        topo_levels: Level in DAG (0 = headwaters) [n_nodes]

        # Edge properties (reach characteristics)
        edge_from: Source node index for each edge [n_edges]
        edge_to: Destination node index for each edge [n_edges]
        edge_lengths: Reach length in meters [n_edges]
        edge_slopes: Reach slope (m/m) [n_edges]
        edge_widths: Channel width in meters [n_edges]
        edge_mannings: Manning's n roughness [n_edges]

        # Connectivity structure
        downstream_idx: Index of downstream node (-1 if outlet) [n_nodes]
        upstream_count: Number of upstream nodes [n_nodes]
        upstream_indices: Padded upstream node indices [n_nodes, max_upstream]

        # Outlet information
        outlet_idx: Index of the outlet node(s) [n_outlets]
    """
    n_nodes: int
    n_edges: int

    # Node properties
    node_ids: Any  # jnp.ndarray or np.ndarray
    node_areas: Any
    topo_order: Any
    topo_levels: Any

    # Edge properties
    edge_from: Any
    edge_to: Any
    edge_lengths: Any
    edge_slopes: Any
    edge_widths: Any
    edge_mannings: Any

    # Connectivity
    downstream_idx: Any
    upstream_count: Any
    upstream_indices: Any

    # Outlets
    outlet_idx: Any


class NetworkBuilder:
    """
    Builds river network graph from shapefiles.

    Expected shapefile structure:
    - River network shapefile with:
        - Segment ID column (e.g., 'seg_id', 'COMID', 'reach_id')
        - Downstream segment ID column (e.g., 'down_seg', 'toNode', 'NextDownID')
        - Length column (optional, computed from geometry if missing)
        - Slope column (optional, defaults used if missing)
        - Width column (optional, estimated from area if missing)

    - Catchment/GRU shapefile with:
        - GRU ID column matching segment IDs
        - Area column or geometry for area computation
    """

    # Common column name variants
    SEGMENT_ID_COLS = ['seg_id', 'COMID', 'reach_id', 'segid', 'HRU_ID', 'hru_id', 'HRUID']
    DOWNSTREAM_COLS = ['down_seg', 'toNode', 'NextDownID', 'tosegment', 'downstream', 'downSegId']
    LENGTH_COLS = ['length', 'Length', 'LENGTHKM', 'length_m', 'seg_length']
    SLOPE_COLS = ['slope', 'Slope', 'SLOPE', 'seg_slope', 'channel_slope']
    WIDTH_COLS = ['width', 'Width', 'WIDTH', 'channel_width', 'bankfull_width']
    AREA_COLS = ['area', 'Area', 'AREA', 'area_sqkm', 'HRU_area', 'catchment_area']
    MANNINGS_COLS = ['mannings_n', 'n_value', 'roughness', 'manning']

    # Default values for missing properties
    DEFAULT_SLOPE = 0.001  # 0.1% slope
    DEFAULT_MANNINGS = 0.035  # Typical natural channel
    DEFAULT_WIDTH_COEF = 2.5  # Width = coef * area^0.5 (empirical)
    DEFAULT_WIDTH_EXP = 0.4

    def __init__(
        self,
        river_network_path: Optional[Path] = None,
        catchment_path: Optional[Path] = None,
        segment_id_col: Optional[str] = None,
        downstream_col: Optional[str] = None,
        area_col: Optional[str] = None,
        use_jax: bool = True
    ):
        """
        Initialize network builder.

        Args:
            river_network_path: Path to river network shapefile
            catchment_path: Path to catchment/GRU shapefile
            segment_id_col: Column name for segment IDs (auto-detected if None)
            downstream_col: Column name for downstream IDs (auto-detected if None)
            area_col: Column name for catchment areas (auto-detected if None)
            use_jax: Whether to return JAX arrays (requires JAX)
        """
        self.river_network_path = river_network_path
        self.catchment_path = catchment_path
        self.segment_id_col = segment_id_col
        self.downstream_col = downstream_col
        self.area_col = area_col
        self.use_jax = use_jax and HAS_JAX

    def _find_column(self, gdf, candidates: List[str], required: bool = True) -> Optional[str]:
        """Find matching column from list of candidates."""
        for col in candidates:
            if col in gdf.columns:
                return col
        if required:
            raise ValueError(f"Could not find column. Tried: {candidates}. Available: {list(gdf.columns)}")
        return None

    def build_from_shapefiles(
        self,
        river_network_path: Optional[Path] = None,
        catchment_path: Optional[Path] = None
    ) -> RiverNetwork:
        """
        Build river network from shapefiles.

        Args:
            river_network_path: Override path to river network shapefile
            catchment_path: Override path to catchment shapefile

        Returns:
            RiverNetwork namedtuple with all network properties
        """
        import geopandas as gpd

        river_path = river_network_path or self.river_network_path
        catch_path = catchment_path or self.catchment_path

        if river_path is None:
            raise ValueError("River network shapefile path required")

        # Load river network
        river_gdf = gpd.read_file(river_path)

        # Load catchment if available
        catch_gdf = None
        if catch_path and Path(catch_path).exists():
            catch_gdf = gpd.read_file(catch_path)

        return self._build_network(river_gdf, catch_gdf)

    def build_from_geodataframes(
        self,
        river_gdf,
        catchment_gdf = None
    ) -> RiverNetwork:
        """
        Build network directly from GeoDataFrames.

        Args:
            river_gdf: GeoDataFrame with river reaches
            catchment_gdf: Optional GeoDataFrame with catchment polygons

        Returns:
            RiverNetwork namedtuple
        """
        return self._build_network(river_gdf, catchment_gdf)

    def _build_network(self, river_gdf, catchment_gdf) -> RiverNetwork:
        """Internal network building logic."""

        # Identify columns
        seg_id_col = self.segment_id_col or self._find_column(river_gdf, self.SEGMENT_ID_COLS)
        down_col = self.downstream_col or self._find_column(river_gdf, self.DOWNSTREAM_COLS)

        # Get segment IDs and downstream IDs
        seg_ids = river_gdf[seg_id_col].values
        down_ids = river_gdf[down_col].values

        n_nodes = len(seg_ids)

        # Create ID to index mapping
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        # Build downstream index array (-1 for outlets)
        downstream_idx = np.full(n_nodes, -1, dtype=np.int32)
        for i, down_id in enumerate(down_ids):
            if down_id in id_to_idx:
                downstream_idx[i] = id_to_idx[down_id]
            # else: remains -1 (outlet or external connection)

        # Build edge arrays
        edges_from = []
        edges_to = []
        for i, down_id in enumerate(down_ids):
            if down_id in id_to_idx:
                edges_from.append(i)
                edges_to.append(id_to_idx[down_id])

        edge_from = np.array(edges_from, dtype=np.int32)
        edge_to = np.array(edges_to, dtype=np.int32)
        n_edges = len(edge_from)

        # Build upstream connectivity (for flow accumulation)
        upstream_count = np.zeros(n_nodes, dtype=np.int32)
        for to_idx in edge_to:
            upstream_count[to_idx] += 1

        max_upstream = max(upstream_count.max(), 1)
        upstream_indices = np.full((n_nodes, max_upstream), -1, dtype=np.int32)
        upstream_counters = np.zeros(n_nodes, dtype=np.int32)

        for from_idx, to_idx in zip(edge_from, edge_to):
            count = upstream_counters[to_idx]
            upstream_indices[to_idx, count] = from_idx
            upstream_counters[to_idx] += 1

        # Topological sort (Kahn's algorithm)
        topo_order, topo_levels = self._topological_sort(downstream_idx, upstream_count.copy())

        # Extract reach properties
        edge_lengths = self._get_edge_lengths(river_gdf, edge_from)
        edge_slopes = self._get_edge_property(river_gdf, self.SLOPE_COLS, edge_from, self.DEFAULT_SLOPE)
        edge_widths = self._get_edge_widths(river_gdf, catchment_gdf, edge_from, seg_id_col)
        edge_mannings = self._get_edge_property(river_gdf, self.MANNINGS_COLS, edge_from, self.DEFAULT_MANNINGS)

        # Get node areas
        node_areas = self._get_node_areas(river_gdf, catchment_gdf, seg_id_col)

        # Find outlets
        outlet_idx = np.where(downstream_idx == -1)[0].astype(np.int32)

        # Convert to JAX arrays if requested
        if self.use_jax:
            return RiverNetwork(
                n_nodes=n_nodes,
                n_edges=n_edges,
                node_ids=jnp.array(seg_ids),
                node_areas=jnp.array(node_areas),
                topo_order=jnp.array(topo_order),
                topo_levels=jnp.array(topo_levels),
                edge_from=jnp.array(edge_from),
                edge_to=jnp.array(edge_to),
                edge_lengths=jnp.array(edge_lengths),
                edge_slopes=jnp.array(edge_slopes),
                edge_widths=jnp.array(edge_widths),
                edge_mannings=jnp.array(edge_mannings),
                downstream_idx=jnp.array(downstream_idx),
                upstream_count=jnp.array(upstream_count),
                upstream_indices=jnp.array(upstream_indices),
                outlet_idx=jnp.array(outlet_idx),
            )
        else:
            return RiverNetwork(
                n_nodes=n_nodes,
                n_edges=n_edges,
                node_ids=seg_ids,
                node_areas=node_areas,
                topo_order=topo_order,
                topo_levels=topo_levels,
                edge_from=edge_from,
                edge_to=edge_to,
                edge_lengths=edge_lengths,
                edge_slopes=edge_slopes,
                edge_widths=edge_widths,
                edge_mannings=edge_mannings,
                downstream_idx=downstream_idx,
                upstream_count=upstream_count,
                upstream_indices=upstream_indices,
                outlet_idx=outlet_idx,
            )

    def _topological_sort(
        self,
        downstream_idx: np.ndarray,
        in_degree: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform topological sort using Kahn's algorithm.

        Returns nodes ordered from upstream (headwaters) to downstream (outlet).
        Also returns the level of each node in the DAG.

        Args:
            downstream_idx: Downstream node index for each node
            in_degree: Number of upstream nodes for each node

        Returns:
            Tuple of (topo_order, topo_levels)
        """
        n_nodes = len(downstream_idx)

        # Find headwater nodes (no upstream connections)
        queue = list(np.where(in_degree == 0)[0])

        topo_order = []
        topo_levels = np.zeros(n_nodes, dtype=np.int32)

        level_queue = [(node, 0) for node in queue]

        while level_queue:
            node, level = level_queue.pop(0)
            topo_order.append(node)
            topo_levels[node] = level

            # Process downstream node
            down_node = downstream_idx[node]
            if down_node >= 0:
                in_degree[down_node] -= 1
                if in_degree[down_node] == 0:
                    level_queue.append((down_node, level + 1))

        if len(topo_order) != n_nodes:
            warnings.warn(
                f"Topological sort incomplete: {len(topo_order)}/{n_nodes} nodes. "
                "Network may contain cycles or disconnected components."
            )
            # Add any remaining nodes
            remaining = set(range(n_nodes)) - set(topo_order)
            topo_order.extend(remaining)
            for node in remaining:
                topo_levels[node] = topo_levels.max() + 1

        return np.array(topo_order, dtype=np.int32), topo_levels

    def _get_edge_lengths(self, river_gdf, edge_from: np.ndarray) -> np.ndarray:
        """Get reach lengths from shapefile or compute from geometry."""
        length_col = self._find_column(river_gdf, self.LENGTH_COLS, required=False)

        if length_col:
            lengths = river_gdf[length_col].values[edge_from]
            # Convert km to m if needed
            if 'km' in length_col.lower() or np.nanmean(lengths) < 100:
                lengths = lengths * 1000
            return lengths.astype(np.float64)

        # Compute from geometry
        # Ensure projected CRS for accurate length
        if river_gdf.crs and river_gdf.crs.is_geographic:
            river_gdf = river_gdf.to_crs(epsg=3857)  # Web Mercator for global

        lengths = river_gdf.geometry.length.values[edge_from]
        return lengths.astype(np.float64)

    def _get_edge_property(
        self,
        river_gdf,
        col_candidates: List[str],
        edge_from: np.ndarray,
        default: float
    ) -> np.ndarray:
        """Get edge property from shapefile or use default."""
        col = self._find_column(river_gdf, col_candidates, required=False)

        if col:
            values = river_gdf[col].values[edge_from]
            # Replace NaN with default
            values = np.where(np.isnan(values), default, values)
            return values.astype(np.float64)

        return np.full(len(edge_from), default, dtype=np.float64)

    def _get_edge_widths(
        self,
        river_gdf,
        catchment_gdf,
        edge_from: np.ndarray,
        seg_id_col: str
    ) -> np.ndarray:
        """Get channel widths from shapefile or estimate from catchment area."""
        width_col = self._find_column(river_gdf, self.WIDTH_COLS, required=False)

        if width_col:
            widths = river_gdf[width_col].values[edge_from]
            valid_mask = ~np.isnan(widths) & (widths > 0)
            if valid_mask.all():
                return widths.astype(np.float64)

        # Estimate from catchment area using Leopold-Maddock relation
        # W = a * A^b where A is drainage area
        if catchment_gdf is not None:
            areas = self._get_node_areas(river_gdf, catchment_gdf, seg_id_col)
            areas_km2 = areas / 1e6  # Convert m² to km²
            widths = self.DEFAULT_WIDTH_COEF * np.power(areas_km2, self.DEFAULT_WIDTH_EXP)
            return widths[edge_from].astype(np.float64)

        # Default width
        return np.full(len(edge_from), 10.0, dtype=np.float64)  # 10m default

    def _get_node_areas(
        self,
        river_gdf,
        catchment_gdf,
        seg_id_col: str
    ) -> np.ndarray:
        """Get catchment areas for each node."""
        n_nodes = len(river_gdf)

        # Try river_gdf first
        area_col = self._find_column(river_gdf, self.AREA_COLS, required=False)
        if area_col:
            areas = river_gdf[area_col].values
            # Convert km² to m² if values are small
            if np.nanmean(areas) < 1000:  # Likely km²
                areas = areas * 1e6
            return areas.astype(np.float64)

        # Try catchment_gdf
        if catchment_gdf is not None:
            area_col = self._find_column(catchment_gdf, self.AREA_COLS, required=False)
            if area_col:
                # Need to match by segment ID
                catch_id_col = self._find_column(catchment_gdf, self.SEGMENT_ID_COLS, required=False)
                if catch_id_col:
                    catch_areas = dict(zip(
                        catchment_gdf[catch_id_col],
                        catchment_gdf[area_col]
                    ))
                    areas = np.array([
                        catch_areas.get(sid, np.nan)
                        for sid in river_gdf[seg_id_col]
                    ])
                    if np.nanmean(areas) < 1000:
                        areas = areas * 1e6
                    return areas.astype(np.float64)

            # Compute from geometry
            if catchment_gdf.crs and catchment_gdf.crs.is_geographic:
                catchment_gdf = catchment_gdf.to_crs(epsg=3857)

            areas = catchment_gdf.geometry.area.values
            return areas.astype(np.float64)

        # Default: equal areas
        warnings.warn("Could not determine catchment areas, using uniform 100 km²")
        return np.full(n_nodes, 100e6, dtype=np.float64)  # 100 km² default


def create_synthetic_network(
    n_nodes: int = 5,
    topology: str = 'linear',
    use_jax: bool = True
) -> RiverNetwork:
    """
    Create a synthetic river network for testing.

    Args:
        n_nodes: Number of nodes (GRUs)
        topology: Network topology type:
            - 'linear': Simple chain (1 -> 2 -> 3 -> ... -> outlet)
            - 'binary_tree': Binary tree (balanced)
            - 'fishbone': Main stem with tributaries
        use_jax: Whether to return JAX arrays

    Returns:
        RiverNetwork with synthetic properties
    """
    if topology == 'linear':
        return _create_linear_network(n_nodes, use_jax)
    elif topology == 'binary_tree':
        return _create_binary_tree_network(n_nodes, use_jax)
    elif topology == 'fishbone':
        return _create_fishbone_network(n_nodes, use_jax)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def _create_linear_network(n_nodes: int, use_jax: bool) -> RiverNetwork:
    """Create simple linear network."""
    # Node 0 is outlet, others flow downstream sequentially
    # Order: n-1 -> n-2 -> ... -> 1 -> 0 (outlet)

    node_ids = np.arange(n_nodes)
    downstream_idx = np.arange(-1, n_nodes - 1)  # [-1, 0, 1, 2, ...]
    downstream_idx[0] = -1  # Node 0 is outlet

    # Reorder so node n-1 flows to n-2, etc.
    downstream_idx = np.concatenate([[-1], np.arange(n_nodes - 1)])

    # Edges
    n_edges = n_nodes - 1
    edge_from = np.arange(1, n_nodes)  # [1, 2, ..., n-1]
    edge_to = np.arange(0, n_nodes - 1)  # [0, 1, ..., n-2]

    # Properties
    node_areas = np.linspace(10e6, 100e6, n_nodes)  # 10-100 km²
    edge_lengths = np.full(n_edges, 5000.0)  # 5 km reaches
    edge_slopes = np.full(n_edges, 0.001)  # 0.1%
    edge_widths = np.linspace(5.0, 20.0, n_edges)  # 5-20 m
    edge_mannings = np.full(n_edges, 0.035)

    # Upstream connectivity
    upstream_count = np.zeros(n_nodes, dtype=np.int32)
    upstream_count[:-1] = 1  # All except last node have 1 upstream
    upstream_indices = np.full((n_nodes, 1), -1, dtype=np.int32)
    for i in range(1, n_nodes):
        upstream_indices[i-1, 0] = i

    # Topological order (upstream to downstream)
    topo_order = np.arange(n_nodes - 1, -1, -1)  # [n-1, n-2, ..., 0]
    topo_levels = np.arange(n_nodes - 1, -1, -1)  # Same as order for linear

    outlet_idx = np.array([0])

    arrs = (
        node_ids, node_areas, topo_order, topo_levels,
        edge_from, edge_to, edge_lengths, edge_slopes, edge_widths, edge_mannings,
        downstream_idx, upstream_count, upstream_indices, outlet_idx
    )

    if use_jax and HAS_JAX:
        arrs = tuple(jnp.array(a) for a in arrs)

    return RiverNetwork(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_ids=arrs[0],
        node_areas=arrs[1],
        topo_order=arrs[2],
        topo_levels=arrs[3],
        edge_from=arrs[4],
        edge_to=arrs[5],
        edge_lengths=arrs[6],
        edge_slopes=arrs[7],
        edge_widths=arrs[8],
        edge_mannings=arrs[9],
        downstream_idx=arrs[10],
        upstream_count=arrs[11],
        upstream_indices=arrs[12],
        outlet_idx=arrs[13],
    )


def _create_binary_tree_network(n_nodes: int, use_jax: bool) -> RiverNetwork:
    """Create balanced binary tree network."""
    # For binary tree, we need 2^k - 1 nodes for a complete tree
    # Outlet is node 0, children of node i are 2i+1 and 2i+2

    node_ids = np.arange(n_nodes)
    downstream_idx = np.zeros(n_nodes, dtype=np.int32)
    downstream_idx[0] = -1  # Outlet

    for i in range(1, n_nodes):
        downstream_idx[i] = (i - 1) // 2  # Parent node

    # Build edges
    edges_from = []
    edges_to = []
    for i in range(1, n_nodes):
        edges_from.append(i)
        edges_to.append(downstream_idx[i])

    edge_from = np.array(edges_from, dtype=np.int32)
    edge_to = np.array(edges_to, dtype=np.int32)
    n_edges = len(edge_from)

    # Upstream connectivity
    upstream_count = np.zeros(n_nodes, dtype=np.int32)
    for to_idx in edge_to:
        upstream_count[to_idx] += 1

    max_upstream = max(upstream_count.max(), 1)
    upstream_indices = np.full((n_nodes, max_upstream), -1, dtype=np.int32)
    counters = np.zeros(n_nodes, dtype=np.int32)
    for from_idx, to_idx in zip(edge_from, edge_to):
        upstream_indices[to_idx, counters[to_idx]] = from_idx
        counters[to_idx] += 1

    # Topological sort (BFS from leaves)
    in_degree = upstream_count.copy()
    topo_order = []
    topo_levels = np.zeros(n_nodes, dtype=np.int32)
    queue = [(i, 0) for i in range(n_nodes) if in_degree[i] == 0]

    while queue:
        node, level = queue.pop(0)
        topo_order.append(node)
        topo_levels[node] = level
        down = downstream_idx[node]
        if down >= 0:
            in_degree[down] -= 1
            if in_degree[down] == 0:
                queue.append((down, level + 1))

    topo_order = np.array(topo_order, dtype=np.int32)

    # Properties (scale with depth)
    depth = int(np.ceil(np.log2(n_nodes + 1)))
    node_areas = np.array([10e6 * (2 ** (depth - topo_levels[i])) for i in range(n_nodes)])
    edge_lengths = np.full(n_edges, 3000.0)
    edge_slopes = np.full(n_edges, 0.002)
    edge_widths = np.array([5.0 + 3.0 * topo_levels[edge_to[i]] for i in range(n_edges)])
    edge_mannings = np.full(n_edges, 0.035)

    outlet_idx = np.array([0])

    arrs = (
        node_ids, node_areas, topo_order, topo_levels,
        edge_from, edge_to, edge_lengths, edge_slopes, edge_widths, edge_mannings,
        downstream_idx, upstream_count, upstream_indices, outlet_idx
    )

    if use_jax and HAS_JAX:
        arrs = tuple(jnp.array(a) for a in arrs)

    return RiverNetwork(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_ids=arrs[0],
        node_areas=arrs[1],
        topo_order=arrs[2],
        topo_levels=arrs[3],
        edge_from=arrs[4],
        edge_to=arrs[5],
        edge_lengths=arrs[6],
        edge_slopes=arrs[7],
        edge_widths=arrs[8],
        edge_mannings=arrs[9],
        downstream_idx=arrs[10],
        upstream_count=arrs[11],
        upstream_indices=arrs[12],
        outlet_idx=arrs[13],
    )


def _create_fishbone_network(n_nodes: int, use_jax: bool) -> RiverNetwork:
    """Create fishbone network (main stem with tributaries)."""
    # Main stem: nodes 0, 1, 2, ... (n//2)
    # Tributaries join at alternating main stem nodes

    n_main = n_nodes // 2 + 1
    n_tribs = n_nodes - n_main

    node_ids = np.arange(n_nodes)
    downstream_idx = np.zeros(n_nodes, dtype=np.int32)
    downstream_idx[0] = -1  # Outlet

    # Main stem connectivity
    for i in range(1, n_main):
        downstream_idx[i] = i - 1

    # Tributary connectivity
    for i in range(n_tribs):
        trib_node = n_main + i
        main_node = min((i + 1) * 2, n_main - 1)  # Join alternating main nodes
        downstream_idx[trib_node] = main_node

    # Build edges
    edges_from = []
    edges_to = []
    for i in range(n_nodes):
        if downstream_idx[i] >= 0:
            edges_from.append(i)
            edges_to.append(downstream_idx[i])

    edge_from = np.array(edges_from, dtype=np.int32)
    edge_to = np.array(edges_to, dtype=np.int32)
    n_edges = len(edge_from)

    # Upstream connectivity
    upstream_count = np.zeros(n_nodes, dtype=np.int32)
    for to_idx in edge_to:
        upstream_count[to_idx] += 1

    max_upstream = max(upstream_count.max(), 1)
    upstream_indices = np.full((n_nodes, max_upstream), -1, dtype=np.int32)
    counters = np.zeros(n_nodes, dtype=np.int32)
    for from_idx, to_idx in zip(edge_from, edge_to):
        upstream_indices[to_idx, counters[to_idx]] = from_idx
        counters[to_idx] += 1

    # Topological sort
    in_degree = upstream_count.copy()
    topo_order = []
    topo_levels = np.zeros(n_nodes, dtype=np.int32)
    queue = [(i, 0) for i in range(n_nodes) if in_degree[i] == 0]

    while queue:
        node, level = queue.pop(0)
        topo_order.append(node)
        topo_levels[node] = level
        down = downstream_idx[node]
        if down >= 0:
            in_degree[down] -= 1
            if in_degree[down] == 0:
                queue.append((down, level + 1))

    topo_order = np.array(topo_order, dtype=np.int32)

    # Properties
    node_areas = np.zeros(n_nodes)
    node_areas[:n_main] = np.linspace(50e6, 200e6, n_main)  # Main stem: larger
    node_areas[n_main:] = 30e6  # Tributaries: smaller

    edge_lengths = np.full(n_edges, 4000.0)
    edge_slopes = np.full(n_edges, 0.0015)
    edge_widths = np.where(edge_from < n_main, 15.0, 8.0)  # Main wider
    edge_mannings = np.full(n_edges, 0.035)

    outlet_idx = np.array([0])

    arrs = (
        node_ids, node_areas, topo_order, topo_levels,
        edge_from, edge_to, edge_lengths, edge_slopes, edge_widths, edge_mannings,
        downstream_idx, upstream_count, upstream_indices, outlet_idx
    )

    if use_jax and HAS_JAX:
        arrs = tuple(jnp.array(a) for a in arrs)

    return RiverNetwork(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_ids=arrs[0],
        node_areas=arrs[1],
        topo_order=arrs[2],
        topo_levels=arrs[3],
        edge_from=arrs[4],
        edge_to=arrs[5],
        edge_lengths=arrs[6],
        edge_slopes=arrs[7],
        edge_widths=arrs[8],
        edge_mannings=arrs[9],
        downstream_idx=arrs[10],
        upstream_count=arrs[11],
        upstream_indices=arrs[12],
        outlet_idx=arrs[13],
    )
