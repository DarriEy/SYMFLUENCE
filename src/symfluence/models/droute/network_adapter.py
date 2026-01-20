"""
dRoute Network Adapter.

Handles conversion between mizuRoute-compatible topology format and dRoute's
internal network representation. Enables seamless switching between routing
models using the same preprocessed network topology.

The adapter supports:
- Loading mizuRoute NetCDF topology files
- Converting to dRoute's Python API format
- Writing dRoute-specific configuration files
- Validating network topology consistency
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import netCDF4 as nc4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class DRouteNetworkAdapter:
    """
    Adapter for converting mizuRoute topology to dRoute format.

    Supports loading topology from:
    - NetCDF files (mizuRoute format)
    - GeoJSON files
    - CSV files

    The dRoute network representation consists of:
    - Segment IDs and downstream connectivity
    - Segment properties (length, slope, width)
    - HRU-to-segment drainage mapping
    - Optional channel geometry parameters

    Example:
        >>> adapter = DRouteNetworkAdapter(logger)
        >>> network = adapter.load_topology('topology.nc')
        >>> droute_config = adapter.to_droute_format(network)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the network adapter.

        Args:
            logger: Logger instance for status messages
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_topology(
        self,
        topology_path: Path,
        format: str = 'netcdf'
    ) -> Dict[str, Any]:
        """
        Load network topology from file.

        Args:
            topology_path: Path to topology file
            format: File format ('netcdf', 'geojson', 'csv')

        Returns:
            Dictionary containing network topology data:
            - seg_ids: Segment IDs
            - down_seg_ids: Downstream segment IDs
            - slopes: Segment slopes
            - lengths: Segment lengths (m)
            - hru_ids: HRU IDs
            - hru_to_seg: HRU-to-segment mapping
            - hru_areas: HRU areas (m^2)

        Raises:
            FileNotFoundError: If topology file doesn't exist
            ValueError: If format is unsupported
        """
        topology_path = Path(topology_path)

        if not topology_path.exists():
            raise FileNotFoundError(f"Topology file not found: {topology_path}")

        if format == 'netcdf':
            return self._load_netcdf_topology(topology_path)
        elif format == 'geojson':
            return self._load_geojson_topology(topology_path)
        elif format == 'csv':
            return self._load_csv_topology(topology_path)
        else:
            raise ValueError(f"Unsupported topology format: {format}")

    def _load_netcdf_topology(self, path: Path) -> Dict[str, Any]:
        """Load mizuRoute-format NetCDF topology."""
        self.logger.debug(f"Loading NetCDF topology from {path}")

        if not HAS_XARRAY:
            if not HAS_NETCDF4:
                raise ImportError("Either xarray or netCDF4 required to load NetCDF topology")
            return self._load_netcdf_topology_nc4(path)

        with xr.open_dataset(path) as ds:
            topology = {
                'seg_ids': ds['segId'].values.astype(int),
                'down_seg_ids': ds['downSegId'].values.astype(int),
                'slopes': ds['slope'].values.astype(float),
                'lengths': ds['length'].values.astype(float),
                'hru_ids': ds['hruId'].values.astype(int),
                'hru_to_seg': ds['hruToSegId'].values.astype(int),
                'hru_areas': ds['area'].values.astype(float),
            }

            # Optional: load width if available
            if 'width' in ds:
                topology['widths'] = ds['width'].values.astype(float)

        self.logger.info(
            f"Loaded topology: {len(topology['seg_ids'])} segments, "
            f"{len(topology['hru_ids'])} HRUs"
        )

        return topology

    def _load_netcdf_topology_nc4(self, path: Path) -> Dict[str, Any]:
        """Fallback: Load NetCDF using netCDF4 directly."""
        with nc4.Dataset(path, 'r') as ds:
            topology = {
                'seg_ids': ds.variables['segId'][:].astype(int),
                'down_seg_ids': ds.variables['downSegId'][:].astype(int),
                'slopes': ds.variables['slope'][:].astype(float),
                'lengths': ds.variables['length'][:].astype(float),
                'hru_ids': ds.variables['hruId'][:].astype(int),
                'hru_to_seg': ds.variables['hruToSegId'][:].astype(int),
                'hru_areas': ds.variables['area'][:].astype(float),
            }

            if 'width' in ds.variables:
                topology['widths'] = ds.variables['width'][:].astype(float)

        return topology

    def _load_geojson_topology(self, path: Path) -> Dict[str, Any]:
        """Load GeoJSON topology (segments as LineStrings)."""
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas required to load GeoJSON topology")

        self.logger.debug(f"Loading GeoJSON topology from {path}")

        gdf = gpd.read_file(path)

        # Map column names (flexible naming)
        seg_id_col = self._find_column(gdf, ['segId', 'seg_id', 'SEGID', 'id'])
        down_seg_col = self._find_column(gdf, ['downSegId', 'down_seg_id', 'DOWNSEGID', 'downstream'])
        slope_col = self._find_column(gdf, ['slope', 'SLOPE'])
        length_col = self._find_column(gdf, ['length', 'LENGTH', 'len'])

        topology = {
            'seg_ids': gdf[seg_id_col].values.astype(int),
            'down_seg_ids': gdf[down_seg_col].values.astype(int),
            'slopes': gdf[slope_col].values.astype(float) if slope_col else np.full(len(gdf), 0.001),
            'lengths': gdf[length_col].values.astype(float) if length_col else gdf.geometry.length.values,
            'hru_ids': gdf[seg_id_col].values.astype(int),  # Default: same as seg_ids
            'hru_to_seg': gdf[seg_id_col].values.astype(int),  # Default: 1:1 mapping
            'hru_areas': np.full(len(gdf), 1e6),  # Default: 1 km^2
        }

        return topology

    def _load_csv_topology(self, path: Path) -> Dict[str, Any]:
        """Load CSV topology."""
        import pandas as pd

        self.logger.debug(f"Loading CSV topology from {path}")

        df = pd.read_csv(path)

        seg_id_col = self._find_column(df, ['segId', 'seg_id', 'SEGID', 'id'])
        down_seg_col = self._find_column(df, ['downSegId', 'down_seg_id', 'DOWNSEGID', 'downstream'])

        topology = {
            'seg_ids': df[seg_id_col].values.astype(int),
            'down_seg_ids': df[down_seg_col].values.astype(int),
            'slopes': df.get('slope', pd.Series(np.full(len(df), 0.001))).values.astype(float),
            'lengths': df.get('length', pd.Series(np.full(len(df), 1000.0))).values.astype(float),
            'hru_ids': df[seg_id_col].values.astype(int),
            'hru_to_seg': df[seg_id_col].values.astype(int),
            'hru_areas': df.get('area', pd.Series(np.full(len(df), 1e6))).values.astype(float),
        }

        return topology

    def _find_column(self, df, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def to_droute_format(
        self,
        topology: Dict[str, Any],
        routing_method: str = 'muskingum_cunge',
        routing_dt: int = 3600
    ) -> Dict[str, Any]:
        """
        Convert topology to dRoute's expected format.

        Args:
            topology: Network topology dictionary from load_topology()
            routing_method: Routing method to use
            routing_dt: Routing timestep in seconds

        Returns:
            Dictionary formatted for dRoute Python API
        """
        # Build adjacency list for downstream connectivity
        seg_ids = topology['seg_ids']
        down_seg_ids = topology['down_seg_ids']

        # Create ID-to-index mapping
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        # Build downstream index array (-1 for outlets)
        downstream_idx = np.full(len(seg_ids), -1, dtype=int)
        for i, down_id in enumerate(down_seg_ids):
            if down_id in id_to_idx:
                downstream_idx[i] = id_to_idx[down_id]

        # Find outlet segments
        outlet_mask = downstream_idx == -1
        outlet_indices = np.where(outlet_mask)[0].tolist()

        # Compute routing order (topological sort)
        routing_order = self._compute_routing_order(downstream_idx)

        droute_config = {
            'n_segments': len(seg_ids),
            'segment_ids': seg_ids.tolist(),
            'downstream_idx': downstream_idx.tolist(),
            'outlet_indices': outlet_indices,
            'routing_order': routing_order,
            'slopes': topology['slopes'].tolist(),
            'lengths': topology['lengths'].tolist(),
            'routing_method': routing_method,
            'routing_dt': routing_dt,
            'hru_ids': topology['hru_ids'].tolist(),
            'hru_to_seg_idx': [id_to_idx.get(sid, -1) for sid in topology['hru_to_seg']],
            'hru_areas': topology['hru_areas'].tolist(),
        }

        # Add optional width if available
        if 'widths' in topology:
            droute_config['widths'] = topology['widths'].tolist()
        else:
            # Estimate width from contributing area (Leopold & Maddock, 1953)
            # W = a * Q^b, approximate with sqrt(upstream_area)
            droute_config['widths'] = self._estimate_channel_widths(topology)

        return droute_config

    def _compute_routing_order(self, downstream_idx: np.ndarray) -> List[int]:
        """
        Compute topological routing order (upstream to downstream).

        Args:
            downstream_idx: Array of downstream segment indices (-1 for outlets)

        Returns:
            List of segment indices in routing order
        """
        n_segments = len(downstream_idx)

        # Count incoming edges (upstream connections)
        in_degree = np.zeros(n_segments, dtype=int)
        for down_idx in downstream_idx:
            if down_idx >= 0:
                in_degree[down_idx] += 1

        # Start with headwater segments (no upstream)
        queue = [i for i in range(n_segments) if in_degree[i] == 0]
        routing_order = []

        while queue:
            current = queue.pop(0)
            routing_order.append(current)

            down_idx = downstream_idx[current]
            if down_idx >= 0:
                in_degree[down_idx] -= 1
                if in_degree[down_idx] == 0:
                    queue.append(down_idx)

        return routing_order

    def _estimate_channel_widths(self, topology: Dict[str, Any]) -> List[float]:
        """
        Estimate channel widths from contributing areas.

        Uses empirical relationship: W ~ sqrt(A)
        """
        # Simple estimate: width proportional to sqrt of contributing area
        hru_to_seg = topology['hru_to_seg']
        hru_areas = topology['hru_areas']
        seg_ids = topology['seg_ids']

        # Sum contributing area for each segment
        seg_areas = {}
        for hru_id, seg_id, area in zip(topology['hru_ids'], hru_to_seg, hru_areas):
            if seg_id not in seg_areas:
                seg_areas[seg_id] = 0.0
            seg_areas[seg_id] += area

        # Estimate width: W = 4.8 * Q^0.5 where Q ~ sqrt(A)
        # Simplified: W ~ 2 * A^0.25 (in meters, A in m^2)
        widths = []
        for seg_id in seg_ids:
            area = seg_areas.get(seg_id, 1e6)  # Default 1 km^2
            width = max(1.0, 2.0 * (area ** 0.25))  # Min 1m width
            widths.append(width)

        return widths

    def write_droute_config(
        self,
        droute_config: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Write dRoute configuration to YAML file.

        Args:
            droute_config: Configuration from to_droute_format()
            output_path: Path to write YAML file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required to write dRoute config")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Separate network data from routing config
        network_config = {
            'network': {
                'n_segments': droute_config['n_segments'],
                'segment_ids': droute_config['segment_ids'],
                'downstream_idx': droute_config['downstream_idx'],
                'outlet_indices': droute_config['outlet_indices'],
                'routing_order': droute_config['routing_order'],
            },
            'geometry': {
                'slopes': droute_config['slopes'],
                'lengths': droute_config['lengths'],
                'widths': droute_config['widths'],
            },
            'hru_mapping': {
                'hru_ids': droute_config['hru_ids'],
                'hru_to_seg_idx': droute_config['hru_to_seg_idx'],
                'hru_areas': droute_config['hru_areas'],
            },
            'routing': {
                'method': droute_config['routing_method'],
                'dt': droute_config['routing_dt'],
            },
        }

        with open(output_path, 'w') as f:
            yaml.dump(network_config, f, default_flow_style=False)

        self.logger.info(f"Wrote dRoute config to {output_path}")

    def validate_topology(self, topology: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate network topology for common issues.

        Args:
            topology: Network topology dictionary

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        warnings = []
        is_valid = True

        seg_ids = set(topology['seg_ids'])
        down_seg_ids = topology['down_seg_ids']

        # Check for disconnected segments
        for i, down_id in enumerate(down_seg_ids):
            if down_id != 0 and down_id not in seg_ids:
                warnings.append(
                    f"Segment {topology['seg_ids'][i]} references non-existent "
                    f"downstream segment {down_id}"
                )
                is_valid = False

        # Check for cycles
        # Simple cycle detection using visited tracking
        visited = set()
        for start_idx in range(len(topology['seg_ids'])):
            path = set()
            idx = start_idx
            while idx >= 0 and idx not in visited:
                if idx in path:
                    warnings.append(
                        f"Cycle detected involving segment {topology['seg_ids'][idx]}"
                    )
                    is_valid = False
                    break
                path.add(idx)

                # Find downstream index
                down_id = down_seg_ids[idx]
                if down_id == 0:
                    break
                try:
                    idx = list(topology['seg_ids']).index(down_id)
                except ValueError:
                    break

            visited.update(path)

        # Check for zero/negative lengths
        for i, length in enumerate(topology['lengths']):
            if length <= 0:
                warnings.append(
                    f"Segment {topology['seg_ids'][i]} has invalid length: {length}"
                )

        # Check for zero/negative slopes
        for i, slope in enumerate(topology['slopes']):
            if slope <= 0:
                warnings.append(
                    f"Segment {topology['seg_ids'][i]} has invalid slope: {slope}"
                )

        return is_valid, warnings


__all__ = ['DRouteNetworkAdapter']
