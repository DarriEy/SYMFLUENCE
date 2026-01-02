"""
Graph-based operations for river network analysis.

Provides NetworkX graph construction and upstream basin tracing.
Eliminates code duplication across GeofabricDelineator and GeofabricSubsetter.

Supports multiple hydrofabric formats:
- MERIT: COMID with up1, up2, up3 columns
- TDX: streamID/LINKNO with USLINKNO1, USLINKNO2 columns
- NWS: COMID with toCOMID column (reverse direction)

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any, Dict, Set
import networkx as nx
import geopandas as gpd


class RiverGraphProcessor:
    """
    Graph operations for river network topology.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def build_river_graph(
        rivers: gpd.GeoDataFrame,
        fabric_config: Dict[str, Any]
    ) -> nx.DiGraph:
        """
        Build a directed graph representing the river network.

        The graph direction depends on the hydrofabric type:
        - MERIT/TDX: Edges point downstream (upstream → current)
        - NWS: Edges point upstream (current → downstream) - reverse direction

        Args:
            rivers: River network GeoDataFrame
            fabric_config: Configuration dict with keys:
                - 'river_id_col': Column name for river segment ID
                - 'upstream_cols': List of upstream column names
                - 'upstream_default': Default value indicating no upstream link

        Returns:
            Directed graph of the river network

        Example fabric_config:
            MERIT: {'river_id_col': 'COMID', 'upstream_cols': ['up1', 'up2', 'up3'],
                    'upstream_default': -9999}
            TDX: {'river_id_col': 'LINKNO', 'upstream_cols': ['USLINKNO1', 'USLINKNO2'],
                  'upstream_default': -9999}
            NWS: {'river_id_col': 'COMID', 'upstream_cols': ['toCOMID'],
                  'upstream_default': 0}
        """
        G = nx.DiGraph()

        for _, row in rivers.iterrows():
            current_basin = row[fabric_config['river_id_col']]

            for up_col in fabric_config['upstream_cols']:
                upstream_basin = row[up_col]

                # Skip if no upstream link
                if upstream_basin != fabric_config['upstream_default']:
                    # NWS uses reverse direction (toCOMID points downstream)
                    if fabric_config['upstream_cols'] == ['toCOMID']:
                        G.add_edge(current_basin, upstream_basin)
                    else:
                        # MERIT/TDX: upstream → current
                        G.add_edge(upstream_basin, current_basin)

        return G

    @staticmethod
    def find_upstream_basins(
        basin_id: Any,
        G: nx.DiGraph,
        logger: Any
    ) -> Set:
        """
        Find all upstream basins for a given basin ID.

        Uses NetworkX ancestors to trace all basins upstream of the given basin.
        The result includes the basin itself.

        Args:
            basin_id: ID of the basin to find upstream basins for
            G: Directed graph of the river network
            logger: Logger instance for warnings

        Returns:
            Set of upstream basin IDs (including the given basin)
        """
        if G.has_node(basin_id):
            # Get all ancestors (upstream basins)
            upstream_basins = nx.ancestors(G, basin_id)
            # Include the basin itself
            upstream_basins.add(basin_id)
        else:
            logger.warning(f"Basin ID {basin_id} not found in the river network.")
            upstream_basins = set()

        return upstream_basins
