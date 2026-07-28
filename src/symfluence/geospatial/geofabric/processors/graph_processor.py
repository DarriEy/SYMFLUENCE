# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Graph-based operations for river network analysis.

Provides NetworkX graph construction and upstream basin tracing.
Eliminates code duplication across GeofabricDelineator and GeofabricSubsetter.

Supports multiple hydrofabric formats:
- MERIT: COMID with up1, up2, up3 columns
- TDX: streamID/LINKNO with USLINKNO1, USLINKNO2 columns
- NWS: divide_id with toid column (reverse direction)

Refactored from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Set

import geopandas as gpd
import networkx as nx


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
        - NWS/TauDEM: Edges point upstream (current → downstream) if configured as 'downstream'

        Args:
            rivers: River network GeoDataFrame
            fabric_config: Configuration dict with keys:
                - 'river_id_col': Column name for river segment ID
                - 'upstream_cols': List of column names (upstream or downstream)
                - 'upstream_default': Default value indicating no link
                - 'direction': 'upstream' (default) or 'downstream'

        Returns:
            Directed graph of the river network
        """
        G = nx.DiGraph()

        # Determine flow direction handling
        # 'upstream': upstream_cols contain IDs of upstream segments (flow: upstream -> current)
        # 'downstream': upstream_cols contain IDs of downstream segments (flow: current -> downstream)
        direction = fabric_config.get('direction', 'upstream')

        # Auto-detect downstream-pointer columns (NWS toCOMID, HydroSHEDS NEXT_DOWN)
        downstream_cols = {'toCOMID', 'toid', 'NEXT_DOWN'}
        if fabric_config.get('upstream_cols') and set(fabric_config['upstream_cols']) <= downstream_cols:
            direction = 'downstream'

        for _, row in rivers.iterrows():
            current_basin = row[fabric_config['river_id_col']]
            G.add_node(current_basin)

            for up_col in fabric_config['upstream_cols']:
                linked_basin = row[up_col]

                # Skip if no link
                if linked_basin != fabric_config['upstream_default']:
                    if direction == 'downstream':
                        # Flow: current -> linked (downstream)
                        # We want the graph edges to represent flow direction?
                        # Usually river graphs are directed downstream.
                        # Wait, find_upstream_basins uses nx.ancestors.
                        # nx.ancestors(G, n) returns all nodes having a path to n.
                        # If edges are upstream -> downstream (A -> B), then ancestors of B includes A.
                        # So G should be directed A -> B (downstream).

                        # If direction is 'downstream' (current -> linked_basin),
                        # then we add edge (current, linked).
                        G.add_edge(current_basin, linked_basin)
                    else:
                        # If direction is 'upstream' (linked_basin -> current),
                        # then we add edge (linked, current).
                        G.add_edge(linked_basin, current_basin)

        return G

    @staticmethod
    def drop_degenerate_reaches(
        rivers: gpd.GeoDataFrame,
        id_col: str = 'LINKNO',
        ds_col: str = 'DSLINKNO',
        us_cols: Sequence[str] = ('USLINKNO1', 'USLINKNO2'),
        null_value: int = -1,
        logger: Any = None,
    ) -> gpd.GeoDataFrame:
        """Remove zero-length connector reaches, rewiring topology around them.

        TauDEM encodes the stream network as a strictly *binary* tree: every link
        carries at most two upstream links (``USLINKNO1``/``USLINKNO2``). Where
        three or more streams meet at a single point, that encoding is impossible,
        so TauDEM splits the confluence into two binary junctions joined by a
        **zero-length link** — start point == end point, ``Length``, ``Slope`` and
        ``StraightL`` all 0, and upstream contributing area equal to downstream.

        Such a link is a topological placeholder, not a river reach. It has no
        catchment polygon of its own, so it breaks the one-segment-per-GRU
        invariant (the Iceland national domain delineates 1,894 GRUs but 1,895
        TauDEM links), and it is actively harmful downstream: routing schemes that
        divide by reach length to obtain a celerity or travel time see a zero
        denominator.

        Dropping the link alone would disconnect everything above it from the
        outlet, so its upstream links are first rewired to its downstream link.
        Chains of consecutive connectors resolve transitively.

        ``us_cols`` are repaired on a best-effort basis. Collapsing a connector can
        leave a junction with three upstream links, which the two TauDEM slots
        cannot represent; those slots are then set to ``null_value`` and a warning
        is logged. This is safe for delineated geofabrics — their topology is read
        from ``ds_col`` (see :meth:`_get_fabric_config`) — but callers that consume
        the binary-tree fields directly should not rely on them afterwards.

        Args:
            rivers: River network GeoDataFrame (TauDEM ``basin-streams`` layout).
            id_col: Segment ID column.
            ds_col: Downstream-segment pointer column.
            us_cols: Upstream-segment pointer columns to repair, if present.
            null_value: Sentinel meaning "no link" (TauDEM uses -1).
            logger: Optional logger.

        Returns:
            The network with connector reaches removed and topology rewired. The
            input is returned unchanged when no degenerate reach is present.
        """
        if rivers is None or len(rivers) == 0:
            return rivers
        if id_col not in rivers.columns or ds_col not in rivers.columns:
            return rivers

        # Degeneracy is a collapsed bounding box: every vertex at one point, so
        # minx == maxx and miny == maxy. Equivalent to zero length but computed
        # without .length, which warns on geographic CRSs (and would need a
        # reprojection to silence, for a test whose answer is CRS-independent).
        geom = rivers.geometry
        bounds = geom.bounds
        is_degenerate = (
            geom.isna()
            | geom.is_empty
            | ((bounds['minx'] == bounds['maxx']) & (bounds['miny'] == bounds['maxy']))
        )
        if not is_degenerate.any():
            return rivers

        drop_ids = set(rivers.loc[is_degenerate, id_col])
        ds_map = dict(zip(rivers[id_col], rivers[ds_col]))

        def _resolve(seg):
            """Follow ds pointers until reaching a surviving link (cycle-safe)."""
            seen = set()
            while seg in drop_ids:
                if seg in seen:  # malformed input: connectors pointing in a loop
                    return null_value
                seen.add(seg)
                seg = ds_map.get(seg, null_value)
            return seg

        cleaned = rivers.loc[~is_degenerate].copy()
        cleaned[ds_col] = cleaned[ds_col].map(_resolve).astype(rivers[ds_col].dtype)

        # Repair the binary-tree upstream slots on links that referenced a dropped
        # connector. Recomputed from the rewired ds pointers, so the two views of
        # the topology cannot disagree.
        present_us = [c for c in us_cols if c in cleaned.columns]
        if present_us:
            affected = cleaned[present_us].isin(drop_ids).any(axis=1)
            if affected.any():
                overflowed = []
                for idx in cleaned.index[affected]:
                    link_id = cleaned.at[idx, id_col]
                    upstream = cleaned.loc[cleaned[ds_col] == link_id, id_col].tolist()
                    if len(upstream) > len(present_us):
                        overflowed.append(link_id)
                        upstream = []
                    for slot, col in enumerate(present_us):
                        value = upstream[slot] if slot < len(upstream) else null_value
                        cleaned.at[idx, col] = value
                if overflowed and logger:
                    logger.warning(
                        f"Collapsing zero-length connectors left {len(overflowed)} "
                        f"junction(s) with more than {len(present_us)} upstream links; "
                        f"{'/'.join(present_us)} cleared to {null_value} for segment(s) "
                        f"{overflowed[:5]}. Downstream topology ({ds_col}) is unaffected."
                    )

        if logger:
            logger.info(
                f"Removed {len(drop_ids)} zero-length connector reach(es) from the "
                f"river network ({len(rivers)} -> {len(cleaned)} segments); upstream "
                f"links rewired to their downstream reach. These are TauDEM "
                f"binary-tree placeholders at 3-way confluences, not river reaches."
            )

        return cleaned

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
