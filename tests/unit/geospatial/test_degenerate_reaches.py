# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for RiverGraphProcessor.drop_degenerate_reaches.

TauDEM encodes the stream network as a binary tree, so a confluence of three or
more streams is split into two junctions joined by a zero-length connector link.
That connector has no catchment polygon, which breaks the one-segment-per-GRU
invariant, and a zero reach length is a division hazard for routing schemes that
derive travel time from length. These tests pin the collapse and, above all, the
topology rewiring: dropping a connector without repointing its upstream links
would silently amputate the whole network above it.
"""
from __future__ import annotations

import pytest

gpd = pytest.importorskip("geopandas")

from shapely.geometry import LineString, Point  # noqa: E402

from symfluence.geospatial.geofabric.processors.graph_processor import (  # noqa: E402
    RiverGraphProcessor,
)

pytestmark = [pytest.mark.unit]


def _network(rows):
    """Build a river-network GeoDataFrame from (id, ds, us1, us2, geom) tuples."""
    return gpd.GeoDataFrame(
        {
            "LINKNO": [r[0] for r in rows],
            "DSLINKNO": [r[1] for r in rows],
            "USLINKNO1": [r[2] for r in rows],
            "USLINKNO2": [r[3] for r in rows],
        },
        geometry=[r[4] for r in rows],
        crs="EPSG:4326",
    )


def _trijunction():
    """Three streams (1, 2, 3) meeting at one point, TauDEM-style.

    Links 1 and 2 join at the zero-length connector 10, which immediately joins
    link 3 at link 4. Mirrors Iceland's LINKNO 1663.
    """
    return _network([
        (1, 10, -1, -1, LineString([(0, 0), (1, 1)])),
        (2, 10, -1, -1, LineString([(2, 0), (1, 1)])),
        (10, 4, 1, 2, LineString([(1, 1), (1, 1)])),   # zero-length connector
        (3, 4, -1, -1, LineString([(0, 2), (1, 1)])),
        (4, -1, 10, 3, LineString([(1, 1), (1, 3)])),  # outlet
    ])


def test_connector_is_dropped():
    cleaned = RiverGraphProcessor.drop_degenerate_reaches(_trijunction())

    assert len(cleaned) == 4
    assert 10 not in set(cleaned["LINKNO"])


def test_upstream_links_are_rewired_to_the_surviving_reach():
    """The whole point of the guard: 1 and 2 must reach the outlet, not dangle."""
    cleaned = RiverGraphProcessor.drop_degenerate_reaches(_trijunction())

    ds = dict(zip(cleaned["LINKNO"], cleaned["DSLINKNO"]))
    assert ds[1] == 4
    assert ds[2] == 4
    assert ds[3] == 4
    assert ds[4] == -1

    # No pointer may reference a segment that no longer exists.
    assert set(cleaned["DSLINKNO"]) - set(cleaned["LINKNO"]) == {-1}


def test_every_segment_still_drains_to_an_outlet():
    """Guards against the failure mode of dropping without rewiring."""
    cleaned = RiverGraphProcessor.drop_degenerate_reaches(_trijunction())
    ds = dict(zip(cleaned["LINKNO"], cleaned["DSLINKNO"]))

    for start in cleaned["LINKNO"]:
        seg, hops = start, 0
        while seg != -1 and hops <= len(cleaned):
            seg, hops = ds[seg], hops + 1
        assert seg == -1, f"segment {start} does not drain to an outlet"


def test_overflowing_binary_slots_are_cleared_and_warned(caplog):
    """Collapsing the connector gives link 4 three upstream links, which the two
    TauDEM slots cannot hold; they must be cleared rather than left dangling."""

    class _Log:
        def __init__(self):
            self.warnings = []
            self.infos = []

        def warning(self, msg):
            self.warnings.append(msg)

        def info(self, msg):
            self.infos.append(msg)

    logger = _Log()
    cleaned = RiverGraphProcessor.drop_degenerate_reaches(_trijunction(), logger=logger)

    row = cleaned[cleaned["LINKNO"] == 4].iloc[0]
    assert row["USLINKNO1"] == -1
    assert row["USLINKNO2"] == -1
    assert any("more than 2 upstream links" in w for w in logger.warnings)
    assert any("1 zero-length connector" in i for i in logger.infos)

    # No upstream slot may reference a dropped segment.
    assert 10 not in set(cleaned["USLINKNO1"]) | set(cleaned["USLINKNO2"])


def test_binary_slots_are_repopulated_when_they_still_fit():
    """A connector between two ordinary joins leaves only two upstream links,
    so the slots are recomputed rather than cleared."""
    net = _network([
        (1, 10, -1, -1, LineString([(0, 0), (1, 1)])),
        (10, 2, 1, -1, LineString([(1, 1), (1, 1)])),  # zero-length connector
        (2, -1, 10, -1, LineString([(1, 1), (1, 3)])),
    ])

    cleaned = RiverGraphProcessor.drop_degenerate_reaches(net)

    row = cleaned[cleaned["LINKNO"] == 2].iloc[0]
    assert row["USLINKNO1"] == 1
    assert row["USLINKNO2"] == -1


def test_chained_connectors_resolve_transitively():
    """Two connectors back to back must collapse to the first real reach."""
    net = _network([
        (1, 10, -1, -1, LineString([(0, 0), (1, 1)])),
        (10, 11, 1, -1, LineString([(1, 1), (1, 1)])),  # connector
        (11, 2, 10, -1, LineString([(1, 1), (1, 1)])),  # connector
        (2, -1, 11, -1, LineString([(1, 1), (1, 3)])),
    ])

    cleaned = RiverGraphProcessor.drop_degenerate_reaches(net)

    assert set(cleaned["LINKNO"]) == {1, 2}
    assert cleaned[cleaned["LINKNO"] == 1].iloc[0]["DSLINKNO"] == 2


def test_clean_network_is_returned_untouched():
    net = _network([
        (1, 2, -1, -1, LineString([(0, 0), (1, 1)])),
        (2, -1, 1, -1, LineString([(1, 1), (1, 3)])),
    ])

    cleaned = RiverGraphProcessor.drop_degenerate_reaches(net)

    assert cleaned is net


def test_is_idempotent():
    once = RiverGraphProcessor.drop_degenerate_reaches(_trijunction())
    twice = RiverGraphProcessor.drop_degenerate_reaches(once)

    assert twice is once
    assert len(twice) == len(once)


def test_empty_and_missing_geometry_count_as_degenerate():
    net = _network([
        (1, 10, -1, -1, LineString([(0, 0), (1, 1)])),
        (10, 2, 1, -1, Point().buffer(0)),  # empty geometry
        (2, -1, 10, -1, LineString([(1, 1), (1, 3)])),
    ])

    cleaned = RiverGraphProcessor.drop_degenerate_reaches(net)

    assert set(cleaned["LINKNO"]) == {1, 2}
    assert cleaned[cleaned["LINKNO"] == 1].iloc[0]["DSLINKNO"] == 2


def test_segment_id_dtype_is_preserved():
    """Shapefile round-trips are dtype-sensitive; int32 must not widen to object."""
    net = _trijunction()
    net["DSLINKNO"] = net["DSLINKNO"].astype("int32")

    cleaned = RiverGraphProcessor.drop_degenerate_reaches(net)

    assert cleaned["DSLINKNO"].dtype == net["DSLINKNO"].dtype


def test_missing_topology_columns_are_tolerated():
    """Non-TauDEM fabrics lacking the pointer columns pass through unchanged."""
    net = gpd.GeoDataFrame(
        {"COMID": [1, 2]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 1)])],
        crs="EPSG:4326",
    )

    assert RiverGraphProcessor.drop_degenerate_reaches(net) is net
