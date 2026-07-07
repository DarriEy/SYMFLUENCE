# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Topology tests for the PIHM semi-distributed grid mesh generator.

``PIHMPreProcessor._build_grid_mesh`` produces the element/node/river tables
for ``PIHM_HILLSLOPE_BANDS > 1``. MM-PIHM only surfaces topology errors at
runtime (or worse, as silently wrong physics), so these tests pin the
structural invariants: element/node bookkeeping, neighbor reciprocity, area
conservation, and the outlet-first river numbering the result extractor
depends on.
"""
from __future__ import annotations

import math

import pytest

from symfluence.models.pihm.preprocessor import PIHMPreProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]

AREA_M2 = 2.21e9  # Bow at Banff
SOIL_DEPTH = 2.0


def _build(n_rows):
    # _build_grid_mesh is self-independent; call it unbound.
    return PIHMPreProcessor._build_grid_mesh(None, AREA_M2, SOIL_DEPTH, n_rows)


def _parse(ele_lines, node_lines):
    """Parse the generated tables into elements and node coordinates."""
    assert ele_lines[0].startswith("NUMELE\t")
    n_ele = int(ele_lines[0].split("\t")[1])
    elements = {}
    for line in ele_lines[2:]:
        if line.startswith("NUMNODE"):
            n_node = int(line.split("\t")[1])
            break
        idx, n1, n2, n3, a1, a2, a3 = (int(v) for v in line.split("\t"))
        elements[idx] = ((n1, n2, n3), (a1, a2, a3))
    nodes = {}
    for line in node_lines[1:]:
        parts = line.split("\t")
        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
    return n_ele, elements, n_node, nodes


@pytest.mark.parametrize("n_rows", [1, 2, 3, 5])
def test_mesh_bookkeeping(n_rows):
    """Element/node counts must match the declared NUMELE/NUMNODE."""
    ele_lines, node_lines, n_ele, river = _build(n_rows)
    n_declared, elements, n_node, nodes = _parse(ele_lines, node_lines)
    N, M = n_rows, max(2, 2 * n_rows)
    assert n_ele == n_declared == len(elements) == 4 * M * N
    assert n_node == len(nodes) == (M + 1) * (2 * N + 1)
    assert sorted(elements) == list(range(1, n_ele + 1))
    assert sorted(nodes) == list(range(1, n_node + 1))


@pytest.mark.parametrize("n_rows", [1, 2, 3, 5])
def test_element_nodes_valid_and_distinct(n_rows):
    ele_lines, node_lines, n_ele, _ = _build(n_rows)
    _, elements, n_node, _ = _parse(ele_lines, node_lines)
    for idx, ((n1, n2, n3), _nabrs) in elements.items():
        assert len({n1, n2, n3}) == 3, f"degenerate element {idx}"
        for n in (n1, n2, n3):
            assert 1 <= n <= n_node, f"element {idx} references node {n}"


@pytest.mark.parametrize("n_rows", [1, 2, 3, 5])
def test_neighbor_reciprocity(n_rows):
    """If element A lists B as a neighbor, B must list A back."""
    ele_lines, node_lines, n_ele, _ = _build(n_rows)
    _, elements, _, _ = _parse(ele_lines, node_lines)
    for idx, (_nodes, nabrs) in elements.items():
        for n in nabrs:
            assert 0 <= n <= n_ele, f"element {idx} references neighbor {n}"
            if n > 0:
                assert idx in elements[n][1], (
                    f"element {idx} lists {n} as neighbor, but not vice versa"
                )


@pytest.mark.parametrize("n_rows", [1, 2, 3, 5])
def test_neighbors_share_an_edge(n_rows):
    """Every declared neighbor pair must share exactly two nodes (an edge)."""
    ele_lines, node_lines, _, _ = _build(n_rows)
    _, elements, _, _ = _parse(ele_lines, node_lines)
    for idx, (nds, nabrs) in elements.items():
        for n in nabrs:
            if n > 0:
                shared = set(nds) & set(elements[n][0])
                assert len(shared) == 2, (
                    f"elements {idx} and {n} declared neighbors but share "
                    f"{len(shared)} nodes"
                )


@pytest.mark.parametrize("n_rows", [1, 3])
def test_total_area_conserved(n_rows):
    """The triangles must tile the catchment area exactly."""
    ele_lines, node_lines, _, _ = _build(n_rows)
    _, elements, _, nodes = _parse(ele_lines, node_lines)
    total = 0.0
    for (n1, n2, n3), _nabrs in elements.values():
        (x1, y1), (x2, y2), (x3, y3) = (nodes[n][:2] for n in (n1, n2, n3))
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
        assert area > 0.0
        total += area
    # node coordinates are written at cm precision, so allow the rounding
    # residual; a real topology error would be off by whole cells (>>1e-6)
    assert total == pytest.approx(AREA_M2, rel=1e-6)


@pytest.mark.parametrize("n_rows", [1, 2, 3, 5])
def test_river_outlet_first_cascade(n_rows):
    """Segment 1 must be the outlet; every other segment drains strictly
    toward a lower index (the convention the result extractor relies on to
    read column 1 of the river output as total basin discharge)."""
    ele_lines, node_lines, n_ele, river = _build(n_rows)
    _, elements, n_node, nodes = _parse(ele_lines, node_lines)
    M = max(2, 2 * n_rows)
    assert len(river) == M
    downs = [seg[2] for seg in river]
    assert downs[0] == -3, "segment 1 must be the outlet (DOWN=-3)"
    assert downs.count(-3) == 1, "exactly one outlet"
    for k, (frm, to, down, left, right) in enumerate(river, start=1):
        assert 1 <= frm <= n_node and 1 <= to <= n_node
        assert 1 <= left <= n_ele and 1 <= right <= n_ele
        assert left != right, "MM-PIHM requires LEFT != RIGHT"
        if k > 1:
            assert 1 <= down < k, (
                f"segment {k} must drain toward a lower index, got {down}"
            )
        # channel nodes lie on the centreline
        assert nodes[frm][1] == pytest.approx(0.0)
        assert nodes[to][1] == pytest.approx(0.0)


def test_outlet_is_lowest_channel_point():
    """The outlet segment must sit at the low end of the channel profile."""
    ele_lines, node_lines, _, river = _build(3)
    _, _, _, nodes = _parse(ele_lines, node_lines)
    outlet = river[0]
    surface_z = {n: nodes[n][3] for seg in river for n in seg[:2]}
    outlet_z = min(nodes[outlet[0]][3], nodes[outlet[1]][3])
    assert outlet_z == min(surface_z.values())


def test_bands_one_matches_lumped_element_count():
    """n_rows=1 gives the minimal grid (M=2): 8 elements, 2 channel segments."""
    ele_lines, _, n_ele, river = _build(1)
    assert n_ele == 8
    assert len(river) == 2
