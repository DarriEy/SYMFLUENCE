# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Catchment-area shapefile selection must be deterministic and artifact-safe.

Area detection took ``list(dir.glob("*.shp"))[0]``, whose order is
filesystem-dependent. The same domain resolved the canonical basin shapefile
on one machine and a leftover ``..._lumped_temp.shp`` scratch file (~0.9 m²)
on another, silently corrupting every discharge on the second — Q = runoff x
area, so a wrong area is a wrong hydrograph with no error.
"""
from __future__ import annotations

from pathlib import Path

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_text("")


def test_canonical_wins_over_temp(tmp_path):
    _touch(tmp_path, "Bow_riverBasins_lumped_temp.shp",
           "Bow_riverBasins_lumped.shp")
    order = [p.name for p in StreamflowEvaluator._ordered_area_shapefiles(tmp_path)]
    assert order[0] == "Bow_riverBasins_lumped.shp"
    assert order[-1] == "Bow_riverBasins_lumped_temp.shp"


def test_ordering_is_deterministic_regardless_of_glob_order(tmp_path):
    # create in "wrong" order; result must not depend on creation/fs order
    _touch(tmp_path, "z_riverBasins_lumped.shp", "a_riverBasins_lumped.shp")
    a = [p.name for p in StreamflowEvaluator._ordered_area_shapefiles(tmp_path)]
    b = [p.name for p in StreamflowEvaluator._ordered_area_shapefiles(tmp_path)]
    assert a == b
    # shortest stem first, then lexical — stable
    assert a == sorted(a, key=lambda s: (len(Path(s).stem), Path(s).stem))


def test_temp_only_is_still_returned_as_last_resort(tmp_path):
    # if the only file is a scratch artifact, don't hide it entirely — a
    # caller that validates areas can still reject it, but selection must not
    # silently return nothing when a file exists.
    _touch(tmp_path, "Bow_riverBasins_lumped_temp.shp")
    order = StreamflowEvaluator._ordered_area_shapefiles(tmp_path)
    assert len(order) == 1


def test_missing_directory_returns_empty(tmp_path):
    assert StreamflowEvaluator._ordered_area_shapefiles(tmp_path / "nope") == []


def test_tmp_suffix_also_deferred(tmp_path):
    _touch(tmp_path, "basin.tmp.shp", "basin.shp")
    order = [p.name for p in StreamflowEvaluator._ordered_area_shapefiles(tmp_path)]
    assert order[0] == "basin.shp"
