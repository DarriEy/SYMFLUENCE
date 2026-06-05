# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the shared basin/catchment shapefile finders in path_resolver.

These back the codebase-wide fix for the flat-path assumption: discretization
writes the catchment under ``catchment/{definition_method}/{experiment_id}/
{domain}_HRUs_{disc}.shp`` (suffix sometimes upper-cased, GRUs -> GRUS), and
the river-basins outline flat under ``river_basins/{domain}_riverBasins_*.shp``.
"""
from __future__ import annotations

import pytest

from symfluence.core.path_resolver import (
    find_basin_shapefile,
    find_catchment_subfile,
    find_river_basins_shapefile,
)

pytestmark = [pytest.mark.unit]

DOMAIN = "Bow_at_Banff"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


# --------------------------------------------------------------------------
# find_basin_shapefile
# --------------------------------------------------------------------------

def test_basin_resolves_nested_with_casing_and_cross_experiment(tmp_path):
    shp = tmp_path / "shapefiles"
    # Catchment written under a DIFFERENT experiment id, with upper-cased GRUS.
    target = _touch(shp / "catchment" / "lumped" / "exp_other" / f"{DOMAIN}_HRUs_GRUS.shp")
    # Config points at a different experiment id; deep search must still find it.
    assert find_basin_shapefile(shp, DOMAIN, "lumped", "exp_config") == target


def test_basin_prefers_correct_experiment_dir(tmp_path):
    shp = tmp_path / "shapefiles"
    correct = _touch(shp / "catchment" / "lumped" / "exp_a" / f"{DOMAIN}_HRUs_GRUs.shp")
    _touch(shp / "catchment" / "lumped" / "exp_b" / f"{DOMAIN}_HRUs_GRUs.shp")
    assert find_basin_shapefile(shp, DOMAIN, "lumped", "exp_a") == correct


def test_basin_falls_back_to_river_basins(tmp_path):
    shp = tmp_path / "shapefiles"
    rb = _touch(shp / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp")
    assert find_basin_shapefile(shp, DOMAIN, "lumped", "exp") == rb


def test_basin_river_basins_disabled_returns_none(tmp_path):
    shp = tmp_path / "shapefiles"
    _touch(shp / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp")
    assert find_basin_shapefile(shp, DOMAIN, "lumped", "exp", include_river_basins=False) is None


def test_basin_prefers_catchment_over_river_basins(tmp_path):
    shp = tmp_path / "shapefiles"
    catchment = _touch(shp / "catchment" / "lumped" / "exp" / f"{DOMAIN}_HRUs_GRUs.shp")
    _touch(shp / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp")
    assert find_basin_shapefile(shp, DOMAIN, "lumped", "exp") == catchment


def test_basin_honours_explicit_path_and_name(tmp_path):
    shp = tmp_path / "shapefiles"
    custom = _touch(tmp_path / "custom" / "basin.shp")
    assert find_basin_shapefile(
        shp, DOMAIN, "lumped", "exp",
        explicit_path=str(tmp_path / "custom"), explicit_name="basin.shp",
    ) == custom


def test_basin_not_found_returns_none(tmp_path):
    assert find_basin_shapefile(tmp_path / "shapefiles", DOMAIN, "lumped", "exp") is None


def test_basin_required_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="define_domain"):
        find_basin_shapefile(tmp_path / "shapefiles", DOMAIN, "lumped", "exp", required=True)


# --------------------------------------------------------------------------
# find_catchment_subfile (specifically-named artifacts, e.g. delineated)
# --------------------------------------------------------------------------

def test_subfile_nested(tmp_path):
    shp = tmp_path / "shapefiles"
    name = f"{DOMAIN}_catchment_delineated.shp"
    target = _touch(shp / "catchment" / "lumped" / "exp" / name)
    assert find_catchment_subfile(shp, "lumped", "exp", name) == target


def test_subfile_deep_search_other_experiment(tmp_path):
    shp = tmp_path / "shapefiles"
    name = f"{DOMAIN}_catchment_delineated.shp"
    target = _touch(shp / "catchment" / "lumped" / "exp_other" / name)
    # Configured experiment differs; deep search finds it.
    assert find_catchment_subfile(shp, "lumped", "exp_config", name) == target


def test_subfile_legacy_flat(tmp_path):
    shp = tmp_path / "shapefiles"
    name = f"{DOMAIN}_catchment_delineated.shp"
    target = _touch(shp / "catchment" / name)
    assert find_catchment_subfile(shp, "lumped", "exp", name) == target


def test_subfile_not_found(tmp_path):
    assert find_catchment_subfile(tmp_path / "shapefiles", "lumped", "exp", "x.shp") is None


# --------------------------------------------------------------------------
# find_river_basins_shapefile
# --------------------------------------------------------------------------

def test_river_basins_found(tmp_path):
    shp = tmp_path / "shapefiles"
    rb = _touch(shp / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp")
    assert find_river_basins_shapefile(shp, DOMAIN) == rb


def test_river_basins_not_found_none(tmp_path):
    assert find_river_basins_shapefile(tmp_path / "shapefiles", DOMAIN) is None


def test_river_basins_required_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="define_domain"):
        find_river_basins_shapefile(tmp_path / "shapefiles", DOMAIN, required=True)
