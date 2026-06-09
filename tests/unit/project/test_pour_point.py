# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests for multi-outlet pour point shapefile creation.

POUR_POINT_COORDS is the primary, most-downstream outlet (id 0) that defines the
domain extent. POUR_POINT_ADDITIONAL_COORDS lists interior outlets (id 1..N) that
are written into the same outlets shapefile so TauDEM breaks the stream network at
each, aligning subbasin/GRU boundaries with interior gauges. Additional outlets
only apply to semidistributed/distributed delineation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.project.project_manager import ProjectManager

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_config(tmp_path: Path, **overrides) -> SymfluenceConfig:
    base = dict(
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        SYMFLUENCE_CODE_DIR=str(tmp_path / "code"),
        DOMAIN_NAME="pp_test",
        DEM_NAME="default",
        DEM_PATH="default",
        DOMAIN_DEFINITION_METHOD="semidistributed",
        CATCHMENT_PATH="default",
        CATCHMENT_SHP_NAME="default",
        CATCHMENT_SHP_GRUID="GRU_ID",
        CATCHMENT_SHP_HRUID="HRU_ID",
        SUB_GRID_DISCRETIZATION="GRUs",
        EXPERIMENT_ID="test",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        FORCING_DATASET="ERA5",
        HYDROLOGICAL_MODEL="SUMMA",
        POUR_POINT_COORDS="51.1722/-115.5717",
    )
    base.update(overrides)
    return SymfluenceConfig(**base)


def _create(tmp_path: Path, **overrides) -> gpd.GeoDataFrame:
    cfg = _make_config(tmp_path, **overrides)
    pm = ProjectManager(cfg, logging.getLogger("test_pour_point"))
    out = pm.create_pour_point()
    assert out is not None and out.exists()
    return gpd.read_file(out)


def test_single_pour_point_has_id_zero(tmp_path):
    """A lone POUR_POINT_COORDS yields one outlet with id 0 in lon/lat order."""
    gdf = _create(tmp_path)
    assert list(gdf['id']) == [0]
    # Point takes (lon, lat); shapefile x=lon, y=lat
    assert gdf.geometry.iloc[0].x == pytest.approx(-115.5717)
    assert gdf.geometry.iloc[0].y == pytest.approx(51.1722)


def test_additional_outlets_string_form(tmp_path):
    """Comma-separated POUR_POINT_ADDITIONAL_COORDS become ids 1..N."""
    gdf = _create(
        tmp_path,
        POUR_POINT_ADDITIONAL_COORDS="51.40/-116.00, 51.60/-116.20",
    )
    assert list(gdf['id']) == [0, 1, 2]
    assert gdf.geometry.iloc[1].x == pytest.approx(-116.00)
    assert gdf.geometry.iloc[1].y == pytest.approx(51.40)
    assert gdf.geometry.iloc[2].y == pytest.approx(51.60)


def test_additional_outlets_list_form(tmp_path):
    """A YAML list of 'lat/lon' pairs is accepted too."""
    gdf = _create(
        tmp_path,
        POUR_POINT_ADDITIONAL_COORDS=["51.40/-116.00", "51.60/-116.20"],
    )
    assert list(gdf['id']) == [0, 1, 2]


def test_additional_outlets_ignored_for_lumped(tmp_path):
    """Interior outlets are dropped for non-distributed methods (single basin)."""
    gdf = _create(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="lumped",
        POUR_POINT_ADDITIONAL_COORDS="51.40/-116.00",
    )
    assert list(gdf['id']) == [0]


def test_invalid_additional_coords_rejected_at_config_time(tmp_path):
    """Malformed interior outlet coords fail config validation up front."""
    from symfluence.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        _make_config(tmp_path, POUR_POINT_ADDITIONAL_COORDS="51.40,-116.00")


def test_out_of_range_additional_coords_rejected(tmp_path):
    """Latitude outside [-90, 90] is rejected."""
    from symfluence.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        _make_config(tmp_path, POUR_POINT_ADDITIONAL_COORDS="151.0/-116.0")
