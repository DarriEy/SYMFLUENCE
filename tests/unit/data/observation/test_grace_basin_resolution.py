# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the shared observation-handler basin-shapefile resolver.

Regression coverage for the case where domain discretization writes the
catchment under the nested, experiment-scoped layout
(``shapefiles/catchment/{definition_method}/{experiment_id}/
{domain}_HRUs_{discretization}.shp``) rather than a flat
``{domain}_catchment.shp``. The resolver (``BaseObservationHandler.
_resolve_catchment_shapefile``) must find the nested file, fall back to the
river_basins outline, and honour explicit config — for every observation
handler, not just GRACE.
"""
import logging

import pytest

from symfluence.data.observation.handlers.grace import GRACEHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]

DOMAIN = "Bow_at_Banff_multivar"
EXPERIMENT = "bow_exp3_streamflow_tws_joint"


def _config(tmp_path, **overrides):
    cfg = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": DOMAIN,
        "EXPERIMENT_ID": EXPERIMENT,
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-10 00:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "lumped",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
    }
    cfg.update(overrides)
    return cfg


def _handler(tmp_path, **overrides):
    return GRACEHandler(_config(tmp_path, **overrides), logging.getLogger("test_grace_basin"))


def _touch_shp(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")  # resolver only checks existence
    return path


def test_resolves_nested_discretized_catchment(tmp_path):
    """The canonical nested HRU shapefile (incl. upper-cased GRUS) is found."""
    handler = _handler(tmp_path)
    nested = (
        handler.project_shapefiles_dir
        / "catchment" / "lumped" / EXPERIMENT
        / f"{DOMAIN}_HRUs_GRUS.shp"
    )
    _touch_shp(nested)

    assert handler._resolve_catchment_shapefile(required=True) == nested


def test_falls_back_to_river_basins(tmp_path):
    """When no catchment shapefile exists, the river_basins outline is used."""
    handler = _handler(tmp_path)
    rb = handler.project_shapefiles_dir / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp"
    _touch_shp(rb)

    assert handler._resolve_catchment_shapefile(required=True) == rb


def test_prefers_catchment_over_river_basins(tmp_path):
    """A discretized catchment outranks the river_basins fallback."""
    handler = _handler(tmp_path)
    nested = (
        handler.project_shapefiles_dir
        / "catchment" / "lumped" / EXPERIMENT
        / f"{DOMAIN}_HRUs_GRUS.shp"
    )
    _touch_shp(nested)
    _touch_shp(handler.project_shapefiles_dir / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp")

    assert handler._resolve_catchment_shapefile(required=True) == nested


def test_honours_explicit_catchment_path_and_name(tmp_path):
    """Explicit CATCHMENT_PATH + CATCHMENT_SHP_NAME take precedence."""
    custom_dir = tmp_path / "my_shapes"
    custom = _touch_shp(custom_dir / "basin.shp")
    handler = _handler(
        tmp_path,
        CATCHMENT_PATH=str(custom_dir),
        CATCHMENT_SHP_NAME="basin.shp",
    )

    assert handler._resolve_catchment_shapefile(required=True) == custom


def test_raises_clear_error_when_nothing_found(tmp_path):
    """A helpful FileNotFoundError is raised when no basin exists."""
    handler = _handler(tmp_path)
    with pytest.raises(FileNotFoundError, match="define_domain"):
        handler._resolve_catchment_shapefile(required=True)


def test_not_required_returns_none(tmp_path):
    """Without required=True, a missing basin resolves to None (no raise)."""
    handler = _handler(tmp_path)
    assert handler._resolve_catchment_shapefile(required=False) is None


def test_shared_resolver_works_for_non_grace_handler(tmp_path):
    """The resolver lives on the base class, so any handler benefits."""
    from symfluence.data.observation.handlers.era5_land import ERA5LandHandler

    handler = ERA5LandHandler(_config(tmp_path), logging.getLogger("test_era5_basin"))
    nested = (
        handler.project_shapefiles_dir
        / "catchment" / "lumped" / EXPERIMENT
        / f"{DOMAIN}_HRUs_GRUS.shp"
    )
    _touch_shp(nested)

    assert handler._resolve_catchment_shapefile() == nested


def test_load_catchment_gdf_reads_resolved_shapefile(tmp_path):
    """_load_catchment_gdf returns a GeoDataFrame for the resolved basin."""
    import geopandas as gpd
    from shapely.geometry import box

    handler = _handler(tmp_path)
    nested = (
        handler.project_shapefiles_dir
        / "catchment" / "lumped" / EXPERIMENT
        / f"{DOMAIN}_HRUs_GRUS.shp"
    )
    nested.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(nested)

    gdf = handler._load_catchment_gdf()
    assert gdf is not None
    assert len(gdf) == 1


def test_load_catchment_gdf_returns_none_when_missing(tmp_path):
    """_load_catchment_gdf returns None (warning path) when no basin exists."""
    handler = _handler(tmp_path)
    assert handler._load_catchment_gdf() is None
