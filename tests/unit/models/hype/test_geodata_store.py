# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE GeoData manager reads soil/land/elevation stats from the MRDS.

``_load_gis_stats`` now prefers the model-ready attributes store, reconstructing
the per-basin soil (USGS_*), landcover (IGBP_*) and elevation frames that the SLC
pipeline consumes, and falling back to gistool CSVs / intersection shapefiles.
The store values originate from the same intersection shapefiles, so SLC output
is unchanged.
"""
from __future__ import annotations

import logging
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip("netCDF4")  # noqa: N816
gpd = pytest.importorskip("geopandas")
pytest.importorskip("xarray")

from shapely.geometry import box

from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager

HRU_IDS = [1, 2, 3]


def _catchment_shp(tmp_path):
    path = tmp_path / "shapefiles" / "catchment" / "test_HRUs_lumped.shp"
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({
        "HRU_ID": HRU_IDS,
        "HRU_area": np.full(3, 2.0e6),
        "geometry": [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs="EPSG:4326").to_file(path)


def _soil_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_soilgrids" / "catchment_with_soilclass.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"HRU_ID": HRU_IDS,
            "USGS_0": [0.1, 0.1, 0.1], "USGS_1": [0.7, 0.0, 0.0],
            "USGS_2": [0.2, 0.8, 0.1], "USGS_3": [0.0, 0.1, 0.8]}
    data["geometry"] = [box(i, 50, i + 1, 51) for i in range(3)]
    gpd.GeoDataFrame(data, crs="EPSG:4326").to_file(path)


def _land_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_landclass" / "catchment_with_landclass.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"HRU_ID": HRU_IDS}
    for i in range(1, 6):
        data[f"IGBP_{i}"] = [0.0, 0.0, 0.0]
    data["IGBP_2"] = [0.9, 0.1, 0.1]
    data["IGBP_4"] = [0.1, 0.9, 0.9]
    data["geometry"] = [box(i, 50, i + 1, 51) for i in range(3)]
    gpd.GeoDataFrame(data, crs="EPSG:4326").to_file(path)


def _dem_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_dem" / "catchment_with_dem.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({
        "HRU_ID": HRU_IDS, "elev_mean": [800.0, 1200.0, 2000.0],
        "geometry": [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs="EPSG:4326").to_file(path)


def _manager(tmp_path):
    mgr = HYPEGeoDataManager.__new__(HYPEGeoDataManager)
    mgr.config = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path.parent),
        "DOMAIN_NAME": "test",
    }
    mgr.logger = logging.getLogger("test_hype_geodata")
    return mgr


def _build(tmp_path):
    # project_dir = <data_dir>/domain_test ; store lives under it.
    assert AttributesNetCDFBuilder(project_dir=tmp_path, domain_name="test").build() is not None


def test_load_gis_stats_from_store(tmp_path):
    project_dir = tmp_path / "domain_test"
    _catchment_shp(project_dir)
    _soil_shp(project_dir)
    _land_shp(project_dir)
    _dem_shp(project_dir)
    _build(project_dir)

    soil, land, elev = _manager(project_dir)._load_gis_stats_from_store("HRU_ID")

    assert list(soil.index) == HRU_IDS
    assert "USGS_1" in soil.columns and soil.loc[1, "USGS_1"] == pytest.approx(0.7)
    assert "IGBP_2" in land.columns and land.loc[1, "IGBP_2"] == pytest.approx(0.9)
    assert elev.loc[3, "elev_mean"] == pytest.approx(2000.0)


def test_store_frames_drive_slc_processing(tmp_path):
    project_dir = tmp_path / "domain_test"
    _catchment_shp(project_dir)
    _soil_shp(project_dir)
    _land_shp(project_dir)
    _dem_shp(project_dir)
    _build(project_dir)

    mgr = _manager(project_dir)
    soil, land, elev = mgr._load_gis_stats_from_store("HRU_ID")
    base_df = pd.DataFrame({"subid": HRU_IDS})

    slc_df, _ = mgr._process_slc(base_df, land, soil, threshold=0.05)

    # Dominant soil: USGS_1 (hru1), USGS_2 (hru2), USGS_3 (hru3).
    # Active land (>0.05): IGBP_2 and/or IGBP_4. SLC = (landcover, soil) combos.
    combos = set(zip(slc_df["landcover"], slc_df["soil"]))
    assert (2, 1) in combos   # hru1: IGBP_2 x USGS_1
    assert (4, 2) in combos   # hru2: IGBP_4 x USGS_2
    assert (4, 3) in combos   # hru3: IGBP_4 x USGS_3
    assert all(slc_df["soil"] >= 1)  # HYPE requires soil >= 1


def test_returns_none_when_store_absent(tmp_path):
    assert _manager(tmp_path / "domain_test")._load_gis_stats_from_store("HRU_ID") is None
