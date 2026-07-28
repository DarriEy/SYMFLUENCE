# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for GDALProcessor.raster_to_polygon (gagewatershed basin ID fix).

Regression cover for the lumped-delineation bug where the polygonizer hardcoded a
filter for ``ID == 1``. TauDEM's ``gagewatershed`` labels each basin with the
*gauge's* id from ``moveoutletstostreams``, which is 0-based, so a single-gauge run
produces a watershed of value 0 and the filter raised on every lumped delineation.
The caller then silently fell back to dissolving streamnet sub-watersheds, whose
outlet polygon extends downstream past the gauge — Bow at Banff came out at
2248 km^2 instead of 2208 km^2, with the gauge 3.7 km inside the basin boundary.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("osgeo")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from symfluence.geospatial.geofabric.processors.gdal_processor import GDALProcessor  # noqa: E402

pytestmark = [pytest.mark.unit]

NODATA = -2147483647


def _write(path, arr, nodata=NODATA):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=str(arr.dtype), crs="EPSG:4326", transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def test_polygonizes_zero_valued_gagewatershed(tmp_path):
    """A single-gauge gagewatershed raster is labelled 0, not 1, and must survive."""
    # Left half is the basin (gauge id 0), right half is nodata.
    arr = np.array([[0, 0, NODATA, NODATA], [0, 0, NODATA, NODATA]], dtype="int32")
    _write(tmp_path / "watershed.tif", arr)

    gp = GDALProcessor(logging.getLogger("t"))
    out = tmp_path / "basin.shp"
    gp.raster_to_polygon(tmp_path / "watershed.tif", out)

    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf["ID"].tolist() == [0]
    # The basin covers the two left-hand columns: 2 cols x 2 rows of 1x1 cells.
    assert gdf.geometry.area.sum() == pytest.approx(4.0)


def test_excludes_nodata_region(tmp_path):
    """Nodata must never be polygonized into a basin, whatever the basin IDs are."""
    arr = np.array([[3, 3, NODATA, NODATA], [3, 3, NODATA, NODATA]], dtype="int32")
    _write(tmp_path / "watershed.tif", arr)

    gp = GDALProcessor(logging.getLogger("t"))
    out = tmp_path / "basin.shp"
    gp.raster_to_polygon(tmp_path / "watershed.tif", out)

    gdf = gpd.read_file(out)
    assert gdf["ID"].tolist() == [3]
    assert NODATA not in gdf["ID"].tolist()
    assert gdf.geometry.area.sum() == pytest.approx(4.0)


def test_keeps_every_basin_for_multi_id_rasters(tmp_path):
    """streamnet's elv-watersheds.tif holds many sub-basins; all must be retained."""
    arr = np.array([[0, 1, 2, NODATA], [0, 1, 2, NODATA]], dtype="int32")
    _write(tmp_path / "elv-watersheds.tif", arr)

    gp = GDALProcessor(logging.getLogger("t"))
    out = tmp_path / "watersheds.shp"
    gp.raster_to_polygon(tmp_path / "elv-watersheds.tif", out)

    gdf = gpd.read_file(out)
    assert sorted(gdf["ID"].tolist()) == [0, 1, 2]


def test_raises_when_raster_is_all_nodata(tmp_path):
    arr = np.full((2, 4), NODATA, dtype="int32")
    _write(tmp_path / "watershed.tif", arr)

    gp = GDALProcessor(logging.getLogger("t"))
    with pytest.raises(ValueError, match="No valid"):
        gp.raster_to_polygon(tmp_path / "watershed.tif", tmp_path / "basin.shp")
