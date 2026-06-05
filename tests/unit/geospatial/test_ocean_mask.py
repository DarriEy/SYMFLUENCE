# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit test for GDALProcessor._mask_ocean_watersheds (coastal tentacle fix)."""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from symfluence.geospatial.geofabric.processors.gdal_processor import GDALProcessor  # noqa: E402

pytestmark = [pytest.mark.unit]


def _write(path, arr, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=str(arr.dtype), crs="EPSG:4326", transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def test_mask_ocean_sets_sea_cells_to_nodata(tmp_path):
    # Left half is land (elev 10, watershed 1), right half is sea (elev 0, watershed 2 = tentacle).
    ws = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype="int32")
    elev = np.array([[10.0, 10.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0]], dtype="float32")
    _write(tmp_path / "elv-watersheds.tif", ws, nodata=-2147483647)
    _write(tmp_path / "elv-fel.tif", elev)

    gp = GDALProcessor(logging.getLogger("t"))
    out = gp._mask_ocean_watersheds(tmp_path, sea_level=0.0)

    with rasterio.open(out) as src:
        masked = src.read(1)
        nd = src.nodata
    assert (masked[:, :2] == 1).all()       # land watershed preserved
    assert (masked[:, 2:] == nd).all()      # ocean watershed -> nodata


def test_mask_ocean_missing_dem_returns_original(tmp_path):
    ws = np.array([[1, 1]], dtype="int32")
    _write(tmp_path / "elv-watersheds.tif", ws, nodata=-2147483647)
    gp = GDALProcessor(logging.getLogger("t"))
    out = gp._mask_ocean_watersheds(tmp_path, sea_level=0.0)
    assert out.endswith("elv-watersheds.tif")  # falls back, no crash
