# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Nodata handling for the resampling elevation calculator.

A DEM that declares a nodata value (or the -9999 crop fill) must be excluded from
a geometry's mean elevation. Left in, nodata-coded pixels (often 0.0) drag a
coarse forcing cell's mean toward sea level and fabricate a lapse-rate cooling.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from symfluence.data.preprocessing.resampling.elevation_calculator import (  # noqa: E402
    ElevationCalculator,
)


def _write_dem(path, array, nodata):
    transform = from_origin(0.0, array.shape[0], 1.0, 1.0)  # 1-degree pixels, origin (0,0)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def test_declared_nodata_pixels_excluded_from_mean(tmp_path):
    # Half the grid is real elevation (1500 m), half is nodata coded as 0.0.
    arr = np.full((4, 4), 1500.0, dtype="float32")
    arr[:, 2:] = 0.0  # right half is nodata
    dem = tmp_path / "dem.tif"
    _write_dem(dem, arr, nodata=0.0)

    # Polygon covering the whole raster.
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0.0, 0.0, 4.0, 4.0)], crs="EPSG:4326")

    calc = ElevationCalculator(logging.getLogger("test.elev"))
    elevations = calc.calculate(gdf, dem)

    # The 0.0 nodata pixels must be excluded -> mean is the real elevation, not ~750.
    assert elevations[0] == pytest.approx(1500.0)


def test_all_nodata_geometry_reports_sentinel(tmp_path):
    arr = np.zeros((3, 3), dtype="float32")  # entirely nodata
    dem = tmp_path / "dem_allnodata.tif"
    _write_dem(dem, arr, nodata=0.0)

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0.0, 0.0, 3.0, 3.0)], crs="EPSG:4326")
    calc = ElevationCalculator(logging.getLogger("test.elev"))
    elevations = calc.calculate(gdf, dem)

    # No valid pixels -> stays at the -9999 sentinel (never collapses to 0.0).
    assert elevations[0] == -9999
