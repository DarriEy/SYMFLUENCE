import logging

import numpy as np
import xarray as xr

from symfluence.data.acquisition.handlers.era5 import (
    _prepare_bbox_for_era5,
    _subset_era5_bbox,
)


def _make_dummy_ds(lat_vals, lon_vals):
    data = xr.DataArray(
        np.zeros((len(lat_vals), len(lon_vals))),
        dims=("latitude", "longitude"),
        coords={"latitude": lat_vals, "longitude": lon_vals},
    )
    return xr.Dataset({"tmp": data})


def test_prepare_bbox_expands_small_extent():
    ds = _make_dummy_ds(
        np.array([50.0, 49.75, 49.5]),
        np.array([10.0, 10.25, 10.5]),
    )
    bbox = {"lat_min": 49.999, "lat_max": 50.001, "lon_min": 10.1, "lon_max": 10.12}

    info = _prepare_bbox_for_era5(ds, bbox, logging.getLogger("test"))

    assert info["lat_max"] - info["lat_min"] >= info["lat_resolution"] - 1e-6
    assert info["lon_max"] - info["lon_min"] >= info["lon_resolution"] - 1e-6
    assert info["wrap_longitude"] is False


def test_subset_handles_longitude_wrap():
    ds = xr.Dataset(
        {"tmp": (("time", "latitude", "longitude"), np.ones((1, 2, 3)))},
        coords={
            "time": [0],
            "latitude": [1.0, 0.5],
            "longitude": [0.0, 0.5, 359.5],
        },
    )
    bbox = {"lat_min": 0.4, "lat_max": 1.1, "lon_min": -1.0, "lon_max": 1.0}

    info = _prepare_bbox_for_era5(ds, bbox, logging.getLogger("test"))

    subset = _subset_era5_bbox(
        ds,
        info["lat_min"],
        info["lat_max"],
        info["lon_min"],
        info["lon_max"],
        info["wrap_longitude"],
        info["lat_descending"],
        (info["lon_min_value"], info["lon_max_value"]),
    )

    assert info["wrap_longitude"] is True
    assert subset.sizes.get("longitude", 0) == 3
    assert subset.sizes.get("latitude", 0) == 2
