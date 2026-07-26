# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavior checks for the model-facing NetCDF encoding contract."""
from __future__ import annotations

import xarray as xr

from symfluence.core.modeling.netcdf_utils import create_minimal_encoding, create_netcdf_encoding


def test_create_netcdf_encoding_applies_defaults_and_overrides():
    dataset = xr.Dataset(
        {"flow": ("time", [1.0, 2.0]), "gruId": ("gru", [7])},
        coords={"time": [0, 1], "gru": [1]},
    )

    encoding = create_netcdf_encoding(
        dataset,
        int_vars={"gruId": "int32"},
        custom_encoding={"flow": {"complevel": 6, "shuffle": True}},
    )

    assert encoding["flow"] == {
        "dtype": "float32",
        "_FillValue": -9999.0,
        "zlib": True,
        "complevel": 6,
        "shuffle": True,
    }
    assert encoding["gruId"] == {
        "dtype": "int32",
        "_FillValue": None,
        "zlib": True,
        "complevel": 4,
    }
    assert encoding["time"] == {"dtype": "float64", "_FillValue": None}
    assert encoding["gru"] == {"dtype": "int32"}


def test_create_minimal_encoding_only_disables_fill_values():
    dataset = xr.Dataset({"flow": ("time", [1.0])}, coords={"time": [0]})

    assert create_minimal_encoding(dataset) == {
        "flow": {"_FillValue": None},
        "time": {"_FillValue": None},
    }
    assert create_minimal_encoding(dataset, preserve_fill=True) == {
        "time": {"_FillValue": None},
    }
