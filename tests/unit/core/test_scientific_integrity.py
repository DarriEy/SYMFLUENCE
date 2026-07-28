# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.scientific_integrity import (
    BalanceCheck,
    ScientificIntegrityError,
    validate_dataset,
)


def _dataset(values=(1.0, 2.0)) -> xr.Dataset:
    ds = xr.Dataset(
        {"flow": ("time", list(values))},
        coords={"time": pd.date_range("2025-01-01", periods=len(values), freq="h")},
    )
    ds["flow"].attrs["units"] = "m3 s-1"
    return ds


def test_valid_dataset_passes():
    validate_dataset(_dataset())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_fail(bad):
    with pytest.raises(ScientificIntegrityError, match="NaN or infinite"):
        validate_dataset(_dataset((1.0, bad)))


def test_duplicate_time_fails():
    ds = _dataset()
    ds = ds.assign_coords(time=[ds.time.values[0], ds.time.values[0]])
    with pytest.raises(ScientificIntegrityError, match="strictly increasing"):
        validate_dataset(ds)


def test_missing_units_fail():
    ds = _dataset()
    ds["flow"].attrs.clear()
    with pytest.raises(ScientificIntegrityError, match="missing units"):
        validate_dataset(ds)


def test_mass_balance_failure():
    ds = xr.Dataset({"rain": ("time", [10.0]), "runoff": ("time", [4.0])})
    for variable in ds.data_vars.values():
        variable.attrs["units"] = "mm"
    with pytest.raises(ScientificIntegrityError, match="Mass-balance"):
        validate_dataset(ds, balance=BalanceCheck(inputs=["rain"], outputs=["runoff"]))
