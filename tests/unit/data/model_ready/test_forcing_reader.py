# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the canonical forcing reader's CFIF (CF standard name) contract.

``open_canonical_forcing`` must return forcing under canonical CF names
(``precipitation_flux``, ``air_temperature``, ...) regardless of whether the
source used SUMMA-native shorthand (``pptrate``/``airtemp``) or other dataset
aliases, and must expose the timestep via ``ds.attrs['timestep_seconds']``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")
import xarray as xr

from symfluence.data.model_ready.cf_conventions import (
    CANONICAL_FORCING,
    resolve_forcing_var,
)
from symfluence.data.model_ready.forcing_reader import (
    forcing_timestep_seconds,
    open_canonical_forcing,
)

CF_NAMES = {
    "precipitation_flux", "air_temperature", "surface_downwelling_shortwave_flux",
    "surface_downwelling_longwave_flux", "wind_speed", "specific_humidity",
    "surface_air_pressure",
}


def _write(tmp_path, data_vars):
    times = pd.date_range("2020-01-01", periods=8, freq="D")
    ds = xr.Dataset(
        {k: ("time", np.linspace(1, 8, 8)) for k in data_vars},
        coords={"time": times},
    )
    path = tmp_path / "forcing.nc"
    ds.to_netcdf(path)
    return path


def test_canonical_forcing_is_keyed_by_cf_names():
    assert set(CANONICAL_FORCING) == CF_NAMES
    # Each entry still records its SUMMA-native shorthand for model-native layers.
    assert CANONICAL_FORCING["precipitation_flux"]["summa"] == "pptrate"
    assert CANONICAL_FORCING["air_temperature"]["summa"] == "airtemp"


def test_summa_source_names_are_renamed_to_cf(tmp_path):
    path = _write(tmp_path, ["pptrate", "airtemp", "SWRadAtm"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds
    assert "air_temperature" in ds
    assert "surface_downwelling_shortwave_flux" in ds
    assert "pptrate" not in ds and "airtemp" not in ds


def test_era5_style_aliases_are_renamed_to_cf(tmp_path):
    path = _write(tmp_path, ["tp", "t2m", "ssrd"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds
    assert "air_temperature" in ds
    assert "surface_downwelling_shortwave_flux" in ds


def test_already_cf_named_source_is_unchanged(tmp_path):
    path = _write(tmp_path, ["precipitation_flux", "air_temperature"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds and "air_temperature" in ds


def test_timestep_inferred_from_daily_axis(tmp_path):
    path = _write(tmp_path, ["pptrate"])
    ds = open_canonical_forcing(path)
    assert forcing_timestep_seconds(ds) == pytest.approx(86400.0)


def test_resolve_forcing_var_accepts_cf_key_and_aliases(tmp_path):
    path = _write(tmp_path, ["pptrate"])
    ds = xr.open_dataset(path)
    try:
        # Resolves the SUMMA source under the CF canonical key.
        assert resolve_forcing_var(ds, "precipitation_flux") == "pptrate"
    finally:
        ds.close()
