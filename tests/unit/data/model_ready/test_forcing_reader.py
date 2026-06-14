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


# ---------------------------------------------------------------------------
# resample_canonical_forcing
# ---------------------------------------------------------------------------
from symfluence.data.model_ready.forcing_reader import resample_canonical_forcing  # noqa: E402


def _forcing(times, pptrate, airtemp):
    return xr.Dataset(
        {
            "precipitation_flux": ("time", np.asarray(pptrate, dtype="f8")),
            "air_temperature": ("time", np.asarray(airtemp, dtype="f8")),
        },
        coords={"time": pd.DatetimeIndex(times)},
    )


def test_resample_is_noop_when_already_at_target():
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    ds = _forcing(times, np.full(6, 1e-4), np.full(6, 283.15))
    out = resample_canonical_forcing(ds, 3600)
    assert out["time"].size == 6
    np.testing.assert_array_equal(out["precipitation_flux"].values, ds["precipitation_flux"].values)


def test_resample_3hourly_to_hourly_conserves_precip_total():
    # 4 source steps at 3h spacing, constant rate.
    times = pd.date_range("2020-01-01", periods=4, freq="3h")
    rate = 2e-4
    ds = _forcing(times, np.full(4, rate), np.full(4, 283.15))

    out = resample_canonical_forcing(ds, 3600)

    # 4 intervals * 3 h = 12 hourly steps covering the full source span.
    assert out["time"].size == 12
    assert float(out.attrs["timestep_seconds"]) == 3600.0
    # Rate is a step function -> every hour carries the source rate.
    np.testing.assert_allclose(out["precipitation_flux"].values, rate, rtol=1e-9)
    # Accumulated total conserved: source sum(rate*3h) == hourly sum(rate*1h).
    src_total = float((ds["precipitation_flux"] * 10800).sum())
    out_total = float((out["precipitation_flux"] * 3600).sum())
    assert out_total == pytest.approx(src_total, rel=1e-9)


def test_resample_interpolates_state_variables():
    # Temperature ramps 280 -> 286 over two 3h steps; hourly interp is linear.
    times = pd.date_range("2020-01-01", periods=3, freq="3h")
    ds = _forcing(times, np.full(3, 1e-4), [280.0, 283.0, 286.0])

    out = resample_canonical_forcing(ds, 3600)
    temps = out["air_temperature"].values

    # First hour equals the first source value; interior is linearly interpolated.
    assert temps[0] == pytest.approx(280.0)
    assert temps[1] == pytest.approx(281.0, abs=1e-6)   # 1/3 of the way 280->283
    assert temps[3] == pytest.approx(283.0, abs=1e-6)   # the second source point
    assert not np.isnan(temps).any()


def test_resample_daily_to_hourly_conserves_precip():
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    rate = 1e-4
    ds = _forcing(times, np.full(3, rate), np.full(3, 283.15))
    out = resample_canonical_forcing(ds, 3600)
    assert out["time"].size == 72  # 3 days * 24 h
    out_total = float((out["precipitation_flux"] * 3600).sum())
    src_total = float((ds["precipitation_flux"] * 86400).sum())
    assert out_total == pytest.approx(src_total, rel=1e-9)
