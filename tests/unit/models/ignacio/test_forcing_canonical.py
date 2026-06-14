# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the IGNACIO (fire) preprocessor.

IGNACIO converted CF ``precipitation_flux`` to its weather CSV with a hardcoded
``* 3600`` (hourly assumption). With the canonical reader it scales by the
declared timestep instead, so daily forcing is no longer undercounted 24x.
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.ignacio.preprocessor import IGNACIOPreProcessor


@pytest.fixture
def ignacio_config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "IGNACIO",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
    })


def _write_cf_forcing(forcing_dir, pptrate_value, *, n=10):
    forcing_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-02", periods=n, freq="D")
    xr.Dataset(
        {
            "precipitation_flux": ("time", np.full(n, pptrate_value, dtype="f8")),  # kg m-2 s-1
            "air_temperature": ("time", np.full(n, 283.15, dtype="f8")),            # K -> 10 degC
            "wind_speed": ("time", np.full(n, 2.0, dtype="f8")),
        },
        coords={"time": times},
    ).to_netcdf(forcing_dir / "ERA5_forcing.nc")


def test_precip_scaled_by_declared_timestep(ignacio_config):
    pp = IGNACIOPreProcessor(ignacio_config, Mock())
    pp.ignacio_input_dir.mkdir(parents=True, exist_ok=True)
    _write_cf_forcing(pp.forcing_basin_path, 1e-4)  # daily, 1e-4 kg/m2/s

    weather_csv = pp._convert_era5_to_weather_csv({})

    assert weather_csv is not None and weather_csv.exists()
    df = pd.read_csv(weather_csv)
    # 1e-4 kg/m2/s * 86400 s = 8.64 mm per step (old *3600 gave 0.36)
    assert df["PRECIP"].mean() == pytest.approx(8.64, rel=1e-3)
    assert df["TEMP"].mean() == pytest.approx(10.0, abs=1e-3)
