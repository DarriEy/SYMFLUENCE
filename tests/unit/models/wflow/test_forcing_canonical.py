# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the Wflow preprocessor.

Wflow previously hardcoded ``precip * 3600`` (an hourly assumption) regardless of
the actual forcing timestep, undercounting precipitation on daily/3-hourly data.
With the canonical reader, precipitation comes from CF ``precipitation_flux``
(kg m-2 s-1) scaled by the *declared* timestep, and temperature from
``air_temperature`` (K).
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.wflow.preprocessor import WflowPreProcessor

_PROPS = {"lat": 51.0, "lon": -115.0, "area_m2": 2.0e9, "elev": 1500.0}


@pytest.fixture
def wflow_config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "WFLOW",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
    })


@pytest.fixture
def mock_logger():
    return Mock()


def _write_cf_forcing(forcing_dir, pptrate_value, *, n=10):
    forcing_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-02", periods=n, freq="D")
    xr.Dataset(
        {
            "precipitation_flux": ("time", np.full(n, pptrate_value, dtype="f8")),  # kg m-2 s-1
            "air_temperature": ("time", np.full(n, 283.15, dtype="f8")),            # K -> 10 degC
        },
        coords={"time": times},
    ).to_netcdf(forcing_dir / "forcing.nc")


def test_precip_uses_declared_timestep_not_hardcoded_hour(wflow_config, mock_logger):
    """Daily forcing must scale by 86400 s, not the old hardcoded 3600."""
    pp = WflowPreProcessor(wflow_config, mock_logger)
    pp._create_directory_structure()
    _write_cf_forcing(pp.forcing_basin_path, 1e-4)  # daily, 1e-4 kg/m2/s

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    ds = xr.open_dataset(pp.forcing_out_dir / "forcing.nc")
    try:
        # 1e-4 kg/m2/s * 86400 s = 8.64 mm/day  (old *3600 gave 0.36)
        assert float(ds["precip"].isel(y=0, x=0).mean()) == pytest.approx(8.64, rel=1e-3)
        assert float(ds["temp"].isel(y=0, x=0).mean()) == pytest.approx(10.0, abs=1e-3)
    finally:
        ds.close()
