# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the CWatM preprocessor.

CWatM's old ingestion guessed precipitation units from value ranges (a 3-way
branch on ``precip_raw.max()``). With the canonical reader, ``pptrate`` is
always a rate in kg m-2 s-1, so precipitation is deterministically
``pptrate * timestep / 1000`` m per step and temperature is ``airtemp - 273.15``.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.cwatm.preprocessor import CWatMPreProcessor

_PROPS = {"lat": 51.0, "lon": -115.0, "area_m2": 2.0e9, "elev": 1500.0}


@pytest.fixture
def cwatm_config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "CWATM",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
    })


@pytest.fixture
def mock_logger():
    return Mock()


def _write_canonical_forcing(forcing_dir, pptrate_value, *, n=10):
    forcing_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-02", periods=n, freq="D")
    xr.Dataset(
        {
            "pptrate": ("time", np.full(n, pptrate_value, dtype="f8")),  # kg m-2 s-1
            "airtemp": ("time", np.full(n, 283.15, dtype="f8")),         # K -> 10 degC
        },
        coords={"time": times},
    ).to_netcdf(forcing_dir / "forcing.nc")


def _run(pp):
    pp._create_directory_structure()
    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()


def test_precip_in_m_per_day_and_temp_celsius(cwatm_config, mock_logger):
    pp = CWatMPreProcessor(cwatm_config, mock_logger)
    pp._create_directory_structure()
    _write_canonical_forcing(pp.forcing_basin_path, 1e-4)  # daily, 1e-4 kg/m2/s

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    precip = xr.open_dataset(pp.forcing_out_dir / "precipitation.nc")
    tavg = xr.open_dataset(pp.forcing_out_dir / "tavg.nc")
    try:
        # 1e-4 kg/m2/s * 86400 s / 1000 = 8.64e-3 m/day
        assert float(precip["precipitation"].isel(lat=0, lon=0).mean()) == pytest.approx(8.64e-3, rel=1e-3)
        assert float(tavg["tavg"].isel(lat=0, lon=0).mean()) == pytest.approx(10.0, abs=1e-3)
    finally:
        precip.close()
        tavg.close()


def test_high_rate_precip_not_misclassified_as_depth(cwatm_config, mock_logger):
    """An extreme rate (0.5 kg/m2/s) must be scaled by the timestep. The old
    3-way value-range guess would have treated it as mm/day (÷1000)."""
    pp = CWatMPreProcessor(cwatm_config, mock_logger)
    pp._create_directory_structure()
    _write_canonical_forcing(pp.forcing_basin_path, 0.5)

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    precip = xr.open_dataset(pp.forcing_out_dir / "precipitation.nc")
    try:
        # canonical: 0.5 * 86400 / 1000 = 43.2 m/day ; legacy guess -> 5e-4
        assert float(precip["precipitation"].isel(lat=0, lon=0).mean()) == pytest.approx(43.2, rel=1e-3)
    finally:
        precip.close()
