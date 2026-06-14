# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the PCR-GLOBWB preprocessor.

Like CWatM, PCR-GLOBWB used a 3-way value-range guess for precipitation units.
With the canonical reader ``pptrate`` is always kg m-2 s-1, so precipitation is
deterministically ``pptrate * timestep / 1000`` m/day and temperature is
``airtemp - 273.15`` degC.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.pcrglobwb.preprocessor import PCRGLOBWBPreProcessor

_PROPS = {"lat": 51.0, "lon": -115.0, "area_m2": 2.0e9, "elev": 1500.0}


@pytest.fixture
def pcrglobwb_config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "PCRGLOBWB",
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


def _prepare(pp):
    """Construct dirs and inject a small grid (skip the bounds-based setup)."""
    pp._create_directory_structure()
    pp.grid_lats = np.array([51.5, 51.0, 50.5])
    pp.grid_lons = np.array([-115.5, -115.0, -114.5])
    pp.nrows = 3
    pp.ncols = 3


def test_precip_m_per_day_and_temp_celsius(pcrglobwb_config, mock_logger):
    pp = PCRGLOBWBPreProcessor(pcrglobwb_config, mock_logger)
    _prepare(pp)
    _write_canonical_forcing(pp.forcing_basin_path, 1e-4)

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    precip = xr.open_dataset(pp.forcing_out_dir / "precipitation.nc")
    temp = xr.open_dataset(pp.forcing_out_dir / "temperature.nc")
    try:
        assert float(precip["precipitation"].isel(lat=0, lon=0).mean()) == pytest.approx(8.64e-3, rel=1e-3)
        assert float(temp["temperature"].isel(lat=0, lon=0).mean()) == pytest.approx(10.0, abs=1e-3)
    finally:
        precip.close()
        temp.close()


def test_high_rate_precip_not_misclassified_as_depth(pcrglobwb_config, mock_logger):
    pp = PCRGLOBWBPreProcessor(pcrglobwb_config, mock_logger)
    _prepare(pp)
    _write_canonical_forcing(pp.forcing_basin_path, 0.5)

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    precip = xr.open_dataset(pp.forcing_out_dir / "precipitation.nc")
    try:
        # canonical: 0.5 * 86400 / 1000 = 43.2 m/day ; legacy guess -> 5e-4
        assert float(precip["precipitation"].isel(lat=0, lon=0).mean()) == pytest.approx(43.2, rel=1e-3)
    finally:
        precip.close()
