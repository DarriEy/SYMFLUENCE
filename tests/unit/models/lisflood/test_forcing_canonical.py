# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the LISFLOOD preprocessor.

Verifies that ``_generate_forcing`` reads forcing through the model-ready
canonical reader (``open_canonical_forcing``): precipitation is taken from the
canonical ``pptrate`` (kg m-2 s-1) and scaled by the declared timestep to
mm/step, and temperature from ``airtemp`` (K) converted to degC — without the
old value-range heuristics (``precip.max() < 0.1`` / ``temp.mean() > 100``).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.lisflood.preprocessor import LisfloodPreProcessor

_PROPS = {"lat": 51.0, "lon": -115.0, "area_m2": 2.0e9, "elev": 1500.0}


def _write_canonical_forcing(forcing_dir, pptrate_value, *, n=10, timestep_attr=None):
    """Write a tiny canonical forcing NetCDF into the preprocessor's forcing dir."""
    forcing_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-02", periods=n, freq="D")
    ds = xr.Dataset(
        {
            "pptrate": ("time", np.full(n, pptrate_value, dtype="f8")),   # kg m-2 s-1
            "airtemp": ("time", np.full(n, 283.15, dtype="f8")),          # K -> 10 degC
        },
        coords={"time": times},
    )
    if timestep_attr is not None:
        ds.attrs["timestep_seconds"] = float(timestep_attr)
    ds.to_netcdf(forcing_dir / "forcing.nc")
    return forcing_dir


def test_precip_scaled_by_timestep_and_temp_to_celsius(lisflood_config, mock_logger, setup_lisflood_directories):
    """Realistic canonical forcing → pptrate * dt mm/day, airtemp - 273.15 degC."""
    pp = LisfloodPreProcessor(lisflood_config, mock_logger)
    pp._create_directory_structure()
    _write_canonical_forcing(pp.forcing_basin_path, 1e-4)  # 1e-4 kg/m2/s, daily

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    pr = xr.open_dataset(pp.forcing_out_dir / "pr.nc")
    ta = xr.open_dataset(pp.forcing_out_dir / "ta.nc")
    try:
        # daily timestep: 1e-4 kg/m2/s * 86400 s = 8.64 mm/day
        assert float(pr["pr"].isel(y=0, x=0).mean()) == pytest.approx(8.64, rel=1e-3)
        assert float(ta["ta"].isel(y=0, x=0).mean()) == pytest.approx(10.0, abs=1e-3)
    finally:
        pr.close()
        ta.close()


def test_high_rate_precip_is_not_treated_as_depth(lisflood_config, mock_logger, setup_lisflood_directories):
    """An (extreme) rate whose magnitude exceeds the old 0.1 guard must still be
    scaled by the timestep — the canonical contract says pptrate is always a
    rate. The legacy `precip.max() < 0.1` heuristic would have left it unscaled."""
    pp = LisfloodPreProcessor(lisflood_config, mock_logger)
    pp._create_directory_structure()
    _write_canonical_forcing(pp.forcing_basin_path, 0.5)  # 0.5 kg/m2/s (extreme), daily

    with patch.object(pp, "_get_catchment_properties", return_value=_PROPS):
        pp._generate_forcing()

    pr = xr.open_dataset(pp.forcing_out_dir / "pr.nc")
    try:
        # canonical: 0.5 * 86400 = 43200 ; legacy heuristic would have left 0.5
        assert float(pr["pr"].isel(y=0, x=0).mean()) == pytest.approx(43200.0, rel=1e-3)
    finally:
        pr.close()
