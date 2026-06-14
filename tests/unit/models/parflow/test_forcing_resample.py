# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow forcing ingestion resamples non-hourly source to its hourly cadence.

ParFlow's rainfall pipeline equates one forcing step with one hour (mm/hr rates,
day-grouping by step count). A 3-hourly source must therefore be resampled to
hourly at ingestion so the cadence assumption holds; an hourly source is
unchanged.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.parflow.preprocessor import ParFlowPreProcessor


def _config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-01-06 00:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "PARFLOW",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 10800,
    })


def _write_forcing(pp, freq, n):
    from symfluence.core.mixins.project import resolve_forcing_basin_path
    fdir = resolve_forcing_basin_path(pp.project_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=n, freq=freq)
    # (time, hru) shape with a single HRU — ParFlow selects column 0.
    xr.Dataset(
        {
            "precipitation_flux": (("time", "hru"), np.full((n, 1), 1e-4)),
            "air_temperature": (("time", "hru"), np.full((n, 1), 283.15)),
        },
        coords={"time": times, "hru": [0]},
    ).to_netcdf(fdir / "ERA5_remapped.nc")


def _cache_step_seconds(settings_dir):
    cache = np.load(settings_dir / "hourly_forcing_cache.npz")
    times = pd.DatetimeIndex(cache["times"].astype("datetime64[ns]"))
    diffs = np.diff(times.values).astype("timedelta64[s]").astype(int)
    assert len(set(diffs)) == 1, f"non-uniform spacing: {set(diffs)}"
    return int(diffs[0])


def test_three_hourly_source_is_resampled_to_hourly(tmp_path):
    pp = ParFlowPreProcessor(_config(tmp_path), Mock())
    _write_forcing(pp, "3h", n=40)  # 3-hourly source
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()

    with patch.object(pp, "_get_latitude", return_value=51.0):
        result = pp._prepare_daily_rainfall(settings_dir)

    assert result is not None
    # Cache must be hourly (3600 s), not the 3-hourly source spacing (10800 s).
    assert _cache_step_seconds(settings_dir) == 3600


def test_hourly_source_passes_through_unchanged(tmp_path):
    pp = ParFlowPreProcessor(_config(tmp_path), Mock())
    _write_forcing(pp, "h", n=120)  # hourly source
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()

    with patch.object(pp, "_get_latitude", return_value=51.0):
        result = pp._prepare_daily_rainfall(settings_dir)

    assert result is not None
    assert _cache_step_seconds(settings_dir) == 3600
