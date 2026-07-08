# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the forcing-remap stale-cache guard in FileProcessor.

A structurally-valid remap output must be reprocessed when the source forcing
gained variables the cached output lacks (e.g. RDRS v2.1 -> CaSR v3.2 adding
wind and radiation), and FORCE_RUN_ALL_STEPS must force reprocessing
unconditionally. See file_processor.filter_processed_files / _output_is_current.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.preprocessing.resampling.file_processor import FileProcessor

_ALL_VARS = [
    "air_temperature", "precipitation_flux", "surface_air_pressure",
    "specific_humidity", "wind_speed",
    "surface_downwelling_shortwave_flux", "surface_downwelling_longwave_flux",
]
_STALE_VARS = ["air_temperature", "precipitation_flux",
               "surface_air_pressure", "specific_humidity"]  # no wind/radiation

_CFG = {"DOMAIN_NAME": "TestDom", "FORCING_DATASET": "RDRS"}


def _make_nc(path, var_names):
    # >100 KB so it clears FileValidator.MIN_FILE_SIZE (not metadata-only).
    t = pd.date_range("2002-04-01", periods=8000, freq="h")
    ds = xr.Dataset(
        {v: ("time", np.arange(len(t), dtype="f4")) for v in var_names},
        coords={"time": t},
    )
    ds.to_netcdf(path, engine="h5netcdf")


@pytest.fixture
def fp(tmp_path):
    return FileProcessor(config=dict(_CFG), output_dir=tmp_path / "out",
                         logger=logging.getLogger("test"))


def test_output_is_current_true_when_output_covers_source(fp, tmp_path):
    src = tmp_path / "src.nc"
    out = tmp_path / "out.nc"
    _make_nc(src, ["air_temperature", "precipitation_flux", "wind_speed"])
    _make_nc(out, _ALL_VARS)  # superset
    assert fp._output_is_current(src, out) is True


def test_output_is_current_false_when_stale(fp, tmp_path):
    src = tmp_path / "src.nc"
    out = tmp_path / "out.nc"
    _make_nc(src, _ALL_VARS)        # has wind + radiation
    _make_nc(out, _STALE_VARS)      # missing them
    assert fp._output_is_current(src, out) is False


def test_filter_reprocesses_stale_output(fp, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "RDRS_monthly_200204.nc"
    _make_nc(src, _ALL_VARS)
    out = fp.determine_output_filename(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    _make_nc(out, _STALE_VARS)      # valid but stale

    remaining = fp.filter_processed_files([src])

    assert remaining == [src]       # flagged for reprocessing
    assert not out.exists()         # stale output deleted


def test_filter_skips_current_output(fp, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "RDRS_monthly_200204.nc"
    _make_nc(src, _ALL_VARS)
    out = fp.determine_output_filename(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    _make_nc(out, _ALL_VARS)        # complete

    remaining = fp.filter_processed_files([src])

    assert remaining == []          # up to date -> skipped
    assert out.exists()


def test_force_run_all_steps_reprocesses_even_when_current(tmp_path):
    fp = FileProcessor(
        config={**_CFG, "FORCE_RUN_ALL_STEPS": True},
        output_dir=tmp_path / "out", logger=logging.getLogger("test"))
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "RDRS_monthly_200204.nc"
    _make_nc(src, _ALL_VARS)
    out = fp.determine_output_filename(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    _make_nc(out, _ALL_VARS)        # current AND complete

    remaining = fp.filter_processed_files([src])

    assert remaining == [src]       # forced -> reprocess anyway
    assert not out.exists()
