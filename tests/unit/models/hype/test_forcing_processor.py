# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for HYPE forcing processing (lumped-domain edge cases)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from symfluence.models.hype.forcing_processor import HYPEForcingProcessor


def _make_processor(tmp_path):
    out = tmp_path / "settings"
    out.mkdir()
    return HYPEForcingProcessor(
        config={},
        logger=logging.getLogger("test_hype_forcing"),
        forcing_input_dir=tmp_path / "in",
        output_path=out,
        cache_path=tmp_path / "cache",
    )


def _write_forcing_chunk(path, start, periods, base_value):
    """Write a small (hru=1) forcing NetCDF like the remapped basin-averaged store."""
    times = pd.date_range(start, periods=periods, freq="h")
    # air_temperature encodes the timestamp index (base_value + hour) so we can
    # verify ordering and first-occurrence dedup after the merge.
    data = (base_value + np.arange(periods, dtype="float64")).reshape(periods, 1)
    ds = xr.Dataset(
        {"air_temperature": (("time", "hru"), data)},
        coords={"time": times},
    )
    ds["hru"] = ("hru", [0])
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()


def test_merge_without_cdo_is_time_sorted_deduped_and_fast(tmp_path, monkeypatch):
    """The no-CDO xarray fallback must merge >=2 files into a time-sorted,
    duplicate-free series (matching `cdo mergetime`) and finish quickly.

    Regression for the native-Windows hang: the old lazy
    open_mfdataset(combine='nested', concat_dim='time').sortby('time') path did
    not complete when input files had overlapping/identical time coordinates.
    """
    import time as _time

    import symfluence.models.hype.forcing_processor as fp

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (tmp_path / "cache").mkdir()

    # Two chunks whose time ranges OVERLAP: part1 = hours 0..71, part2 = hours
    # 48..119. Hours 48..71 are duplicated across both files. A correct merge
    # yields 120 unique hourly steps, strictly increasing, first-occurrence wins.
    _write_forcing_chunk(in_dir / "part1.nc", "2002-01-01", 72, base_value=0.0)
    _write_forcing_chunk(in_dir / "part2.nc", "2002-01-03", 72, base_value=1000.0)

    # Force the CDO-absent branch: make cdo.Cdo() raise (caught -> xarray fallback).
    def _no_cdo(*args, **kwargs):
        raise OSError("cdo binary not available (simulated native Windows)")

    monkeypatch.setattr(fp.cdo, "Cdo", _no_cdo)

    proc = _make_processor(tmp_path)

    t0 = _time.time()
    merged = proc._merge_forcing_files()
    elapsed = _time.time() - t0

    assert merged is not None and merged.exists()
    # Must be quick; generous ceiling so it fails only on a true hang/regression.
    assert elapsed < 30, f"no-CDO merge took {elapsed:.1f}s (expected << 30s)"

    with xr.open_dataset(merged, engine="h5netcdf") as ds:
        tv = ds["time"].values
        # 72 + 72 with 24 overlapping hours dropped -> 120 unique steps.
        assert len(tv) == 120
        assert np.all(tv[1:] > tv[:-1]), "merged time axis must be strictly increasing"
        assert len(np.unique(tv)) == len(tv), "duplicate timesteps must be dropped"
        # First-occurrence wins for the overlap: hour 48 came from part1 (value 48),
        # not part2 (value 1024).
        temps = ds["air_temperature"].values.reshape(-1)
        assert temps[0] == 0.0
        assert temps[48] == 48.0


def test_lumped_forcing_with_singleton_spatial_dim(tmp_path):
    """A lumped store may carry a singleton spatial dim whose level name we do not
    unstack on (e.g. 'gru'). That previously left a MultiIndex of tuples and made
    pd.to_datetime raise "<class 'tuple'> is not convertible to datetime". The
    daily conversion must instead collapse the singleton dim and emit one subbasin.
    """
    times = pd.date_range("2000-01-01", periods=48, freq="h")
    # dims (time, gru) with gru size 1, and no hruId coordinate -> falls to the
    # lumped branch with a (time, gru) MultiIndex in to_series().
    data = np.ones((len(times), 1), dtype="float32")
    ds = xr.Dataset(
        {"airtemp": (("time", "gru"), data)},
        coords={"time": times, "gru": [0]},
    )
    nc = tmp_path / "merged.nc"
    ds.to_netcdf(nc, engine="h5netcdf")

    out_txt = tmp_path / "Tobs.txt"
    proc = _make_processor(tmp_path)
    proc._convert_hourly_to_daily(
        input_file_name=nc,
        variable_in="airtemp",
        variable_out="airtemp",
        var_id="hruId",
        stat="mean",
        output_file_name_txt=out_txt,
    )

    assert out_txt.exists()
    written = pd.read_csv(out_txt, sep="\t")
    # Two calendar days, parseable dates, and a single subbasin column promoted to id 1.
    assert len(written) == 2
    pd.to_datetime(written["time"])  # must not raise
    assert [c for c in written.columns if c != "time"] == ["1"]
