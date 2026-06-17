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
