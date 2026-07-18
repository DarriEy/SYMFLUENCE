# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Source-driven completeness contract of the model-ready store.

``data/model_ready`` is created unconditionally at project init, so bare
directory existence must never count as complete, and every basin-averaged
forcing source file must actually be materialized in the store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")
import xarray as xr

from symfluence.data.model_ready.store_builder import ModelReadyStoreBuilder


def _write_forcing(path):
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    xr.Dataset(
        {"airtemp": ("time", np.full(4, 283.15))}, coords={"time": times}
    ).to_netcdf(path)


@pytest.fixture
def builder(tmp_path):
    return ModelReadyStoreBuilder(project_dir=tmp_path, domain_name="test")


def test_missing_store_dir_is_incomplete(builder):
    assert not builder.is_store_complete()


def test_bare_store_dir_from_project_init_is_incomplete(builder, tmp_path):
    (tmp_path / "data" / "model_ready").mkdir(parents=True)
    assert not builder.is_store_complete()


def test_partially_materialized_forcings_are_incomplete(builder, tmp_path):
    src = tmp_path / "data" / "forcing" / "basin_averaged_data"
    src.mkdir(parents=True)
    _write_forcing(src / "a.nc")
    _write_forcing(src / "b.nc")

    store = tmp_path / "data" / "model_ready" / "forcings"
    store.mkdir(parents=True)
    _write_forcing(store / "a.nc")  # b.nc never materialized

    assert not builder.is_store_complete()


def test_fully_materialized_forcings_are_complete(builder, tmp_path):
    src = tmp_path / "data" / "forcing" / "basin_averaged_data"
    src.mkdir(parents=True)
    _write_forcing(src / "a.nc")

    store = tmp_path / "data" / "model_ready" / "forcings"
    store.mkdir(parents=True)
    _write_forcing(store / "a.nc")

    assert builder.is_store_complete()


def test_corrupt_observations_artifact_is_incomplete(builder, tmp_path):
    pytest.importorskip("netCDF4")
    src = tmp_path / "data" / "forcing" / "basin_averaged_data"
    src.mkdir(parents=True)
    _write_forcing(src / "a.nc")
    store = tmp_path / "data" / "model_ready" / "forcings"
    store.mkdir(parents=True)
    _write_forcing(store / "a.nc")

    obs_dir = tmp_path / "data" / "model_ready" / "observations"
    obs_dir.mkdir(parents=True)
    (obs_dir / "test_observations.nc").write_bytes(b"not a netcdf file")

    assert not builder.is_store_complete()
