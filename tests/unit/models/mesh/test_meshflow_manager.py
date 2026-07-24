"""
Tests for the meshflow manager's CDO compatibility shim.

meshflow merges the MESH forcing chunks by shelling out to the CDO binary.
CDO has no Windows build, so the merge is transparently replaced with an
xarray implementation whenever no CDO executable can be resolved.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from symfluence.models.mesh.preprocessing.meshflow_manager import (  # noqa: E402
    _XarrayMergetime,
    find_cdo_binary,
)


@pytest.fixture
def forcing_chunks(tmp_path: Path) -> Path:
    """Three single-day forcing chunks written to disk out of time order."""
    for offset in (48, 0, 24):
        times = pd.date_range("2001-01-01", periods=24, freq="h") + pd.Timedelta(hours=offset)
        xr.Dataset(
            {
                "air_temperature": (
                    ("time", "subbasin"),
                    np.full((24, 2), offset, dtype="f4"),
                ),
                "precipitation": (
                    ("time", "subbasin"),
                    np.full((24, 2), 1.0, dtype="f4"),
                ),
            },
            coords={"time": times, "subbasin": np.array([1, 2], dtype="i4")},
        ).to_netcdf(tmp_path / f"chunk_{offset:03d}.nc")
    return tmp_path


class TestFindCdoBinary:
    """Tests for CDO binary resolution."""

    def test_prefers_cdo_environment_variable(self, tmp_path, monkeypatch):
        """A CDO env var naming an existing file wins, as python-cdo requires."""
        binary = tmp_path / "cdo.exe"
        binary.write_text("")
        monkeypatch.setenv("CDO", str(binary))

        assert find_cdo_binary() == str(binary)

    def test_ignores_cdo_environment_variable_that_is_not_a_file(self, tmp_path, monkeypatch):
        """python-cdo only honors CDO when it is a file, so neither do we."""
        monkeypatch.setenv("CDO", str(tmp_path / "missing"))

        with patch("shutil.which", return_value=None):
            assert find_cdo_binary() is None

    def test_falls_back_to_path_lookup(self, monkeypatch):
        """PATH resolution covers cdo.exe via PATHEXT on Windows."""
        monkeypatch.delenv("CDO", raising=False)

        with patch("shutil.which", return_value=r"C:\tools\cdo.exe") as which:
            assert find_cdo_binary() == r"C:\tools\cdo.exe"
        which.assert_called_once_with("cdo")


class TestXarrayMergetime:
    """Tests for the xarray stand-in for cdo.Cdo().mergetime."""

    def test_merges_glob_in_time_order(self, forcing_chunks):
        """Chunks are concatenated along time regardless of filename order."""
        merged = _XarrayMergetime().mergetime(
            input=str(forcing_chunks / "*.nc"),
            returnXArray=["air_temperature", "precipitation"],
        )

        assert merged.sizes["time"] == 72
        assert merged.indexes["time"].is_monotonic_increasing
        assert float(merged["air_temperature"][0, 0]) == 0.0
        assert float(merged["air_temperature"][-1, 0]) == 48.0

    def test_returns_only_requested_variables(self, forcing_chunks):
        """returnXArray selects variables, matching python-cdo's behaviour."""
        merged = _XarrayMergetime().mergetime(
            input=str(forcing_chunks / "*.nc"),
            returnXArray=["air_temperature"],
        )

        assert sorted(merged.data_vars) == ["air_temperature"]

    def test_accepts_an_explicit_file_list(self, forcing_chunks):
        """CDO also takes a sequence of paths rather than a glob."""
        files = sorted(str(p) for p in forcing_chunks.glob("*.nc"))
        merged = _XarrayMergetime().mergetime(input=files, returnXArray=["air_temperature"])

        assert merged.sizes["time"] == 72

    def test_writes_to_output_when_asked(self, forcing_chunks, tmp_path):
        """The output= form writes a merged file and returns its path."""
        target = tmp_path / "merged.nc"
        result = _XarrayMergetime().mergetime(
            input=str(forcing_chunks / "*.nc"), output=str(target)
        )

        assert result == str(target)
        assert target.exists()

    def test_no_matching_files_raises_model_execution_error(self, tmp_path):
        """An empty match reports a MESH-domain error, not a bare OS error."""
        from symfluence.core.exceptions import ModelExecutionError

        with pytest.raises(ModelExecutionError, match="No forcing files matched"):
            _XarrayMergetime().mergetime(input=str(tmp_path / "*.nc"))

    def test_merged_result_is_not_dask_backed(self, forcing_chunks):
        """The merge must return NumPy-backed data, never a Dask graph.

        Regression guard for the runaway where ``open_mfdataset`` inherited the
        forcing's on-disk time chunking (one chunk per timestep). meshflow then
        stacked transpose / convert_calendar / pint ops on that graph and the
        final ``to_netcdf`` spun for 11+ h inside Dask graph optimisation. A
        NumPy-backed merge keeps that graph from ever existing.
        """
        merged = _XarrayMergetime().mergetime(
            input=str(forcing_chunks / "*.nc"),
            returnXArray=["air_temperature", "precipitation"],
        )

        for name, var in merged.variables.items():
            assert not hasattr(var.data, "npartitions"), (
                f"{name!r} is Dask-backed; the merge must materialise to NumPy "
                "to avoid the graph-optimisation runaway"
            )
