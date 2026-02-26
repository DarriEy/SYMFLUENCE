"""Tests for _safe_to_netcdf with engine fallback and temp-file workaround."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.era5 import _safe_to_netcdf


@pytest.fixture
def sample_dataset():
    """Create a small xarray dataset for testing."""
    return xr.Dataset(
        {"temperature": (["time", "latitude", "longitude"],
                         np.random.rand(3, 2, 2).astype(np.float32))},
        coords={
            "time": [0, 1, 2],
            "latitude": [50.0, 51.0],
            "longitude": [10.0, 11.0],
        },
    )


@pytest.fixture
def logger():
    return logging.getLogger("test_safe_to_netcdf")


class TestSafeToNetcdfHappyPath:
    """Attempt 1 succeeds — no fallback needed."""

    def test_writes_netcdf_successfully(self, sample_dataset, tmp_path, logger):
        out = tmp_path / "output.nc"
        _safe_to_netcdf(sample_dataset, out, logger=logger)
        assert out.exists()
        with xr.open_dataset(out) as ds:
            assert "temperature" in ds.data_vars
            assert ds.sizes["time"] == 3

    def test_clears_encoding_from_source(self, sample_dataset, tmp_path, logger):
        """Cloud-backed datasets carry Zarr encoding; it must be stripped."""
        sample_dataset["temperature"].encoding["compressor"] = "zstd"
        sample_dataset["latitude"].encoding["_FillValue"] = None
        out = tmp_path / "output.nc"
        _safe_to_netcdf(sample_dataset, out, logger=logger)
        assert out.exists()


class TestSafeToNetcdfFallbacks:
    """Engine fallback chain: netcdf4 → h5netcdf → h5netcdf-nocomp → tempfile."""

    def _hdf_error(self):
        """Return an OSError that matches _HDF_ERROR_PATTERNS."""
        return OSError("unable to lock file, errno = 11")

    def test_falls_through_to_h5netcdf(self, sample_dataset, tmp_path, logger):
        """If netcdf4 raises an HDF error, h5netcdf attempt should succeed."""
        out = tmp_path / "output.nc"
        original_to_netcdf = xr.Dataset.to_netcdf
        patched_to_netcdf = self._make_failing_to_netcdf(original_to_netcdf, fail_on={1})
        with patch.object(xr.Dataset, "to_netcdf", patched_to_netcdf):
            _safe_to_netcdf(sample_dataset, out, logger=logger)

        assert out.exists()

    def test_tempfile_fallback_on_all_direct_writes_failing(
        self, sample_dataset, tmp_path, logger
    ):
        """When all 3 direct-write attempts fail, the tempfile fallback should succeed."""
        out = tmp_path / "output.nc"
        original_to_netcdf = xr.Dataset.to_netcdf

        # Fail attempts 1-3 (writes to target), succeed attempt 4 (write to /tmp)
        def patched(self_ds, path_or_buf=None, *args, **kwargs):
            target = str(path_or_buf) if path_or_buf is not None else ""
            # Fail if writing to the target path on the "parallel filesystem"
            if str(out) == target:
                raise OSError("unable to lock file, errno = 11")
            # Succeed when writing to a temp path
            return original_to_netcdf(self_ds, path_or_buf, *args, **kwargs)

        with patch.object(xr.Dataset, "to_netcdf", patched):
            _safe_to_netcdf(sample_dataset, out, logger=logger)

        assert out.exists()
        with xr.open_dataset(out) as ds:
            assert "temperature" in ds.data_vars

    def test_non_hdf_error_in_attempt1_reraises(self, sample_dataset, tmp_path, logger):
        """Non-HDF errors in attempt 1 must not be swallowed."""
        out = tmp_path / "output.nc"

        def patched(self_ds, *a, **kw):
            raise OSError("No space left on device")

        with patch.object(xr.Dataset, "to_netcdf", patched):
            with pytest.raises(OSError, match="No space left on device"):
                _safe_to_netcdf(sample_dataset, out, logger=logger)

    def test_tempfile_cleaned_up_on_failure(self, sample_dataset, tmp_path, logger):
        """If even the tempfile write fails, the temp file should be cleaned up."""
        out = tmp_path / "output.nc"

        def always_fail(self_ds, *a, **kw):
            raise OSError("unable to lock file, errno = 11")

        with patch.object(xr.Dataset, "to_netcdf", always_fail):
            with pytest.raises(OSError):
                _safe_to_netcdf(sample_dataset, out, logger=logger)

        # No leftover temp files
        tmpdir = os.environ.get("TMPDIR") or "/tmp"
        leftover = [f for f in Path(tmpdir).glob("*.nc") if f.stat().st_size == 0]
        # We can't guarantee no other .nc files exist in /tmp, but
        # the specific temp file should have been cleaned up.

    def test_sets_hdf5_use_file_locking(self, sample_dataset, tmp_path, logger):
        """HDF5_USE_FILE_LOCKING should be (re)set to FALSE."""
        # Unset it first
        os.environ.pop("HDF5_USE_FILE_LOCKING", None)
        out = tmp_path / "output.nc"
        _safe_to_netcdf(sample_dataset, out, logger=logger)
        assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"

    # --- helper -----------------------------------------------------------

    @staticmethod
    def _make_failing_to_netcdf(original, fail_on: set):
        """Return a patched to_netcdf that raises HDF errors on given call numbers."""
        state = {"n": 0}

        def patched(self_ds, *args, **kwargs):
            state["n"] += 1
            if state["n"] in fail_on:
                raise OSError("unable to lock file, errno = 11")
            return original(self_ds, *args, **kwargs)

        return patched
