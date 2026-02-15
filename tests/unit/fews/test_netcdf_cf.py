"""Tests for FEWS NetCDF-CF reader and writer."""

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.exceptions import FEWSAdapterError
from symfluence.fews.netcdf_cf import read_fews_netcdf, write_fews_netcdf


class TestReadFEWSNetCDF:
    def test_read_basic(self, sample_netcdf):
        ds = read_fews_netcdf(sample_netcdf)
        assert "P.obs" in ds.data_vars
        assert "T.obs" in ds.data_vars
        ds.close()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FEWSAdapterError, match="not found"):
            read_fews_netcdf(tmp_path / "nonexistent.nc")


class TestWriteFEWSNetCDF:
    def test_write_basic(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"discharge": ("time", np.random.rand(len(times)))},
            coords={"time": times},
        )
        path = tmp_path / "output.nc"
        write_fews_netcdf(ds, path)
        assert path.exists()

    def test_cf_conventions(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"Q": ("time", np.ones(len(times)))},
            coords={"time": times},
        )
        path = tmp_path / "cf.nc"
        write_fews_netcdf(ds, path)

        ds2 = xr.open_dataset(path)
        assert ds2.attrs["Conventions"] == "CF-1.6"
        ds2.close()

    def test_round_trip(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-01T06:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {
                "precip": ("time", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "temp": ("time", [280.0, 281.0, 282.0, 283.0, 284.0, 285.0]),
            },
            coords={"time": times},
        )
        path = tmp_path / "roundtrip.nc"
        write_fews_netcdf(ds, path)

        ds2 = read_fews_netcdf(path)
        np.testing.assert_allclose(ds2["precip"].values, ds["precip"].values, rtol=1e-5)
        ds2.close()

    def test_global_attrs(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
        ds = xr.Dataset({"Q": ("time", np.ones(len(times)))}, coords={"time": times})
        path = tmp_path / "attrs.nc"
        write_fews_netcdf(ds, path, global_attrs={"source": "FEWS test"})

        ds2 = xr.open_dataset(path)
        assert ds2.attrs["source"] == "FEWS test"
        ds2.close()

    def test_creates_parent_dir(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
        ds = xr.Dataset({"Q": ("time", np.ones(len(times)))}, coords={"time": times})
        path = tmp_path / "subdir" / "deep" / "output.nc"
        write_fews_netcdf(ds, path)
        assert path.exists()
