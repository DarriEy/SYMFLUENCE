"""Isolated HDF5 backend compatibility probe.

This module is executed in a subprocess by :mod:`symfluence.core.hdf5_safety`.
Keeping the probe out of the host process prevents a failed combination of
wheel-bundled HDF5 libraries from poisoning the application interpreter.
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def probe_hdf5_backends() -> None:
    """Round-trip tiny datasets through both HDF5-backed xarray engines."""
    import h5py  # noqa: F401
    import netCDF4  # noqa: F401
    import numpy as np
    import xarray as xr

    dataset = xr.Dataset(
        {"flow": (("time",), np.array([1.25, 2.5], dtype="float64"))},
        coords={"time": np.array([0, 1], dtype="int32")},
    )
    with tempfile.TemporaryDirectory(prefix="symfluence-hdf5-probe-") as tmp:
        root = Path(tmp)
        for engine in ("netcdf4", "h5netcdf"):
            path = root / f"{engine}.nc"
            dataset.to_netcdf(path, engine=engine)
            with xr.open_dataset(path, engine=engine) as reopened:
                np.testing.assert_allclose(reopened["flow"].values, [1.25, 2.5])


if __name__ == "__main__":
    probe_hdf5_backends()
