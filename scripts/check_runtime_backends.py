#!/usr/bin/env python3
"""Validate the built-in storage backends with tiny local round trips."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# This process is the explicit compatibility probe. Avoid recursively spawning
# the startup detector when importing the package below.
os.environ["SYMFLUENCE_HDF5_PROBE"] = "1"


def _check_installed_wheel() -> None:
    import symfluence

    package_path = Path(symfluence.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"runtime contract must resolve from an installed wheel, got {package_path}"
        )
    print(f"symfluence={package_path}")


def _check_zarr() -> None:
    import numpy as np
    import xarray as xr
    import zarr

    dataset = xr.Dataset(
        {"temperature": (("x",), np.array([273.15, 274.15], dtype="float64"))},
        coords={"x": np.array([0, 1], dtype="int32")},
    )
    with tempfile.TemporaryDirectory(prefix="symfluence-zarr-probe-") as tmp:
        store = Path(tmp) / "sample.zarr"
        dataset.to_zarr(store, mode="w")
        with xr.open_zarr(store) as reopened:
            np.testing.assert_allclose(
                reopened["temperature"].values, [273.15, 274.15]
            )
    print(f"zarr={zarr.__version__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-installed-wheel",
        action="store_true",
        help="fail unless symfluence resolves from site-packages",
    )
    args = parser.parse_args()

    if args.require_installed_wheel:
        _check_installed_wheel()

    from symfluence.core.hdf5_probe import probe_hdf5_backends
    from symfluence.data.acquisition.handlers.hrrr import HRRRAcquirer

    probe_hdf5_backends()
    _check_zarr()

    import h5py
    import netCDF4

    print(
        "hdf5 backends="
        f"h5py {h5py.__version__}/HDF5 {h5py.version.hdf5_version}; "
        f"netCDF4 {netCDF4.__version__}/HDF5 {netCDF4.__hdf5libversion__}"
    )
    print(f"HRRR handler={HRRRAcquirer.__module__}.{HRRRAcquirer.__name__}")
    print("runtime backend contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
