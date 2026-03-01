# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS NetCDF-CF reader and writer.

Reads and writes CF-1.6 compliant NetCDF files using xarray and the existing
``create_netcdf_encoding()`` utility for consistent encoding.
"""

from pathlib import Path
from typing import Dict, Optional

import xarray as xr

from .exceptions import FEWSAdapterError


def read_fews_netcdf(path: Path) -> xr.Dataset:
    """Read a FEWS-exported NetCDF-CF file into an xarray Dataset.

    Args:
        path: Path to the NetCDF file

    Returns:
        xr.Dataset

    Raises:
        FEWSAdapterError: If the file cannot be read
    """
    path = Path(path)
    if not path.is_file():
        raise FEWSAdapterError(f"FEWS NetCDF file not found: {path}")

    try:
        with xr.open_dataset(path) as ds:
            return ds.load()
    except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
        raise FEWSAdapterError(f"Failed to read FEWS NetCDF {path}: {exc}") from exc


def write_fews_netcdf(
    dataset: xr.Dataset,
    path: Path,
    *,
    compression: bool = True,
    complevel: int = 4,
    global_attrs: Optional[Dict[str, str]] = None,
) -> None:
    """Write an xarray Dataset as CF-1.6 compliant NetCDF for FEWS.

    Uses SYMFLUENCE's ``create_netcdf_encoding()`` for consistent encoding.

    Args:
        dataset: Dataset to write
        path: Output file path
        compression: Enable zlib compression
        complevel: Compression level (1-9)
        global_attrs: Additional global attributes

    Raises:
        FEWSAdapterError: If writing fails
    """
    path = Path(path)

    try:
        from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
    except ImportError:
        create_netcdf_encoding = None

    # Add CF conventions
    ds = dataset.copy()
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["institution"] = "SYMFLUENCE FEWS Adapter"
    if global_attrs:
        ds.attrs.update(global_attrs)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if create_netcdf_encoding is not None:
            encoding = create_netcdf_encoding(
                ds,
                compression=compression,
                complevel=complevel,
            )
        else:
            # Fallback if netcdf_utils not available
            encoding = {}
            for var in ds.data_vars:
                encoding[var] = {
                    "zlib": compression,
                    "complevel": complevel,
                    "_FillValue": -9999.0,
                }

        ds.to_netcdf(path, encoding=encoding)

    except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
        raise FEWSAdapterError(f"Failed to write FEWS NetCDF to {path}: {exc}") from exc
