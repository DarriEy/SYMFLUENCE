# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Canonical forcing reader for the model-ready store.

Single entry point every model adapter should use to load basin-averaged
forcing. It guarantees the canonical SYMFLUENCE vocabulary (pptrate, airtemp,
SWRadAtm, LWRadAtm, windspd, spechum, airpres) regardless of the source dataset
(CARRA, ERA5, ...) and exposes the forcing timestep via ``ds.attrs['timestep_seconds']``,
so adapters never re-parse raw variable names, units, or guess the timestep.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import xarray as xr

from .cf_conventions import CANONICAL_FORCING, resolve_forcing_var

logger = logging.getLogger(__name__)


def open_canonical_forcing(
    forcing_files: Union[Path, List[Path]],
) -> xr.Dataset:
    """Open model-ready forcing and return it under canonical names.

    - Aliased source variables (e.g. ``precipitation_flux`` -> ``pptrate``) are
      renamed to the canonical vocabulary.
    - ``ds.attrs['timestep_seconds']`` is set from the store's declared attribute
      or, failing that, inferred from the time axis.

    Args:
        forcing_files: a single NetCDF path or a list of them.

    Returns:
        xarray.Dataset with canonical variable names and a timestep_seconds attr.
    """
    files = [forcing_files] if isinstance(forcing_files, (str, Path)) else list(forcing_files)
    files = [Path(f) for f in files]
    if len(files) == 1:
        ds = xr.open_dataset(files[0])
    else:
        try:
            ds = xr.open_mfdataset(files, combine='by_coords', data_vars='minimal',
                                   coords='minimal', compat='override')
        except (ValueError, OSError):
            ds = xr.concat([xr.open_dataset(f) for f in sorted(files)], dim='time')

    # Rename any aliased source variable to its canonical name.
    rename = {}
    for canonical in CANONICAL_FORCING:
        src = resolve_forcing_var(ds, canonical)
        if src is not None and src != canonical:
            rename[src] = canonical
    if rename:
        ds = ds.rename(rename)

    # Declared timestep wins; otherwise infer from the time axis.
    ts = ds.attrs.get('timestep_seconds')
    if ts is None and 'time' in ds and ds['time'].size > 1:
        ts = float(np.diff(ds['time'].values[:2])[0] / np.timedelta64(1, 's'))
    if ts is not None:
        ds.attrs['timestep_seconds'] = float(ts)
    return ds


def forcing_timestep_seconds(ds: xr.Dataset, default: float = 3600.0) -> float:
    """Return the canonical forcing timestep in seconds (declared or inferred)."""
    ts = ds.attrs.get('timestep_seconds')
    if ts is not None:
        return float(ts)
    if 'time' in ds and ds['time'].size > 1:
        return float(np.diff(ds['time'].values[:2])[0] / np.timedelta64(1, 's'))
    return default
