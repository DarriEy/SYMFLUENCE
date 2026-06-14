# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Canonical forcing reader for the model-ready store.

Single entry point every model adapter should use to load basin-averaged
forcing. It guarantees the canonical CFIF vocabulary — CF standard names with
underscores (precipitation_flux, air_temperature, surface_downwelling_shortwave_flux,
surface_downwelling_longwave_flux, wind_speed, specific_humidity, surface_air_pressure)
— regardless of the source dataset (CARRA, ERA5, ...) and exposes the forcing
timestep via ``ds.attrs['timestep_seconds']``, so adapters never re-parse raw
variable names, units, or guess the timestep. Model-native layers (e.g. SUMMA's
Fortran binary) translate these to their shorthand via cfif.CFIF_TO_SUMMA_MAPPING.
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
    """Open model-ready forcing and return it under canonical CFIF names.

    - Aliased source variables (e.g. ``pptrate`` -> ``precipitation_flux``) are
      renamed to the canonical CF vocabulary.
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


def resample_canonical_forcing(ds: xr.Dataset, target_seconds: float) -> xr.Dataset:
    """Resample a CF-named forcing dataset to a fixed target timestep.

    Some model pipelines assume a fixed forcing cadence (e.g. hourly) and equate
    "one timestep" with "one hour" in their downstream aggregation. Feeding such
    a pipeline a coarser source (CARRA 3-hourly, daily) silently miscounts the
    cadence. This resamples the source to ``target_seconds`` so the assumption
    holds for any input:

    - **Rate** variables (``kind == 'rate'``, e.g. ``precipitation_flux``) are
      held constant across each source interval — a step function (``ffill`` on
      upsample, interval ``mean`` on downsample) that conserves the accumulated
      total.
    - **State** variables (``kind == 'state'``, e.g. ``air_temperature``) are
      linearly interpolated on upsample, interval-averaged on downsample.

    It is a no-op when the source already matches ``target_seconds`` — so an
    hourly source (ERA5) is returned byte-for-byte unchanged.

    Args:
        ds: forcing Dataset under canonical CF names (from open_canonical_forcing).
        target_seconds: desired timestep in seconds (e.g. 3600 for hourly).

    Returns:
        Resampled Dataset with ``target_seconds`` on ``ds.attrs['timestep_seconds']``.
    """
    if 'time' not in ds or ds['time'].size < 2:
        return ds

    src_seconds = forcing_timestep_seconds(ds)
    if abs(src_seconds - target_seconds) < 1.0:
        return ds  # no-op — source already at the target cadence

    import pandas as pd

    src_times = pd.DatetimeIndex(ds['time'].values)
    freq = f'{int(target_seconds)}s'

    def _is_rate(var: str) -> bool:
        spec = CANONICAL_FORCING.get(var)
        return bool(spec) and spec.get('kind') == 'rate'

    if target_seconds < src_seconds:
        # Upsample to a finer cadence. Extend the target to cover the final
        # source interval fully so step-filled rates conserve their total.
        end = (src_times[-1]
               + pd.Timedelta(seconds=src_seconds)
               - pd.Timedelta(seconds=target_seconds))
        target_times = pd.date_range(start=src_times[0], end=end, freq=freq)
        if len(target_times) == 0:
            return ds
        rate_vars = [v for v in ds.data_vars if _is_rate(str(v))]
        state_vars = [v for v in ds.data_vars if not _is_rate(str(v))]
        parts = []
        if rate_vars:
            # Step function: each source rate spans its whole interval.
            parts.append(ds[rate_vars].reindex(time=target_times, method='ffill'))
        if state_vars:
            # Linear interpolation in the interior; hold the last value across
            # the extrapolated tail (interp yields NaN beyond the source range).
            interp = ds[state_vars].interp(time=target_times).ffill('time')
            parts.append(interp)
        out = xr.merge(parts) if parts else ds.reindex(time=target_times, method='ffill')
    else:
        # Downsample to a coarser cadence: mean over each interval works for both
        # a mean rate and an averaged state.
        out = ds.resample(time=freq).mean()

    out.attrs.update(ds.attrs)
    out.attrs['timestep_seconds'] = float(target_seconds)
    return out
