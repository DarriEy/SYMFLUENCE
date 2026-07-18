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

from .cf_conventions import CANONICAL_FORCING

logger = logging.getLogger(__name__)

# Deltas within this tolerance (seconds) of each other count as one cadence.
_CADENCE_TOLERANCE_S = 1.0


def _has_datetime_axis(ds: xr.Dataset) -> bool:
    """Report whether *ds* has a datetime64 time axis long enough to measure."""
    return (
        'time' in ds
        and ds['time'].size > 1
        and np.issubdtype(np.asarray(ds['time'].values).dtype, np.datetime64)
    )


def validated_timestep_seconds(times: np.ndarray, context: str = 'forcing') -> float:
    """Infer the cadence (seconds) from a full time axis, refusing irregular ones.

    The whole axis is checked — not just the first pair of timestamps — because a
    single leading delta cannot distinguish an hourly axis from one with a gap
    ([00:00, 01:00, 04:00] is not hourly). An axis that is non-monotonic, has
    duplicate timestamps, or mixes different deltas would silently corrupt
    accumulated precipitation and timestep counts downstream, so it is rejected.

    Args:
        times: datetime64 time coordinate values (size >= 2).
        context: label for error messages (e.g. the source file).

    Returns:
        The constant timestep in seconds.

    Raises:
        ValueError: if the axis is non-monotonic, has duplicates, or is irregular.
    """
    deltas = np.diff(times) / np.timedelta64(1, 's')
    if np.any(deltas <= 0):
        bad = int(np.argmax(deltas <= 0))
        raise ValueError(
            f"Time axis of {context} is not strictly increasing at index {bad + 1} "
            f"({times[bad]} -> {times[bad + 1]}): duplicate or out-of-order timestamps. "
            "Fix or regenerate the forcing store before using it."
        )
    if float(deltas.max() - deltas.min()) > _CADENCE_TOLERANCE_S:
        bad = int(np.argmax(np.abs(deltas - deltas[0]) > _CADENCE_TOLERANCE_S))
        raise ValueError(
            f"Time axis of {context} is irregular: steps range from {deltas.min():.0f} s "
            f"to {deltas.max():.0f} s (first differing interval at {times[bad]} -> "
            f"{times[bad + 1]}). Refusing to treat it as a fixed cadence — gaps would "
            "silently distort accumulated forcing. Regularize the axis or rebuild the store."
        )
    return float(np.median(deltas))


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
            # Filename sort order need not be chronological; sort the axis so
            # the regularity validation below sees the true cadence.
            ds = xr.concat([xr.open_dataset(f) for f in sorted(files)], dim='time').sortby('time')

    # Resolve each canonical variable, coalescing across every spelling present.
    # A partially-regenerated store can carry more than one spelling of the same
    # variable spread across files (e.g. 'precipitation_flux' in some, 'pptrate' in
    # others); the by-coords merge then leaves each spelling NaN wherever the other
    # supplied the data. Fill the canonical name from its aliases so every timestep
    # gets a value where any source provided one. With a single spelling present
    # (the clean-store case) this reduces to a plain rename.
    for canonical, spec in CANONICAL_FORCING.items():
        # spec['aliases'] already includes the canonical name; dict.fromkeys
        # de-duplicates while preserving order so the canonical stays the primary.
        candidates = dict.fromkeys([canonical, *spec['aliases']])  # type: ignore[misc]
        present = [name for name in candidates if name in ds]
        if not present:
            continue
        primary = present[0]
        if len(present) > 1:
            filled = ds[primary]
            for alt in present[1:]:
                filled = filled.where(~filled.isnull(), ds[alt])
            ds[primary] = filled
            ds = ds.drop_vars(present[1:])
        if primary != canonical:
            ds = ds.rename({primary: canonical})

    # The measured axis is authoritative: validate the whole axis and cross-check
    # any declared attribute against it rather than trusting metadata blindly.
    # Non-datetime64 axes (cftime calendars, undecoded numerics) can't be
    # validated this way and fall back to the declared attribute.
    ts = ds.attrs.get('timestep_seconds')
    if _has_datetime_axis(ds):
        measured = validated_timestep_seconds(
            ds['time'].values, context=', '.join(f.name for f in files[:3])
        )
        if ts is not None and abs(float(ts) - measured) > _CADENCE_TOLERANCE_S:
            logger.warning(
                "Declared timestep_seconds=%s disagrees with the time axis (%.0f s); "
                "using the measured cadence", ts, measured,
            )
        ts = measured
    if ts is not None:
        ds.attrs['timestep_seconds'] = float(ts)
    return ds


def forcing_timestep_seconds(ds: xr.Dataset, default: float = 3600.0) -> float:
    """Return the canonical forcing timestep in seconds (measured or declared).

    The full time axis, when present, is validated and wins over the declared
    ``timestep_seconds`` attribute — a stale or wrong attribute must not let an
    irregular or coarser axis masquerade as the declared cadence.
    """
    ts = ds.attrs.get('timestep_seconds')
    if _has_datetime_axis(ds):
        measured = validated_timestep_seconds(ds['time'].values)
        if ts is not None and abs(float(ts) - measured) > _CADENCE_TOLERANCE_S:
            logger.warning(
                "Declared timestep_seconds=%s disagrees with the time axis (%.0f s); "
                "using the measured cadence", ts, measured,
            )
        return measured
    if ts is not None:
        return float(ts)
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
