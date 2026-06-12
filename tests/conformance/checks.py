# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Provider-agnostic verification helpers for conformance items 3 and 4.

These helpers are the *checkable* form of the contract rules documented in
``symfluence.data.backends.contract``:

* item 3 (honesty): what a backend's manifest claims must match what is in
  the delivered files — :func:`verify_honesty`. Silent truncation (the
  WSC-pagination bug class) is an honesty failure, never a warning.
* item 4 (window/bbox semantics): the delivered time axis must lie in the
  half-open UTC window ``[start, end)`` — :func:`check_window_semantics` —
  and the delivered grid must cover the requested bbox within one native
  pixel per edge — :func:`check_bbox_semantics`.

Every checker returns a list of human-readable problems (empty = conformant)
so tests can assert emptiness and induced-violation fixtures can assert the
right problem is reported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

#: Schemas whose files carry CFIF variable names, making the manifest's
#: ``variables_delivered`` directly checkable against NetCDF contents.
#: NATIVE_RAW files keep native upstream names (rename map owned by
#: preprocessing), so their declared CFIF set is NOT file-checkable.
CFIF_NAMED_SCHEMAS = {'canonical-v1'}

_NETCDF_SUFFIXES = ('.nc', '.nc4')


def _utc_index(times: Iterable[Any]) -> pd.DatetimeIndex:
    """Normalize delivered timestamps to a tz-aware UTC index.

    Naive timestamps ARE UTC per the contract; tz-aware ones are converted.
    """
    return pd.DatetimeIndex(pd.to_datetime(list(times), utc=True)).sort_values()


# ----------------------------------------------------------------------
# Item 4 — window semantics
# ----------------------------------------------------------------------

def check_window_semantics(times: Iterable[Any], window: Tuple[str, str]) -> List[str]:
    """Check a delivered time axis against the half-open UTC window rule.

    Contract rule (see WINDOW SEMANTICS in ``contract.py``): every delivered
    timestep ``t`` must satisfy ``start <= t < end`` in UTC. The rule exists
    because USGS NWIS treats ``endDT`` as inclusive of the end day, so
    backends disagreed on the final boundary bin until the inclusivity rule
    was pinned down.
    """
    idx = _utc_index(times)
    if idx.empty:
        return ['delivered time axis is empty']
    start = pd.to_datetime(window[0], utc=True)
    end = pd.to_datetime(window[1], utc=True)

    problems: List[str] = []
    early = idx[idx < start]
    if not early.empty:
        problems.append(
            f'{len(early)} timestep(s) before window start {window[0]} '
            f'(earliest: {early.min().isoformat()})'
        )
    late = idx[idx >= end]
    if not late.empty:
        problems.append(
            f'{len(late)} timestep(s) at/after window end {window[1]} '
            f'(latest: {late.max().isoformat()}); the window is half-open [start, end)'
        )
    return problems


def check_no_truncation(times: Iterable[Any], window: Tuple[str, str]) -> List[str]:
    """Check that a delivered time axis spans the requested window (item 3).

    The WSC-pagination bug class: an upstream that silently stops paging
    delivers a prefix of the requested record while the manifest claims the
    full window. With the native step inferred from the axis itself, the
    first timestep must fall within one step of the window start and the
    last within two steps of the (exclusive) window end. Axes with fewer
    than two timesteps carry no inferable step and are not checked.
    """
    idx = _utc_index(times)
    if len(idx) < 2:
        return []  # cannot infer the native step; truncation not checkable
    step = pd.Series(idx).diff().median()
    start = pd.to_datetime(window[0], utc=True)
    end = pd.to_datetime(window[1], utc=True)

    problems: List[str] = []
    if idx.min() > start + step:
        problems.append(
            f'delivered time axis starts at {idx.min().isoformat()}, more than one '
            f'native step ({step}) after the window start {window[0]} (truncated head)'
        )
    if idx.max() < end - 2 * step:
        problems.append(
            f'delivered time axis stops at {idx.max().isoformat()} but the window runs '
            f'to {window[1]} (silent truncation — the WSC-pagination bug class)'
        )
    return problems


# ----------------------------------------------------------------------
# Item 4 — bbox semantics
# ----------------------------------------------------------------------

def _check_axis_coverage(name: str, centers: Sequence[float], lo: float, hi: float) -> List[str]:
    values = sorted(float(v) for v in centers)
    if not values:
        return [f'delivered {name} axis is empty']
    if len(values) == 1:
        # Single cell: no resolution is inferable; the center must at least
        # fall inside the requested range.
        if not (lo <= values[0] <= hi):
            return [f'single delivered {name} cell center {values[0]} lies outside [{lo}, {hi}]']
        return []

    res = float(pd.Series(values).diff().median())
    cover_lo = values[0] - res / 2.0
    cover_hi = values[-1] + res / 2.0

    problems: List[str] = []
    for edge_name, delivered, requested, sign in (
        ('min', cover_lo, lo, +1),
        ('max', cover_hi, hi, -1),
    ):
        # sign +1: shortfall means delivered > requested (gap below lo);
        # sign -1: shortfall means delivered < requested (gap below hi).
        shortfall = (delivered - requested) * sign
        if shortfall > res:
            problems.append(
                f'delivered {name} extent {edge_name} edge {delivered:.4f} falls short of the '
                f'requested edge {requested:.4f} by more than one native pixel ({res:.4f})'
            )
        elif shortfall < -res:
            problems.append(
                f'delivered {name} extent {edge_name} edge {delivered:.4f} overshoots the '
                f'requested edge {requested:.4f} by more than one native pixel ({res:.4f})'
            )
    return problems


def check_bbox_semantics(
    lats: Sequence[float],
    lons: Sequence[float],
    bbox: Tuple[float, float, float, float],
) -> List[str]:
    """Check delivered cell centers against the one-native-pixel bbox rule.

    Contract rule (see BBOX SEMANTICS in ``contract.py``): each edge of the
    delivered extent (outermost centers ± half a pixel) must lie within one
    native pixel of the requested edge. Outward snapping to the native grid
    (the ``dem.py`` tile-merge clip) and cell-center subsetting are both
    within tolerance.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        _check_axis_coverage('latitude', lats, lat_min, lat_max)
        + _check_axis_coverage('longitude', lons, lon_min, lon_max)
    )


# ----------------------------------------------------------------------
# Item 3 — honesty (manifest vs delivered files)
# ----------------------------------------------------------------------

def delivered_axes(paths: Iterable[Path]) -> dict:
    """Extract time/lat/lon axes from delivered NetCDF outputs (union).

    Returns a dict with any of the keys ``time``/``lat``/``lon`` that could
    be found across the readable NetCDF files.
    """
    import xarray as xr

    axes: dict = {}
    for path in paths:
        p = Path(path)
        if p.suffix.lower() not in _NETCDF_SUFFIXES or not p.is_file():
            continue
        with xr.open_dataset(p) as ds:
            for key, names in (
                ('time', ('time',)),
                ('lat', ('lat', 'latitude', 'y')),
                ('lon', ('lon', 'longitude', 'x')),
            ):
                for name in names:
                    if name in ds.coords:
                        values = list(pd.Series(ds[name].values))
                        axes.setdefault(key, []).extend(values)
                        break
    return axes


def verify_honesty(
    result: Any,
    manifest: Mapping[str, Any],
    *,
    window: Optional[Tuple[str, str]] = None,
) -> List[str]:
    """Item 3 verification helper: do the delivered files match the claims?

    Checks, in order of increasing specificity:

    * every manifest path exists and is non-empty;
    * the manifest's ``variables_delivered`` agrees with the returned result;
    * every delivered NetCDF opens;
    * for CFIF-named schemas (:data:`CFIF_NAMED_SCHEMAS`) every claimed
      variable is present in the delivered NetCDF files;
    * for ``obs-csv-v1``, every delivered CSV has a header containing
      ``datetime``;
    * when *window* is given and a time axis exists, the axis spans the
      window (:func:`check_no_truncation` — the WSC-pagination class).

    NATIVE_RAW variable names are NOT checked (native upstream names; the
    CFIF rename map is owned by preprocessing) — see the same limitation
    documented on ``NativeBackend._verify_delivery``.
    """
    import xarray as xr

    problems: List[str] = []

    paths = [Path(p) for p in manifest['paths']]
    readable_nc: List[Path] = []
    for p in paths:
        if not p.exists():
            problems.append(f'manifest path does not exist: {p}')
            continue
        if p.is_file() and p.stat().st_size == 0:
            problems.append(f'manifest path is empty: {p}')
            continue
        if p.suffix.lower() in _NETCDF_SUFFIXES:
            readable_nc.append(p)

    if sorted(manifest['variables_delivered']) != sorted(result.variables_delivered):
        problems.append(
            'manifest variables_delivered disagrees with the returned AcquisitionResult'
        )

    file_vars: set = set()
    times: List[Any] = []
    for p in readable_nc:
        try:
            with xr.open_dataset(p) as ds:
                file_vars |= {str(v) for v in ds.data_vars}
                if 'time' in ds.coords:
                    times.extend(pd.Series(ds['time'].values))
        except (OSError, ValueError) as exc:
            problems.append(f'unreadable NetCDF output {p.name}: {exc}')

    if manifest['schema'] in CFIF_NAMED_SCHEMAS and readable_nc:
        missing = set(manifest['variables_delivered']) - file_vars
        if missing:
            problems.append(
                f'variables claimed in the manifest but absent from the delivered '
                f'files: {sorted(missing)}'
            )

    if manifest['schema'] == 'obs-csv-v1':
        for p in paths:
            if p.suffix.lower() != '.csv' or not p.is_file():
                continue
            header = p.read_text(encoding='utf-8').splitlines()[:1]
            if not header or 'datetime' not in header[0]:
                problems.append(f"CSV output {p.name} lacks a 'datetime' header column")

    if window is not None and times:
        problems.extend(check_no_truncation(times, window))

    return problems
