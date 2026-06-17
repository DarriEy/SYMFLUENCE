# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 4 — window/bbox semantics, identical across backends.

The contract rules (normative text in ``symfluence.data.backends.contract``):

* window: half-open UTC ``[start, end)`` — every delivered timestep ``t``
  satisfies ``start <= t < end`` in UTC (naive == UTC). Rationale: USGS NWIS
  treats ``endDT`` inclusively, so backends disagreed on the final boundary
  bin until the rule was pinned down (the USGS boundary-bin discrepancy).
* bbox: each edge of the delivered extent must lie within one native pixel
  of the requested edge (outward grid-snap à la the ``dem.py`` tile-merge
  clip, or cell-center subsetting — both within tolerance).

Synthetic axes prove the checkers accept compliant deliveries and reject
each violation class; registered backends are then held to the same rule via
their offline fixtures.
"""
from __future__ import annotations

import pandas as pd
import pytest

from .checks import check_bbox_semantics, check_window_semantics, delivered_axes
from .conftest import CONFORMANCE_BBOX, CONFORMANCE_WINDOW

pytestmark = [pytest.mark.unit]

_WINDOW = ('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z')


# ----------------------------------------------------------------------
# Window rule against synthetic time axes
# ----------------------------------------------------------------------

class TestWindowSemantics:
    def test_compliant_axis_passes(self):
        times = pd.date_range('2020-01-01 00:00', '2020-01-01 23:00', freq='h')
        assert check_window_semantics(times, _WINDOW) == []

    def test_axis_including_the_end_instant_fails(self):
        """The boundary bin at `end` belongs to the NEXT window: [start, end)."""
        times = pd.date_range('2020-01-01 00:00', '2020-01-02 00:00', freq='h')
        problems = check_window_semantics(times, _WINDOW)
        assert any('half-open' in p for p in problems), problems

    def test_axis_starting_before_the_window_fails(self):
        times = pd.date_range('2019-12-31 23:00', '2020-01-01 23:00', freq='h')
        problems = check_window_semantics(times, _WINDOW)
        assert any('before window start' in p for p in problems), problems

    def test_tz_aware_axis_is_compared_in_utc(self):
        from datetime import timedelta, timezone

        # 00:30+01:00 == 2019-12-31T23:30Z — outside the window even though
        # the local clock reads inside it. UTC comparison must catch this.
        times = pd.date_range(
            '2020-01-01 00:30', periods=3, freq='h', tz=timezone(timedelta(hours=1))
        )
        problems = check_window_semantics(times, _WINDOW)
        assert any('before window start' in p for p in problems), problems

        # The same instants expressed in UTC+0 within the window pass.
        ok = pd.date_range('2020-01-01 00:30', periods=3, freq='h', tz='UTC')
        assert check_window_semantics(ok, _WINDOW) == []

    def test_empty_axis_fails(self):
        assert check_window_semantics([], _WINDOW) == ['delivered time axis is empty']


# ----------------------------------------------------------------------
# Bbox rule against synthetic grids
# ----------------------------------------------------------------------

class TestBboxSemantics:
    BBOX = (50.0, -116.0, 51.0, -115.0)  # (lat_min, lon_min, lat_max, lon_max)

    def test_exact_cover_passes(self):
        # 0.25° centers whose cell edges hit the bbox edges exactly.
        lats = [50.125 + 0.25 * i for i in range(4)]
        lons = [-115.875 + 0.25 * i for i in range(4)]
        assert check_bbox_semantics(lats, lons, self.BBOX) == []

    def test_cell_center_subsetting_passes(self):
        # Centers strictly inside the bbox (sel(slice) style): the extent
        # falls short by half a pixel per edge — within tolerance.
        lats = [50.25, 50.5, 50.75]
        lons = [-115.75, -115.5, -115.25]
        assert check_bbox_semantics(lats, lons, self.BBOX) == []

    def test_outward_snap_of_one_pixel_passes(self):
        """The dem.py tile-merge clip snaps outward to the native grid."""
        lats = [49.875 + 0.25 * i for i in range(6)]   # extends 1 pixel beyond both edges
        lons = [-116.125 + 0.25 * i for i in range(6)]
        assert check_bbox_semantics(lats, lons, self.BBOX) == []

    def test_shortfall_beyond_one_pixel_fails(self):
        # Delivered grid stops 1.5 pixels short of lat_max.
        lats = [50.125, 50.375]
        lons = [-115.875 + 0.25 * i for i in range(4)]
        problems = check_bbox_semantics(lats, lons, self.BBOX)
        assert any('latitude' in p and 'falls short' in p for p in problems), problems

    def test_overshoot_beyond_one_pixel_fails(self):
        # Delivered grid extends ~2 pixels past lon_max.
        lats = [50.125 + 0.25 * i for i in range(4)]
        lons = [-115.875 + 0.25 * i for i in range(6)]  # ... up to -114.625
        problems = check_bbox_semantics(lats, lons, self.BBOX)
        assert any('longitude' in p and 'overshoots' in p for p in problems), problems

    def test_single_cell_must_be_inside_the_bbox(self):
        assert check_bbox_semantics([50.5], [-115.5], self.BBOX) == []
        problems = check_bbox_semantics([52.0], [-115.5], self.BBOX)
        assert any('outside' in p for p in problems), problems


# ----------------------------------------------------------------------
# Registered backends, held to the same rule via their offline fixtures
# ----------------------------------------------------------------------

def test_delivered_axes_obey_window_and_bbox(offline_acquisition):
    request, result = offline_acquisition

    axes = delivered_axes(result.paths)
    if 'time' not in axes:
        pytest.skip(
            f'offline fixture for this backend delivers no NetCDF time axis; '
            f'window semantics not checkable for {result.backend!r} offline'
        )

    problems = check_window_semantics(axes['time'], CONFORMANCE_WINDOW)
    if 'lat' in axes and 'lon' in axes:
        problems += check_bbox_semantics(axes['lat'], axes['lon'], CONFORMANCE_BBOX)

    assert problems == [], (
        f'backend {result.backend!r} violates the window/bbox contract rules:\n- '
        + '\n- '.join(problems)
    )
