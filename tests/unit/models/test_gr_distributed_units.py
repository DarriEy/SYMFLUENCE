# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit correctness for distributed GR streamflow extraction.

``GRPostProcessor._extract_distributed_streamflow`` used to end both of its
branches with a single ``convert_mm_per_day_to_cms`` call (``* area_km2 /
86.4``), which was wrong for each branch in a different way:

* the **routed** branch reads mizuRoute output, already m³/s, and was inflated
  by ``area_km2 / 86.4``;
* the **unrouted** branch reads GR's own ``q_routed``, a per-GRU depth rate in
  **m/s** (the runner divides mm/day by ``1000 * 86400``), and was both summed
  across GRUs instead of area-weighted and then treated as mm/day.

The source even carried the open question in a comment: "If mizuRoute, it might
be in m3/s already depending on config, but typically routing input is mm/day
and output is m3/s?". These tests settle it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = [pytest.mark.unit]

EXPERIMENT_ID = 'gr_units_001'
DOMAIN_NAME = 'testdom'

# The textbook conversion the lumped path uses and this must agree with:
#   Q(m3/s) = Q(mm/day) * Area(km2) / 86.4
MM_DAY_TO_CMS = 86.4


class _NS:
    """Minimal stand-in for a typed config node."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_postprocessor(tmp_path: Path, *, areas_m2=None, total_area_km2=None):
    """Build a GRPostProcessor without running __init__ (which requires rpy2).

    Only the attributes the extraction helpers touch are populated, so the real
    arithmetic under test runs unmodified.
    """
    from symfluence.models.gr.postprocessor import GRPostProcessor

    import logging

    obj = GRPostProcessor.__new__(GRPostProcessor)
    obj.project_dir = tmp_path
    obj.domain_name = DOMAIN_NAME
    obj.logger = logging.getLogger('test.gr.postprocessor')
    obj.config = _NS(
        domain=_NS(experiment_id=EXPERIMENT_ID),
        model=_NS(
            gr=_NS(routing_integration='none'),
            mizuroute=_NS(routing_var='default'),
        ),
    )

    def _get_config_value(getter, default=None, **kwargs):
        try:
            value = getter()
        except (AttributeError, TypeError):
            value = None
        return default if value is None else value

    obj._get_config_value = _get_config_value
    obj._gru_areas_m2 = lambda ds: areas_m2
    obj.get_catchment_area_km2 = lambda: total_area_km2
    return obj


def _write_runs_def(tmp_path: Path, runoff_ms: np.ndarray, gru_ids) -> Path:
    """Write a GR distributed output file in the runner's exact layout."""
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'GR'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{DOMAIN_NAME}_{EXPERIMENT_ID}_runs_def.nc"

    times = pd.date_range('2010-01-01', periods=runoff_ms.shape[0], freq='D')
    ds = xr.Dataset(
        {
            'gruId': ('gru', np.asarray(gru_ids)),
            # units 'm/s', exactly as GRRunner writes it
            'q_routed': (('time', 'gru'), runoff_ms, {'units': 'm/s'}),
        },
        coords={'time': times, 'gru': np.arange(runoff_ms.shape[1])},
    )
    ds.to_netcdf(path)
    return path


def _write_mizuroute_output(tmp_path: Path, flow_cms: np.ndarray) -> Path:
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'mizuRoute'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{EXPERIMENT_ID}_routed.nc"

    times = pd.date_range('2010-01-01', periods=flow_cms.shape[0], freq='D')
    ds = xr.Dataset(
        {'IRFroutedRunoff': (('time', 'seg'), flow_cms, {'units': 'm3/s'})},
        coords={'time': times, 'seg': np.arange(flow_cms.shape[1])},
    )
    ds.to_netcdf(path)
    return path


def test_unrouted_matches_the_textbook_mm_per_day_conversion(tmp_path):
    """The strongest check: agree with the lumped path's known-good physics.

    A single 100 km² GRU producing 2 mm/day must yield the same discharge the
    lumped branch would compute, 2 * 100 / 86.4 = 2.3148 m³/s -- reached here
    through the m/s representation GR actually writes to disk.
    """
    mm_per_day = 2.0
    area_km2 = 100.0
    runoff_ms = np.full((3, 1), mm_per_day / (1000.0 * 86400.0))
    _write_runs_def(tmp_path, runoff_ms, gru_ids=[1])

    pp = _make_postprocessor(tmp_path, areas_m2=np.array([area_km2 * 1e6]))
    series = pp._unrouted_streamflow_cms()

    expected = mm_per_day * area_km2 / MM_DAY_TO_CMS
    assert series is not None
    assert series.values == pytest.approx(expected)
    assert expected == pytest.approx(2.3148148, rel=1e-6)


def test_unrouted_weights_grus_by_area(tmp_path):
    """Q = sum_i runoff_i * area_i, not a plain sum of depth rates."""
    runoff_ms = np.array([[1e-7, 3e-7]])  # two GRUs, one timestep
    areas_m2 = np.array([2.0e6, 5.0e6])
    _write_runs_def(tmp_path, runoff_ms, gru_ids=[1, 2])

    pp = _make_postprocessor(tmp_path, areas_m2=areas_m2)
    series = pp._unrouted_streamflow_cms()

    expected = 1e-7 * 2.0e6 + 3e-7 * 5.0e6
    assert series.values == pytest.approx(expected)
    # A plain sum-then-scale would have produced something entirely different.
    assert series.values[0] != pytest.approx((1e-7 + 3e-7) * 100.0 / MM_DAY_TO_CMS)


def test_unrouted_equal_area_fallback_uses_basin_mean(tmp_path):
    """With no per-GRU areas, mean x total area is the equal-area equivalent."""
    runoff_ms = np.array([[1e-7, 3e-7]])
    total_area_km2 = 10.0
    _write_runs_def(tmp_path, runoff_ms, gru_ids=[1, 2])

    pp = _make_postprocessor(
        tmp_path, areas_m2=None, total_area_km2=total_area_km2
    )
    series = pp._unrouted_streamflow_cms()

    expected = np.mean([1e-7, 3e-7]) * total_area_km2 * 1e6
    assert series.values == pytest.approx(expected)


def test_routed_output_passes_through_unconverted(tmp_path):
    """mizuRoute output is already m³/s and must not be scaled by area/86.4."""
    flow_cms = np.array([[1.0, 5.0], [2.0, 6.0]])  # (time, seg)
    _write_mizuroute_output(tmp_path, flow_cms)

    # An area is available, so a regression would silently rescale by it.
    pp = _make_postprocessor(tmp_path, total_area_km2=1000.0)
    series = pp._routed_streamflow_cms()

    assert series is not None
    # Outlet is the last segment.
    assert list(series.values) == pytest.approx([5.0, 6.0])
    assert isinstance(series.index, pd.DatetimeIndex)


def test_missing_runoff_variable_reports_rather_than_raises(tmp_path):
    """A file without the configured variable returns None, not a KeyError."""
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'GR'
    out_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2010-01-01', periods=2, freq='D')
    xr.Dataset(
        {'something_else': (('time',), np.zeros(2))}, coords={'time': times}
    ).to_netcdf(out_dir / f"{DOMAIN_NAME}_{EXPERIMENT_ID}_runs_def.nc")

    pp = _make_postprocessor(tmp_path, areas_m2=None, total_area_km2=1.0)
    assert pp._unrouted_streamflow_cms() is None
