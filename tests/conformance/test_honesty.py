# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 3 — honesty: ``variables_delivered`` must match the files.

Two layers:

* every registered backend's offline acquisition must pass
  :func:`tests.conformance.checks.verify_honesty` (manifest claims vs the
  delivered files, including window-coverage truncation when a time axis is
  present);
* induced-violation fixtures prove the checker catches the WSC-pagination
  class of bug — a backend that writes fewer timesteps or fewer variables
  than its manifest claims must FAIL conformance, never pass silently.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    AcquisitionRequest,
    AcquisitionResult,
    SchemaId,
    read_manifest,
    write_manifest,
)

from .checks import verify_honesty
from .conftest import CONFORMANCE_BBOX, CONFORMANCE_WINDOW

pytestmark = [pytest.mark.unit]


# ----------------------------------------------------------------------
# Registered backends: offline acquisition must verify clean
# ----------------------------------------------------------------------

def test_delivery_matches_manifest_claims(offline_acquisition):
    request, result = offline_acquisition
    manifest = read_manifest(request.target_dir)

    problems = verify_honesty(result, manifest, window=request.window)
    assert problems == [], (
        'delivered files do not match the manifest claims:\n- ' + '\n- '.join(problems)
    )


# ----------------------------------------------------------------------
# Induced violations: the checker must catch a lying backend
# ----------------------------------------------------------------------

class _SyntheticBackend:
    """Fixture backend writing CANONICAL_V1 NetCDF — honest by default.

    ``claim_unwritten_variable`` makes the manifest claim a variable that is
    never written; ``truncate_timesteps`` delivers only the first half of the
    requested window while still claiming success (the WSC-pagination class).
    """

    name = 'fixture'
    interface_version = PROTOCOL_VERSION

    def __init__(self, *, claim_unwritten_variable: bool = False,
                 truncate_timesteps: bool = False) -> None:
        self.claim_unwritten_variable = claim_unwritten_variable
        self.truncate_timesteps = truncate_timesteps

    def capabilities(self):  # pragma: no cover — not consulted in these tests
        return ()

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        import numpy as np
        import pandas as pd
        import xarray as xr

        target_dir = Path(request.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / 'fixture.nc'

        periods = 12 if self.truncate_timesteps else 24
        # tz-naive UTC time axis (naive == UTC per the contract; NetCDF
        # cannot encode tz-aware datetimes anyway).
        start = pd.to_datetime(request.window[0], utc=True).tz_localize(None)
        time = pd.date_range(start, periods=periods, freq='h')
        claimed = frozenset({'air_temperature', 'precipitation_flux'})
        written = sorted(claimed - ({'precipitation_flux'} if self.claim_unwritten_variable else set()))

        ds = xr.Dataset(
            {name: (('time',), np.zeros(time.size, dtype='float32')) for name in written},
            coords={'time': time},
        )
        ds.to_netcdf(out)

        result = AcquisitionResult(
            paths=(out,),
            schema=SchemaId.CANONICAL_V1,
            dataset_id=request.dataset_id,
            backend=self.name,
            provenance={'source': 'synthetic'},
            variables_delivered=claimed,
        )
        write_manifest(result, target_dir)
        return result


def _acquire_synthetic(tmp_path, **kwargs):
    backend = _SyntheticBackend(**kwargs)
    request = AcquisitionRequest(
        dataset_id='SYNTH',
        bbox=CONFORMANCE_BBOX,
        window=CONFORMANCE_WINDOW,
        variables=None,
        target_dir=tmp_path / 'raw_data',
    )
    result = backend.acquire(request)
    return request, result, read_manifest(request.target_dir)


def test_honest_synthetic_backend_passes(tmp_path):
    request, result, manifest = _acquire_synthetic(tmp_path)
    assert verify_honesty(result, manifest, window=request.window) == []


def test_claiming_an_unwritten_variable_fails_conformance(tmp_path):
    request, result, manifest = _acquire_synthetic(tmp_path, claim_unwritten_variable=True)
    problems = verify_honesty(result, manifest, window=request.window)
    assert any('precipitation_flux' in p and 'absent' in p for p in problems), problems


def test_induced_timestep_truncation_fails_conformance(tmp_path):
    """The WSC-pagination bug class: half the window delivered, full window claimed."""
    request, result, manifest = _acquire_synthetic(tmp_path, truncate_timesteps=True)
    problems = verify_honesty(result, manifest, window=request.window)
    assert any('truncation' in p for p in problems), problems


def test_manifest_disagreeing_with_result_fails_conformance(tmp_path):
    import dataclasses

    request, result, manifest = _acquire_synthetic(tmp_path)
    dishonest = dataclasses.replace(
        result, variables_delivered=frozenset({'air_temperature'}),
    )
    problems = verify_honesty(dishonest, manifest, window=request.window)
    assert any('disagrees' in p for p in problems), problems


def test_missing_and_empty_paths_fail_conformance(tmp_path):
    request, result, manifest = _acquire_synthetic(tmp_path)
    delivered = Path(result.paths[0])

    delivered.write_bytes(b'')  # empty
    problems = verify_honesty(result, manifest, window=request.window)
    assert any('empty' in p for p in problems), problems

    delivered.unlink()  # missing
    problems = verify_honesty(result, manifest, window=request.window)
    assert any('does not exist' in p for p in problems), problems
