# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance for ObservationBackend implementations (contract 0.2.0).

Items 1-5 applied to the observation flavour, parameterized over every
backend registered under ``R.observation_backends`` (none in a bare install
— the whole module then skips with a reason; the csfs plugin registers
``community``). All tests are offline: upstream access is mocked at the
``csfs.fetch_observations_sync`` seam.

Item 4 is the headline: the UTC half-open ``[start, end)`` window rule
against the USGS provider with a mocked NWIS-shaped upstream that returns
the *inclusive* end boundary bin — the delivered OBS_CSV_V1 artifact must
exclude it (the boundary-bin discrepancy that motivated the rule).
"""
from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    AcquisitionResult,
    ObservationBackend,
    ObservationCapability,
    ObservationRequest,
    SchemaId,
    is_compatible,
    read_manifest,
)

from .checks import check_window_semantics, verify_honesty
from .conftest import CONFORMANCE_WINDOW

pytestmark = [pytest.mark.unit]

_PARITY_GRADE_RE = re.compile(r'^(bit-identical|value-identical:.+)$')


def _observation_backend_names() -> list:
    import symfluence  # noqa: F401 — bootstrap registries/plugins
    import symfluence.data.backends  # noqa: F401
    from symfluence.core.registries import R

    names = R.observation_backends.keys()
    if not names:
        return [pytest.param(
            '__none__',
            marks=pytest.mark.skip(
                reason='no observation backends registered '
                       '(community packages not installed)'
            ),
        )]
    return names


@pytest.fixture(params=_observation_backend_names())
def observation_backend_name(request) -> str:
    return request.param


@pytest.fixture
def observation_backend(observation_backend_name, minimal_config):
    """Materialize the registered observation backend (version-skew aware)."""
    from symfluence.core.registries import R
    from symfluence.data.backends.contract import PROTOCOL_VERSION

    entry = R.observation_backends.get(observation_backend_name)
    if entry is None:  # pragma: no cover — registry mutated mid-session
        pytest.skip(f'observation backend {observation_backend_name!r} no longer registered')
    if isinstance(entry, type):
        materialized = entry(
            minimal_config, logging.getLogger(f'conformance.obs.{observation_backend_name}'))
    else:
        materialized = entry
    version = getattr(materialized, 'interface_version', '')
    if not is_compatible(version):
        pytest.skip(
            f'observation backend {observation_backend_name!r} targets '
            f'interface_version {version!r}, incompatible with protocol '
            f'{PROTOCOL_VERSION}; inert until it re-targets'
        )
    return materialized


# ----------------------------------------------------------------------
# Offline fixture providers
# ----------------------------------------------------------------------

@contextlib.contextmanager
def _community_observation_fixture(*, include_end_boundary_bin: bool = False):
    """Mock the CSFS facade fetch with an NWIS-shaped hourly record.

    ``include_end_boundary_bin=True`` appends the observation at exactly the
    window end instant — what an inclusive-``endDT`` upstream (USGS NWIS)
    returns — so item 4 can assert the backend trims it from the protocol
    delivery.
    """
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    csfs = pytest.importorskip('csfs', reason='observation backend requires the csfs package')
    from csfs.core.models import Observation, QualityFlag, TimeSeriesChunk

    station = 'usgs:06191500'
    start = datetime(2020, 1, 1, tzinfo=UTC)
    hours = 25 if include_end_boundary_bin else 24
    observations = [
        Observation(
            station_id=station,
            timestamp=start + timedelta(hours=i),
            discharge_m3s=10.0 + i,
            quality=QualityFlag.GOOD,
        )
        for i in range(hours)
    ]
    chunk = TimeSeriesChunk(
        station_id=station, provider='usgs',
        observations=observations, fetched_at=datetime.now(UTC),
    )
    calls = []

    def _fetch(provider, station_id, *, start, end, config=None):
        calls.append((provider, station_id, start, end))
        return chunk

    with patch.object(csfs, 'fetch_observations_sync', side_effect=_fetch):
        yield {'provider_id': 'USGS', 'station_ids': ('06191500',), 'calls': calls}


OFFLINE_OBSERVATION_FIXTURES = {
    'community': _community_observation_fixture,
}


@pytest.fixture
def observation_fixture(observation_backend):
    provider = OFFLINE_OBSERVATION_FIXTURES.get(observation_backend.name)
    if provider is None:
        pytest.skip(
            f'no offline fixture provider registered for observation backend '
            f'{observation_backend.name!r}; add one to test_observation_backends.py'
        )
    return provider


def _request(tmp_path, facts) -> ObservationRequest:
    return ObservationRequest(
        provider_id=facts['provider_id'],
        station_ids=facts['station_ids'],
        kind='streamflow',
        window=CONFORMANCE_WINDOW,
        target_dir=tmp_path / 'obs_delivery',
    )


# ----------------------------------------------------------------------
# Item 1 — contract shape
# ----------------------------------------------------------------------

def test_backend_satisfies_the_protocol(observation_backend):
    assert isinstance(observation_backend, ObservationBackend)
    assert isinstance(observation_backend.name, str) and observation_backend.name
    assert observation_backend.name == observation_backend.name.lower()


def test_observation_capabilities_are_well_formed(observation_backend):
    caps = observation_backend.capabilities()
    assert caps, f'backend {observation_backend.name!r} claims no providers'

    seen: set[str] = set()
    for cap in caps:
        assert isinstance(cap, ObservationCapability)
        assert isinstance(cap.provider_id, str) and cap.provider_id
        key = cap.provider_id.lower()
        assert key not in seen, f'duplicate capability for {cap.provider_id!r}'
        seen.add(key)

        assert isinstance(cap.kinds, frozenset) and cap.kinds
        assert all(isinstance(k, str) and k for k in cap.kinds)
        assert isinstance(cap.station_id_scheme, str) and cap.station_id_scheme
        assert isinstance(cap.auth, frozenset)
        if cap.temporal is not None:
            assert len(cap.temporal) == 2
        if cap.parity_grade is not None:
            assert _PARITY_GRADE_RE.match(cap.parity_grade), (
                f'{cap.provider_id}: malformed parity_grade {cap.parity_grade!r}'
            )
        assert isinstance(cap.notes, str)


# ----------------------------------------------------------------------
# Items 2 + 3 — schema validity and honesty
# ----------------------------------------------------------------------

def test_acquire_writes_a_valid_obs_csv_v1_delivery(observation_backend, observation_fixture, tmp_path):
    with observation_fixture() as facts:
        request = _request(tmp_path, facts)
        result = observation_backend.acquire(request)

    assert isinstance(result, AcquisitionResult)
    assert result.schema is SchemaId.OBS_CSV_V1
    assert result.backend == observation_backend.name
    assert result.dataset_id == facts['provider_id']
    assert result.paths and all(Path(p).suffix == '.csv' for p in result.paths)

    manifest_path = request.target_dir / MANIFEST_FILENAME
    assert manifest_path.exists(), 'backend must write the sidecar manifest'
    manifest = read_manifest(manifest_path)
    assert manifest['schema'] == 'obs-csv-v1'
    assert manifest['paths'] == [str(p) for p in result.paths]

    # Item 3: claims vs files (CSV header, non-emptiness, window coverage).
    problems = verify_honesty(result, manifest, window=request.window)
    assert problems == [], '\n- '.join([''] + problems)


# ----------------------------------------------------------------------
# Item 4 — the UTC [start, end) rule against a mocked inclusive upstream
# ----------------------------------------------------------------------

def test_inclusive_end_boundary_bin_is_trimmed_from_the_delivery(
    observation_backend, observation_fixture, tmp_path
):
    """A mocked NWIS-shaped upstream returns the end-instant bin (inclusive
    endDT); the OBS_CSV_V1 delivery must obey half-open [start, end)."""
    with observation_fixture(include_end_boundary_bin=True) as facts:
        request = _request(tmp_path, facts)
        result = observation_backend.acquire(request)

    times: list = []
    for path in result.paths:
        times.extend(pd.to_datetime(pd.read_csv(path)['datetime']))

    problems = check_window_semantics(times, CONFORMANCE_WINDOW)
    assert problems == [], (
        f'backend {observation_backend.name!r} delivered the inclusive end '
        f'boundary bin:\n- ' + '\n- '.join(problems)
    )
    end = pd.to_datetime(CONFORMANCE_WINDOW[1]).tz_localize(None)
    assert end not in set(pd.to_datetime(times)), 'the end instant belongs to the NEXT window'


# ----------------------------------------------------------------------
# Item 5 — idempotency (re-acquire == no second upstream fetch)
# ----------------------------------------------------------------------

def test_identical_reacquisition_does_not_refetch_upstream(
    observation_backend, observation_fixture, tmp_path
):
    with observation_fixture() as facts:
        first = observation_backend.acquire(_request(tmp_path, facts))
        assert len(facts['calls']) == 1

        second = observation_backend.acquire(_request(tmp_path, facts))
        assert len(facts['calls']) == 1, (
            'identical re-acquisition must reuse the existing raw delivery'
        )

    assert [str(p) for p in second.paths] == [str(p) for p in first.paths]
