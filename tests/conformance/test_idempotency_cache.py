# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 5 — idempotency + cache keys.

* Re-acquiring an identical request must not hit upstream a second time:
  the framework's raw-forcing cache serves the repeat (a fixture backend
  counts its ``acquire()`` calls through the real ``AcquisitionService``
  protocol path).
* Cache keys are structural over ``(dataset, bbox, window, variables,
  schema, backend)``: entries from different backends or schemas must never
  collide with each other or with the legacy ``DATA_ACCESS``-salted entries.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    AcquisitionResult,
    DatasetCapability,
    GridClass,
    SchemaId,
)

pytestmark = [pytest.mark.unit]


class _CountingBackend:
    """Protocol backend counting upstream fetches (single-file .nc output)."""

    name = 'community'
    interface_version = PROTOCOL_VERSION

    def __init__(self):
        self.acquire_calls = 0
        self.target_dirs: list[Path] = []

    def capabilities(self):
        return (DatasetCapability(
            dataset_id='ERA5',
            grid_class=GridClass.REGULAR_LATLON,
            schema=SchemaId.CANONICAL_V1,
            variables=frozenset({'air_temperature', 'precipitation_flux'}),
            temporal=None,
            auth=frozenset(),
            parity_grade='bit-identical',
        ),)

    def acquire(self, request):
        self.acquire_calls += 1
        self.target_dirs.append(Path(request.target_dir))
        out = Path(request.target_dir) / 'counting_era5.nc'
        out.write_bytes(b'payload')  # never opened: cache stores/copies bytes
        return AcquisitionResult(
            paths=(out,),
            schema=SchemaId.CANONICAL_V1,
            dataset_id=request.dataset_id,
            backend=self.name,
            provenance={'source': 'counting-fixture'},
            variables_delivered=frozenset({'air_temperature'}),
        )


def _service(tmp_path, time_end='2020-01-02 00:00'):
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.data.acquisition.acquisition_service import AcquisitionService

    config = SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'idempotency',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': time_end,
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'BOUNDING_BOX_COORDS': '51.3/-115.8/50.9/-115.2',
        'DATA_ACCESS': 'community',
    })
    return AcquisitionService(config, logging.getLogger('conformance.idempotency'))


def test_identical_reacquisition_does_not_refetch_upstream(tmp_path, register_backend):
    """Re-acquire with an identical request == cache hit, zero upstream calls."""
    backend = _CountingBackend()
    register_backend('community', backend)

    _service(tmp_path).acquire_forcings()
    assert backend.acquire_calls == 1

    # Wipe the delivered raw data; the repeat must restore from cache, not refetch.
    raw_dir = backend.target_dirs[0]
    for f in raw_dir.iterdir():
        f.unlink()

    _service(tmp_path).acquire_forcings()
    assert backend.acquire_calls == 1, 'identical re-acquisition must be served from cache'
    assert (raw_dir / 'counting_era5.nc').exists()


def test_changed_request_is_a_cache_miss(tmp_path, register_backend):
    """The key is structural: a different window must reach upstream again."""
    backend = _CountingBackend()
    register_backend('community', backend)

    _service(tmp_path).acquire_forcings()
    assert backend.acquire_calls == 1

    _service(tmp_path, time_end='2020-01-03 00:00').acquire_forcings()
    assert backend.acquire_calls == 2, 'a different window must not reuse the cache entry'


def test_cache_keys_are_structural_over_backend_and_schema(tmp_path):
    """(dataset, bbox, window, variables, schema, backend) — pairwise distinct salts."""
    from symfluence.data.cache.forcing_cache import RawForcingCache

    cache = RawForcingCache(cache_root=tmp_path / 'c')
    args = dict(
        dataset='ERA5', bbox='51.3/-115.8/50.9/-115.2',
        time_start='2020-01-01 00:00', time_end='2020-01-02 00:00', variables=None,
    )

    salts = [
        'CLOUD',                      # legacy DATA_ACCESS salts
        'COMMUNITY',
        'native:native-raw',          # protocol path: backend:schema
        'community:canonical-v1',
        'community:native-raw',       # same backend, different schema
        'native:canonical-v1',        # same schema, different backend
    ]
    keys = [cache.generate_cache_key(**args, backend=salt) for salt in salts]
    assert len(set(keys)) == len(keys), 'cache keys must differ across backends AND schemas'

    # Determinism: the same structural request always maps to the same key.
    again = [cache.generate_cache_key(**args, backend=salt) for salt in salts]
    assert keys == again
