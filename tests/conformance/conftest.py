# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance suite for AcquisitionBackend implementations (design §4).

Provider-agnostic: parameterized over every backend registered under
``R.acquisition_backends`` (today: ``native`` only; community backends join
in Phase A step 3). All tests are offline.

Implemented items (Phase A skeleton):
  1. Contract shape       — test_contract_shape.py
  2. Schema validity      — test_schema_validity.py (minimal NATIVE_RAW manifest)
  +  Extraction readiness — test_extraction_readiness.py (design §5 guard)

Deferred to later phases: 3 honesty, 4 window/bbox semantics,
5 idempotency + cache, 6 parity (live-marked).
"""
from __future__ import annotations

import logging

import pytest


def _backend_names() -> list[str]:
    import symfluence  # noqa: F401 — bootstrap registries/plugins
    import symfluence.data.backends  # noqa: F401 — registers the native backend
    from symfluence.core.registries import R

    return R.acquisition_backends.keys()


@pytest.fixture(params=_backend_names())
def backend_name(request) -> str:
    return request.param


@pytest.fixture
def minimal_config(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig

    return SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'conformance',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'BOUNDING_BOX_COORDS': '51.3/-115.8/50.9/-115.2',
    })


@pytest.fixture
def backend(backend_name, minimal_config):
    """Materialize the registered backend (classes get (config, logger))."""
    from symfluence.core.registries import R

    entry = R.acquisition_backends.get(backend_name)
    if entry is None:  # pragma: no cover — registry mutated mid-session
        pytest.skip(f'backend {backend_name!r} no longer registered')
    if isinstance(entry, type):
        return entry(minimal_config, logging.getLogger(f'conformance.{backend_name}'))
    return entry
