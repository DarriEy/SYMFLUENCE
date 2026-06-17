# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance for AttributeBackend implementations (contract 0.3.0).

Parameterized over every backend registered under ``R.attribute_backends``
(none in a bare install — the module then skips with a reason; the cas plugin
registers ``community``). All tests are offline: CAS upstream access is mocked
at the ``cas.batch_extract_sync`` seam, and geometries are supplied inline so
no shapefile/geopandas dependency is needed.

Items covered, attribute-flavored:
  1. Contract shape       — backend satisfies the AttributeBackend protocol;
                            capabilities are well-formed.
  2. Schema validity      — acquire() writes a valid HRU_STATS_V1 CSV delivery
                            and the sidecar manifest validates.
  3. Honesty              — manifest paths match the delivered files; the CSV
                            has one row per requested HRU.
  + Parity semantics      — declared parity_grade is the tolerance form
                            ("value-within:<tol>"), NOT bit-identical, because
                            zonal stats are resampling-dependent (the normative
                            attribute parity rule, contract 0.3.0).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    AcquisitionResult,
    AttributeBackend,
    AttributeCapability,
    AttributeRequest,
    SchemaId,
    is_compatible,
    read_manifest,
)

pytestmark = [pytest.mark.unit]

#: Attribute parity is tolerance-based by design (zonal stats are not
#: bitwise-reproducible). bit-identical is therefore DISALLOWED here.
_ATTR_PARITY_GRADE_RE = re.compile(r'^value-within:.+$')


def _attribute_backend_names() -> list:
    import symfluence  # noqa: F401 — bootstrap registries/plugins
    import symfluence.data.backends  # noqa: F401
    from symfluence.core.registries import R

    names = R.attribute_backends.keys()
    if not names:
        return [pytest.param(
            '__none__',
            marks=pytest.mark.skip(
                reason='no attribute backends registered '
                       '(community packages not installed)'
            ),
        )]
    return names


@pytest.fixture(params=_attribute_backend_names())
def attribute_backend_name(request) -> str:
    return request.param


@pytest.fixture
def attribute_config(tmp_path):
    """A minimal config that opts CAS in (CAS_DATASETS), so it claims capabilities."""
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
        'CAS_DATASETS': 'isric_soilgrids:clay_0-5cm,terraclimate:aridity',
    })


@pytest.fixture
def attribute_backend(attribute_backend_name, attribute_config):
    from symfluence.core.registries import R
    from symfluence.data.backends.contract import PROTOCOL_VERSION

    entry = R.attribute_backends.get(attribute_backend_name)
    if entry is None:  # pragma: no cover — registry mutated mid-session
        pytest.skip(f'attribute backend {attribute_backend_name!r} no longer registered')
    if isinstance(entry, type):
        materialized = entry(
            attribute_config, logging.getLogger(f'conformance.attr.{attribute_backend_name}'))
    else:
        materialized = entry
    version = getattr(materialized, 'interface_version', '')
    if not is_compatible(version):
        pytest.skip(
            f'attribute backend {attribute_backend_name!r} targets '
            f'interface_version {version!r}, incompatible with protocol '
            f'{PROTOCOL_VERSION}; inert until it re-targets'
        )
    return materialized


# ----------------------------------------------------------------------
# Offline fixture: mock cas.batch_extract_sync with two scalar attributes
# ----------------------------------------------------------------------

def _cas_offline_fixture():
    """Patch cas.batch_extract_sync; return (provider_id, hru_ids, geometries)."""
    from unittest.mock import patch

    cas = pytest.importorskip('cas', reason='attribute backend requires the cas package')
    from cas.core.models import (
        AttributeResponse,
        AttributeResult,
        BatchAttributeResponse,
        QualityFlag,
    )

    hru_ids = (1, 2)
    geometries = (
        {'type': 'Polygon', 'coordinates': [[[-115.7, 50.95], [-115.6, 50.95],
                                             [-115.6, 51.05], [-115.7, 51.05], [-115.7, 50.95]]]},
        {'type': 'Polygon', 'coordinates': [[[-115.5, 51.1], [-115.4, 51.1],
                                             [-115.4, 51.2], [-115.5, 51.2], [-115.5, 51.1]]]},
    )

    def _result(dataset_id, units, value):
        return AttributeResult(
            dataset_id=dataset_id, variable=dataset_id.split(':')[-1], value=value,
            units=units, aggregation='mean', quality=QualityFlag.GOOD,
            coverage_fraction=1.0, pixel_count=42, provider=dataset_id.split(':')[0],
        )

    def _batch_extract_sync(request):
        responses = []
        for i, _geom in enumerate(request.geometries):
            responses.append(AttributeResponse(
                request_id=f'req-{i}', geometry_hash=f'h{i}',
                results=[
                    _result('isric_soilgrids:clay_0-5cm', 'g/kg', 155.0 + i),
                    _result('terraclimate:aridity', '1', 1.6 + i * 0.1),
                ],
                elapsed_ms=1,
            ))
        return BatchAttributeResponse(
            request_id='batch', responses=responses,
            total_geometries=len(responses), total_results=2 * len(responses), elapsed_ms=1,
        )

    patcher = patch.object(cas, 'batch_extract_sync', side_effect=_batch_extract_sync)
    return patcher, {'provider_id': 'CAS', 'hru_ids': hru_ids, 'geometries': geometries}


OFFLINE_ATTRIBUTE_FIXTURES = {'community': _cas_offline_fixture}


@pytest.fixture
def attribute_fixture(attribute_backend):
    provider = OFFLINE_ATTRIBUTE_FIXTURES.get(attribute_backend.name)
    if provider is None:
        pytest.skip(
            f'no offline fixture provider registered for attribute backend '
            f'{attribute_backend.name!r}; add one to test_attribute_backends.py'
        )
    return provider


def _request(tmp_path, facts) -> AttributeRequest:
    return AttributeRequest(
        provider_id=facts['provider_id'],
        attribute_ids=(),
        hru_ids=facts['hru_ids'],
        geometries=facts['geometries'],
        catchment_path=None,
        target_dir=tmp_path / 'attr_delivery',
        lumped=False,
    )


# ----------------------------------------------------------------------
# Item 1 — contract shape
# ----------------------------------------------------------------------

def test_backend_satisfies_the_protocol(attribute_backend):
    assert isinstance(attribute_backend, AttributeBackend)
    assert isinstance(attribute_backend.name, str) and attribute_backend.name
    assert attribute_backend.name == attribute_backend.name.lower()


def test_attribute_capabilities_are_well_formed(attribute_backend):
    caps = attribute_backend.capabilities()
    assert caps, f'backend {attribute_backend.name!r} claims no providers'

    seen: set[str] = set()
    for cap in caps:
        assert isinstance(cap, AttributeCapability)
        assert isinstance(cap.provider_id, str) and cap.provider_id
        key = cap.provider_id.lower()
        assert key not in seen, f'duplicate capability for {cap.provider_id!r}'
        seen.add(key)

        assert isinstance(cap.attribute_ids, frozenset) and cap.attribute_ids
        assert all(isinstance(a, str) and a for a in cap.attribute_ids)
        assert cap.output_kind == 'per_hru_stats'
        assert cap.schema is SchemaId.HRU_STATS_V1
        assert isinstance(cap.auth, frozenset)
        # Parity semantics: tolerance-based, never bit-identical.
        if cap.parity_grade is not None:
            assert _ATTR_PARITY_GRADE_RE.match(cap.parity_grade), (
                f'{cap.provider_id}: attribute parity_grade must be the tolerance '
                f'form "value-within:<tol>", got {cap.parity_grade!r}'
            )
        assert isinstance(cap.notes, str)


# ----------------------------------------------------------------------
# Items 2 + 3 — schema validity and honesty
# ----------------------------------------------------------------------

def test_acquire_writes_a_valid_hru_stats_v1_delivery(attribute_backend, attribute_fixture, tmp_path):
    patcher, facts = attribute_fixture()
    with patcher:
        request = _request(tmp_path, facts)
        result = attribute_backend.acquire(request)

    assert isinstance(result, AcquisitionResult)
    assert result.schema is SchemaId.HRU_STATS_V1
    assert result.backend == attribute_backend.name
    assert result.dataset_id == facts['provider_id']
    assert result.paths and all(Path(p).suffix == '.csv' for p in result.paths)

    manifest_path = request.target_dir / MANIFEST_FILENAME
    assert manifest_path.exists(), 'backend must write the sidecar manifest'
    manifest = read_manifest(manifest_path)
    assert manifest['schema'] == 'hru-stats-v1'
    assert manifest['paths'] == [str(p) for p in result.paths]

    # Honesty: one row per requested HRU, with an hru_id column.
    df = pd.read_csv(result.paths[0])
    assert 'hru_id' in df.columns
    assert len(df) == len(facts['hru_ids'])
