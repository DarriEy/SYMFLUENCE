# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 1 — contract shape.

Capabilities well-formed; declared variables ⊆ schema vocabulary; semver
compatibility. Parameterized over every registered backend.
"""
from __future__ import annotations

import re

import pytest

from symfluence.data.backends.contract import (
    AcquisitionBackend,
    DatasetCapability,
    GridClass,
    SchemaId,
    is_compatible,
)

pytestmark = [pytest.mark.unit]

_PARITY_GRADE_RE = re.compile(r'^(bit-identical(:.+)?|value-identical:.+)$')

#: Schemas whose variable vocabulary is the CFIF forcing vocabulary.
_CFIF_SCHEMAS = {SchemaId.NATIVE_RAW, SchemaId.CANONICAL_V1}


def _cfif_vocabulary() -> frozenset:
    from symfluence.data.preprocessing.cfif import CFIF_VARIABLES

    return frozenset(CFIF_VARIABLES)


def test_backend_satisfies_the_protocol(backend):
    assert isinstance(backend, AcquisitionBackend)
    assert isinstance(backend.name, str) and backend.name
    assert backend.name == backend.name.lower(), 'backend names are lowercase ids'


def test_interface_version_is_compatible_semver(backend):
    assert is_compatible(backend.interface_version), (
        f'backend {backend.name!r} targets interface_version '
        f'{backend.interface_version!r}, incompatible with this framework'
    )


def test_capabilities_are_well_formed(backend):
    caps = backend.capabilities()
    assert caps, f'backend {backend.name!r} claims no datasets'

    seen: set[str] = set()
    for cap in caps:
        assert isinstance(cap, DatasetCapability)
        assert isinstance(cap.dataset_id, str) and cap.dataset_id
        key = cap.dataset_id.lower()
        assert key not in seen, f'duplicate capability for {cap.dataset_id!r}'
        seen.add(key)

        assert isinstance(cap.grid_class, GridClass)
        assert isinstance(cap.schema, SchemaId)
        assert isinstance(cap.variables, frozenset)
        assert cap.variables, f'{cap.dataset_id}: empty variable declaration'
        assert all(isinstance(v, str) and v for v in cap.variables)
        assert isinstance(cap.auth, frozenset)
        assert all(isinstance(a, str) and a for a in cap.auth)

        if cap.temporal is not None:
            assert len(cap.temporal) == 2
            assert all(isinstance(t, str) and t for t in cap.temporal)

        if cap.parity_grade is not None:
            assert _PARITY_GRADE_RE.match(cap.parity_grade), (
                f'{cap.dataset_id}: malformed parity_grade {cap.parity_grade!r}'
            )

        assert isinstance(cap.notes, str)


def test_declared_variables_are_in_the_schema_vocabulary(backend):
    vocabulary = _cfif_vocabulary()
    for cap in backend.capabilities():
        if cap.schema not in _CFIF_SCHEMAS:
            continue
        unknown = set(cap.variables) - vocabulary
        assert not unknown, (
            f'{backend.name}/{cap.dataset_id}: variables not in the CFIF '
            f'vocabulary: {sorted(unknown)}'
        )
