# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 2 — schema validity (minimal NATIVE_RAW manifest spec).

acquire() output must validate against the declared SchemaId spec, offline,
using a synthetic fixture per backend (shared offline fixture providers live
in ``conftest.py``). Phase A defines the NATIVE_RAW spec minimally as the
sidecar ``acquisition_manifest.json`` written by the backend's wrapping layer
(handlers themselves stay untouched): the manifest must exist next to the raw
files, validate against the manifest schema, and agree with the returned
:class:`AcquisitionResult`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    AcquisitionResult,
    SchemaId,
    read_manifest,
)

pytestmark = [pytest.mark.unit]


def test_acquire_output_validates_against_declared_schema(backend, offline_acquisition):
    request, result = offline_acquisition

    assert isinstance(result, AcquisitionResult)
    assert result.backend == backend.name
    assert result.dataset_id == request.dataset_id
    assert isinstance(result.schema, SchemaId)
    assert result.paths, 'acquire() must report at least one output path'
    for path in result.paths:
        assert Path(path).exists(), f'reported path does not exist: {path}'

    # Declared-schema sidecar manifest: present, valid, and consistent with
    # the result (resumed runs dispatch on this, never on file sniffing).
    manifest_path = request.target_dir / MANIFEST_FILENAME
    assert manifest_path.exists(), 'backend wrapping layer must write the sidecar manifest'
    manifest = read_manifest(manifest_path)  # validates against the spec
    assert manifest['schema'] == str(result.schema)
    assert manifest['dataset_id'] == result.dataset_id
    assert manifest['backend'] == result.backend
    assert manifest['paths'] == [str(p) for p in result.paths]
    assert manifest['variables_delivered'] == sorted(result.variables_delivered)
