# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Declared-schema dataset-handler dispatch off the acquisition manifest.

Manifest present with a non-native schema -> the SCHEMA-keyed handler serves;
manifest missing or schema 'native-raw' -> the existing per-dataset lookup,
bit-identical (migration safety for legacy native raw directories).
"""
from __future__ import annotations

import json
import logging

import pytest

from symfluence.core.registries import R
from symfluence.data.backends.contract import MANIFEST_FILENAME
from symfluence.data.preprocessing.dataset_handlers.schema_dispatch import (
    read_declared_schema,
    resolve_dataset_handler,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]

logger = logging.getLogger('test_schema_dispatch')


class _FakeHandler:
    """Constructor-compatible with DatasetRegistry.get_handler instantiation."""

    def __init__(self, config, logger, project_dir, **kwargs):
        self.config = config
        self.project_dir = project_dir
        self.kwargs = kwargs


class _SchemaHandler(_FakeHandler):
    pass


class _DatasetHandler(_FakeHandler):
    pass


@pytest.fixture
def handler_slots():
    """Temporarily occupy dataset-handler registry slots; restore afterwards."""
    saved: list[tuple[str, object]] = []

    def _occupy(key: str, cls) -> None:
        original = R.dataset_handlers.get(key)
        saved.append((key, original))
        if original is not None:
            R.dataset_handlers.remove(key)
        R.dataset_handlers.add(key, cls)

    yield _occupy

    for key, original in reversed(saved):
        R.dataset_handlers.remove(key)
        if original is not None:
            R.dataset_handlers.add(key, original)


def _write_manifest(project_dir, schema: str) -> None:
    raw_dir = project_dir / 'data' / 'forcing' / 'raw_data'
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / MANIFEST_FILENAME).write_text(json.dumps({
        'manifest_version': 1,
        'interface_version': '0.1.0',
        'dataset_id': 'TESTDS',
        'schema': schema,
        'backend': 'community',
        'paths': [str(raw_dir / 'x.nc')],
        'variables_delivered': ['air_temperature'],
        'provenance': {},
    }))


_CONFIG = {'DOMAIN_NAME': 'dispatch_test', 'FORCING_TIME_STEP_SIZE': 3600}


class TestReadDeclaredSchema:
    def test_missing_manifest_returns_none(self, tmp_path):
        assert read_declared_schema(tmp_path) is None

    def test_declared_schema_is_returned(self, tmp_path):
        _write_manifest(tmp_path, 'canonical-v1')
        assert read_declared_schema(tmp_path) == 'canonical-v1'

    def test_legacy_raw_dir_layout_is_found(self, tmp_path):
        raw_dir = tmp_path / 'forcing' / 'raw_data'
        raw_dir.mkdir(parents=True)
        (raw_dir / MANIFEST_FILENAME).write_text(json.dumps({'schema': 'canonical-v1'}))
        assert read_declared_schema(tmp_path) == 'canonical-v1'

    def test_corrupt_manifest_raises_instead_of_guessing(self, tmp_path):
        raw_dir = tmp_path / 'data' / 'forcing' / 'raw_data'
        raw_dir.mkdir(parents=True)
        (raw_dir / MANIFEST_FILENAME).write_text('{not json')
        with pytest.raises(ValueError, match='Unreadable acquisition manifest'):
            read_declared_schema(tmp_path)


class TestResolveDatasetHandler:
    def test_canonical_schema_resolves_the_schema_keyed_handler(self, tmp_path, handler_slots):
        handler_slots('canonical-v1', _SchemaHandler)
        handler_slots('testds', _DatasetHandler)
        _write_manifest(tmp_path, 'canonical-v1')

        handler = resolve_dataset_handler('TESTDS', _CONFIG, logger, tmp_path)
        assert isinstance(handler, _SchemaHandler)

    def test_native_raw_schema_uses_per_dataset_lookup(self, tmp_path, handler_slots):
        handler_slots('canonical-v1', _SchemaHandler)
        handler_slots('testds', _DatasetHandler)
        _write_manifest(tmp_path, 'native-raw')

        handler = resolve_dataset_handler('TESTDS', _CONFIG, logger, tmp_path)
        assert isinstance(handler, _DatasetHandler)

    def test_no_manifest_means_legacy_per_dataset_lookup(self, tmp_path, handler_slots):
        """Migration safety: raw dirs without a manifest are legacy native files."""
        handler_slots('canonical-v1', _SchemaHandler)
        handler_slots('testds', _DatasetHandler)

        handler = resolve_dataset_handler('TESTDS', _CONFIG, logger, tmp_path)
        assert isinstance(handler, _DatasetHandler)

    def test_declared_schema_without_registered_handler_is_a_hard_error(
        self, tmp_path, handler_slots
    ):
        handler_slots('testds', _DatasetHandler)
        _write_manifest(tmp_path, 'obs-csv-v1')  # valid schema id, no handler key
        with pytest.raises(ValueError, match="declares schema 'obs-csv-v1'"):
            resolve_dataset_handler('TESTDS', _CONFIG, logger, tmp_path)

    def test_handler_kwargs_pass_through(self, tmp_path, handler_slots):
        handler_slots('canonical-v1', _SchemaHandler)
        _write_manifest(tmp_path, 'canonical-v1')
        handler = resolve_dataset_handler(
            'TESTDS', _CONFIG, logger, tmp_path, forcing_timestep_seconds=10800)
        assert handler.kwargs['forcing_timestep_seconds'] == 10800
