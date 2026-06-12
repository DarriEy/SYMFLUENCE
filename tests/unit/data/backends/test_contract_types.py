# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Contract types: construction, immutability, manifest round-trip, semver."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    PROTOCOL_VERSION,
    AcquisitionBackend,
    AcquisitionRequest,
    AcquisitionResult,
    CredentialContext,
    DatasetCapability,
    GridClass,
    SchemaId,
    build_manifest,
    is_compatible,
    parse_version,
    read_manifest,
    validate_manifest,
    write_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _result(**overrides) -> AcquisitionResult:
    kwargs = dict(
        paths=(Path('/data/raw/file.nc'),),
        schema=SchemaId.NATIVE_RAW,
        dataset_id='ERA5',
        backend='native',
        provenance={'handler': 'x.Y', 'acquired_at': '2026-06-12T00:00:00+00:00'},
        variables_delivered=frozenset({'air_temperature', 'precipitation_flux'}),
    )
    kwargs.update(overrides)
    return AcquisitionResult(**kwargs)


class TestContractTypes:
    def test_enums_match_design(self):
        assert {g.value for g in GridClass} == {'regular_latlon', 'projected', 'station', 'vector'}
        assert {s.value for s in SchemaId} == {'native-raw', 'canonical-v1', 'obs-csv-v1', 'hru-stats-v1'}

    def test_capability_is_frozen(self):
        cap = DatasetCapability(
            dataset_id='ERA5', grid_class=GridClass.REGULAR_LATLON, schema=SchemaId.NATIVE_RAW,
            variables=frozenset({'air_temperature'}), temporal=None, auth=frozenset(), parity_grade=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.dataset_id = 'OTHER'  # type: ignore[misc]

    def test_request_defaults(self, tmp_path):
        req = AcquisitionRequest(
            dataset_id='ERA5', bbox=None, window=None, variables=None, target_dir=tmp_path,
        )
        assert isinstance(req.credentials, CredentialContext)
        assert not req.credentials.has('earthdata')
        assert req.options == {}

    def test_credential_context_is_passive_carrier(self):
        ctx = CredentialContext(providers={'earthdata': {'username': 'u', 'password': 'p'}})
        assert ctx.has('earthdata')
        assert not ctx.has('cds')
        assert ctx.get('earthdata')['username'] == 'u'
        assert ctx.get('cds') == {}

    def test_protocol_is_runtime_checkable(self):
        class Dummy:
            name = 'dummy'
            interface_version = PROTOCOL_VERSION

            def capabilities(self):
                return ()

            def acquire(self, request):
                raise NotImplementedError

        assert isinstance(Dummy(), AcquisitionBackend)
        assert not isinstance(object(), AcquisitionBackend)


class TestSemver:
    def test_parse(self):
        assert parse_version('1.2.3') == (1, 2, 3)
        with pytest.raises(ValueError):
            parse_version('1.2')
        with pytest.raises(ValueError):
            parse_version('not-a-version')

    def test_pre_1_0_minor_is_breaking(self):
        assert is_compatible(PROTOCOL_VERSION)
        major, minor, patch = parse_version(PROTOCOL_VERSION)
        assert is_compatible(f'{major}.{minor}.{patch + 1}')
        assert not is_compatible(f'{major}.{minor + 1}.0')
        assert not is_compatible(f'{major + 1}.0.0')
        assert not is_compatible('garbage')


class TestManifest:
    def test_round_trip(self, tmp_path):
        result = _result(warnings=('partial month at end',))
        manifest_path = write_manifest(result, tmp_path)
        assert manifest_path == tmp_path / MANIFEST_FILENAME

        payload = read_manifest(tmp_path)  # directory form
        assert payload == read_manifest(manifest_path)  # file form
        assert payload['dataset_id'] == 'ERA5'
        assert payload['schema'] == 'native-raw'
        assert payload['backend'] == 'native'
        assert payload['paths'] == ['/data/raw/file.nc']
        assert payload['variables_delivered'] == ['air_temperature', 'precipitation_flux']
        assert payload['warnings'] == ['partial month at end']
        assert payload['interface_version'] == PROTOCOL_VERSION

    def test_build_manifest_is_json_serializable(self):
        json.dumps(build_manifest(_result()))

    def test_validate_rejects_bad_payloads(self):
        good = build_manifest(_result())
        validate_manifest(good)

        with pytest.raises(ValueError, match='mapping'):
            validate_manifest(['not', 'a', 'mapping'])

        missing = dict(good)
        del missing['schema']
        with pytest.raises(ValueError, match="missing field 'schema'"):
            validate_manifest(missing)

        bad_schema = dict(good, schema='no-such-schema')
        with pytest.raises(ValueError, match='unknown schema id'):
            validate_manifest(bad_schema)

        bad_paths = dict(good, paths=['ok', ''])
        with pytest.raises(ValueError, match='paths'):
            validate_manifest(bad_paths)
