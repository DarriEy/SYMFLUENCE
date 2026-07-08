# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NativeObservationBackend: capabilities against the live registry; acquire over a mocked handler."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    ObservationRequest,
    Redistribution,
    SchemaId,
    read_manifest,
)
from symfluence.data.backends.errors import (
    AcquisitionError,
    DatasetUnsupported,
    IntegrityError,
)
from symfluence.data.backends.native import _OBSERVATION_FACTS, NativeObservationBackend

pytestmark = [pytest.mark.unit, pytest.mark.quick]

logger = logging.getLogger('test_native_observation_backend')


class _FakeStreamflowHandler:
    """Stub mimicking BaseObservationHandler.acquire()/process()."""

    __module__ = 'symfluence.tests.fake_observation_handler'

    def __init__(self, processed_dir: Path):
        self._processed_dir = processed_dir

    def acquire(self) -> Path:
        raw = self._processed_dir / 'raw.txt'
        raw.write_text('raw', encoding='utf-8')
        return raw

    def process(self, input_path: Path) -> Path:
        out = self._processed_dir / 'usgs_streamflow_processed.csv'
        out.write_text('datetime,value,quality_flag\n2020-01-01T00:00:00Z,1.0,A\n', encoding='utf-8')
        return out


class TestCapabilities:
    def test_usgs_is_declared_with_open_public_domain_posture(self):
        caps = {c.provider_id: c for c in NativeObservationBackend().capabilities()}
        assert 'USGS' in caps
        usgs = caps['USGS']
        assert 'streamflow' in usgs.kinds
        assert usgs.redistribution == Redistribution.OPEN
        assert usgs.data_license == 'public-domain'
        # Native is the parity reference — exempt from the gate.
        assert usgs.parity_grade is None

    def test_only_providers_with_a_registered_handler_are_claimed(self):
        # Every declared provider must resolve to a real native handler key,
        # so the capability claim never outruns what native can serve.
        from symfluence.data.backends.native import _native_handler_key
        for provider in _OBSERVATION_FACTS:
            claimed = any(c.provider_id == provider for c in NativeObservationBackend().capabilities())
            assert claimed == (_native_handler_key(provider) is not None)

    def test_interface_version_tracks_protocol(self):
        assert NativeObservationBackend().interface_version == PROTOCOL_VERSION


class TestAcquire:
    def _request(self, tmp_path, provider='USGS'):
        return ObservationRequest(
            provider_id=provider,
            station_ids=('06191500',),
            kind='streamflow',
            window=('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z'),
            target_dir=tmp_path,
        )

    def test_happy_path_wraps_handler_output_and_writes_manifest(self, minimal_config, tmp_path):
        handler = _FakeStreamflowHandler(tmp_path)
        backend = NativeObservationBackend(config=minimal_config, logger=logger)
        with patch(
            'symfluence.data.observation.registry.ObservationRegistry.get_handler',
            return_value=handler,
        ):
            result = backend.acquire(self._request(tmp_path))

        assert result.schema == SchemaId.OBS_CSV_V1
        assert result.backend == 'native'
        assert result.dataset_id == 'USGS'
        assert result.redistribution == Redistribution.OPEN
        assert result.data_license == 'public-domain'
        assert result.paths and result.paths[0].name == 'usgs_streamflow_processed.csv'

        manifest = read_manifest(tmp_path)
        assert manifest['backend'] == 'native'
        assert manifest['schema'] == 'obs-csv-v1'
        assert manifest['redistribution'] == 'open'

    def test_unknown_provider_raises_dataset_unsupported(self, minimal_config, tmp_path):
        backend = NativeObservationBackend(config=minimal_config, logger=logger)
        with pytest.raises(DatasetUnsupported):
            backend.acquire(self._request(tmp_path, provider='NO_SUCH_PROVIDER'))

    def test_get_handler_failure_maps_to_dataset_unsupported(self, minimal_config, tmp_path):
        backend = NativeObservationBackend(config=minimal_config, logger=logger)
        with patch(
            'symfluence.data.observation.registry.ObservationRegistry.get_handler',
            side_effect=DataAcquisitionError('no handler'),
        ), pytest.raises(DatasetUnsupported):
            backend.acquire(self._request(tmp_path))

    def test_empty_output_is_an_integrity_error(self, minimal_config, tmp_path):
        class EmptyHandler(_FakeStreamflowHandler):
            def process(self, input_path):
                out = tmp_path / 'usgs_streamflow_processed.csv'
                out.write_text('', encoding='utf-8')
                return out

        backend = NativeObservationBackend(config=minimal_config, logger=logger)
        with patch(
            'symfluence.data.observation.registry.ObservationRegistry.get_handler',
            return_value=EmptyHandler(tmp_path),
        ), pytest.raises(IntegrityError):
            backend.acquire(self._request(tmp_path))

    def test_acquire_without_config_is_an_error(self, tmp_path):
        with pytest.raises(AcquisitionError):
            NativeObservationBackend().acquire(self._request(tmp_path))
