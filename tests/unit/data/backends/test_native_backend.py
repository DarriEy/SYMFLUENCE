# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NativeBackend: capabilities against the live registry; acquire over a mocked handler."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R
from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    AcquisitionRequest,
    GridClass,
    SchemaId,
    read_manifest,
)
from symfluence.data.backends.errors import (
    AcquisitionError,
    AuthRequired,
    DatasetUnsupported,
    UpstreamOutage,
)
from symfluence.data.backends.native import _FORCING_FACTS, NativeBackend

from .conftest import FakeNativeHandler, write_minimal_netcdf

pytestmark = [pytest.mark.unit]

logger = logging.getLogger('test_native_backend')


def _request(tmp_path, dataset='CHIRPS', **overrides) -> AcquisitionRequest:
    kwargs = dict(
        dataset_id=dataset,
        bbox=(50.9, -115.8, 51.3, -115.2),
        window=('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z'),
        variables=None,
        target_dir=tmp_path / 'raw_data',
    )
    kwargs.update(overrides)
    return AcquisitionRequest(**kwargs)


class TestCapabilities:
    def test_every_capability_is_backed_by_a_registered_handler(self):
        backend = NativeBackend()
        caps = backend.capabilities()
        assert caps, 'native backend must claim at least one forcing dataset'
        for cap in caps:
            handler_cls = R.acquisition_handlers.get(cap.dataset_id)
            assert handler_cls is not None, cap.dataset_id

    def test_capability_shape_matches_static_facts(self):
        backend = NativeBackend()
        for cap in backend.capabilities():
            assert cap.schema is SchemaId.NATIVE_RAW
            assert isinstance(cap.grid_class, GridClass)
            assert cap.parity_grade is None  # native is the parity reference
            assert isinstance(cap.variables, frozenset) and cap.variables
            assert isinstance(cap.auth, frozenset)

    def test_registered_facts_datasets_appear(self):
        """Every facts-table dataset with a registered handler is claimed."""
        backend = NativeBackend()
        claimed = {c.dataset_id for c in backend.capabilities()}
        for primary, facts in _FORCING_FACTS.items():
            for key in (primary, *facts.aliases):
                if R.acquisition_handlers.get(key) is not None:
                    assert key in claimed

    def test_unregistered_facts_entries_are_not_claimed(self, occupy_handler_slot):
        original = R.acquisition_handlers.get('CHIRPS')
        if original is None:
            pytest.skip('CHIRPS not registered in this environment')
        R.acquisition_handlers.remove('CHIRPS')
        try:
            claimed = {c.dataset_id for c in NativeBackend().capabilities()}
            assert 'CHIRPS' not in claimed
        finally:
            R.acquisition_handlers.add('CHIRPS', original)


class TestAcquire:
    def test_happy_path_wraps_handler_output_and_writes_manifest(
        self, minimal_config, occupy_handler_slot, tmp_path
    ):
        occupy_handler_slot('CHIRPS', FakeNativeHandler)
        backend = NativeBackend(minimal_config, logger)
        request = _request(tmp_path)

        result = backend.acquire(request)

        assert result.schema is SchemaId.NATIVE_RAW
        assert result.backend == 'native'
        assert result.dataset_id == 'CHIRPS'
        assert result.paths == (tmp_path / 'raw_data' / 'fake_raw.nc',)
        assert result.paths[0].exists()
        assert 'fake_native_handler' in result.provenance['handler']
        assert 'acquired_at' in result.provenance
        # Declared variables fall back to the static capability table
        assert result.variables_delivered == frozenset({'precipitation_flux'})

        manifest = read_manifest(tmp_path / 'raw_data')
        assert manifest['dataset_id'] == 'CHIRPS'
        assert manifest['schema'] == 'native-raw'
        assert manifest['backend'] == 'native'

    def test_requested_variables_win_over_static_declaration(
        self, minimal_config, occupy_handler_slot, tmp_path
    ):
        occupy_handler_slot('CHIRPS', FakeNativeHandler)
        backend = NativeBackend(minimal_config, logger)
        request = _request(tmp_path, variables=frozenset({'precipitation_flux'}))

        result = backend.acquire(request)
        assert result.variables_delivered == frozenset({'precipitation_flux'})

    def test_directory_output_collects_files_excluding_manifest(
        self, minimal_config, occupy_handler_slot, tmp_path
    ):
        class DirHandler(FakeNativeHandler):
            def download(self, output_dir):
                for name in ('b.nc', 'a.nc'):
                    write_minimal_netcdf(output_dir / name)
                return output_dir

        occupy_handler_slot('CHIRPS', DirHandler)
        result = NativeBackend(minimal_config, logger).acquire(_request(tmp_path))

        names = [p.name for p in result.paths]
        assert names == ['a.nc', 'b.nc']
        assert MANIFEST_FILENAME not in names

    def test_unknown_dataset_raises_dataset_unsupported_with_cause(self, minimal_config, tmp_path):
        backend = NativeBackend(minimal_config, logger)
        with pytest.raises(DatasetUnsupported) as excinfo:
            backend.acquire(_request(tmp_path, dataset='NO_SUCH_DATASET'))
        assert isinstance(excinfo.value.__cause__, DataAcquisitionError)
        assert excinfo.value.dataset_id == 'NO_SUCH_DATASET'

    @pytest.mark.parametrize(
        'raised, expected',
        [
            (ValueError('bad bbox'), AcquisitionError),
            (RuntimeError('boom'), AcquisitionError),
            (DataAcquisitionError('legacy failure'), AcquisitionError),
            (ConnectionError('refused'), UpstreamOutage),
            (TimeoutError('slow upstream'), UpstreamOutage),
            (PermissionError('403'), AuthRequired),
        ],
    )
    def test_handler_failures_map_onto_taxonomy_preserving_cause(
        self, minimal_config, occupy_handler_slot, tmp_path, raised, expected
    ):
        class FailingHandler(FakeNativeHandler):
            def download(self, output_dir):
                raise raised

        occupy_handler_slot('CHIRPS', FailingHandler)
        backend = NativeBackend(minimal_config, logger)

        with pytest.raises(expected) as excinfo:
            backend.acquire(_request(tmp_path))
        assert excinfo.value.__cause__ is raised
        # New taxonomy stays catchable at existing DataAcquisitionError sites
        assert isinstance(excinfo.value, DataAcquisitionError)

    def test_protocol_errors_pass_through_unwrapped(
        self, minimal_config, occupy_handler_slot, tmp_path
    ):
        original = UpstreamOutage('already mapped', upstream='CHIRPS')

        class MappedHandler(FakeNativeHandler):
            def download(self, output_dir):
                raise original

        occupy_handler_slot('CHIRPS', MappedHandler)
        with pytest.raises(UpstreamOutage) as excinfo:
            NativeBackend(minimal_config, logger).acquire(_request(tmp_path))
        assert excinfo.value is original

    def test_acquire_without_config_is_an_error(self, tmp_path):
        with pytest.raises(AcquisitionError, match='requires a framework config'):
            NativeBackend().acquire(_request(tmp_path))

    def test_no_manifest_written_on_failure(self, minimal_config, occupy_handler_slot, tmp_path):
        class FailingHandler(FakeNativeHandler):
            def download(self, output_dir):
                raise ValueError('nope')

        occupy_handler_slot('CHIRPS', FailingHandler)
        with pytest.raises(AcquisitionError):
            NativeBackend(minimal_config, logger).acquire(_request(tmp_path))
        assert not (tmp_path / 'raw_data' / MANIFEST_FILENAME).exists()

    def test_honesty_verification_recorded_in_provenance(
        self, minimal_config, occupy_handler_slot, tmp_path
    ):
        """Item 3: what was cheaply verified is recorded, auditable, in the manifest."""
        occupy_handler_slot('CHIRPS', FakeNativeHandler)
        result = NativeBackend(minimal_config, logger).acquire(_request(tmp_path))

        assert 'netcdf-readable' in result.provenance['honesty_checks']
        assert result.provenance['netcdf_variables'] == 'precip'
        manifest = read_manifest(tmp_path / 'raw_data')
        assert manifest['provenance']['netcdf_variables'] == 'precip'

    @pytest.mark.parametrize(
        'corruption, match',
        [
            ('empty', 'empty'),
            ('garbage', 'unreadable NetCDF'),
            ('missing', 'does not exist'),
        ],
    )
    def test_dishonest_delivery_raises_integrity_error(
        self, minimal_config, occupy_handler_slot, tmp_path, corruption, match
    ):
        """The WSC-pagination bug class: partial/garbage delivery must not look like success."""
        from symfluence.data.backends.errors import IntegrityError

        class DishonestHandler(FakeNativeHandler):
            def download(self, output_dir):
                out = Path(output_dir) / 'fake_raw.nc'
                if corruption == 'empty':
                    out.write_bytes(b'')
                elif corruption == 'garbage':
                    out.write_bytes(b'not-a-netcdf')
                # 'missing': report a path that was never written
                return out

        occupy_handler_slot('CHIRPS', DishonestHandler)
        with pytest.raises(IntegrityError, match=match):
            NativeBackend(minimal_config, logger).acquire(_request(tmp_path))
        assert not (tmp_path / 'raw_data' / MANIFEST_FILENAME).exists(), (
            'no manifest may be written for a delivery that failed verification'
        )

    def test_handler_instantiated_exactly_as_the_service_does(
        self, minimal_config, tmp_path
    ):
        """acquire() must go through AcquisitionRegistry.get_handler(dataset, config, logger)."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        with patch.object(
            AcquisitionRegistry, 'get_handler',
            side_effect=lambda name, config, log: FakeNativeHandler(config, log),
        ) as get_handler:
            NativeBackend(minimal_config, logger).acquire(_request(tmp_path))

        get_handler.assert_called_once_with('CHIRPS', minimal_config, logger)
