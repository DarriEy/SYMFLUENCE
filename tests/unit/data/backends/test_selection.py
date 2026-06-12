# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""select_backend(): priority, override, capability-decline fallthrough, parity policy."""
from __future__ import annotations

import logging

import pytest

from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    DatasetCapability,
    GridClass,
    SchemaId,
)
from symfluence.data.backends.errors import DatasetUnsupported
from symfluence.data.backends.selection import select_backend

from .conftest import FakeNativeHandler

pytestmark = [pytest.mark.unit, pytest.mark.quick]

logger = logging.getLogger('test_selection')

#: Dataset used throughout; the fixture below guarantees the native backend
#: claims it regardless of which community plugins are installed in the env.
DATASET = 'CHIRPS'


def _capability(dataset=DATASET, parity_grade='bit-identical', variables=('precipitation_flux',)):
    return DatasetCapability(
        dataset_id=dataset,
        grid_class=GridClass.REGULAR_LATLON,
        schema=SchemaId.CANONICAL_V1,
        variables=frozenset(variables),
        temporal=None,
        auth=frozenset(),
        parity_grade=parity_grade,
    )


class FakeCommunityBackend:
    name = 'community'
    interface_version = PROTOCOL_VERSION

    def __init__(self, caps=()):
        self._caps = tuple(caps)
        self.acquired = []

    def capabilities(self):
        return self._caps

    def acquire(self, request):
        self.acquired.append(request)
        raise NotImplementedError('selection tests never acquire')


@pytest.fixture
def native_claims_dataset(occupy_handler_slot):
    """Guarantee DATASET resolves to an in-tree handler (env-independent)."""
    occupy_handler_slot(DATASET, FakeNativeHandler)


class TestPriority:
    def test_default_data_access_resolves_native(self, native_claims_dataset):
        assert select_backend(DATASET, {}, logger=logger).name == 'native'

    @pytest.mark.parametrize('access', ['cloud', 'CLOUD', 'MAF', 'maf'])
    def test_non_community_access_never_consults_community(
        self, native_claims_dataset, register_backend, access
    ):
        community = FakeCommunityBackend([_capability()])
        register_backend('community', community)
        selected = select_backend(DATASET, {'DATA_ACCESS': access}, logger=logger)
        assert selected.name == 'native'

    @pytest.mark.parametrize('access', ['community', 'COMMUNITY'])
    def test_community_access_prefers_community(
        self, native_claims_dataset, register_backend, access
    ):
        community = FakeCommunityBackend([_capability()])
        register_backend('community', community)
        selected = select_backend(DATASET, {'DATA_ACCESS': access}, logger=logger)
        assert selected is community

    def test_community_access_without_community_backend_falls_to_native(
        self, native_claims_dataset
    ):
        selected = select_backend(DATASET, {'DATA_ACCESS': 'community'}, logger=logger)
        assert selected.name == 'native'

    def test_no_backend_can_serve_raises(self):
        with pytest.raises(DatasetUnsupported) as excinfo:
            select_backend('NO_SUCH_DATASET', {}, logger=logger)
        assert 'NO_SUCH_DATASET' in str(excinfo.value)


class TestOverride:
    def test_dataset_backend_key_pins_native_under_community_access(
        self, native_claims_dataset, register_backend
    ):
        community = FakeCommunityBackend([_capability()])
        register_backend('community', community)
        config = {'DATA_ACCESS': 'community', f'{DATASET}_BACKEND': 'native'}
        assert select_backend(DATASET, config, logger=logger).name == 'native'

    def test_dataset_backend_key_pins_community_under_cloud_access(
        self, native_claims_dataset, register_backend
    ):
        community = FakeCommunityBackend([_capability()])
        register_backend('community', community)
        config = {'DATA_ACCESS': 'cloud', f'{DATASET}_BACKEND': 'community'}
        assert select_backend(DATASET, config, logger=logger) is community

    def test_hyphenated_dataset_uses_sanitized_flat_key(self, register_backend):
        community = FakeCommunityBackend([_capability(dataset='NEX-GDDP')])
        register_backend('community', community)
        config = {'NEX_GDDP_BACKEND': 'community'}
        assert select_backend('NEX-GDDP', config, logger=logger) is community

    def test_pinned_backend_that_cannot_serve_raises_without_fallthrough(
        self, native_claims_dataset, register_backend
    ):
        community = FakeCommunityBackend([])  # claims nothing
        register_backend('community', community)
        config = {f'{DATASET}_BACKEND': 'community'}
        with pytest.raises(DatasetUnsupported, match='pinned'):
            select_backend(DATASET, config, logger=logger)

    def test_pinned_unregistered_backend_raises(self, native_claims_dataset):
        config = {f'{DATASET}_BACKEND': 'community'}
        with pytest.raises(DatasetUnsupported, match='not registered'):
            select_backend(DATASET, config, logger=logger)


class TestCapabilityDecline:
    def test_unclaimed_dataset_falls_through_to_native_with_info_log(
        self, native_claims_dataset, register_backend, caplog
    ):
        community = FakeCommunityBackend([])  # declines everything
        register_backend('community', community)
        with caplog.at_level(logging.INFO, logger='symfluence.data.backends.selection'):
            selected = select_backend(DATASET, {'DATA_ACCESS': 'community'})
        assert selected.name == 'native'
        assert any('declined' in record.message for record in caplog.records)
        assert any('does not claim' in record.getMessage() for record in caplog.records)

    def test_unservable_variables_fall_through(self, native_claims_dataset, register_backend):
        community = FakeCommunityBackend([_capability(variables=('air_temperature',))])
        register_backend('community', community)
        selected = select_backend(
            DATASET, {'DATA_ACCESS': 'community'},
            variables=frozenset({'precipitation_flux'}), logger=logger,
        )
        assert selected.name == 'native'

    def test_window_outside_declared_coverage_falls_through(
        self, native_claims_dataset, register_backend
    ):
        cap = DatasetCapability(
            dataset_id=DATASET, grid_class=GridClass.REGULAR_LATLON, schema=SchemaId.CANONICAL_V1,
            variables=frozenset({'precipitation_flux'}), temporal=('2002-01-01', '2020-01-01'),
            auth=frozenset(), parity_grade='bit-identical',
        )
        community = FakeCommunityBackend([cap])
        register_backend('community', community)
        selected = select_backend(
            DATASET, {'DATA_ACCESS': 'community'},
            window=('1999-01-01', '2000-01-01'), logger=logger,
        )
        assert selected.name == 'native'

    def test_incompatible_interface_version_falls_through(
        self, native_claims_dataset, register_backend
    ):
        community = FakeCommunityBackend([_capability()])
        community.interface_version = '99.0.0'
        register_backend('community', community)
        selected = select_backend(DATASET, {'DATA_ACCESS': 'community'}, logger=logger)
        assert selected.name == 'native'


class TestParityPolicy:
    def test_ungraded_dataset_refused_from_non_native_backend(
        self, native_claims_dataset, register_backend
    ):
        community = FakeCommunityBackend([_capability(parity_grade=None)])
        register_backend('community', community)
        selected = select_backend(DATASET, {'DATA_ACCESS': 'community'}, logger=logger)
        assert selected.name == 'native'

    @pytest.mark.parametrize('flag', [True, 'true', 'TRUE', 'yes', '1'])
    def test_allow_ungated_backends_permits_ungraded(
        self, native_claims_dataset, register_backend, flag
    ):
        community = FakeCommunityBackend([_capability(parity_grade=None)])
        register_backend('community', community)
        config = {'DATA_ACCESS': 'community', 'ALLOW_UNGATED_BACKENDS': flag}
        assert select_backend(DATASET, config, logger=logger) is community

    @pytest.mark.parametrize('flag', [False, 'false', '0', 'no', ''])
    def test_falsy_allow_ungated_still_refuses(
        self, native_claims_dataset, register_backend, flag
    ):
        community = FakeCommunityBackend([_capability(parity_grade=None)])
        register_backend('community', community)
        config = {'DATA_ACCESS': 'community', 'ALLOW_UNGATED_BACKENDS': flag}
        assert select_backend(DATASET, config, logger=logger).name == 'native'

    def test_native_backend_is_exempt_from_parity_gate(self, native_claims_dataset):
        # Native capabilities are all parity_grade=None (it IS the reference);
        # selection must still hand it the dataset.
        assert select_backend(DATASET, {}, logger=logger).name == 'native'
