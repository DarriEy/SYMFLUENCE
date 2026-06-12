# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""acquire_forcings() protocol seam: inert today, active only for non-native backends.

Zero-behavior-change guard for Phase A: with no non-native backend registered
under the protocol (every existing configuration), the selector is never
consulted and the legacy dispatch — including the community shadow-wrapper
registry pathway — runs unchanged. The protocol path activates only when a
non-native backend registers AND the selector resolves to it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.acquisition.acquisition_service import AcquisitionService
from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    AcquisitionResult,
    DatasetCapability,
    GridClass,
    SchemaId,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


class _BranchTaken(Exception):
    """Sentinel raised by the patched availability check to halt early."""


def _service(tmp_path, **overrides) -> AcquisitionService:
    config_dict = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'seam_test',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'BOUNDING_BOX_COORDS': '51.3/-115.8/50.9/-115.2',
    }
    config_dict.update(overrides)
    return AcquisitionService(SymfluenceConfig(**config_dict), logging.getLogger('test_seam'))


class FakeProtocolBackend:
    name = 'community'
    interface_version = PROTOCOL_VERSION

    def __init__(self):
        self.requests = []

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
        self.requests.append(request)
        out = Path(request.target_dir) / 'community_era5.nc'
        out.write_bytes(b'fake')
        return AcquisitionResult(
            paths=(out,),
            schema=SchemaId.CANONICAL_V1,
            dataset_id=request.dataset_id,
            backend=self.name,
            provenance={'source': 'fake'},
            variables_delivered=frozenset({'air_temperature'}),
            warnings=('synthetic data',),
        )


@pytest.mark.parametrize('access', ['cloud', 'community'])
def test_seam_is_inert_when_only_native_is_registered(tmp_path, access):
    """No non-native protocol backend (today, always) => selector never consulted,
    legacy dispatch reached exactly as before."""
    svc = _service(tmp_path, DATA_ACCESS=access)
    with patch(
        'symfluence.data.backends.selection.select_backend',
        side_effect=AssertionError('selector must not be consulted'),
    ), patch(
        'symfluence.data.acquisition.acquisition_service.check_cloud_access_availability',
        side_effect=_BranchTaken,
    ) as availability:
        with pytest.raises(_BranchTaken):
            svc.acquire_forcings()
    assert availability.called


def test_seam_routes_to_non_native_backend_under_community_access(tmp_path, register_backend):
    backend = FakeProtocolBackend()
    register_backend('community', backend)
    svc = _service(tmp_path, DATA_ACCESS='community')

    with patch(
        'symfluence.data.acquisition.acquisition_service.check_cloud_access_availability',
        side_effect=_BranchTaken,
    ) as availability:
        svc.acquire_forcings()

    assert not availability.called, 'legacy branch must be skipped when the protocol path serves'
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.dataset_id == 'ERA5'
    assert request.bbox == (50.9, -115.8, 51.3, -115.2)  # (lat_min, lon_min, lat_max, lon_max)
    assert request.target_dir.name == 'raw_data'
    assert (request.target_dir / 'community_era5.nc').exists()


def test_seam_falls_back_to_legacy_when_selector_resolves_native(tmp_path, register_backend):
    """A registered non-native backend that declines leaves the legacy path untouched."""

    class DecliningBackend(FakeProtocolBackend):
        def capabilities(self):
            return ()

    backend = DecliningBackend()
    register_backend('community', backend)
    svc = _service(tmp_path, DATA_ACCESS='community')

    with patch(
        'symfluence.data.acquisition.acquisition_service.check_cloud_access_availability',
        side_effect=_BranchTaken,
    ) as availability:
        with pytest.raises(_BranchTaken):
            svc.acquire_forcings()

    assert availability.called
    assert backend.requests == []


def test_seam_stays_legacy_under_cloud_access_even_with_backend_registered(
    tmp_path, register_backend
):
    backend = FakeProtocolBackend()
    register_backend('community', backend)
    svc = _service(tmp_path, DATA_ACCESS='cloud')

    with patch(
        'symfluence.data.acquisition.acquisition_service.check_cloud_access_availability',
        side_effect=_BranchTaken,
    ) as availability:
        with pytest.raises(_BranchTaken):
            svc.acquire_forcings()

    assert availability.called
    assert backend.requests == []


def test_maf_path_untouched_by_seam(tmp_path, register_backend):
    backend = FakeProtocolBackend()
    register_backend('community', backend)
    svc = _service(tmp_path, DATA_ACCESS='MAF')

    with patch(
        'symfluence.data.acquisition.acquisition_service.datatoolRunner'
    ) as dt:
        dt.supported_datasets.return_value = set()  # force the early MAF error
        with pytest.raises(ValueError, match='DATA_ACCESS: MAF'):
            svc.acquire_forcings()

    assert backend.requests == []
