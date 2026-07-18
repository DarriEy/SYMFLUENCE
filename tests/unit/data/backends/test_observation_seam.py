# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ObservationBackend seam in ObservedDataProcessor (contract 0.2.0).

Same inert-seam discipline as the forcing seam: with no observation backend
registered (every pre-0.2.0 configuration), the selector is never consulted
and the registry-handler/legacy tiers run unchanged. The backend tier
activates only under ``DATA_ACCESS: community`` when a registered backend
claims the provider and survives the capability/parity checks.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.registries import R
from symfluence.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    AcquisitionResult,
    ObservationCapability,
    ObservationRequest,
    SchemaId,
)
from symfluence.data.observation.registry import ObservationRegistry

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _config(tmp_path, provider='USGS', **extra) -> dict:
    config = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'obs_seam',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': '3600',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'STREAMFLOW_DATA_PROVIDER': provider,
        'STATION_ID': '06191500',
    }
    config.update(extra)
    return config


class FakeObservationBackend:
    name = 'community'
    interface_version = PROTOCOL_VERSION

    def __init__(self, parity_grade='bit-identical'):
        self.requests: list[ObservationRequest] = []
        self.parity_grade = parity_grade

    def capabilities(self):
        return (ObservationCapability(
            provider_id='USGS',
            kinds=frozenset({'streamflow'}),
            station_id_scheme='8-digit USGS site number',
            temporal=None,
            auth=frozenset(),
            parity_grade=self.parity_grade,
        ),)

    def acquire(self, request: ObservationRequest) -> AcquisitionResult:
        self.requests.append(request)
        target = Path(request.target_dir)
        target.mkdir(parents=True, exist_ok=True)
        out = target / 'usgs_06191500_obs_v1.csv'
        out.write_text('datetime,value,quality_flag\n2020-01-01 00:00:00,1.0,A\n')
        return AcquisitionResult(
            paths=(out,),
            schema=SchemaId.OBS_CSV_V1,
            dataset_id=request.provider_id,
            backend=self.name,
            provenance={'source': 'fake'},
            variables_delivered=frozenset({'streamflow'}),
        )


@pytest.fixture
def no_observation_backends():
    """Strip every registered observation backend (env-independent), restoring after."""
    removed: list[tuple[str, object]] = []
    for key in list(R.observation_backends.keys()):
        removed.append((key, R.observation_backends.get(key)))
        R.observation_backends.remove(key)

    yield

    for key, entry in removed:
        if R.observation_backends.get(key) is None:
            R.observation_backends.add(key, entry)


@pytest.fixture
def register_observation_backend(no_observation_backends):
    """Register a backend into the (emptied) observation-backend registry."""
    added: list[str] = []

    def _register(name: str, backend) -> None:
        R.observation_backends.add(name, backend)
        added.append(name)

    yield _register

    for name in added:
        R.observation_backends.remove(name)


def test_seam_is_inert_when_no_backend_is_registered(tmp_path, no_observation_backends):
    """Empty registry => selector never consulted, legacy branch reached as before."""
    logger = MagicMock()
    processor = ObservedDataProcessor(_config(tmp_path, DATA_ACCESS='community'), logger)

    with patch(
        'symfluence.data.backends.selection.select_observation_backend',
        side_effect=AssertionError('selector must not be consulted'),
    ), patch.object(ObservationRegistry, 'get_handler') as get_handler:
        processor.process_streamflow_data()

    # Fallthrough: with csfs absent the legacy USGS branch logs; with csfs
    # installed the registry tier would have served — but it was emptied of
    # backends only, so handler dispatch may still occur. Either way, no error.
    logger.error.assert_not_called()
    # The dispatch note is per-item detail: emitted at DEBUG per ADR 0005.
    assert get_handler.called or any(
        'formalized observation handler' in str(call.args[0])
        for call in logger.debug.call_args_list
    )


def test_community_mode_routes_through_the_backend_tier(tmp_path, register_observation_backend):
    backend = FakeObservationBackend()
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(_config(tmp_path, DATA_ACCESS='community'), logger)

    with patch.object(ObservationRegistry, 'get_handler') as get_handler:
        processor.process_streamflow_data()

    get_handler.assert_not_called()
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.provider_id == 'USGS'
    assert request.kind == 'streamflow'
    assert request.station_ids == ('06191500',)
    assert request.window is not None
    assert request.window[0].startswith('2020-01-01')
    assert request.window[1].startswith('2020-01-02')
    assert Path(request.target_dir).name == 'raw_data'
    logger.error.assert_not_called()


def test_legacy_mode_never_consults_the_backend(tmp_path, register_observation_backend):
    backend = FakeObservationBackend()
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(_config(tmp_path), logger)  # no DATA_ACCESS

    processor.process_streamflow_data()

    assert backend.requests == []
    # The dispatch note is per-item detail: emitted at DEBUG per the logging
    # level policy (ADR 0005), never at INFO.
    debug = [str(call.args[0]) for call in logger.debug.call_args_list]
    assert any('formalized observation handler' in m for m in debug)
    logger.error.assert_not_called()


def test_ungraded_provider_is_refused_and_falls_through(tmp_path, register_observation_backend):
    """Parity gate: parity_grade=None on the backend => handler/legacy tier serves."""
    backend = FakeObservationBackend(parity_grade=None)
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(_config(tmp_path, DATA_ACCESS='community'), logger)

    with patch.object(ObservationRegistry, 'get_handler') as get_handler:
        get_handler.side_effect = lambda key, config, log: MagicMock()
        processor.process_streamflow_data()

    assert backend.requests == []
    info = [str(call.args[0]) for call in logger.info.call_args_list]
    assert any('declined' in m for m in info)


def test_allow_ungated_opt_in_serves_through_the_backend(tmp_path, register_observation_backend):
    backend = FakeObservationBackend(parity_grade=None)
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(
        _config(tmp_path, DATA_ACCESS='community', ALLOW_UNGATED_BACKENDS=True), logger
    )

    processor.process_streamflow_data()
    assert len(backend.requests) == 1


def test_provider_pin_to_native_skips_the_backend(tmp_path, register_observation_backend):
    backend = FakeObservationBackend()
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(
        _config(tmp_path, DATA_ACCESS='community', USGS_BACKEND='native'), logger
    )

    with patch.object(ObservationRegistry, 'get_handler') as get_handler:
        get_handler.side_effect = lambda key, config, log: MagicMock()
        processor.process_streamflow_data()

    assert backend.requests == []


def test_unclaimed_provider_falls_through_cleanly(tmp_path, register_observation_backend):
    """A backend that does not claim the provider must not intercept it."""
    backend = FakeObservationBackend()
    register_observation_backend('community', backend)
    logger = MagicMock()
    processor = ObservedDataProcessor(_config(tmp_path, provider='WSC', DATA_ACCESS='community'), logger)

    with patch.object(ObservationRegistry, 'get_handler') as get_handler:
        get_handler.side_effect = lambda key, config, log: MagicMock()
        processor.process_streamflow_data()

    assert backend.requests == []
