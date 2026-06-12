# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance item 2 — schema validity (minimal NATIVE_RAW manifest spec).

acquire() output must validate against the declared SchemaId spec, offline,
using a synthetic fixture per backend. Phase A defines the NATIVE_RAW spec
minimally as the sidecar ``acquisition_manifest.json`` written by the
backend's wrapping layer (handlers themselves stay untouched): the manifest
must exist next to the raw files, validate against the manifest schema, and
agree with the returned :class:`AcquisitionResult`.

Each backend contributes an offline fixture provider below; backends without
one are skipped (their acquire path needs recorded fixtures first).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    AcquisitionRequest,
    AcquisitionResult,
    SchemaId,
    read_manifest,
)

pytestmark = [pytest.mark.unit]


# ----------------------------------------------------------------------
# Offline fixture providers, one per backend
# ----------------------------------------------------------------------

@contextlib.contextmanager
def _native_offline_fixture():
    """Stub the native handler lookup so acquire() runs fully offline."""
    from unittest.mock import patch

    from symfluence.data.acquisition.registry import AcquisitionRegistry

    class StubHandler:
        def __init__(self, config, logger, reporting_manager=None):
            pass

        def download(self, output_dir: Path) -> Path:
            out = Path(output_dir) / 'synthetic_raw.nc'
            out.write_bytes(b'synthetic')
            return out

    with patch.object(
        AcquisitionRegistry, 'get_handler',
        side_effect=lambda name, config, logger: StubHandler(config, logger),
    ):
        yield 'CHIRPS'  # a dataset id the native backend claims


@contextlib.contextmanager
def _community_offline_fixture():
    """Stub cfs.fetch_sync so the community backend's acquire() runs offline.

    Present only when the cfs package is installed (otherwise the community
    backend never registers and this fixture is never consulted).
    """
    from datetime import datetime
    from unittest.mock import patch

    cfs = pytest.importorskip('cfs', reason='community backend requires the cfs package')
    np = pytest.importorskip('numpy')
    xr = pytest.importorskip('xarray')

    from cfs.core.models import BoundingBox, FetchResult, TimeRange

    time = [datetime(2020, 1, 1, h) for h in range(4)]
    lat = np.array([50.95, 51.15])
    lon = np.array([-115.7, -115.4])
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            'air_temperature': (
                ('time', 'latitude', 'longitude'), np.full(shape, 275.0, dtype='float32'),
                {'standard_name': 'air_temperature', 'units': 'K'},
            ),
            'precipitation_flux': (
                ('time', 'latitude', 'longitude'), np.full(shape, 1e-5, dtype='float32'),
                {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'},
            ),
        },
        coords={'time': time, 'latitude': lat, 'longitude': lon},
        attrs={'cfs_schema': 'canonical-v1'},
    )
    fetch_result = FetchResult(
        product_id='era5_arco:single_levels',
        provider='era5_arco',
        variables=['air_temperature', 'precipitation_flux'],
        bbox=BoundingBox(min_lon=-115.8, min_lat=50.9, max_lon=-115.2, max_lat=51.3),
        time_range=TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 1, 2)),
        n_times=4, n_lat=2, n_lon=2, resolution_deg=0.25,
    )
    with patch.object(cfs, 'fetch_sync', return_value=(ds, fetch_result)):
        yield 'ERA5'  # a dataset id the community backend claims


#: backend name -> context manager yielding a claimed dataset id, with the
#: backend's upstream access stubbed for offline execution.
_OFFLINE_FIXTURES = {
    'native': _native_offline_fixture,
    'community': _community_offline_fixture,
}


def test_acquire_output_validates_against_declared_schema(backend, tmp_path):
    fixture_provider = _OFFLINE_FIXTURES.get(backend.name)
    if fixture_provider is None:
        pytest.skip(
            f'no offline fixture provider registered for backend {backend.name!r}; '
            f'add one to tests/conformance/test_schema_validity.py'
        )

    with fixture_provider() as dataset_id:
        request = AcquisitionRequest(
            dataset_id=dataset_id,
            bbox=(50.9, -115.8, 51.3, -115.2),
            window=('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z'),
            variables=None,
            target_dir=tmp_path / 'raw_data',
        )
        result = backend.acquire(request)

    assert isinstance(result, AcquisitionResult)
    assert result.backend == backend.name
    assert result.dataset_id == dataset_id
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
