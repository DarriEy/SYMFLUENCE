# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Conformance suite for AcquisitionBackend implementations (design §4).

Provider-agnostic: parameterized over every backend registered under
``R.acquisition_backends`` (``native`` always; community backends when their
packages are installed — backends without an offline fixture provider are
skipped with a reason, never silently). All tests are offline.

Implemented items:
  1. Contract shape          — test_contract_shape.py
  2. Schema validity         — test_schema_validity.py (minimal NATIVE_RAW manifest)
  3. Honesty                 — test_honesty.py (manifest vs files; induced truncation)
  4. Window/bbox semantics   — test_window_bbox_semantics.py (UTC [start, end); 1-pixel bbox rule)
  5. Idempotency + cache     — test_idempotency_cache.py (no re-fetch; structural keys)
  +  Extraction readiness    — test_extraction_readiness.py (design §5 guard)

Deferred: 6 parity (live-marked; absorbs the /tmp experiments).
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import pytest

#: The canonical offline conformance request. The offline fixture providers
#: below synthesize data that *complies* with the contract rules for exactly
#: this window/bbox, so items 2-4 can assert conformance end to end.
CONFORMANCE_BBOX = (50.9, -115.8, 51.3, -115.2)  # (lat_min, lon_min, lat_max, lon_max)
CONFORMANCE_WINDOW = ('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z')  # UTC [start, end)


def _backend_names() -> list[str]:
    import symfluence  # noqa: F401 — bootstrap registries/plugins
    import symfluence.data.backends  # noqa: F401 — registers the native backend
    from symfluence.core.registries import R

    return R.acquisition_backends.keys()


@pytest.fixture(params=_backend_names())
def backend_name(request) -> str:
    return request.param


@pytest.fixture
def minimal_config(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig

    return SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'conformance',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'BOUNDING_BOX_COORDS': '51.3/-115.8/50.9/-115.2',
    })


@pytest.fixture
def backend(backend_name, minimal_config):
    """Materialize the registered backend (classes get (config, logger)).

    Backends targeting an incompatible interface version are SKIPPED, not
    failed: pre-1.0, a minor contract bump is breaking by design, and a
    third-party backend hardcoding an older target is *detected* as skew by
    the selection layer and declined (it is inert, so conformance of its
    acquire path is moot until it re-targets). The in-tree native backend
    imports PROTOCOL_VERSION and can never skew.
    """
    from symfluence.core.registries import R
    from symfluence.data.backends.contract import PROTOCOL_VERSION, is_compatible

    entry = R.acquisition_backends.get(backend_name)
    if entry is None:  # pragma: no cover — registry mutated mid-session
        pytest.skip(f'backend {backend_name!r} no longer registered')
    if isinstance(entry, type):
        materialized = entry(minimal_config, logging.getLogger(f'conformance.{backend_name}'))
    else:
        materialized = entry
    version = getattr(materialized, 'interface_version', '')
    if not is_compatible(version):
        pytest.skip(
            f'backend {backend_name!r} targets interface_version {version!r}, '
            f'incompatible with protocol {PROTOCOL_VERSION} (pre-1.0 minor bumps '
            f'are breaking); the selection layer declines it, so it is inert '
            f'until it re-targets'
        )
    return materialized


@pytest.fixture
def register_backend():
    """Temporarily register a protocol backend, restoring the slot afterwards."""
    from symfluence.core.registries import R

    displaced: list[tuple[str, object]] = []

    def _register(name: str, backend) -> None:
        original = R.acquisition_backends.get(name)
        displaced.append((name, original))
        if original is not None:
            R.acquisition_backends.remove(name)
        R.acquisition_backends.add(name, backend)

    yield _register

    for name, original in reversed(displaced):
        R.acquisition_backends.remove(name)
        if original is not None:
            R.acquisition_backends.add(name, original)


# ----------------------------------------------------------------------
# Offline fixture providers, one per backend
# ----------------------------------------------------------------------

@contextlib.contextmanager
def _native_offline_fixture():
    """Stub the native handler lookup so acquire() runs fully offline.

    The stub writes a *real* NetCDF whose axes comply with the canonical
    conformance window/bbox: NativeBackend's honesty layer (item 3) verifies
    NetCDF readability, and items 3-4 inspect the delivered axes.
    """
    from unittest.mock import patch

    np = pytest.importorskip('numpy')
    xr = pytest.importorskip('xarray')
    import pandas as pd

    from symfluence.data.acquisition.registry import AcquisitionRegistry

    class StubHandler:
        def __init__(self, config, logger, reporting_manager=None):
            pass

        def download(self, output_dir: Path) -> Path:
            out = Path(output_dir) / 'synthetic_raw.nc'
            time = pd.date_range('2020-01-01', periods=24, freq='h')  # [start, end) hourly
            lat = np.linspace(50.95, 51.25, 4)    # 0.1° centers; extent == bbox
            lon = np.linspace(-115.75, -115.25, 6)
            shape = (time.size, lat.size, lon.size)
            ds = xr.Dataset(
                # Native upstream variable name — NATIVE_RAW keeps raw names.
                {'precip': (('time', 'latitude', 'longitude'),
                            np.zeros(shape, dtype='float32'))},
                coords={'time': time, 'latitude': lat, 'longitude': lon},
            )
            ds.to_netcdf(out)
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
    backend never registers and this fixture is never consulted). The
    synthetic dataset complies with the canonical window/bbox: 24 hourly
    steps in [start, end) and 0.2°/0.3° cell centers within one native pixel
    of the requested bbox edges.
    """
    from datetime import datetime
    from unittest.mock import patch

    cfs = pytest.importorskip('cfs', reason='community backend requires the cfs package')
    np = pytest.importorskip('numpy')
    xr = pytest.importorskip('xarray')

    from cfs.core.models import BoundingBox, FetchResult, TimeRange

    time = [datetime(2020, 1, 1, h) for h in range(24)]
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
        n_times=24, n_lat=2, n_lon=2, resolution_deg=0.25,
    )
    with patch.object(cfs, 'fetch_sync', return_value=(ds, fetch_result)):
        yield 'ERA5'  # a dataset id the community backend claims


#: backend name -> context manager yielding a claimed dataset id, with the
#: backend's upstream access stubbed for offline execution.
OFFLINE_FIXTURES = {
    'native': _native_offline_fixture,
    'community': _community_offline_fixture,
}


@pytest.fixture
def offline_acquisition(backend, tmp_path):
    """Acquire the canonical conformance request offline through *backend*.

    Skip-with-reason when no offline fixture provider exists for the backend
    (new backends must contribute one to join items 2-4).
    """
    from symfluence.data.backends.contract import AcquisitionRequest

    fixture_provider = OFFLINE_FIXTURES.get(backend.name)
    if fixture_provider is None:
        pytest.skip(
            f'no offline fixture provider registered for backend {backend.name!r}; '
            f'add one to tests/conformance/conftest.py'
        )

    with fixture_provider() as dataset_id:
        request = AcquisitionRequest(
            dataset_id=dataset_id,
            bbox=CONFORMANCE_BBOX,
            window=CONFORMANCE_WINDOW,
            variables=None,
            target_dir=tmp_path / 'raw_data',
        )
        result = backend.acquire(request)
    return request, result
