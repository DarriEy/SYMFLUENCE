# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared fixtures for acquisition-backend protocol tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.core.registries import R


@pytest.fixture
def minimal_config(tmp_path):
    """A minimal typed config sufficient for AcquisitionService/handlers."""
    from symfluence.core.config.models import SymfluenceConfig

    return SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'backend_test',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'BOUNDING_BOX_COORDS': '51.3/-115.8/50.9/-115.2',
    })


def write_minimal_netcdf(path: Path, variables: tuple[str, ...] = ('precip',)) -> Path:
    """Write a tiny but *real* NetCDF file (NativeBackend now verifies readability)."""
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {name: (('time',), np.array([1.0, 2.0], dtype='float32')) for name in variables},
        coords={'time': np.array([0, 1], dtype='int64')},
    )
    ds.to_netcdf(path)
    return path


class FakeNativeHandler:
    """Stub handler that mimics BaseAcquisitionHandler.download()."""

    # Looks in-tree to NativeBackend's plugin filter.
    __module__ = 'symfluence.tests.fake_native_handler'

    def __init__(self, config, logger, reporting_manager=None):
        self.config = config
        self.logger = logger

    def download(self, output_dir: Path) -> Path:
        return write_minimal_netcdf(Path(output_dir) / 'fake_raw.nc')


@pytest.fixture
def occupy_handler_slot():
    """Temporarily occupy an R.acquisition_handlers slot, restoring the original."""
    saved: list[tuple[str, object]] = []

    def _occupy(key: str, handler_cls: type) -> None:
        original = R.acquisition_handlers.get(key)
        saved.append((key, original))
        R.acquisition_handlers.remove(key)
        R.acquisition_handlers.add(key, handler_cls)

    yield _occupy

    for key, original in reversed(saved):
        R.acquisition_handlers.remove(key)
        if original is not None:
            R.acquisition_handlers.add(key, original)


@pytest.fixture
def register_backend():
    """Temporarily register a protocol backend, restoring the slot afterwards.

    With community plugins installed in the environment (e.g. the cfs
    package's CommunityForcingBackend), the slot may legitimately be occupied
    — the original entry is displaced for the test and restored after.
    """
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


@pytest.fixture
def only_native_backend():
    """Temporarily strip every non-native protocol backend (env-independent).

    Tests asserting "no non-native backend registered" behavior must hold
    even when community plugins are installed in the environment.
    """
    import symfluence.data.backends  # noqa: F401 — ensure 'native' is registered

    removed: list[tuple[str, object]] = []
    for name in list(R.acquisition_backends.keys()):
        if name != 'native':
            removed.append((name, R.acquisition_backends.get(name)))
            R.acquisition_backends.remove(name)

    yield

    for name, entry in removed:
        if R.acquisition_backends.get(name) is None:
            R.acquisition_backends.add(name, entry)
