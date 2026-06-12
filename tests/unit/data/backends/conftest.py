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


class FakeNativeHandler:
    """Stub handler that mimics BaseAcquisitionHandler.download()."""

    # Looks in-tree to NativeBackend's plugin filter.
    __module__ = 'symfluence.tests.fake_native_handler'

    def __init__(self, config, logger, reporting_manager=None):
        self.config = config
        self.logger = logger

    def download(self, output_dir: Path) -> Path:
        out = Path(output_dir) / 'fake_raw.nc'
        out.write_bytes(b'fake-netcdf')
        return out


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
    """Temporarily register a protocol backend, removing it after the test."""
    injected: list[str] = []

    def _register(name: str, backend) -> None:
        assert name not in R.acquisition_backends, f'refusing to clobber backend {name!r}'
        R.acquisition_backends.add(name, backend)
        injected.append(name)

    yield _register

    for name in injected:
        R.acquisition_backends.remove(name)
