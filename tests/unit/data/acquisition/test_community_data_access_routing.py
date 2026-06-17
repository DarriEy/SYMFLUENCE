# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DATA_ACCESS routing for the community backend.

``DATA_ACCESS: community`` consults the AcquisitionBackend selector first;
datasets no registered non-native protocol backend serves fall through to the
same registry-handler pathway as ``cloud``, never the datatool/MAF runner;
the default (MAF) path must be untouched.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.registries import R
from symfluence.data.acquisition.acquisition_service import AcquisitionService

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.fixture
def only_native_backend():
    """Strip non-native protocol backends so the legacy fallthrough is testable
    regardless of which community plugins are installed in the environment."""
    import symfluence.data.backends  # noqa: F401 — ensure 'native' is registered

    removed = []
    for name in list(R.acquisition_backends.keys()):
        if name != 'native':
            removed.append((name, R.acquisition_backends.get(name)))
            R.acquisition_backends.remove(name)

    yield

    for name, entry in removed:
        if R.acquisition_backends.get(name) is None:
            R.acquisition_backends.add(name, entry)


class _BranchTaken(Exception):
    """Sentinel raised by the patched availability check to halt early."""


def _service(tmp_path, **overrides):
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "routing_test",
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "BOUNDING_BOX_COORDS": "51.3/-115.8/50.9/-115.2",
    }
    config_dict.update(overrides)
    config = SymfluenceConfig(**config_dict)
    return AcquisitionService(config, logging.getLogger("test_routing"))


def _availability_spy(*_args, **_kwargs):
    raise _BranchTaken


@pytest.mark.parametrize("access", ["cloud", "community", "CLOUD", "COMMUNITY"])
def test_cloud_and_community_route_to_registry_branch(tmp_path, access, only_native_backend):
    """Both values must enter the registry branch (availability check called)
    when no non-native protocol backend serves the dataset."""
    svc = _service(tmp_path, DATA_ACCESS=access)
    with patch(
        "symfluence.data.acquisition.acquisition_service.check_cloud_access_availability",
        side_effect=_availability_spy,
    ) as spy:
        with pytest.raises(_BranchTaken):
            svc.acquire_forcings()
    assert spy.called


def test_maf_does_not_touch_registry_branch(tmp_path):
    """Explicit MAF must bypass the availability check and use datatool.

    (The typed-config default for ``data_access`` is ``'cloud'`` —
    ``domain.py:173`` — so MAF must be requested explicitly here.)
    """
    svc = _service(tmp_path, DATA_ACCESS="MAF")
    with patch(
        "symfluence.data.acquisition.acquisition_service.check_cloud_access_availability",
        side_effect=_availability_spy,
    ) as spy, patch(
        "symfluence.data.acquisition.acquisition_service.datatoolRunner"
    ) as dt:
        dt.supported_datasets.return_value = set()  # force the early MAF error
        with pytest.raises(ValueError, match="DATA_ACCESS: MAF"):
            svc.acquire_forcings()
    assert not spy.called


def test_cache_key_distinguishes_community_backend(tmp_path):
    """Same request must not share a cache entry across backends (different
    file formats), while None/'cloud' stay backward-compatible."""
    from symfluence.data.cache.forcing_cache import RawForcingCache

    cache = RawForcingCache(cache_root=tmp_path)
    args = dict(
        dataset="AORC",
        bbox=(39.5, -105.5, 40.0, -105.0),
        time_start="2023-06-01",
        time_end="2023-06-03",
        variables=None,
    )
    legacy = cache.generate_cache_key(**args)
    assert cache.generate_cache_key(**args, backend=None) == legacy
    assert cache.generate_cache_key(**args, backend="cloud") == legacy
    assert cache.generate_cache_key(**args, backend="CLOUD") == legacy
    community = cache.generate_cache_key(**args, backend="COMMUNITY")
    assert community != legacy
