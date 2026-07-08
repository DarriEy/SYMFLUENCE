# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Manager-flow engagement tests for the community backend seams.

These guard the two wiring gaps the Bow e2e found (RESULTS.md Findings 1 & 2):
the community ObservationBackend tier and the AttributeBackend/plugin tier must
engage when driven through the DataManager (``process_observed_data`` /
``run_model_agnostic_preprocessing``), not only when a processor is called
directly. Both run fully offline against fake registered backends.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.registries import R
from symfluence.data import data_manager as dm_mod
from symfluence.data.backends.contract import (
    AcquisitionResult,
    AttributeCapability,
    ObservationCapability,
    SchemaId,
    write_manifest,
)
from symfluence.data.data_manager import DataManager

# ----------------------------------------------------------------------
# Fake backends (registered into R for the duration of a test)
# ----------------------------------------------------------------------

class _FakeObservationBackend:
    name = "community"
    interface_version = "0.2.0"

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger

    def capabilities(self):
        return (
            ObservationCapability(
                provider_id="WSC", kinds=frozenset({"streamflow"}),
                station_id_scheme="WSC station number", temporal=None,
                auth=frozenset(), parity_grade="value-identical:float-repr", notes="",
            ),
        )

    def acquire(self, request):  # pragma: no cover - not reached in these tests
        raise AssertionError("acquire should be stubbed at the processor level")


class _FakeAttributeBackend:
    name = "community"
    interface_version = "0.3.0"

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.acquired = []

    def capabilities(self):
        return (
            AttributeCapability(
                provider_id="CAS", attribute_ids=frozenset({"isric_soilgrids:clay_0-5cm"}),
                output_kind="per_hru_stats", schema=SchemaId.HRU_STATS_V1,
                auth=frozenset(), parity_grade="value-within:resampling-tolerance", notes="",
            ),
        )

    def acquire(self, request):
        self.acquired.append(request)
        target = Path(request.target_dir)
        target.mkdir(parents=True, exist_ok=True)
        out = target / "fake_cas_attributes.csv"
        out.write_text("hru_id,clay_0_5cm\n1,155.0\n")
        result = AcquisitionResult(
            paths=(out,), schema=SchemaId.HRU_STATS_V1, dataset_id="CAS",
            backend=self.name, provenance={"integration": "test"},
            variables_delivered=frozenset({"clay_0_5cm"}),
        )
        write_manifest(result, target)
        return result


def _swap_backend(registry, name, value):
    """Temporarily install *value* under *name*, restoring the prior entry.

    Plugins (cfs/csfs/cas) may already occupy the slot in the shared session
    registry; this saves and restores it so the test never leaks a removed or
    overwritten registration to other tests on the same xdist worker.
    """
    original = registry.get(name)
    if original is not None:
        registry.remove(name)
    registry.add(name, value)

    def _restore():
        if name in registry:
            registry.remove(name)
        if original is not None:
            registry.add(name, original)

    return _restore


@pytest.fixture
def register_obs_backend():
    restore = _swap_backend(R.observation_backends, "community", _FakeObservationBackend)
    try:
        yield
    finally:
        restore()


@pytest.fixture
def register_attr_backend():
    backend = _FakeAttributeBackend()
    restore = _swap_backend(R.attribute_backends, "community", backend)
    try:
        yield backend
    finally:
        restore()


def _community_config(tmp_path, **extra):
    return SymfluenceConfig.from_minimal(
        domain_name="bow", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        DATA_ACCESS="community", ALLOW_UNGATED_BACKENDS=False, **extra,
    )


def _manager(cfg):
    logger = logging.getLogger("test.community_flow")
    with patch.object(dm_mod, "AcquisitionService", MagicMock(__name__="AcquisitionService")), \
         patch.object(dm_mod, "EMEarthIntegrator", MagicMock(__name__="EMEarthIntegrator")), \
         patch.object(dm_mod, "VariableHandler", MagicMock(__name__="VariableHandler")):
        return DataManager(cfg, logger, reporting_manager=None)


# ----------------------------------------------------------------------
# Finding 1 — observation backend engages in the manager flow
# ----------------------------------------------------------------------

def test_community_streamflow_routes_through_processor_backend_tier(tmp_path, register_obs_backend):
    cfg = _community_config(tmp_path, STREAMFLOW_DATA_PROVIDER="WSC")
    dm = _manager(cfg)

    # The pre-check the manager uses must report the backend serves WSC.
    assert dm._observation_backend_serves("WSC") is True

    # Drive the manager flow; assert process_streamflow_data IS invoked (the
    # only home of the backend tier) and WSC is NOT pre-empted into the
    # registry-handler additional_obs loop.
    with patch.object(dm_mod, "ObservedDataProcessor") as proc_cls, \
         patch.object(dm, "acquire_observations"), \
         patch.object(dm, "_ensure_multi_gauge_dataset_present"):
        proc = proc_cls.return_value
        dm.process_observed_data()
        proc.process_streamflow_data.assert_called_once()


def test_non_community_mode_keeps_native_formalized_path(tmp_path, register_obs_backend):
    """DATA_ACCESS != community: backend pre-check declines, native path unchanged."""
    cfg = SymfluenceConfig.from_minimal(
        domain_name="bow", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        DATA_ACCESS="cloud", STREAMFLOW_DATA_PROVIDER="WSC",
    )
    dm = _manager(cfg)
    assert dm._observation_backend_serves("WSC") is False


# ----------------------------------------------------------------------
# Finding 2 — attribute backend + plugin loop engage in the manager flow
# ----------------------------------------------------------------------

def test_community_attribute_backend_engages_and_writes_cas_group(tmp_path, register_attr_backend):
    cfg = _community_config(tmp_path, CAS_DATASETS="isric_soilgrids:clay_0-5cm")
    dm = _manager(cfg)

    # Patch the plugin loop so this test isolates the backend tier (the plugin
    # seam is covered separately); it must still be CALLED.
    with patch.object(dm, "_run_attribute_plugins") as plugins:
        dm._run_community_attribute_pipeline()

    backend = register_attr_backend
    assert backend.acquired, "the attribute backend was never invoked in the manager flow"
    assert backend.acquired[0].provider_id == "CAS"

    # The backend wrote a per-HRU CSV under data/attributes/cas/ — the AttributesNetCDFBuilder ingest dir.
    cas_dir = dm.project_dir / "data" / "attributes" / "cas"
    assert list(cas_dir.glob("*_attributes.csv")), "no cas-group CSV produced for the store"

    # The plugin loop ran too, with CAS excluded (backend is canonical).
    plugins.assert_called_once()
    (_args, kwargs), = [(plugins.call_args.args, plugins.call_args.kwargs)]
    excluded = kwargs.get("exclude_providers") or (plugins.call_args.args[0] if plugins.call_args.args else set())
    assert "cas" in {str(p).lower() for p in excluded}


def test_attribute_pipeline_is_noop_outside_community_mode(tmp_path, register_attr_backend):
    cfg = SymfluenceConfig.from_minimal(
        domain_name="bow", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path), DATA_ACCESS="cloud",
        CAS_DATASETS="isric_soilgrids:clay_0-5cm",
    )
    dm = _manager(cfg)
    dm._run_community_attribute_pipeline()
    assert register_attr_backend.acquired == [], "backend must not run outside community mode"
