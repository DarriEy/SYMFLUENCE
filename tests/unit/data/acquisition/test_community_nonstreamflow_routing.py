# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Community routing for NON-streamflow observations (GRACE TWS, ...).

Generalizes the streamflow-only ObservationBackend routing to the other
observation kinds: under ``DATA_ACCESS: community`` a registered backend (e.g.
COS) serving ``(provider, kind)`` acquires + reduces the observation and writes
the canonical processed file the evaluators read, so the native
``evaluation.<kind>.download`` handler is skipped. Runs fully offline against a
fake TWS backend.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.registries import R
from symfluence.data.acquisition.acquisition_service import AcquisitionService
from symfluence.data.backends.contract import (
    AcquisitionResult,
    ObservationCapability,
    SchemaId,
)
from symfluence.data.observation.paths import tws_default_observation_path

pytestmark = [pytest.mark.unit]


class _FakeTWSBackend:
    """COS-like backend: serves provider 'grace' / kind 'tws', ungraded."""

    name = "community-observation"
    interface_version = "0.3.0"

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.acquired = []

    def capabilities(self):
        return (
            ObservationCapability(
                provider_id="grace", kinds=frozenset({"tws"}),
                station_id_scheme="basin-mean", temporal=None, auth=frozenset(),
                parity_grade=None, notes="ungated COS-like TWS",
            ),
        )

    def acquire(self, request):
        self.acquired.append(request)
        from pathlib import Path
        target = Path(request.target_dir)
        target.mkdir(parents=True, exist_ok=True)
        out = target / "cos_grace_basin_obs_v1.csv"
        # OBS_CSV_V1 TWS shape: datetime, tws_anomaly_mm, uncertainty_mm
        out.write_text(
            "datetime,tws_anomaly_mm,uncertainty_mm\n"
            "2004-01-15,12.3,2.0\n2004-02-15,-4.5,2.0\n2004-03-15,7.8,2.0\n"
        )
        return AcquisitionResult(
            paths=(out,), schema=SchemaId.OBS_CSV_V1, dataset_id="grace",
            backend=self.name, provenance={"integration": "test"},
            variables_delivered=frozenset({"tws_anomaly_mm"}),
        )


def _swap(registry, name, value):
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
def register_tws_backend():
    backend = _FakeTWSBackend()
    restore = _swap(R.observation_backends, "community-observation", backend)
    try:
        yield backend
    finally:
        restore()


def _service(tmp_path, **extra):
    cfg = SymfluenceConfig.from_minimal(
        domain_name="bow", model="SUMMA",
        EXPERIMENT_TIME_START="2002-04-01 00:00", EXPERIMENT_TIME_END="2017-12-31 23:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path), DATA_ACCESS="community", **extra,
    )
    return AcquisitionService(cfg, logging.getLogger("test.cos_routing"))


def test_grace_routes_through_backend_and_writes_canonical_tws(tmp_path, register_tws_backend):
    svc = _service(tmp_path, ALLOW_UNGATED_BACKENDS=True)
    handled = svc._route_community_nonstreamflow_obs(["GRACE"])

    assert "GRACE" in handled, "GRACE should be served by the community TWS backend"
    assert register_tws_backend.acquired, "backend.acquire was never called"
    req = register_tws_backend.acquired[0]
    assert req.provider_id == "grace" and req.kind == "tws"

    # Canonical processed file the TWS evaluator reads, with the GRACE column.
    out_path = tws_default_observation_path(svc.project_dir, "bow")
    assert out_path.exists(), f"canonical TWS file not written at {out_path}"
    df = pd.read_csv(out_path)
    assert "grace_jpl_anomaly" in df.columns
    assert len(df) == 3


def test_ungated_backend_declined_without_allow_flag(tmp_path, register_tws_backend):
    # COS is parity_grade=None with no posture -> the gate refuses it unless
    # ALLOW_UNGATED_BACKENDS: true. GRACE then stays on the native handler.
    svc = _service(tmp_path, ALLOW_UNGATED_BACKENDS=False)
    handled = svc._route_community_nonstreamflow_obs(["GRACE"])
    assert handled == set()
    assert register_tws_backend.acquired == []


def test_noop_outside_community_mode(tmp_path, register_tws_backend):
    cfg = SymfluenceConfig.from_minimal(
        domain_name="bow", model="SUMMA",
        EXPERIMENT_TIME_START="2002-04-01 00:00", EXPERIMENT_TIME_END="2017-12-31 23:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path), DATA_ACCESS="cloud", ALLOW_UNGATED_BACKENDS=True,
    )
    svc = AcquisitionService(cfg, logging.getLogger("test.cos_routing"))
    handled = svc._route_community_nonstreamflow_obs(["GRACE"])
    assert handled == set()
    assert register_tws_backend.acquired == []


def test_unmapped_obs_is_left_for_native(tmp_path, register_tws_backend):
    svc = _service(tmp_path, ALLOW_UNGATED_BACKENDS=True)
    # SMAP is not in the community mapping -> untouched (native path).
    handled = svc._route_community_nonstreamflow_obs(["SMAP"])
    assert handled == set()


class _FakeObsBackend:
    """COS-like backend serving one (provider, kind), delivering OBS_CSV_V1."""

    name = "community-observation"
    interface_version = "0.3.0"

    def __init__(self, provider, kind, value=254.0, config=None, logger=None):
        self._provider = provider
        self._kind = kind
        self._value = value
        self.acquired = []

    def capabilities(self):
        return (
            ObservationCapability(
                provider_id=self._provider, kinds=frozenset({self._kind}),
                station_id_scheme="x", temporal=None, auth=frozenset(),
                parity_grade=None, notes="ungated COS-like",
            ),
        )

    def acquire(self, request):
        self.acquired.append(request)
        from pathlib import Path
        target = Path(request.target_dir)
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"cos_{self._provider}_obs_v1.csv"
        # Canonical OBS_CSV_V1 delivery: datetime, value, quality_flag.
        out.write_text(
            "datetime,value,quality_flag\n"
            f"2010-01-15,{self._value},good\n2010-02-15,{self._value},good\n"
        )
        return AcquisitionResult(
            paths=(out,), schema=SchemaId.OBS_CSV_V1, dataset_id=self._provider,
            backend=self.name, provenance={"integration": "test"},
            variables_delivered=frozenset({self._kind}),
        )


@pytest.mark.parametrize(
    "native_key,provider,kind,path_helper,out_column,scale",
    [
        ("SNOTEL", "snotel", "swe", "swe_default_observation_path", "swe", 1.0 / 25.4),
        ("MODIS_SNOW", "modis_sca", "snow_cover", "snow_cover_default_observation_path", "sca", 1.0),
        ("MODIS_ET", "mod16_et", "et", "modis_et_default_observation_path", "et_mm_day", 1.0),
        ("FLUXNET_ET", "fluxnet_et", "et", "fluxnet_et_default_observation_path", "et_mm_day", 1.0),
        ("USGS_GW", "usgs_gw", "groundwater", "groundwater_default_observation_path", "depth", 1.0),
    ],
)
def test_each_kind_routes_and_writes_canonical(
    tmp_path, native_key, provider, kind, path_helper, out_column, scale
):
    """Each mapped kind acquires via the backend and writes the exact file +
    column its evaluator reads, with the kind's value scale applied."""
    from symfluence.data.observation import paths as obs_paths

    backend = _FakeObsBackend(provider, kind, value=254.0)
    restore = _swap(R.observation_backends, "community-observation", backend)
    try:
        svc = _service(tmp_path, ALLOW_UNGATED_BACKENDS=True)
        handled = svc._route_community_nonstreamflow_obs([native_key])

        assert native_key in handled
        assert backend.acquired and backend.acquired[0].provider_id == provider
        assert backend.acquired[0].kind == kind

        out_path = getattr(obs_paths, path_helper)(svc.project_dir, "bow")
        assert out_path.exists(), f"canonical file not written at {out_path}"
        df = pd.read_csv(out_path)
        assert out_column in df.columns
        # SWE is written in inches (254 mm -> 10 in); the rest are identity-scaled.
        assert df[out_column].iloc[0] == pytest.approx(254.0 * scale)
    finally:
        restore()
