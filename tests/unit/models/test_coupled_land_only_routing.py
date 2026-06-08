# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Coupled chain routing-ownership tests.

Guards the land-only mode introduced for joint SUMMA+dRoute calibration: in a coupled chain the
UPSTREAM land model (SUMMA) must NOT run its own routing (mizuRoute); the downstream objective
routing model (dRoute) routes the runoff. Crucially, this must NOT change standalone SUMMA, whose
SUMMA -> mizuRoute coupling stays intact (the ``skip_routing`` flag is only ever set inside the
coupled chain).
"""
from __future__ import annotations

import logging
from pathlib import Path

from symfluence.models.coupled.calibration.worker import CoupledModelWorker
from symfluence.models.summa.calibration.worker import SUMMAWorker


def _coupled_config():
    return {
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'ROUTING_MODEL': 'DROUTE',
        'DOMAIN_DEFINITION_METHOD': 'subset',
        'CALIBRATION_VARIABLE': 'streamflow',
    }


def test_coupled_chain_marks_upstream_model_no_route(tmp_path):
    """SUMMA (upstream) is told to skip routing; dRoute (objective) routes and reads SUMMA output."""
    cfg = _coupled_config()
    worker = CoupledModelWorker(config=cfg, logger=logging.getLogger('t'))
    # Pin the SUMMA(land) -> DROUTE(objective) chain directly. The constructor derives _models from
    # models with a *registered parameter manager*, and the optional dRoute plugin is absent on a
    # clean env (CI), where DROUTE would be dropped and _models collapse to ['SUMMA']. Forcing the
    # chain keeps this a focused unit test of _run_sequential's skip_routing wiring rather than of
    # dRoute plugin registration (covered separately).
    worker._models = ['SUMMA', 'DROUTE']
    worker.objective_model = 'DROUTE'

    seen = {}

    class _FakeWorker:
        def __init__(self, name):
            self.name = name

        def run_model(self, config, settings_dir, output_dir, **kw):
            seen[self.name] = kw
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return True

    worker._worker = lambda m: _FakeWorker(m)  # type: ignore[assignment]

    ok = worker._run_sequential(cfg, tmp_path / 'settings' / 'COUPLED', tmp_path / 'out')
    assert ok is True

    # Upstream land model must not self-route.
    assert seen['SUMMA'].get('skip_routing') is True
    # Objective routing model must route, and gets this iteration's SUMMA output as its source.
    assert 'skip_routing' not in seen['DROUTE'] or seen['DROUTE'].get('skip_routing') is not True
    assert seen['DROUTE'].get('coupling_source_dir') is not None


def test_summa_worker_skips_mizuroute_when_flagged(tmp_path, monkeypatch):
    """With skip_routing set, SUMMA returns right after SUMMA runs -- mizuRoute is never invoked."""
    calls = {'summa': 0, 'mizu': 0}

    def _fake_summa(*_a, **_k):
        calls['summa'] += 1
        return True

    def _fake_mizu(*_a, **_k):
        calls['mizu'] += 1
        return True

    monkeypatch.setattr('symfluence.optimization.workers.summa._run_summa_worker', _fake_summa)
    monkeypatch.setattr('symfluence.optimization.workers.summa._run_mizuroute_worker', _fake_mizu)

    cfg = _coupled_config()
    worker = SUMMAWorker(config=cfg, logger=logging.getLogger('t'))
    settings = tmp_path / 'SUMMA'
    settings.mkdir(parents=True)
    (settings / 'fileManager.txt').write_text('x')

    ok = worker.run_model(cfg, settings, tmp_path / 'out', sim_dir=tmp_path / 'out', skip_routing=True)
    assert ok is True
    assert calls['summa'] == 1
    assert calls['mizu'] == 0, "mizuRoute must not run in coupled land-only mode"


def test_summa_worker_runs_mizuroute_standalone(tmp_path, monkeypatch):
    """Without the flag (standalone calibration), SUMMA -> mizuRoute coupling is preserved."""
    calls = {'summa': 0, 'mizu': 0}

    def _fake_summa(*_a, **_k):
        calls['summa'] += 1
        return True

    def _fake_mizu(*_a, **_k):
        calls['mizu'] += 1
        return True

    monkeypatch.setattr('symfluence.optimization.workers.summa._run_summa_worker', _fake_summa)
    monkeypatch.setattr('symfluence.optimization.workers.summa._run_mizuroute_worker', _fake_mizu)

    cfg = _coupled_config()  # subset domain + streamflow => needs_routing() is True
    worker = SUMMAWorker(config=cfg, logger=logging.getLogger('t'))
    assert worker.needs_routing(cfg) is True
    settings = tmp_path / 'SUMMA'
    settings.mkdir(parents=True)
    (settings / 'fileManager.txt').write_text('x')

    ok = worker.run_model(cfg, settings, tmp_path / 'out', sim_dir=tmp_path / 'out')
    assert ok is True
    assert calls['summa'] == 1
    assert calls['mizu'] == 1, "standalone SUMMA must still drive mizuRoute"
