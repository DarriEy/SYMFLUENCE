# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""A config that reports success but produces nothing must be visible.

`collect_metrics.py` reports what completed and has no notion of what was
supposed to run. Seven native HYPE calibrations recorded ``exit_code=0`` while
the model never launched, wrote no ``*_best_params.json``, and were therefore
skipped by the collector — so they were *absent* from the comparison rather
than shown as failures, and the tally read "207 OK, 1 DIFF" while eight cells
were missing. Absence looked like success.

These tests pin the reconciliation, including the two ways it could become
useless: crying wolf over runs that simply have not been harvested yet, and
demanding calibration cells from configs that only produce a discretisation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MOD_PATH = (Path(__file__).resolve().parents[3] / "examples" / "paper_case_studies"
             / "repro_status" / "completeness.py")


@pytest.fixture(scope="module")
def mod():
    if not _MOD_PATH.exists():
        pytest.skip(f"not found: {_MOD_PATH}")
    spec = importlib.util.spec_from_file_location("repro_completeness", _MOD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["repro_completeness"] = module
    spec.loader.exec_module(module)
    return module


def _config(path: Path, *, eid: str, model: str, calibrates: bool, domain="D"):
    steps = ["setup_project", "define_domain", "discretize_domain"]
    if calibrates:
        steps.append("calibrate_model")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "system:\n  workflow_steps:\n"
        + "".join(f"  - {s}\n" for s in steps)
        + f"domain:\n  name: {domain}\n"
        f"experiment:\n  experiment_id: {eid}\n"
        f"model:\n  hydrological_model: {model}\n",
        encoding="utf-8")
    return path


def _results(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["config,exit_code,seconds,finished_at"]
    lines += [f"{c},{code},10,{when}" for c, code, when in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics(path: Path, keys, collected="2026-07-21T23:00:00Z"):
    lines = ["experiment_id,model,domain,metric,best_score,collected_at"]
    lines += [f"{k},M,dom,KGE,0.5,{collected}" for k in keys]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_success_without_a_metrics_cell_is_reported_missing(mod, tmp_path):
    """The regression: exit 0, nothing produced, previously invisible."""
    _config(tmp_path / "cfg" / "a.yaml", eid="exp_a", model="HYPE", calibrates=True)
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/a.yaml", "0", "2026-07-21T20:00:00Z")])
    _metrics(tmp_path / "metrics.csv", [])  # collector saw nothing

    findings = mod.reconcile(tmp_path / "res", tmp_path / "metrics.csv", tmp_path)
    assert ("MISSING", "cfg/a.yaml", "exp_a@HYPE", "0") in findings


def test_a_run_finished_after_the_last_harvest_is_pending_not_missing(mod, tmp_path):
    """A gate that cries wolf is a gate people switch off."""
    _config(tmp_path / "cfg" / "a.yaml", eid="exp_a", model="HYPE", calibrates=True)
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/a.yaml", "0", "2026-07-22T09:00:00Z")])  # after collection
    _metrics(tmp_path / "metrics.csv", ["other@M"], collected="2026-07-21T23:00:00Z")

    verdicts = {v for v, _c, _k, _x in
                mod.reconcile(tmp_path / "res", tmp_path / "metrics.csv", tmp_path)}
    assert "PENDING" in verdicts and "MISSING" not in verdicts


def test_geometry_only_configs_are_not_expected_to_score(mod, tmp_path):
    """Configs stopping at discretize_domain produce shapes, not scores."""
    _config(tmp_path / "cfg" / "geo.yaml", eid="exp_g", model="SUMMA",
            calibrates=False, domain="Bow_lumped")
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/geo.yaml", "0", "2026-07-21T20:00:00Z")])
    path = tmp_path / "metrics.csv"
    path.write_text(
        "experiment_id,model,domain,metric,best_score,collected_at\n"
        "domain_Bow_lumped:catchment/x.shp,,dom,n_features,1,2026-07-21T23:00:00Z\n",
        encoding="utf-8")

    verdicts = {v for v, _c, _k, _x in mod.reconcile(tmp_path / "res", path, tmp_path)}
    assert verdicts == {"OK"}, "a discretisation config must not be demanded a score"


def test_honest_failures_are_distinguished_from_silent_ones(mod, tmp_path):
    """A non-zero exit was already visible; don't relabel it."""
    _config(tmp_path / "cfg" / "a.yaml", eid="exp_a", model="HYPE", calibrates=True)
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/a.yaml", "1", "2026-07-21T20:00:00Z")])
    _metrics(tmp_path / "metrics.csv", [])

    verdicts = {v for v, _c, _k, _x in
                mod.reconcile(tmp_path / "res", tmp_path / "metrics.csv", tmp_path)}
    assert verdicts == {"FAILED"}


def test_a_retry_supersedes_the_attempt_before_it(mod, tmp_path):
    """results_*.csv appends; the last row is the outcome that counts."""
    _config(tmp_path / "cfg" / "a.yaml", eid="exp_a", model="HYPE", calibrates=True)
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/a.yaml", "1", "2026-07-21T19:00:00Z"),
              ("cfg/a.yaml", "0", "2026-07-21T20:00:00Z")])
    _metrics(tmp_path / "metrics.csv", ["exp_a@HYPE"])

    verdicts = {v for v, _c, _k, _x in
                mod.reconcile(tmp_path / "res", tmp_path / "metrics.csv", tmp_path)}
    assert verdicts == {"OK"}


def test_a_cell_nobody_attempted_is_flagged_orphan(mod, tmp_path):
    """Stale rows from a renamed or removed config should not pass silently."""
    _config(tmp_path / "cfg" / "a.yaml", eid="exp_a", model="HYPE", calibrates=True)
    _results(tmp_path / "res" / "results_01.csv",
             [("cfg/a.yaml", "0", "2026-07-21T20:00:00Z")])
    _metrics(tmp_path / "metrics.csv", ["exp_a@HYPE", "ghost@OLD"])

    findings = mod.reconcile(tmp_path / "res", tmp_path / "metrics.csv", tmp_path)
    assert ("ORPHAN", "", "ghost@OLD", "") in findings
