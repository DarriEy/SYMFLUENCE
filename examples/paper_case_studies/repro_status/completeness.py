#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Reconcile configs *attempted* against cells *present* in the metrics.

`collect_metrics.py` reports what completed. It has no notion of what was
supposed to run, so a config that exits 0 while producing nothing does not
appear as a failure — it does not appear at all.

That is not hypothetical. Seven native HYPE calibrations recorded
``exit_code=0`` while the model never launched: zero ``hyss_*.log`` files, no
``*_best_params.json``, no ``final_evaluation/``. The collector skipped them
for having no results, so they were simply absent from ``comparison.csv``,
and the tally read "207 OK, 1 DIFF" while eight cells were missing. Absence
looked like success.

This closes that gap by joining two sources that already exist:

* the run harness's ``results_*.csv`` — every config attempted, with its exit
  code (the ground truth for "was this supposed to produce a cell?");
* ``metrics_<platform>.csv`` — the cells that actually landed.

Verdicts, per config:

  OK       exit 0 and a metrics row exists — the normal case
  MISSING  exit 0 but no metrics row  <-- the silent failure this exists for
  PENDING  exit 0, no row, but it finished after the last collection — simply
           not harvested yet, NOT a failure
  FAILED   non-zero exit and no row — an honest failure, already visible
  ORPHAN   a metrics row nothing attempted (stale data from an earlier config)

Usage:
    completeness.py --results <dir-with-results_*.csv> \
                    --metrics metrics_native_windows.csv \
                    --configs <paper_case_studies/configs>
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# experiment_id and hydrological_model, read without a YAML dependency: these
# configs are flat enough that two anchored regexes beat adding a parser to a
# reporting script that must run anywhere the campaign runs.
_EID = re.compile(r"^\s*experiment_id:\s*['\"]?([^'\"#\s]+)", re.M)
_MODEL = re.compile(r"^\s*hydrological_model:\s*['\"]?([^'\"#\s]+)", re.M)
_DOMAIN = re.compile(r"^\s*name:\s*['\"]?([^'\"#\s]+)", re.M)
# Configs are self-scoping via workflow_steps. One that stops at
# discretize_domain produces geometry rows (keyed "domain_X:path"), never a
# calibration cell — demanding one would report a failure for work nobody
# asked for.
_CALIBRATES = re.compile(r"^\s*-\s*calibrate_model\s*$", re.M)


def config_expectation(config_path: Path):
    """What this config should produce: ("calibration", key) or ("geometry", domain).

    Mirrors collect_metrics.scan() for the calibration case:
    "<experiment_id>@<model>". experiment_id alone is not unique — all 17
    model-ensemble configs share "run_1".
    """
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    if not _CALIBRATES.search(text):
        domain = _DOMAIN.search(text)
        return ("geometry", domain.group(1)) if domain else (None, None)
    eid = _EID.search(text)
    model = _MODEL.search(text)
    if not eid:
        return None, None
    key = f"{eid.group(1)}@{model.group(1)}" if model else eid.group(1)
    return "calibration", key


def _parse_time(text: str):
    """Parse an ISO timestamp to an aware datetime, or None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def attempted(results_dir: Path):
    """Every config the harness ran: last exit code and finish time."""
    runs: dict = {}
    for csv_path in sorted(results_dir.glob("results_*.csv")):
        try:
            rows = list(csv.DictReader(csv_path.read_text(
                encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            continue
        for row in rows:
            cfg = (row.get("config") or "").strip()
            if not cfg:
                continue
            # Last write wins: a retry supersedes the attempt before it.
            runs[cfg] = ((row.get("exit_code") or "").strip(),
                         _parse_time(row.get("finished_at", "")))
    return runs


def last_collected(metrics_csv: Path):
    """Newest collected_at in the metrics, or None.

    A run that finished after this has simply not been harvested yet.
    Without this check the reconciliation flags every recent success as a
    silent failure, and a gate that cries wolf is a gate people switch off.
    """
    try:
        rows = list(csv.DictReader(metrics_csv.read_text(
            encoding="utf-8", errors="replace").splitlines()))
    except OSError:
        return None
    stamps = [t for t in (_parse_time(r.get("collected_at", "")) for r in rows) if t]
    return max(stamps) if stamps else None


def present(metrics_csv: Path):
    """Cell keys that actually landed in the metrics."""
    try:
        rows = list(csv.DictReader(metrics_csv.read_text(
            encoding="utf-8", errors="replace").splitlines()))
    except OSError:
        return set()
    keys = {(r.get("experiment_id") or "").strip() for r in rows}
    calibration = {k for k in keys if ":" not in k}
    # Geometry rows are keyed "domain_<name>:<relative shapefile path>".
    domains = {k.split(":", 1)[0] for k in keys if ":" in k}
    return calibration, domains


def reconcile(results_dir: Path, metrics_csv: Path, repo_root: Path):
    have, have_domains = present(metrics_csv)
    runs = attempted(results_dir)
    harvested_at = last_collected(metrics_csv)
    findings = []
    expected_keys = set()

    for cfg, (code, finished) in sorted(runs.items()):
        cfg_path = repo_root / cfg
        kind, key = config_expectation(cfg_path)
        if kind is None:
            findings.append(("UNREADABLE", cfg, "", code))
            continue
        if kind == "geometry":
            # Expect a discretisation, not a score.
            landed = f"domain_{key}" in have_domains or key in have_domains
            verdict = "OK" if landed else ("FAILED" if code != "0" else "MISSING")
            findings.append((verdict, cfg, f"domain_{key}:*", code))
            continue
        expected_keys.add(key)
        if key in have:
            findings.append(("OK", cfg, key, code))
        elif code != "0":
            findings.append(("FAILED", cfg, key, code))
        elif harvested_at and finished and finished > harvested_at:
            # Finished after the last harvest: absent because nobody has
            # looked yet, not because it produced nothing.
            findings.append(("PENDING", cfg, key, code))
        else:
            # The case this tool exists for: reported success, produced nothing.
            findings.append(("MISSING", cfg, key, code))

    for key in sorted(have - expected_keys):
        findings.append(("ORPHAN", "", key, ""))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path,
                    help="directory holding the harness's results_*.csv")
    ap.add_argument("--metrics", required=True, type=Path,
                    help="metrics_<platform>.csv to reconcile against")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[3],
                    help="repo root that config paths are relative to")
    ap.add_argument("--quiet-ok", action="store_true",
                    help="only print anomalies")
    args = ap.parse_args()

    findings = reconcile(args.results, args.metrics, args.repo_root)
    counts: dict = {}
    for verdict, cfg, key, code in findings:
        counts[verdict] = counts.get(verdict, 0) + 1
        if args.quiet_ok and verdict in ("OK", "PENDING"):
            continue
        name = Path(cfg).name if cfg else "-"
        detail = f"exit={code}" if code else ""
        print(f"{verdict:<10} {name:<42} {key:<34} {detail}")

    print("\nsummary: " + "  ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))

    missing = counts.get("MISSING", 0)
    if missing:
        print(f"\n{missing} config(s) reported success but produced no metrics "
              f"cell. These are silent failures: they are absent from the "
              f"comparison rather than shown as failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
