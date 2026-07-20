#!/usr/bin/env python
"""Collect per-run best scores from a SYMFLUENCE_data tree into a platform CSV.

Each reproduction machine runs this against its own data root and commits only
its own metrics_<platform>.csv, so concurrent updates from several machines
never conflict.

Usage:
    python collect_metrics.py --platform macos --root /path/to/SYMFLUENCE_data
    python collect_metrics.py --platform wsl --root ~/repos/SYMFLUENCE_data
    python collect_metrics.py --platform native_windows --root C:/Users/me/repos/SYMFLUENCE_data

Output: metrics_<platform>.csv next to this script with columns
    experiment_id,domain,metric,best_score,best_iteration,collected_at
Rows are keyed by experiment_id; re-running refreshes values in place.
"""
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def code_provenance():
    """Version + commit of the SYMFLUENCE checkout doing the collecting.

    Run artifacts record no code version, so a dataset produced before a
    behaviour-changing fix is indistinguishable from a platform difference
    (observed: a Mac dataset predating PR #309 read as a cross-platform
    DIFF). Recording the commit here, plus each run's completion time,
    makes that skew visible in the comparison instead of a mystery.
    """
    here = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(here.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    try:
        from symfluence import __version__ as ver
    except Exception:
        ver = ""
    return ver, commit


def scan(root: Path, version: str = "", commit: str = ""):
    for domain in sorted(root.glob("domain_*")):
        opt = domain / "optimization"
        if not opt.is_dir():
            continue
        for bp in sorted(opt.rglob("*_best_params.json")):
            # best_params.json is rewritten on every improvement DURING
            # calibration; harvesting it mid-run records a partial score
            # (observed: a DDS best at eval 212/1000 flagged as a spurious
            # cross-platform DIFF). The final_evaluation directory is only
            # written when the calibration completes — use it as the gate.
            if not (bp.parent / "final_evaluation").is_dir():
                continue
            try:
                d = json.loads(bp.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            eid = d.get("experiment_id")
            score = d.get("best_score")
            if eid is None or score is None:
                continue
            # Qualify the id with the model. experiment_id is NOT unique:
            # all 17 configs of the model-ensemble experiment share
            # experiment_id "run_1", so keying on it alone collapses 17
            # different models into one row (last scan wins) and compares
            # whichever model happened to be scanned last on each machine.
            # The model is the directory under optimization/.
            try:
                model = bp.relative_to(opt).parts[0]
            except ValueError:
                model = ""
            key = f"{eid}@{model}" if model else eid
            yield {
                "experiment_id": key,
                "model": model,
                "domain": domain.name,
                "metric": d.get("metric", "?"),
                "best_score": f"{float(score):.10g}",
                "best_iteration": d.get("best_iteration", ""),
                "run_completed": datetime.fromtimestamp(
                    bp.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symfluence_version": version,
                "code_commit": commit,
                "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True,
                    help="e.g. macos, wsl, native_windows")
    ap.add_argument("--root", required=True, help="SYMFLUENCE_data directory")
    args = ap.parse_args()

    out = Path(__file__).parent / f"metrics_{args.platform}.csv"
    fields = ["experiment_id", "model", "domain", "metric", "best_score",
              "best_iteration", "run_completed", "symfluence_version",
              "code_commit", "collected_at"]
    version, commit = code_provenance()

    existing = {}
    if out.exists():
        with out.open(newline="") as f:
            for row in csv.DictReader(f):
                # tolerate rows written before the provenance columns existed
                existing[row["experiment_id"]] = {k: row.get(k, "") for k in fields}

    n_new = 0
    seen = set()
    for row in scan(Path(args.root).expanduser(), version, commit):
        seen.add(row["experiment_id"])
        prev = existing.get(row["experiment_id"])
        if prev is None or prev["best_score"] != row["best_score"]:
            existing[row["experiment_id"]] = row
            n_new += 1
        else:
            # value unchanged: keep the original collected_at, but backfill
            # provenance for rows written before those columns existed
            for k in ("run_completed", "symfluence_version", "code_commit"):
                if not prev.get(k) and row.get(k):
                    prev[k] = row[k]

    # Drop rows the scan no longer accepts. A row harvested before the
    # completeness gate existed (or from a domain since wiped and re-run)
    # otherwise persists forever and is compared as if current — exactly how
    # a mid-run snapshot survived into the shared table and read as a DIFF.
    stale = [eid for eid in existing if eid not in seen]
    for eid in stale:
        del existing[eid]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eid in sorted(existing):
            w.writerow(existing[eid])

    print(f"{out.name}: {len(existing)} runs "
          f"({n_new} new/updated, {len(stale)} stale dropped)")


if __name__ == "__main__":
    main()
