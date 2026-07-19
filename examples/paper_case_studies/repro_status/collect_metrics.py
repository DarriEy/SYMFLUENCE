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
from datetime import datetime, timezone
from pathlib import Path


def scan(root: Path):
    for domain in sorted(root.glob("domain_*")):
        opt = domain / "optimization"
        if not opt.is_dir():
            continue
        for bp in sorted(opt.rglob("*_best_params.json")):
            try:
                d = json.loads(bp.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            eid = d.get("experiment_id")
            score = d.get("best_score")
            if eid is None or score is None:
                continue
            yield {
                "experiment_id": eid,
                "domain": domain.name,
                "metric": d.get("metric", "?"),
                "best_score": f"{float(score):.10g}",
                "best_iteration": d.get("best_iteration", ""),
                "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True,
                    help="e.g. macos, wsl, native_windows")
    ap.add_argument("--root", required=True, help="SYMFLUENCE_data directory")
    args = ap.parse_args()

    out = Path(__file__).parent / f"metrics_{args.platform}.csv"
    fields = ["experiment_id", "domain", "metric", "best_score",
              "best_iteration", "collected_at"]

    existing = {}
    if out.exists():
        with out.open(newline="") as f:
            for row in csv.DictReader(f):
                existing[row["experiment_id"]] = row

    n_new = 0
    for row in scan(Path(args.root).expanduser()):
        prev = existing.get(row["experiment_id"])
        if prev is None or prev["best_score"] != row["best_score"]:
            existing[row["experiment_id"]] = row
            n_new += 1
        # keep original collected_at when value unchanged

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eid in sorted(existing):
            w.writerow(existing[eid])

    print(f"{out.name}: {len(existing)} runs ({n_new} new/updated)")


if __name__ == "__main__":
    main()
