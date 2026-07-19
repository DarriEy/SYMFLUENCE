#!/usr/bin/env python
"""Compare metrics_<platform>.csv files pairwise and against paper references.

Reads every metrics_*.csv beside this script, writes comparison.csv, and
prints one line per NEW verdict (state kept in .compare_state.json, which is
git-ignored — each machine tracks its own reporting state):

  CMP <experiment_id> <platA>=<v> <platB>=<v> delta=<d> [OK|DIFF]
  REF <platform>:<rule> value=<v> expected=<lo>..<hi> n=<count> [OK|DIFF]

Tolerance: 0.02 KGE (04_calibration_ensemble/README.md cross-platform bound).
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / ".compare_state.json"
OUT = HERE / "comparison.csv"
TOL = 0.02
TOP_TIER = ("de", "cmaes", "dds", "adam")

# Metrics bounded near [0, 1], where 0.02 is a meaningful absolute difference.
# Anything else (RMSE, MAE, ...) is a magnitude in the units of the variable —
# for the forcing ensemble that is streamflow RMSE in the hundreds, so an
# absolute 0.02 would flag every pair of runs no matter how well they agree.
BOUNDED_METRICS = {"KGE", "KGEP", "KGENP", "NSE", "LOGNSE", "R2", "VE"}
REL_TOL = 0.02


def tolerance_for(metric: str, va: float, vb: float) -> float:
    """Absolute tolerance for bounded scores, relative for magnitude metrics."""
    if (metric or "").upper() in BOUNDED_METRICS:
        return TOL
    scale = max(abs(va), abs(vb))
    return REL_TOL * scale if scale else TOL


def load_explained():
    """experiment_id -> note for divergences with a completed diagnosis."""
    f = HERE / "explained_divergences.csv"
    if not f.exists():
        return {}
    with f.open(newline="") as fh:
        return {r["experiment_id"]: r["note"] for r in csv.DictReader(fh)}


def load_platforms():
    platforms = {}
    for f in sorted(HERE.glob("metrics_*.csv")):
        name = f.stem.replace("metrics_", "")
        rows = {}
        with f.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows[row["experiment_id"]] = (row.get("metric", "?"),
                                                  float(row["best_score"]))
                except (KeyError, ValueError):
                    continue
        platforms[name] = rows
    return platforms


def ref_checks(scores):
    checks = []
    tt = [v for e, (_, v) in scores.items()
          if e.startswith("cal_ensemble_") and e.rsplit("_", 1)[-1] in TOP_TIER]
    if len(tt) >= 8:
        checks.append(("exp04-top-tier-mean-calib-KGE",
                       sum(tt) / len(tt), 0.867 - TOL, 0.872 + TOL, len(tt)))
    if "cal_ensemble_sacsma_bayesian_opt" in scores:
        # The README documents this run as an expected optimizer failure
        # (KGE ~ -0.03). The exact failure value is noise; the check is that
        # it stays far below the ~0.87 success tier, not a tight window.
        checks.append(("exp04-sacsma-bayesopt-fails",
                       scores["cal_ensemble_sacsma_bayesian_opt"][1],
                       -10.0, 0.3, 1))
    return checks


def main():
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}

    platforms = load_platforms()
    explained = load_explained()
    rows, verdicts = [], {}

    for pa, pb in itertools.combinations(sorted(platforms), 2):
        a, b = platforms[pa], platforms[pb]
        for eid in sorted(set(a) & set(b)):
            metric = a[eid][0]
            va, vb = a[eid][1], b[eid][1]
            delta = abs(va - vb)
            status = "OK" if delta <= tolerance_for(metric, va, vb) else "DIFF"
            if status == "DIFF" and eid in explained:
                # diagnosed divergence (see explained_divergences.csv) —
                # keep it visible but do not re-alert
                status = "EXPLAINED"
            key = f"CMP {pa}|{pb} {eid}"
            verdicts[key] = (status,
                             f"CMP {eid} metric={metric} {pa}={va:.4f} "
                             f"{pb}={vb:.4f} delta={delta:.4f} {status}")
            rows.append({"experiment_id": eid, "metric": metric,
                         "platform_a": pa, "value_a": f"{va:.6f}",
                         "platform_b": pb, "value_b": f"{vb:.6f}",
                         "delta": f"{delta:.6f}", "status": status})

    for pname, scores in sorted(platforms.items()):
        for name, val, lo, hi, count in ref_checks(scores):
            status = "OK" if lo <= val <= hi else "DIFF"
            key = f"REF {pname} {name}"
            verdicts[key] = (status,
                             f"REF {pname}:{name} value={val:.4f} "
                             f"expected={lo:.3f}..{hi:.3f} n={count} {status}")
            rows.append({"experiment_id": name, "metric": "reference",
                         "platform_a": pname, "value_a": f"{val:.6f}",
                         "platform_b": "paper", "value_b": f"{lo:.3f}..{hi:.3f}",
                         "delta": "", "status": status})

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment_id", "metric",
                                          "platform_a", "value_a",
                                          "platform_b", "value_b",
                                          "delta", "status"])
        w.writeheader()
        w.writerows(rows)

    for key, (status, line) in sorted(verdicts.items()):
        if prev.get(key) != status:
            print(line, flush=True)
    STATE.write_text(json.dumps({k: s for k, (s, _) in verdicts.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
