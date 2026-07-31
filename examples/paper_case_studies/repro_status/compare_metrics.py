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
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / ".compare_state.json"
OUT = HERE / "comparison.csv"
TOL = 0.02
TOP_TIER = ("de", "cmaes", "dds", "adam")
# The eight models of experiment 04 (04_calibration_ensemble/README.md)
EXP04_MODELS = frozenset({"hbv", "hechms", "sacsma", "topmodel", "xinanjiang",
                          "fuse", "summa", "hype"})

# Metrics bounded near [0, 1], where 0.02 is a meaningful absolute difference.
# Anything else (RMSE, MAE, ...) is a magnitude in the units of the variable —
# for the forcing ensemble that is streamflow RMSE in the hundreds, so an
# absolute 0.02 would flag every pair of runs no matter how well they agree.
BOUNDED_METRICS = {"KGE", "KGEP", "KGENP", "NSE", "LOGNSE", "R2", "VE"}
REL_TOL = 0.02


# Grid-derived discretizations clip a regular grid to the delineated basin and
# then drop cells below 10% of a full cell (grid_delineator.py). That threshold
# quantizes a continuous quantity — the basin boundary — into an integer count,
# so a sub-percent boundary difference flips whole cells. Measured on the Bow
# 1 km domain: the smallest kept cell sat at 0.10098 of a cell and 6 cells lay
# within 1% of the cutoff, turning a boundary difference far below any
# hydrological significance into 2329 vs 2332.
GRID_COUNT_REL_TOL = 0.005


def is_grid_derived(experiment_id: str) -> bool:
    """True for counts produced by clipping a regular grid to the basin.

    Keyed on the shapefile path (``<domain>:<relative path>``), which carries
    the discretization: grid domains write under ``catchment/distributed/`` and
    name their outputs ``*_distributed_<cellsize>``.

    Sub-basin discretizations are excluded explicitly. They are NOT grids, they
    reproduce exactly today, and they must keep being held exact — but they are
    spelled both ``semidistributed`` and ``semi_distributed`` in the tree, and
    the latter contains ``_distributed_``. Checked against every n_features id
    present in the committed platform CSVs.
    """
    eid = (experiment_id or "").lower()
    if re.search(r"semi[_-]?distributed", eid):
        return False
    # A grid is identified by its cell-size suffix (``_distributed_1000m``) or
    # by living under the ``distributed`` discretization directory.
    return bool(re.search(r"_distributed_\d+\s*m", eid)) or "/distributed/" in eid


def tolerance_for(metric: str, va: float, vb: float, experiment_id: str = "") -> float:
    """Absolute tolerance for bounded scores, relative for magnitude metrics."""
    if (metric or "").lower() == "n_features":
        # Feature counts (experiment 01, Table 1) are integers with no numerical
        # noise, so any mismatch is a real divergence — EXCEPT for grid-derived
        # counts, where a hard sliver threshold sits on top of a continuous
        # boundary (see GRID_COUNT_REL_TOL). Everything else stays exact: the
        # sub-basin, elevation-band, land-class and elev x aspect counts all
        # reproduce bit-identically across platforms and should keep doing so.
        if is_grid_derived(experiment_id):
            return GRID_COUNT_REL_TOL * max(abs(va), abs(vb))
        return 0.0
    if (metric or "").upper() in BOUNDED_METRICS:
        return TOL
    scale = max(abs(va), abs(vb))
    return REL_TOL * scale if scale else TOL


def load_explained():
    """experiment_id -> note for divergences with a completed diagnosis."""
    f = HERE / "explained_divergences.csv"
    if not f.exists():
        return {}
    # Notes are appended from several machines; the Windows box writes
    # cp1252 (observed: 0x97 em-dash), which is invalid UTF-8.
    for enc in ("utf-8", "cp1252"):
        try:
            with f.open(newline="", encoding=enc) as fh:
                return {r["experiment_id"]: r["note"] for r in csv.DictReader(fh)}
        except UnicodeDecodeError:
            continue
    return {}


def load_platforms():
    """platform -> {experiment_id: (metric, score, code_commit, run_completed)}.

    The commit and completion date travel with every row so a difference can be
    attributed before it is blamed on the platform: see ``staleness_note``.
    """
    platforms = {}
    for f in sorted(HERE.glob("metrics_*.csv")):
        name = f.stem.replace("metrics_", "")
        rows = {}
        with f.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows[row["experiment_id"]] = (row.get("metric", "?"),
                                                  float(row["best_score"]),
                                                  (row.get("code_commit") or "").strip(),
                                                  (row.get("run_completed") or "").strip())
                except (KeyError, ValueError):
                    continue
        platforms[name] = rows
    return platforms


def load_platform_status():
    """platform -> (state, note) from platform_status.csv; {} if absent.

    ``frozen`` marks a baseline whose machine no longer exists, so it can never
    be re-collected. Those rows stay useful as history but must not be read as
    live platform comparisons: any behaviour change merged after their commit
    shows up as a difference forever.
    """
    f = HERE / "platform_status.csv"
    if not f.exists():
        return {}
    out = {}
    for enc in ("utf-8", "cp1252"):
        try:
            with f.open(newline="", encoding=enc) as fh:
                for r in csv.DictReader(fh):
                    out[r["platform"]] = ((r.get("state") or "live").strip(),
                                          (r.get("note") or "").strip())
            return out
        except UnicodeDecodeError:
            continue
    return out


def staleness_note(ra, rb, pa: str, pb: str, status=None) -> str:
    """Explain a difference by baseline age, or '' if the rows are comparable.

    A row produced before a behaviour-changing fix is not a platform
    divergence, but it prints identically to one. This is not hypothetical: the
    Iceland river-network counts read 1894 vs 1895 across every platform until
    the commits were checked, at which point all four reference rows turned out
    to predate `fix(geofabric): drop TauDEM zero-length connector reaches`
    (#388, 2026-07-27) -- the newer number was simply correct. The repro README
    added code_commit/run_completed for exactly this reason; nothing was
    reading them.
    """
    status = status if status is not None else {}
    ca, cb = (ra[2] or ""), (rb[2] or "")
    da, db = (ra[3] or "")[:10], (rb[3] or "")[:10]
    if not ca or not cb or ca == cb:
        return ""
    older, newer = (pa, pb) if da <= db else (pb, pa)
    older_commit = ca[:8] if older == pa else cb[:8]
    older_date = da if older == pa else db
    detail = (f"different code_commit ({pa}={ca[:8]}@{da or '?'} "
              f"{pb}={cb[:8]}@{db or '?'})")
    state, note = status.get(older, ("live", ""))
    if state == "frozen":
        # Telling someone to re-collect a decommissioned machine is advice they
        # cannot act on; say what is actually true instead.
        return (f"{detail}; {older} is a FROZEN baseline at {older_commit}"
                f"@{older_date or '?'} and cannot be re-collected"
                f"{' -- ' + note.rstrip('.') if note else ''}. Differences against newer "
                f"code are expected; compare {newer} against a live platform "
                f"instead of treating this as a platform divergence")
    return (f"{detail}; {older} may predate a behaviour change present in "
            f"{newer} -- re-collect {older} before treating this as a platform "
            f"divergence")


def ref_checks(scores):
    checks = []
    # The paper's top-tier mean is an ensemble-wide statistic over all eight
    # models. Evaluating it on a partial set compares different populations:
    # a run order that finishes the high-scoring models first (HBV ~0.94)
    # yields a mean far above the full-ensemble value regardless of
    # correctness. Only evaluate once every model is represented.
    tt, models = [], set()
    for e, row in scores.items():
        v = row[1]
        if not e.startswith("cal_ensemble_"):
            continue
        rest = e[len("cal_ensemble_"):]
        model = rest.split("_", 1)[0]
        if rest.rsplit("_", 1)[-1] in TOP_TIER:
            tt.append(v)
            models.add(model)
    if EXP04_MODELS <= models:
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
    pstatus = load_platform_status()
    rows, verdicts = [], {}

    for pa, pb in itertools.combinations(sorted(platforms), 2):
        a, b = platforms[pa], platforms[pb]
        for eid in sorted(set(a) & set(b)):
            metric = a[eid][0]
            va, vb = a[eid][1], b[eid][1]
            delta = abs(va - vb)
            status = "OK" if delta <= tolerance_for(metric, va, vb, eid) else "DIFF"
            note = ""
            if status == "DIFF" and eid in explained:
                # diagnosed divergence (see explained_divergences.csv) —
                # keep it visible but do not re-alert
                status = "EXPLAINED"
            elif status == "DIFF":
                # Attribute before blaming the platform: rows built from
                # different code are not evidence of a platform difference.
                note = staleness_note(a[eid], b[eid], pa, pb, pstatus)
                if note:
                    status = "STALE?"
            key = f"CMP {pa}|{pb} {eid}"
            verdicts[key] = (status,
                             f"CMP {eid} metric={metric} {pa}={va:.4f} "
                             f"{pb}={vb:.4f} delta={delta:.4f} {status}"
                             + (f" | {note}" if note else ""))
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
