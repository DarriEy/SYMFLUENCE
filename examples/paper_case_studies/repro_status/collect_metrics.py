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
    experiment_id,domain,metric,best_score,eval_score,best_iteration,collected_at
Rows are keyed by experiment_id; re-running refreshes values in place.

best_score is the CALIBRATION objective — the endpoint of a stochastic search
(e.g. 1000 DDS iterations). Across architectures (ARM64 vs x86_64) the models
produce floating-point-different objective values, so the optimiser takes a
different accept/reject path and lands on a different-but-comparable optimum;
on models with a rough response surface that makes best_score diverge well
beyond the 0.02 tolerance even when nothing is wrong. eval_score is the
out-of-sample metric at the found optimum, which is what actually answers
"did the two machines reproduce each other" — two different calibration paths
that both generalise to ~0.87 ARE a reproduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# --- geospatial provenance -------------------------------------------------
# The DEM and the GDAL/PROJ/GEOS stack decide delineation geometry, so a change
# in either shifts discretization counts in a way that is otherwise
# indistinguishable from a genuine platform difference — the same ambiguity
# symfluence_version/code_commit were added to resolve for code.
#
# Hashing is done here with hashlib rather than via rasterio/GDAL to keep this
# script dependency-free: every reproduction machine runs it outside any model
# environment. Library versions cannot be read that way, so they are taken from
# the run manifest, which the run itself wrote from inside its environment.

_DEM_CACHE: dict = {}


def dem_sha256(domain: Path):
    """SHA-256 of a domain's source DEM, or None. Cached per domain."""
    key = str(domain)
    if key in _DEM_CACHE:
        return _DEM_CACHE[key]
    digest = None
    dem_dir = domain / "data" / "attributes" / "elevation" / "dem"
    try:
        for dem in sorted(dem_dir.glob("*.tif")):
            h = hashlib.sha256()
            with dem.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
            break
    except OSError:
        digest = None
    _DEM_CACHE[key] = digest
    return digest


_STACK_CACHE: dict = {}


def geospatial_stack(domain: Path):
    """{'gdal','proj','geos'} for a domain, read from its newest run manifest."""
    key = str(domain)
    if key in _STACK_CACHE:
        return _STACK_CACHE[key]
    stack = {}
    try:
        manifests = sorted(domain.glob("**/run_manifest.json"),
                           key=lambda m: m.stat().st_mtime, reverse=True)
        for m in manifests:
            doc = json.loads(m.read_text())
            if doc.get("geospatial"):
                stack = doc["geospatial"]
                break
    except (OSError, json.JSONDecodeError):
        stack = {}
    _STACK_CACHE[key] = stack
    return stack


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


# Where each objective metric lands in the final_evaluation JSON. Four schemas
# are in use: plain names (FUSE/HYPE/SUMMA/...), Eval_ prefixes
# (HBV/TOPMODEL), _Eval suffixes (the PRMS-family optimizers: PRMS/mHM/SWAT/
# CRHM/GSFLOW/MESH/WRF-Hydro/WATFLOOD write KGE_Eval), and the correlation
# objective stored as 'correlation'/'r' (the TWS runs). Look up the
# objective's own family so eval_score is the out-of-sample value of the
# SAME metric that was calibrated.
_EVAL_KEYS = {
    "KGE": ("Eval_KGE", "KGE_Eval", "KGE"),
    "NSE": ("Eval_NSE", "NSE_Eval", "NSE"),
    "RMSE": ("Eval_RMSE", "RMSE_Eval", "RMSE"),
    "CORRELATION": ("Eval_correlation", "correlation", "Eval_r", "r"),
    "R": ("Eval_r", "r", "Eval_correlation", "correlation"),
}


def evaluation_score(best_params_path: Path, metric: str):
    """Out-of-sample metric at the found optimum, or "" if unavailable.

    Read from the sibling <run>_final_evaluation.json's evaluation_metrics.
    Returns "" when the run has no evaluation period (e.g. a config with only
    a calibration window) rather than guessing.
    """
    fe = next(best_params_path.parent.glob("*_final_evaluation.json"), None)
    if fe is None:
        return ""
    try:
        em = (json.loads(fe.read_text()).get("evaluation_metrics") or {})
    except (OSError, ValueError, json.JSONDecodeError):
        return ""
    for key in _EVAL_KEYS.get((metric or "").upper(), (f"Eval_{metric}", metric)):
        val = em.get(key)
        if isinstance(val, (int, float)):
            return f"{float(val):.10g}"
    return ""


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
                "eval_score": evaluation_score(bp, d.get("metric", "")),
                "best_iteration": d.get("best_iteration", ""),
                "run_completed": datetime.fromtimestamp(
                    bp.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symfluence_version": version,
                "code_commit": commit,
                "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }


# Domain-geometry categories counted for experiment 01 (Table 1): GRUs come
# from river_basins, HRUs from catchment, routing segments from river_network.
GEOMETRY_DIRS = ("catchment", "river_basins", "river_network")


def dbf_record_count(shp: Path):
    """Feature count of a shapefile, read from its .dbf header.

    Bytes 4:8 of a DBF header are the little-endian record count. Reading it
    directly keeps the collector dependency-free (no geopandas/fiona), which
    matters because every reproduction machine — including Windows — runs
    this script outside any model environment.
    """
    try:
        with shp.with_suffix(".dbf").open("rb") as f:
            f.seek(4)
            return struct.unpack("<I", f.read(4))[0]
    except (OSError, struct.error):
        return None


def scan_domain_counts(root: Path, version: str = "", commit: str = ""):
    """One row per delineation shapefile: metric n_features, score = count.

    Keyed by domain + shapefile path so every platform counts the same file;
    compare_metrics.py holds n_features rows to exact equality (counts have
    no noise tolerance). `_temp` intermediates are skipped — they are
    delineation scratch files and not present on every machine.
    """
    for domain in sorted(root.glob("domain_*")):
        shp_root = domain / "shapefiles"
        stack = geospatial_stack(domain)
        for cat in GEOMETRY_DIRS:
            for shp in sorted((shp_root / cat).rglob("*.shp")):
                if shp.stem.endswith("_temp"):
                    continue
                n = dbf_record_count(shp)
                if n is None:
                    continue
                rel = shp.relative_to(shp_root).as_posix()
                yield {
                    "experiment_id": f"{domain.name}:{rel}",
                    "model": "geometry",
                    "domain": domain.name,
                    "metric": "n_features",
                    "best_score": str(n),
                    "eval_score": "",
                    "best_iteration": "",
                    "run_completed": datetime.fromtimestamp(
                        shp.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symfluence_version": version,
                    "code_commit": commit,
                    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dem_sha256": (dem_sha256(domain) or "")[:16],
                    "gdal_version": stack.get("gdal", ""),
                    "proj_version": stack.get("proj", ""),
                    "geos_version": stack.get("geos", ""),
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True,
                    help="e.g. macos, wsl, native_windows")
    ap.add_argument("--root", required=True, help="SYMFLUENCE_data directory")
    args = ap.parse_args()

    out = Path(__file__).parent / f"metrics_{args.platform}.csv"
    fields = ["experiment_id", "model", "domain", "metric", "best_score",
              "eval_score", "best_iteration", "run_completed",
              "symfluence_version", "code_commit", "collected_at",
              "dem_sha256", "gdal_version", "proj_version", "geos_version"]
    version, commit = code_provenance()

    existing = {}
    if out.exists():
        with out.open(newline="") as f:
            for row in csv.DictReader(f):
                # tolerate rows written before the provenance columns existed
                existing[row["experiment_id"]] = {k: row.get(k, "") for k in fields}

    root = Path(args.root).expanduser()
    n_new = 0
    seen = set()
    for row in itertools.chain(scan(root, version, commit),
                               scan_domain_counts(root, version, commit)):
        seen.add(row["experiment_id"])
        prev = existing.get(row["experiment_id"])
        if (prev is None or prev["best_score"] != row["best_score"]
                or prev.get("eval_score", "") != row.get("eval_score", "")):
            existing[row["experiment_id"]] = row
            n_new += 1
        else:
            # value unchanged: keep the original collected_at, but backfill
            # provenance for rows written before those columns existed
            for k in ("run_completed", "symfluence_version", "code_commit",
                      "dem_sha256", "gdal_version", "proj_version", "geos_version"):
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
