# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backfill / repair ``<run>_<algo>_final_evaluation.json`` for the Paper-3
reproduction campaign.

WHY THIS TOOL EXISTS
--------------------
The reproduction protocol compares the *held-out* ``eval_score`` (out-of-sample
KGE), not the noisy calibration ``best_score``. The collector
(``collect_metrics.py``) reads, per run, the top-of-run-dir file
``<run>_<algo>_final_evaluation.json`` and pulls ``evaluation_metrics``. Four
models produce a wrong or blank ``eval_score``:

  1. FUSE / HECHMS (generic-optimizer path): the top-level
     ``*_final_evaluation.json`` is missing entirely, or exists with EMPTY
     ``calibration_metrics``/``evaluation_metrics`` -- so ``eval_score`` is blank.
  2. CRHM / SWAT (PRMS-family per-model optimizer): the JSON exists but
     ``evaluation_metrics.KGE_Eval == calibration_metrics.KGE_Calib`` exactly,
     because the eval-period slice was never applied (both windows were scored
     over the same dates).

Every run's model output already spans a period that CONTAINS the held-out
window, so this is a RECOMPUTE -- no hydrological model is re-run. For each run
we read the simulated daily streamflow (m3/s) from the EXISTING
``final_evaluation/`` output, load the observed daily series, and slice it twice
with :func:`align_and_filter` -- once to the calibration window, once to the
evaluation window -- then compute KGE for each with the framework metric.

DEPENDENCIES
------------
Unlike ``collect_metrics.py`` (pure-stdlib), this tool imports the SYMFLUENCE
framework plus xarray / pandas / geopandas, so it MUST run inside the model
env (the ``symfluence`` conda env). Example:

    conda run -n symfluence python backfill_final_eval.py \
        --root /path/to/SYMFLUENCE_data --models fuse --dry-run --verbose

SAFETY
------
* Default is ``--dry-run``: nothing is written; each run's computed calib vs
  eval KGE is printed along with whether it WOULD write.
* Only ``<run>_<algo>_final_evaluation.json`` files are ever written, and only
  under ``--write``. No model output is touched.
* A run is written only when BOTH calib and eval KGE compute to finite values.
  Runs whose sim/obs cannot be read are SKIPPED and reported -- a penalty/echo
  value is never written.
* Idempotent: re-running reproduces the same numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Framework imports -- these require the model env.
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.models.fuse.calibration.metrics_calculation import (
    align_and_filter,
    read_fuse_streamflow,
)

logger = logging.getLogger("backfill_final_eval")

# ---------------------------------------------------------------------------
# Defaults (constant across the Paper-3 campaign; CLI-overridable).
# ---------------------------------------------------------------------------
DEFAULT_CALIBRATION = "2004-01-01, 2007-12-31"
DEFAULT_EVALUATION = "2008-01-01, 2009-12-31"
DEFAULT_SPINUP = "2002-01-01, 2003-12-31"

# Minimum number of overlapping obs/sim points required in a window to trust KGE.
MIN_POINTS = 30

# On-disk optimization sub-directory name per model key.
MODEL_DIRS = {
    "fuse": "FUSE",
    "hechms": "HECHMS",
    "crhm": "CRHM",
    "swat": "SWAT",
}


# ---------------------------------------------------------------------------
# Per-model simulated-streamflow readers.
# Each returns a daily pd.Series in m3/s (datetime index), or None if unreadable.
# The series should span at least the calibration+evaluation window; the caller
# slices to each period. We deliberately do NOT skip a warmup here -- the date
# windows already exclude the spinup years.
# ---------------------------------------------------------------------------
def read_fuse_sim(
    final_eval_dir: Path,
    config: Dict[str, Any],
    project_dir: Path,
    sm: StreamflowMetrics,
) -> Optional[pd.Series]:
    """Read FUSE simulated daily streamflow (m3/s) from the run's output NetCDF.

    Mirrors the FUSE calibration path: locate the ``*_runs_*.nc`` file that the
    generic optimizer left in ``final_evaluation/`` and hand it to the framework
    reader, which handles lumped/distributed mode and the mm/day -> m3/s
    conversion (using the catchment area from the domain shapefile).
    """
    candidates: List[Path] = []
    for pattern in ("*_runs_def.nc", "*_runs_best.nc", "*_runs_pre.nc", "*runs*.nc"):
        candidates.extend(sorted(final_eval_dir.glob(pattern)))
    # De-dupe while preserving priority order.
    seen: set = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]
    if not ordered:
        logger.debug("FUSE: no *_runs_*.nc under %s", final_eval_dir)
        return None

    sim = read_fuse_streamflow(ordered[0], config, project_dir, sm, log=logger)
    if sim is None or sim.empty:
        return None
    # Ensure a clean daily series.
    if not isinstance(sim.index, pd.DatetimeIndex):
        sim.index = pd.to_datetime(sim.index)
    return sim.resample("D").mean()


def read_hechms_sim(
    final_eval_dir: Path,
    config: Dict[str, Any],
    project_dir: Path,
    sm: StreamflowMetrics,
) -> Optional[pd.Series]:
    """Read HEC-HMS simulated daily streamflow (m3/s).

    HEC-HMS in this campaign is the JAX/BMI re-implementation (``jhechms``); it
    has no classic calibration ``worker.calculate_metrics`` to mirror. Instead
    it writes a self-describing ``*hechms_output.csv`` in ``final_evaluation/``
    with columns ``datetime, streamflow_mm_day, streamflow_cms``. We read the
    ``streamflow_cms`` column directly (already in m3/s), which is the
    unambiguous, correct source. A NetCDF twin (``*hechms_output.nc``) is used
    as a fallback.
    """
    csvs = sorted(final_eval_dir.glob("*hechms_output.csv")) or sorted(
        final_eval_dir.glob("*hechms*.csv")
    )
    if csvs:
        df = pd.read_csv(csvs[0], parse_dates=["datetime"], index_col="datetime")
        col = None
        for cand in ("streamflow_cms", "streamflow_m3s", "discharge_cms"):
            if cand in df.columns:
                col = cand
                break
        if col is None:
            # mm/day fallback needs an area conversion.
            if "streamflow_mm_day" in df.columns:
                area_km2 = sm.get_catchment_area(
                    config, project_dir, config.get("DOMAIN_NAME")
                )
                series = sm.convert_runoff_to_discharge(
                    df["streamflow_mm_day"], area_km2, input_units="mm/day"
                )
                return series.resample("D").mean()
            logger.debug("HECHMS: no recognised flow column in %s", csvs[0].name)
            return None
        return df[col].astype(float).resample("D").mean()

    ncs = sorted(final_eval_dir.glob("*hechms_output.nc")) or sorted(
        final_eval_dir.glob("*hechms*.nc")
    )
    if ncs:
        import xarray as xr

        with xr.open_dataset(ncs[0]) as ds:
            for var in ("streamflow_cms", "streamflow_m3s", "discharge_cms"):
                if var in ds.variables:
                    s = ds[var].to_series()
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index)
                    return s.astype(float).resample("D").mean()
        logger.debug("HECHMS: no cms variable in %s", ncs[0].name)
    return None


def read_crhm_sim(
    final_eval_dir: Path,
    config: Dict[str, Any],
    project_dir: Path,
    sm: StreamflowMetrics,
) -> Optional[pd.Series]:
    """Read CRHM simulated daily streamflow (m3/s).

    Mirrors ``CRHMWorker.calculate_metrics``: read ``CRHM_output_*.txt``
    (tab-separated, a units row at line 2), find the ``basinflow(1)`` column,
    convert m3/interval -> m3/s by dividing by the interval seconds (inferred
    from the timestamp spacing, ~3600 for hourly), then resample to daily mean.
    Unlike the worker we do NOT drop a warmup slice -- the date windows handle
    that.
    """
    txts = sorted(final_eval_dir.glob("CRHM_output*.txt"))
    if not txts:
        txts = sorted(final_eval_dir.glob("*output*.csv")) or sorted(
            final_eval_dir.glob("*.csv")
        )
    if not txts:
        logger.debug("CRHM: no CRHM_output_*.txt under %s", final_eval_dir)
        return None

    sim_file = txts[0]
    if sim_file.suffix == ".txt":
        df = pd.read_csv(
            sim_file, sep="\t", parse_dates=True, index_col=0,
            skiprows=[1], encoding="latin-1",
        )
    else:
        df = pd.read_csv(sim_file, parse_dates=True, index_col=0, encoding="latin-1")

    flow_col = None
    for col in df.columns:
        cl = col.lower()
        if "basinflow" in cl or "basin_flow" in cl:
            flow_col = col
            break
    if flow_col is None:
        for col in ("flow", "Flow", "discharge", "Discharge", "Q", "flow_cms"):
            if col in df.columns:
                flow_col = col
                break
    if flow_col is None:
        flow_cols = [c for c in df.columns if "flow" in c.lower()]
        flow_col = flow_cols[0] if flow_cols else None
    if flow_col is None:
        logger.debug("CRHM: no flow column in %s", sim_file.name)
        return None

    s = df[flow_col].astype(float)
    s.index = pd.to_datetime(s.index)

    interval_seconds = 3600.0
    if len(s) > 1:
        dt = (s.index[1] - s.index[0]).total_seconds()
        if dt > 0:
            interval_seconds = dt
    s = s / interval_seconds  # m3/interval -> m3/s
    return s.resample("D").mean()


def read_swat_sim(
    final_eval_dir: Path,
    config: Dict[str, Any],
    project_dir: Path,
    sm: StreamflowMetrics,
) -> Optional[pd.Series]:
    """Read SWAT simulated daily streamflow (m3/s) from ``output.rch``.

    Mirrors ``SWATWorker.calculate_metrics`` / ``_parse_output_rch``: parse the
    outlet reach (reach 1) ``FLOW_OUTcms`` column, then build a daily index.

    TODO / CAVEAT: SWAT's ``output.rch`` carries no absolute dates. The worker
    reconstructs them as ``EXPERIMENT_TIME_START + SWAT_WARMUP_YEARS`` (NYSKIP
    already skips the warmup in the file). This backfill preserves that logic,
    reading ``EXPERIMENT_TIME_START`` / ``TIME_START`` and ``SWAT_WARMUP_YEARS``
    from ``config`` when present. Because no SWAT runs exist in the current data
    root, this path is UNVERIFIED end-to-end; if a future SWAT run's period or
    warmup differs from the config used here, the date reconstruction (and thus
    the calib/eval slice) would be wrong. When the start date cannot be
    determined confidently the run is skipped rather than guessed.
    """
    output_rch = final_eval_dir / "output.rch"
    if not output_rch.exists():
        cand = sorted(final_eval_dir.glob("output.rch")) or sorted(
            final_eval_dir.glob("*.rch")
        )
        if not cand:
            logger.debug("SWAT: no output.rch under %s", final_eval_dir)
            return None
        output_rch = cand[0]

    flow = _parse_swat_output_rch(output_rch, reach_id=1)
    if flow is None or len(flow) == 0:
        return None

    start_str = config.get("EXPERIMENT_TIME_START") or config.get("TIME_START")
    if not start_str:
        logger.warning(
            "SWAT: no EXPERIMENT_TIME_START/TIME_START in config for %s; "
            "cannot date output.rch -- SKIPPING (see TODO in read_swat_sim).",
            final_eval_dir,
        )
        return None
    try:
        start_date = pd.to_datetime(start_str).normalize()
    except (ValueError, TypeError):
        logger.warning("SWAT: unparseable start '%s' -- SKIPPING", start_str)
        return None

    warmup_years = int(config.get("SWAT_WARMUP_YEARS", 2))
    output_start = start_date + pd.DateOffset(years=warmup_years)
    dates = pd.date_range(start=output_start, periods=len(flow), freq="D")
    return pd.Series(flow, index=dates)


def _parse_swat_output_rch(output_rch: Path, reach_id: int = 1) -> Optional[np.ndarray]:
    """Extract FLOW_OUTcms for ``reach_id`` from a SWAT output.rch file."""
    try:
        with open(output_rch, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        header_end = 0
        for i, line in enumerate(lines):
            up = line.upper()
            if "RCH" in up and ("FLOW" in up or "MON" in up):
                header_end = i + 1
                break
        if header_end == 0:
            header_end = 9
        flow_values: List[float] = []
        for line in lines[header_end:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                if parts[0].upper() == "REACH":
                    rch = int(parts[1])
                    flow_col = 6
                else:
                    rch = int(parts[0])
                    flow_col = 5
                if rch == reach_id and len(parts) > flow_col:
                    flow_values.append(max(0.0, float(parts[flow_col])))
            except (ValueError, IndexError):
                continue
        return np.array(flow_values) if flow_values else None
    except Exception as exc:  # noqa: BLE001 -- resilience
        logger.error("SWAT: error parsing %s: %s", output_rch, exc)
        return None


READERS = {
    "fuse": read_fuse_sim,
    "hechms": read_hechms_sim,
    "crhm": read_crhm_sim,
    "swat": read_swat_sim,
}


# ---------------------------------------------------------------------------
# Core per-run logic.
# ---------------------------------------------------------------------------
def _kge_for_window(
    obs: pd.Series,
    sim: pd.Series,
    window: str,
    base_config: Dict[str, Any],
    sm: StreamflowMetrics,
) -> Optional[float]:
    """Slice obs/sim to ``window`` via align_and_filter and return KGE, or None."""
    cfg = dict(base_config)
    cfg["CALIBRATION_PERIOD"] = window  # align_and_filter slices on this key.
    obs_vals, sim_vals = align_and_filter(obs, sim, cfg, log=logger)
    if len(obs_vals) < MIN_POINTS:
        logger.debug("window %s: only %d points (<%d)", window, len(obs_vals), MIN_POINTS)
        return None
    metrics = sm.calculate_metrics(obs_vals, sim_vals, metrics=["kge"])
    kge = metrics.get("kge")
    if kge is None or not np.isfinite(kge):
        return None
    return float(kge)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _best_score(run_dir: Path) -> Optional[float]:
    """Recorded optimisation objective (``best_score``) from ``*_best_params.json``.

    This is the value the optimiser actually maximised. The guard in
    :func:`process_run` compares it against the backfill's recomputed
    calibration-window KGE: they must agree for the backfill to be trustworthy.
    """
    for bp in sorted(run_dir.glob("*_best_params.json")):
        v = _load_json(bp).get("best_score")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _run_metadata(run_dir: Path, domain_name: str) -> Optional[Tuple[str, Dict[str, Any], Path]]:
    """Return (base_name, existing_json_or_bestparams, target_json_path).

    ``base_name`` is ``<experiment_id>_<algo_tag>`` -- the shared stem of both
    ``<base>_final_evaluation.json`` and ``<base>_best_params.json``.
    """
    fe = sorted(run_dir.glob("*_final_evaluation.json"))
    bp = sorted(run_dir.glob("*_best_params.json"))
    if fe:
        base = fe[0].name[: -len("_final_evaluation.json")]
        meta = _load_json(fe[0])
        target = fe[0]
    elif bp:
        base = bp[0].name[: -len("_best_params.json")]
        meta = _load_json(bp[0])
        target = run_dir / f"{base}_final_evaluation.json"
    else:
        return None
    # If only best_params existed, also fold in its data for provenance.
    if bp and "best_params" not in meta:
        meta.setdefault("best_params", _load_json(bp[0]).get("best_params", {}))
    return base, meta, target


def process_run(
    model_key: str,
    run_dir: Path,
    domain_name: str,
    root: Path,
    calib_window: str,
    eval_window: str,
    sm: StreamflowMetrics,
    write: bool,
    calib_tolerance: float,
) -> Dict[str, Any]:
    """Process one run dir. Returns a status dict for reporting."""
    result: Dict[str, Any] = {
        "model": model_key,
        "run_dir": str(run_dir),
        "status": "skipped",
        "reason": "",
        "calib_kge": None,
        "eval_kge": None,
        "best_score": None,
        "calib_vs_best": None,
        "would_write": False,
        "wrote": False,
    }

    meta_info = _run_metadata(run_dir, domain_name)
    if meta_info is None:
        result["reason"] = "no *_final_evaluation.json or *_best_params.json"
        return result
    base, existing, target = meta_info

    final_eval_dir = run_dir / "final_evaluation"
    if not final_eval_dir.is_dir():
        result["reason"] = "no final_evaluation/ output dir"
        return result

    project_dir = root / f"domain_{domain_name}"
    config: Dict[str, Any] = {
        "SYMFLUENCE_DATA_DIR": str(root),
        "DOMAIN_NAME": domain_name,
        "EXPERIMENT_ID": existing.get("experiment_id", base),
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "FUSE_SPATIAL_MODE": "lumped",
        "OBSERVATIONS_PATH": "default",
    }

    # Simulated series (m3/s, daily).
    try:
        sim = READERS[model_key](final_eval_dir, config, project_dir, sm)
    except Exception as exc:  # noqa: BLE001 -- never abort the sweep on one run
        result["reason"] = f"sim read error: {exc}"
        return result
    if sim is None or sim.empty:
        result["reason"] = "simulated series unreadable/empty"
        return result

    # Observed series (m3/s, daily).
    obs_vals, obs_idx = sm.load_observations(config, project_dir, domain_name, resample_freq="D")
    if obs_vals is None or obs_idx is None:
        result["reason"] = "observations unreadable"
        return result
    obs = pd.Series(obs_vals, index=obs_idx, name="discharge_cms")

    calib_kge = _kge_for_window(obs, sim, calib_window, config, sm)
    eval_kge = _kge_for_window(obs, sim, eval_window, config, sm)
    result["calib_kge"] = calib_kge
    result["eval_kge"] = eval_kge

    if calib_kge is None or eval_kge is None:
        result["reason"] = "insufficient overlap in calib and/or eval window"
        return result

    # GUARD: the backfill's calibration-window KGE must reproduce the recorded
    # optimisation best_score. When it does not (delta > tolerance), the run's
    # calibration was scored over a DIFFERENT window than we slice here -- the
    # hallmark of a run produced before the eval-period windowing fix, whose
    # "held-out" period actually leaked into calibration. Writing a held-out
    # eval for such a run would stamp an in-sample number with an out-of-sample
    # label, so REFUSE and flag it for re-run instead of backfill.
    best = _best_score(run_dir)
    result["best_score"] = best
    if best is not None:
        cal_vs_best = abs(calib_kge - best)
        result["calib_vs_best"] = cal_vs_best
        if cal_vs_best > calib_tolerance:
            result["status"] = "flagged"
            result["reason"] = (
                f"calib KGE {calib_kge:.5f} diverges from best_score {best:.5f} "
                f"by {cal_vs_best:.4f} (> tol {calib_tolerance:g}): calibration-"
                f"window mismatch (likely pre-fix leakage) -- NEEDS RE-RUN, not backfill"
            )
            return result

    result["would_write"] = True

    payload = {
        "algorithm": existing.get("algorithm", base.rsplit("_", 1)[-1].upper()),
        "experiment_id": existing.get("experiment_id", config["EXPERIMENT_ID"]),
        "domain_name": existing.get("domain_name", domain_name),
        "calibration_metrics": {"KGE": calib_kge, "KGE_Calib": calib_kge},
        "evaluation_metrics": {"KGE": eval_kge, "KGE_Eval": eval_kge},
        "best_params": existing.get("best_params", {}),
        "timestamp": datetime.now().isoformat(),
        "backfilled_by": "backfill_final_eval.py",
        "backfill_calibration_period": calib_window,
        "backfill_evaluation_period": eval_window,
        "backfill_calib_vs_best_delta": result["calib_vs_best"],
    }
    # Preserve any other pre-existing top-level keys (e.g. best_score, metric).
    for k, v in existing.items():
        payload.setdefault(k, v)

    if write:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        result["wrote"] = True
        result["status"] = "written"
    else:
        result["status"] = "would-write"
    return result


# ---------------------------------------------------------------------------
# Sweep.
# ---------------------------------------------------------------------------
def iter_run_dirs(root: Path, model_key: str):
    """Yield (domain_name, run_dir) for every run dir of ``model_key``."""
    model_subdir = MODEL_DIRS[model_key]
    for domain_dir in sorted(root.glob("domain_*")):
        opt = domain_dir / "optimization" / model_subdir
        if not opt.is_dir():
            continue
        domain_name = domain_dir.name[len("domain_"):]
        for run_dir in sorted(opt.iterdir()):
            if run_dir.is_dir():
                yield domain_name, run_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill/repair *_final_evaluation.json eval_score for the "
        "Paper-3 reproduction campaign (recompute held-out KGE from existing "
        "model output; never re-runs a model).",
    )
    parser.add_argument("--root", required=True, help="SYMFLUENCE_data directory")
    parser.add_argument("--calibration-period", default=DEFAULT_CALIBRATION,
                        help=f"'start, end' (default: {DEFAULT_CALIBRATION})")
    parser.add_argument("--evaluation-period", default=DEFAULT_EVALUATION,
                        help=f"'start, end' (default: {DEFAULT_EVALUATION})")
    parser.add_argument("--spinup-period", default=DEFAULT_SPINUP,
                        help=f"'start, end' (default: {DEFAULT_SPINUP}); informational only")
    parser.add_argument("--models", default="fuse,hechms,crhm,swat",
                        help="comma list of: fuse,hechms,crhm,swat")
    parser.add_argument("--calib-tolerance", type=float, default=0.01,
                        help="max |recomputed calib KGE - recorded best_score| allowed to "
                             "write; larger divergence FLAGS the run as a calibration-window "
                             "mismatch (pre-fix leakage) needing re-run, not backfill (default: 0.01)")
    parser.add_argument("--write", action="store_true",
                        help="actually write JSON (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="explicit dry-run (default); overrides --write if both given")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    # Quiet the noisy framework loggers unless verbose.
    if not args.verbose:
        logging.getLogger("symfluence").setLevel(logging.ERROR)

    write = bool(args.write) and not bool(args.dry_run)
    root = Path(args.root)
    if not root.is_dir():
        logger.error("root not found: %s", root)
        return 2

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODEL_DIRS]
    if bad:
        logger.error("unknown models: %s (valid: %s)", bad, list(MODEL_DIRS))
        return 2

    calib_window = args.calibration_period
    eval_window = args.evaluation_period

    print("=" * 78)
    print(f"backfill_final_eval  mode={'WRITE' if write else 'DRY-RUN'}")
    print(f"  root       : {root}")
    print(f"  calibration: {calib_window}")
    print(f"  evaluation : {eval_window}")
    print(f"  models     : {', '.join(models)}")
    print("=" * 78)

    sm = StreamflowMetrics()
    results: List[Dict[str, Any]] = []

    for model_key in models:
        for domain_name, run_dir in iter_run_dirs(root, model_key):
            res = process_run(
                model_key, run_dir, domain_name, root,
                calib_window, eval_window, sm, write, args.calib_tolerance,
            )
            results.append(res)
            tag = f"[{model_key}] {domain_name}/{run_dir.name}"
            if res["calib_kge"] is not None and res["eval_kge"] is not None:
                delta = res["eval_kge"] - res["calib_kge"]
                if res["wrote"]:
                    verb = "WROTE"
                elif res["would_write"]:
                    verb = "WOULD WRITE"
                elif res["status"] == "flagged":
                    verb = "FLAGGED (no write)"
                else:
                    verb = "no-write"
                line = (
                    f"{tag}\n"
                    f"    calib KGE = {res['calib_kge']:.5f}   "
                    f"eval KGE = {res['eval_kge']:.5f}   "
                    f"(delta = {delta:+.5f})   -> {verb}"
                )
                if res["status"] == "flagged":
                    line += f"\n    ! {res['reason']}"
                print(line)
            else:
                print(f"{tag}\n    SKIP: {res['reason']}")

    # Summary.
    n_total = len(results)
    n_ok = sum(1 for r in results if r["would_write"] or r["wrote"])
    n_wrote = sum(1 for r in results if r["wrote"])
    n_flagged = sum(1 for r in results if r["status"] == "flagged")
    n_skip = n_total - n_ok - n_flagged
    print("-" * 78)
    print(f"runs seen: {n_total}   computable: {n_ok}   "
          f"{'written' if write else 'would-write'}: {n_wrote if write else n_ok}   "
          f"flagged (needs re-run): {n_flagged}   skipped: {n_skip}")
    if n_flagged:
        print("FLAGGED runs (calibration-window mismatch -- re-run, do NOT backfill):")
        for r in results:
            if r["status"] == "flagged":
                print(f"  [{r['model']}] {r['run_dir']}  -- {r['reason']}")
    if n_skip:
        print("skipped runs (sim/obs unreadable):")
        for r in results:
            if r["status"] != "flagged" and not (r["would_write"] or r["wrote"]):
                print(f"  [{r['model']}] {r['run_dir']}  -- {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
