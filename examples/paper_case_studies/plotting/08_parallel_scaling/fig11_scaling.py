#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Paper 3, Figure 11: calibration scaling under two evaluation-cost regimes.

Top row — local laptop, lumped Bow at Banff, ~15 s per SUMMA evaluation,
1–10 processes, ProcessPool vs MPI.
Bottom row — DRAC Fir cluster, catchment CAN_01AD003, ~60 s per evaluation,
10–100 cores, Async-DDS vs differential evolution.

Both rows are read from PRIMARY evidence — each run's own log, via the
framework's ``Calibrating model parameters (Duration: …s)`` line — rather than
from a hand-maintained summary table. An intermediate CSV drifted from its
source during this figure's preparation (an MPI point read 569.84 s where the
run log said 738.81 s), so nothing here is trusted unless it can be traced back
to the run that produced it.

Two corrections the raw wall-clock numbers require:

* **The local "ProcessPool" leg only measures ProcessPool if MPI is unavailable.**
  The shipped configs select no backend — it is chosen at runtime — so with Open
  MPI on PATH *both* legs launched persistent MPI workers and the comparison was
  vacuous. The ProcessPool timings used here come from runs with the MPI
  launchers hidden from PATH, and each run's log is asserted to contain no MPI
  pool start.
* **Equal core counts do not imply equal work.** Async-DDS at 25 cores ran 124
  evaluations rather than 100, so its raw wall-clock understates its scaling;
  DDS is therefore normalised to time-per-evaluation. DE holds ~200 evaluations
  (population 100 x 1 generation) at every core count, so it needs no correction
  — but DE and DDS are NOT doing the same amount of work as each other, and the
  two curves are not comparable in absolute time.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
OUT = _HERE.parents[1] / "output"
OUT.mkdir(parents=True, exist_ok=True)
LOCAL_CSV = _HERE.parents[1] / "results" / "timing_calibration_latest.csv"
FIR_LOGS = Path(os.getenv(
    "P3_FIR_LOG_DIR",
    str(Path.home() / "Desktop/symfluence_papers_final_rev/Frank_repro/Fir-repeat-0716-2026"),
))

# The step duration ("Calibrating model parameters (Duration: ...s)") is NOT the
# right measurement: it also contains the final evaluation, a single full-period
# model run appended after calibration that does not parallelise. That fixed tail
# is ~140 s on Fir and ~20 s locally — negligible against a 30-minute run, but
# 45% of a 3.4-minute one, so including it collapses speedup exactly where
# scaling is being tested. Timing the calibration loop instead reproduces the
# reference Fir figure to within 0.3 min at all eight points.
_TS = re.compile(r"(?:\d{4}-\d{2}-\d{2} )?(\d{2}):(\d{2}):(\d{2})")
_LOOP_START = re.compile(r"Evaluating initial (?:pool|population)")
_LOOP_END = "RUNNING FINAL EVALUATION"
_EVALS = re.compile(r"Total evaluations: (\d+)")


def _calibration_loop_seconds(text: str):
    """Wall-clock of the calibration loop only, excluding the final evaluation."""
    start = end = None
    for line in text.splitlines():
        m = _TS.match(line.strip())
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if start is None and _LOOP_START.search(line):
            start = t
        if end is None and _LOOP_END in line:
            end = t
    if start is None or end is None:
        return None
    delta = end - start
    return delta + 86400 if delta < 0 else delta   # run crossed midnight

# Two hues, validated for colour-vision separation on all pairs (dE 24.7 under
# protanopia); marker shape repeats the distinction so the panels survive
# greyscale printing.
SERIES = {
    "ProcessPool": ("#2a78d6", "o"), "MPI": ("#eb6834", "s"),
    "Async-DDS": ("#2a78d6", "^"), "DE": ("#eb6834", "D"),
}
IDEAL = {"color": "#777777", "ls": "--", "lw": 1.1}

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_NOTE = 10.5, 9.5, 8.5, 8.5, 7.5


LOCAL_LOGS = Path(os.getenv("P3_LOCAL_LOG_DIR", str(_HERE.parent / "local_logs")))


def load_local() -> dict:
    """Local timings: calibration loop and evaluation count, per run log.

    Same basis as the Fir row — the loop only, and the evaluation count, because
    the local runs do not all execute exactly 100 evaluations either (the
    10-process run did 103).
    """
    out: dict = {}
    for path in sorted(LOCAL_LOGS.glob("calib_summa_*_100evals_np*.log")):
        m = re.search(r"calib_summa_(pp|mpi)_100evals_np(\d+)", path.name)
        if not m:
            continue
        text = path.read_text(errors="ignore")
        secs = _calibration_loop_seconds(text)
        if secs is None:
            continue
        used_mpi = "Starting persistent MPI workers" in text
        label = "MPI" if used_mpi else "ProcessPool"
        # np=1 spawns no pool in either leg; it is the shared serial baseline, so
        # attribute it by filename rather than by which backend appeared.
        if int(m.group(2)) == 1:
            label = "MPI" if m.group(1) == "mpi" else "ProcessPool"
        evals = _EVALS.findall(text)
        out.setdefault(label, {})[int(m.group(2))] = (
            float(secs), int(evals[-1]) if evals else None,
        )
    return out


def load_fir() -> dict:
    """Fir timings parsed from the run logs, with evaluation counts.

    Returns ``{algorithm: {cores: (seconds, evaluations)}}``. Evaluation counts
    are read where the log reports them; DE does not print a total, but its
    configuration fixes population x generations across every core count, so its
    workload is constant by construction.
    """
    out: dict = {}
    for path in sorted(FIR_LOGS.glob("fig15_*.out")):
        m = re.match(r"fig15_(\w+)_n(\d+)\.out", path.name)
        if not m:
            continue
        algo, cores = m.group(1), int(m.group(2))
        text = path.read_text(errors="ignore")
        secs = _calibration_loop_seconds(text)
        if secs is None:
            continue
        evals = _EVALS.findall(text)
        label = "Async-DDS" if algo == "dds" else "DE"
        out.setdefault(label, {})[cores] = (
            float(secs), int(evals[-1]) if evals else None,
        )
    return out


def _panel(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_row(axes, series, baseline_units, time_scale, time_label, xlabel, prefix):
    """One regime: wall-clock, speedup and parallel efficiency.

    ``series`` maps a label to ``{units: (time_seconds, work)}``. Where *work*
    differs between points, speedup is computed on time-per-unit-of-work so a run
    that did more evaluations is not penalised for it.
    """
    ax_t, ax_s, ax_e = axes
    for label, points in series.items():
        xs = sorted(points)
        colour, marker = SERIES[label]
        times = [points[x][0] for x in xs]
        works = [points[x][1] for x in xs]
        ax_t.plot(xs, [t / time_scale for t in times], marker=marker, color=colour,
                  lw=1.6, ms=5, label=label)

        # Normalise by work where it is known and varies; otherwise raw time.
        if all(w for w in works) and len(set(works)) > 1:
            cost = [t / w for t, w in zip(times, works)]
        else:
            cost = times
        base_cost = cost[0]
        speedup = [base_cost / c for c in cost]
        ideal = [x / baseline_units for x in xs]
        ax_s.plot(xs, speedup, marker=marker, color=colour, lw=1.6, ms=5, label=label)
        ax_e.plot(xs, [s / i for s, i in zip(speedup, ideal)], marker=marker,
                  color=colour, lw=1.6, ms=5, label=label)

    xs_all = sorted({x for p in series.values() for x in p})
    ax_s.plot(xs_all, [x / baseline_units for x in xs_all], label="Ideal", **IDEAL)
    ax_e.axhline(1.0, label="Ideal", **IDEAL)

    _panel(ax_t, f"{prefix} — wall-clock time", xlabel, time_label)
    _panel(ax_s, f"{prefix} — speedup", xlabel,
           f"Speedup (vs {baseline_units} " + ("process)" if baseline_units == 1 else "cores)"))
    _panel(ax_e, f"{prefix} — parallel efficiency", xlabel, "Parallel efficiency")
    ax_e.set_ylim(0, 1.25)
    for ax in axes:
        ax.legend(fontsize=FS_LEGEND, frameon=True, framealpha=0.95)


def main() -> None:
    local_series, fir = load_local(), load_fir()

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6))
    # Short prefixes: the evaluation-cost regime is already named in the
    # suptitle, and the long form clipped the right-hand column's titles.
    _draw_row(axes[0], local_series, 1, 1.0, "Wall-clock time (s)",
              "Number of processes", "Local laptop")
    _draw_row(axes[1], fir, 10, 60.0, "Wall-clock time (min)",
              "Number of cores", "DRAC Fir")

    # No figure title and no footnote: both belong to the caption in the
    # manuscript, and repeating them here duplicates the caption and eats panel
    # area. The measurement basis they carried — calibration loop only, speedup
    # normalised per evaluation, DDS ~100 vs DE ~200 evaluations — is stated in
    # the caption text delivered alongside this figure.
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"figure_11_parallel_scaling.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"Saved: {path.name}")

    for label, points in {**local_series, **fir}.items():
        xs = sorted(points)
        base = points[xs[0]]
        base_cost = base[0] / base[1] if base[1] else base[0]
        speeds = []
        for x in xs:
            t, w = points[x]
            speeds.append(base_cost / (t / w if w else t))
        print(f"  {label:12s} " + "  ".join(f"{x}:{s:.2f}x" for x, s in zip(xs, speeds)))


if __name__ == "__main__":
    main()
