#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Paper 3, Figure 9(a): calibration-period KGE heatmap for the model x algorithm
matrix (the ~130 combinations of the calibration ensemble).

Rows are the eight hydrological models (five JAX-native above the divider, three
external below); columns are the optimization algorithms. Each cell is the
calibration-period KGE read from that combination's final_evaluation.json.
Gradient methods (ADAM, L-BFGS) exist only for the differentiable JAX models, so
those cells are intentionally blank for the external models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
DATA_DIR = Path(os.getenv("SYMFLUENCE_DATA_DIR", str(_REPO_ROOT.parent / "SYMFLUENCE_data")))
DOMAIN = "Bow_at_Banff_lumped_calibration_ensemble"
OPT = DATA_DIR / f"domain_{DOMAIN}" / "optimization"
OUT = _HERE.parents[1] / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Row order: 5 JAX-native, then 3 external (matches the paper's divider).
JAX_MODELS = ["HBV", "SACSMA", "XINANJIANG", "HECHMS", "TOPMODEL"]
EXTERNAL_MODELS = ["SUMMA", "FUSE", "HYPE"]
MODELS = JAX_MODELS + EXTERNAL_MODELS
MODEL_LABELS = {
    "HBV": "HBV", "SACSMA": "SAC-SMA", "XINANJIANG": "Xinanjiang",
    "HECHMS": "HEC-HMS", "TOPMODEL": "TOPMODEL",
    "SUMMA": "SUMMA", "FUSE": "FUSE", "HYPE": "HYPE",
}

# Column order, grouped by algorithm family (sampling -> evolutionary -> gradient
# -> direct search -> stochastic -> surrogate -> multi-objective/other).
ALGO_ORDER = [
    "DDS", "SCE-UA", "DE", "PSO", "GA", "CMA-ES", "DREAM", "ABC", "GLUE",
    "NSGA-II", "MOEA/D", "Nelder-Mead", "SA", "Basin Hop.", "Bayes. Opt.",
    "ADAM", "L-BFGS",
]
# Normalise the many spellings the run JSONs use to the labels above.
ALIAS = {
    "DDS": "DDS", "SCE-UA": "SCE-UA", "SCEUA": "SCE-UA", "DE": "DE", "PSO": "PSO",
    "GA": "GA", "CMA-ES": "CMA-ES", "CMAES": "CMA-ES", "DREAM": "DREAM",
    "ABC": "ABC", "GLUE": "GLUE", "NSGA-II": "NSGA-II", "NSGA2": "NSGA-II",
    "MOEA/D": "MOEA/D", "MOEAD": "MOEA/D", "NELDER-MEAD": "Nelder-Mead",
    "NELDER_MEAD": "Nelder-Mead", "SA": "SA", "SIMULATED_ANNEALING": "SA",
    "BASIN-HOPPING": "Basin Hop.", "BASIN_HOPPING": "Basin Hop.",
    "BAYESIAN_OPT": "Bayes. Opt.", "BAYESIAN-OPT": "Bayes. Opt.",
    "ADAM": "ADAM", "LBFGS": "L-BFGS", "L-BFGS": "L-BFGS",
}


def _cal_kge_from_dir(algo_dir: Path):
    """Calibration KGE for one run dir, tolerant of the three storage conventions.

    Different model families write the metric differently:
      * ``final_evaluation.json`` -> ``calibration_metrics.KGE`` (SUMMA/FUSE/HYPE/SAC-SMA)
      * ``final_evaluation.json`` -> ``calibration_metrics.Calib_KGE`` (HBV/TOPMODEL)
      * ``best_params.json`` -> ``best_score`` with ``metric == 'KGE'`` (HEC-HMS, and a
        universal fallback when a run has no final-evaluation file).
    Returns (algorithm_label, kge) or (label, None).
    """
    label = None
    kge = None
    fe = list(algo_dir.glob("*final_evaluation.json"))
    if fe:
        try:
            d = json.load(open(fe[0]))
            label = ALIAS.get(str(d.get("algorithm", "")).strip().upper())
            cm = d.get("calibration_metrics") or {}
            for key in ("KGE", "Calib_KGE"):
                if cm.get(key) is not None:
                    kge = float(cm[key])
                    break
        except (OSError, ValueError):
            pass
    if kge is None:
        bp = list(algo_dir.glob("*best_params.json"))
        if bp:
            try:
                b = json.load(open(bp[0]))
                if label is None:
                    label = ALIAS.get(str(b.get("algorithm", "")).strip().upper())
                if b.get("best_score") is not None and "KGE" in str(b.get("metric", "")).upper():
                    kge = float(b["best_score"])
            except (OSError, ValueError):
                pass
    if kge is None:
        # Last resort: the calibration ran but never wrote a final-evaluation file.
        # The best (max) objective in the iteration log is the calibration KGE
        # (the optimiser maximises KGE); ignore penalty scores (<= -100).
        csvs = list(algo_dir.glob("*parallel_iteration_results.csv"))
        if csvs:
            try:
                import csv as _csv
                scores = []
                for row in _csv.DictReader(open(csvs[0])):
                    try:
                        s = float(row.get("score", ""))
                    except (TypeError, ValueError):
                        continue
                    if s > -100:
                        scores.append(s)
                if scores:
                    kge = max(scores)
            except OSError:
                pass
    if label is None:
        label = ALIAS.get(algo_dir.name.split("_cal_ensemble")[0].upper())
    return label, kge


def _load_cal_kge(model: str) -> dict:
    """Map algorithm-label -> calibration KGE for one model."""
    out = {}
    mdir = OPT / model
    if not mdir.is_dir():
        return out
    for algo_dir in sorted(mdir.iterdir()):
        if not algo_dir.is_dir():
            continue
        label, kge = _cal_kge_from_dir(algo_dir)
        if label is not None and kge is not None:
            out[label] = kge
    return out


def main():
    matrix = {m: _load_cal_kge(m) for m in MODELS}
    n_found = sum(len(v) for v in matrix.values())
    # Keep only algorithm columns that at least one model actually has.
    cols = [a for a in ALGO_ORDER if any(a in matrix[m] for m in MODELS)]
    grid = np.full((len(MODELS), len(cols)), np.nan)
    for i, m in enumerate(MODELS):
        for j, a in enumerate(cols):
            if a in matrix[m]:
                grid[i, j] = matrix[m][a]
    print(f"Loaded {n_found} model-algorithm KGE values "
          f"({len(MODELS)} models x {len(cols)} algorithms).")

    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.5, 0.5 * len(MODELS) + 2.0))
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#e6e6e6")  # blank cells (missing combos) in light grey
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m.title() if m != "SUMMA" else "SUMMA")
                        for m in MODELS], fontsize=9)

    # Annotate each populated cell with its KGE.
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            if np.isfinite(grid[i, j]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if (v < 0.25 or v > 0.85) else "black")

    # Divider between the five JAX-native models and the three external ones.
    ax.axhline(len(JAX_MODELS) - 0.5, color="black", linewidth=1.8)
    ax.text(-0.6, (len(JAX_MODELS) - 1) / 2, "JAX-native", rotation=90,
            va="center", ha="center", fontsize=8, fontweight="bold")
    ax.text(-0.6, len(JAX_MODELS) + (len(EXTERNAL_MODELS) - 1) / 2, "External",
            rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("(a) Calibration-period KGE across model × algorithm combinations",
                 fontsize=11, pad=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Calibration KGE", fontsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = OUT / f"figure_09_calibration_heatmap.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved: {p.name}")


if __name__ == "__main__":
    main()
