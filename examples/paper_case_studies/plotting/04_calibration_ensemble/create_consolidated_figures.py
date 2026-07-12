#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Create consolidated publication figures for Section 4.4.

Figure 1: Performance, Generalization & Robustness (3 panels)
Figure 2: Convergence Efficiency (existing fig2)
Figure 3: Parameter Equifinality (existing fig3)
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths
# Data root from SYMFLUENCE_DATA_DIR (default: sibling SYMFLUENCE_data of the
# repo root). The shipped 04_calibration_ensemble configs use domain
# Bow_at_Banff_lumped_calibration_ensemble with experiment ids
# cal_ensemble_<model>_<algo>.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
RESULTS_DIR = _HERE.parent / "results"
PLOTS_DIR = _HERE.parents[1] / "output"

SYMFLUENCE_DATA_DIR = Path(
    os.getenv("SYMFLUENCE_DATA_DIR", str(_REPO_ROOT.parent / "SYMFLUENCE_data"))
)
DOMAIN_NAME = "Bow_at_Banff_lumped_calibration_ensemble"

# Algorithm metadata
ALGORITHMS = {
    "dds": {"label": "DDS", "family": "Sampling", "color": "#1f77b4"},
    "sceua": {"label": "SCE-UA", "family": "Evolutionary", "color": "#ff7f0e"},
    "de": {"label": "DE", "family": "Evolutionary", "color": "#2ca02c"},
    "pso": {"label": "PSO", "family": "Evolutionary", "color": "#d62728"},
    "ga": {"label": "GA", "family": "Evolutionary", "color": "#9467bd"},
    "cmaes": {"label": "CMA-ES", "family": "Evolutionary", "color": "#8c564b"},
    "adam": {"label": "ADAM", "family": "Gradient", "color": "#e377c2"},
    "lbfgs": {"label": "L-BFGS", "family": "Gradient", "color": "#7f7f7f"},
    "nelder_mead": {"label": "Nelder-Mead", "family": "Direct Search", "color": "#bcbd22"},
    "sa": {"label": "SA", "family": "Stochastic", "color": "#17becf"},
    "basin_hopping": {"label": "Basin Hop.", "family": "Stochastic", "color": "#aec7e8"},
    "bayesian_opt": {"label": "Bayes. Opt.", "family": "Surrogate", "color": "#ffbb78"},
}

# Algorithms for robustness analysis
ROBUSTNESS_ALGOS = ["dds", "sceua", "de", "pso", "cmaes", "adam"]


def find_optimization_dir(data_dir: Path, algo_key: str, seed: int = 42):
    """Locate optimization output directory."""
    algo_prefix_map = {
        "dds": "dds", "sceua": "sce-ua", "de": "de", "pso": "pso",
        "ga": "ga", "cmaes": "cma-es", "adam": "adam", "lbfgs": "lbfgs",
        "nelder_mead": "nelder-mead", "sa": "simulated_annealing",
        "basin_hopping": "basin-hopping", "bayesian_opt": "bayesian_opt",
    }

    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    experiment_id = f"cal_ensemble_hbv_{algo_key}{seed_suffix}"  # shipped eids include the model name
    prefix = algo_prefix_map.get(algo_key, algo_key)

    opt_base = data_dir / f"domain_{DOMAIN_NAME}" / "optimization" / "HBV"
    if not opt_base.exists():
        opt_base = data_dir / DOMAIN_NAME / "optimization" / "HBV"

    exact = opt_base / f"{prefix}_{experiment_id}"
    if exact.exists():
        return exact

    # Try scanning
    if opt_base.exists():
        for entry in sorted(opt_base.iterdir()):
            if entry.is_dir() and experiment_id in entry.name:
                return entry
    return None


def load_metrics(opt_dir: Path):
    """Load final evaluation metrics."""
    for pattern in ["*final_evaluation*.json", "*evaluation*.json", "*metrics*.json"]:
        files = list(opt_dir.glob(pattern))
        if files:
            try:
                with open(files[0]) as f:
                    data = json.load(f)
                cm = data.get("calibration_metrics", {})
                em = data.get("evaluation_metrics", {})
                return {
                    "cal_kge": float(cm.get("Calib_KGE", cm.get("KGE", np.nan))),
                    "eval_kge": float(em.get("Eval_KGE", em.get("KGE", np.nan))),
                }
            except Exception:
                continue
    return None


def load_all_metrics():
    """Load metrics for all algorithms."""
    metrics = {}
    for algo_key in ALGORITHMS:
        opt_dir = find_optimization_dir(SYMFLUENCE_DATA_DIR, algo_key)
        if opt_dir:
            m = load_metrics(opt_dir)
            if m:
                metrics[algo_key] = m
    return metrics


def load_robustness_metrics():
    """Load multi-seed metrics for robustness algorithms."""
    multi_seed = {}
    seeds = [42, 1042, 2042, 3042, 4042]

    for algo_key in ROBUSTNESS_ALGOS:
        seed_results = []
        for seed in seeds:
            opt_dir = find_optimization_dir(SYMFLUENCE_DATA_DIR, algo_key, seed=seed)
            if opt_dir:
                m = load_metrics(opt_dir)
                if m:
                    seed_results.append(m)
        if seed_results:
            multi_seed[algo_key] = seed_results
    return multi_seed


def create_figure1(metrics, multi_seed_metrics, output_path):
    """
    Create consolidated Figure 1: Performance, Generalization & Robustness.

    Panel (a): Cal/Eval KGE bar chart
    Panel (b): Cal vs Eval scatter (generalization)
    Panel (c): Robustness box plots
    """
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1.2], wspace=0.3)

    # Sort algorithms by calibration KGE
    sorted_algos = sorted(
        [k for k in ALGORITHMS if k in metrics],
        key=lambda a: metrics[a]["cal_kge"],
        reverse=True
    )

    # =========================================================================
    # Panel (a): Bar chart - Cal/Eval KGE
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(sorted_algos))
    width = 0.35

    cal_vals = [metrics[a]["cal_kge"] for a in sorted_algos]
    eval_vals = [metrics[a]["eval_kge"] for a in sorted_algos]
    colors = [ALGORITHMS[a]["color"] for a in sorted_algos]
    labels = [ALGORITHMS[a]["label"] for a in sorted_algos]

    bars_cal = ax1.bar(x - width/2, cal_vals, width, label="Calibration",
                       color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    bars_eval = ax1.bar(x + width/2, eval_vals, width, label="Evaluation",
                        color=colors, edgecolor="black", linewidth=0.5, alpha=0.5,
                        hatch="//")

    # Reference line at KGE=0.75
    ax1.axhline(y=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.text(len(sorted_algos)-0.5, 0.755, "KGE=0.75", fontsize=8, color="gray", ha="right")

    ax1.set_ylabel("KGE", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0.55, 0.82)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("(a) Algorithm Performance", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # =========================================================================
    # Panel (b): Generalization scatter
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])

    for algo_key in sorted_algos:
        m = metrics[algo_key]
        meta = ALGORITHMS[algo_key]
        ax2.scatter(m["cal_kge"], m["eval_kge"],
                   c=meta["color"], s=120,
                   edgecolors="black", linewidth=0.8,
                   label=meta["label"], zorder=5)

    # 1:1 line
    lims = [0.62, 0.80]
    ax2.plot(lims, lims, "k--", alpha=0.5, linewidth=1, zorder=1)
    ax2.fill_between(lims, lims, [0.62, 0.62], alpha=0.1, color="red", label="Overfitting")
    ax2.fill_between(lims, lims, [0.80, 0.80], alpha=0.1, color="green", label="Generalizes")

    ax2.set_xlabel("Calibration KGE", fontsize=10)
    ax2.set_ylabel("Evaluation KGE", fontsize=10)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect("equal")
    ax2.set_title("(b) Generalization", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    # Annotate ADAM (best generalization)
    adam_m = metrics.get("adam", {})
    if adam_m:
        ax2.annotate("ADAM", (adam_m["cal_kge"], adam_m["eval_kge"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    # =========================================================================
    # Panel (c): Robustness box plots
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])

    robustness_algos_sorted = ["dds", "adam", "cmaes", "de", "sceua", "pso"]
    box_data_cal = []
    box_data_eval = []
    box_labels = []
    box_colors = []

    for algo in robustness_algos_sorted:
        if algo in multi_seed_metrics:
            results = multi_seed_metrics[algo]
            box_data_cal.append([r["cal_kge"] for r in results])
            box_data_eval.append([r["eval_kge"] for r in results])
            box_labels.append(ALGORITHMS[algo]["label"])
            box_colors.append(ALGORITHMS[algo]["color"])

    x_pos = np.arange(len(box_labels))
    width = 0.35

    # Calibration boxes
    bp1 = ax3.boxplot([box_data_cal[i] for i in range(len(box_labels))],
                      positions=x_pos - width/2, widths=width*0.8,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=4))
    for patch, color in zip(bp1["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    # Evaluation boxes
    bp2 = ax3.boxplot([box_data_eval[i] for i in range(len(box_labels))],
                      positions=x_pos + width/2, widths=width*0.8,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=4))
    for patch, color in zip(bp2["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_hatch("//")

    ax3.set_ylabel("KGE", fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=9)
    ax3.set_title("(c) Robustness (5 seeds)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.2, axis="y")
    ax3.set_ylim(0.55, 0.82)

    # Legend for cal/eval
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="gray", alpha=0.9, label="Calibration"),
                       Patch(facecolor="gray", alpha=0.5, hatch="//", label="Evaluation")]
    ax3.legend(handles=legend_elements, loc="lower left", fontsize=8)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("Creating Consolidated Figures for Section 4.4")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading metrics...")
    metrics = load_all_metrics()
    print(f"  Loaded {len(metrics)} algorithms")

    print("\nLoading robustness data...")
    multi_seed = load_robustness_metrics()
    for algo, results in multi_seed.items():
        print(f"  {ALGORITHMS[algo]['label']}: {len(results)} seeds")

    # Create Figure 1
    print("\nGenerating Figure 1 (Performance, Generalization, Robustness)...")
    create_figure1(metrics, multi_seed, PLOTS_DIR / "fig1_consolidated.png")

    print("\n" + "=" * 60)
    print("Done! Figure 2 (convergence) and Figure 3 (parameters) remain as-is.")
    print("=" * 60)


if __name__ == "__main__":
    main()
