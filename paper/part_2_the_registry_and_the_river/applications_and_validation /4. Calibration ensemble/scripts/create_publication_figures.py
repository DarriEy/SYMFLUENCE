#!/usr/bin/env python3
"""
Create publication-ready figures for Section 4.4: Calibration Ensemble.

Generates three main-text figures for the calibration algorithm comparison,
with consistent styling, proper font sizes, and PDF + PNG output.

Figures:
  Fig. 1 - Algorithm performance + generalization (2-panel: KGE bars + Cal vs Eval scatter)
  Fig. 2 - Convergence curves with top performers highlighted
  Fig. 3 - Parameter equifinality heatmap (algorithms ordered by Cal KGE)

Usage:
    python create_publication_figures.py
    python create_publication_figures.py --data-dir /path --format pdf
"""

import argparse
from pathlib import Path
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams
    from matplotlib.lines import Line2D
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Error: matplotlib/pandas required for publication figures")
    exit(1)

# Publication style
rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
})

# Import shared config from analyze_results
from analyze_results import (
    ALGORITHMS, HBV_PARAMS, SYMFLUENCE_DATA_DIR, find_optimization_dir, load_convergence_data, load_best_params,
    load_final_metrics, _get_convergence_xy, _sort_algos_by_cal_kge,
)

STUDY_DIR = Path(__file__).parent.parent
RESULTS_DIR = STUDY_DIR / "results"

# Top performers to highlight in convergence plot (by Cal KGE rank)
N_HIGHLIGHTED = 5

# Short, single-line parameter labels for the heatmap x-axis (no \n breaks)
HEATMAP_PARAM_LABELS = {
    "tt": "TT", "cfmax": "CFMAX", "sfcf": "SFCF",
    "cfr": "CFR", "cwh": "CWH", "fc": "FC", "lp": "LP",
    "beta": "Beta", "k0": "K0", "k1": "K1",
    "k2": "K2", "uzl": "UZL", "perc": "PERC",
    "maxbas": "MAXBAS",
}

# Units shown as a second row beneath parameter names
HEATMAP_PARAM_UNITS = {
    "tt": "(°C)", "cfmax": "(mm/°C/d)", "sfcf": "(-)",
    "cfr": "(-)", "cwh": "(-)", "fc": "(mm)", "lp": "(-)",
    "beta": "(-)", "k0": "(1/d)", "k1": "(1/d)",
    "k2": "(1/d)", "uzl": "(mm)", "perc": "(mm/d)",
    "maxbas": "(d)",
}


def _save_figure(fig, output_path: Path, fmt: str):
    """Save figure in both the requested format and PNG."""
    fig.savefig(output_path.with_suffix(f".{fmt}"), facecolor="white")
    fig.savefig(output_path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path.stem}.{fmt} + .png")


def fig_performance_and_generalization(
    metrics: Dict[str, Dict],
    output_path: Path,
    fmt: str = "pdf",
):
    """
    Fig. 1: Combined performance + generalization figure.

    Panel (a): Paired bar chart of calibration and evaluation KGE for each
    algorithm, sorted by calibration KGE descending. A horizontal reference
    line at KGE = 0.75 marks "good" performance (Knoben et al., 2019).

    Panel (b): Scatter plot of calibration KGE vs evaluation KGE with a 1:1
    line. Points above the line indicate evaluation improvement (negative
    degradation). ADAM is annotated as the key outlier.
    """
    algos = _sort_algos_by_cal_kge(
        [k for k in ALGORITHMS if k in metrics], metrics)
    if not algos:
        return

    fig = plt.figure(figsize=(7.5, 4.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.45)

    # --- Panel (a): KGE bar chart ---
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.text(-0.10, 1.05, "(a)", transform=ax_bar.transAxes,
                fontsize=13, fontweight="bold", va="top")

    x = np.arange(len(algos))
    width = 0.35
    cal_vals = [metrics[a].get("cal_kge", np.nan) for a in algos]
    eval_vals = [metrics[a].get("eval_kge", np.nan) for a in algos]
    colors = [ALGORITHMS[a]["color"] for a in algos]
    labels = [ALGORITHMS[a]["label"] for a in algos]

    ax_bar.bar(x - width / 2, cal_vals, width, color=colors,
               edgecolor="black", linewidth=0.4, alpha=0.9, label="Calibration")
    ax_bar.bar(x + width / 2, eval_vals, width, color=colors,
               edgecolor="black", linewidth=0.4, alpha=0.5, hatch="//",
               label="Evaluation")

    # Reference line at KGE = 0.75
    ax_bar.axhline(y=0.75, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.6, zorder=1)
    ax_bar.text(len(algos) - 0.5, 0.752, "KGE = 0.75", fontsize=6.5,
                color="grey", ha="right", va="bottom")

    ax_bar.set_ylabel("KGE")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=50, ha="right", fontsize=7.5)
    ax_bar.legend(fontsize=7, loc="lower left")
    ax_bar.set_ylim(0.55, 0.82)

    # --- Panel (b): Generalization scatter ---
    ax_sc = fig.add_subplot(gs[1])
    ax_sc.text(-0.18, 1.05, "(b)", transform=ax_sc.transAxes,
               fontsize=13, fontweight="bold", va="top")

    for algo in algos:
        m = metrics[algo]
        meta = ALGORITHMS[algo]
        ck = m.get("cal_kge", np.nan)
        ek = m.get("eval_kge", np.nan)
        if not np.isnan(ck) and not np.isnan(ek):
            ax_sc.scatter(ck, ek, c=meta["color"], s=80, marker=meta["marker"],
                          edgecolors="black", linewidth=0.5, zorder=5)

    # 1:1 line — draw across full axis range
    ax_sc.set_xlim(0.62, 0.80)
    ax_sc.set_ylim(0.60, 0.80)
    ax_sc.plot([0.55, 0.85], [0.55, 0.85], "k--", alpha=0.35, linewidth=0.8,
               zorder=1)

    # Annotate ADAM — the key outlier (best eval KGE despite mid-rank cal KGE)
    if "adam" in metrics:
        adam_ck = metrics["adam"].get("cal_kge", np.nan)
        adam_ek = metrics["adam"].get("eval_kge", np.nan)
        if not np.isnan(adam_ck) and not np.isnan(adam_ek):
            ax_sc.annotate(
                "ADAM",
                xy=(adam_ck, adam_ek),
                xytext=(adam_ck - 0.04, adam_ek + 0.008),
                fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="grey",
                                lw=0.6, connectionstyle="arc3,rad=0.15"),
            )

    # Label regions above/below 1:1 — positioned clear of data
    ax_sc.text(0.04, 0.93, "Eval > Cal", transform=ax_sc.transAxes,
               fontsize=6, color="grey", fontstyle="italic")
    ax_sc.text(0.72, 0.04, "Cal > Eval", transform=ax_sc.transAxes,
               fontsize=6, color="grey", fontstyle="italic")

    ax_sc.set_xlabel("Calibration KGE")
    ax_sc.set_ylabel("Evaluation KGE")
    ax_sc.set_aspect("equal")

    # Build a compact legend using colored markers only (no text labels —
    # readers cross-reference colors/markers with panel a)
    # Instead, use a small 3-col legend outside the data region
    legend_handles = []
    legend_labels = []
    for algo in algos:
        meta = ALGORITHMS[algo]
        h = Line2D([0], [0], marker=meta["marker"], color="w",
                   markerfacecolor=meta["color"], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=5, linestyle="None")
        legend_handles.append(h)
        legend_labels.append(meta["label"])
    ax_sc.legend(legend_handles, legend_labels, fontsize=5.5, ncol=3,
                 loc="lower center", handletextpad=0.1, columnspacing=0.4,
                 framealpha=0.9, borderpad=0.4)

    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, output_path, fmt)


def fig_convergence(
    convergence: Dict[str, pd.DataFrame],
    metrics: Dict[str, Dict],
    output_path: Path,
    fmt: str = "pdf",
):
    """
    Fig. 2: Convergence curves with top performers highlighted.

    The top N_HIGHLIGHTED algorithms (by Cal KGE) are drawn with bold,
    colored lines. Remaining algorithms are rendered as thin grey lines
    to provide context without visual clutter. An inset or log-scale x-axis
    emphasises early convergence behaviour.
    """
    if not convergence:
        return

    # Determine top performers to highlight
    algos_with_conv = [k for k in ALGORITHMS if k in convergence]
    ranked = _sort_algos_by_cal_kge(algos_with_conv, metrics)
    top_set = set(ranked[:N_HIGHLIGHTED])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # First pass: draw background (non-highlighted) algorithms
    for algo_key in ranked:
        if algo_key in top_set:
            continue
        df = convergence[algo_key]
        meta = ALGORITHMS[algo_key]
        x, y_best = _get_convergence_xy(df, algo_key)
        if len(y_best) == 0:
            continue
        ax.plot(x, y_best, color="#b0b0b0", linewidth=0.7, alpha=0.5,
                zorder=2)

    # Second pass: draw highlighted algorithms on top
    for algo_key in ranked:
        if algo_key not in top_set:
            continue
        df = convergence[algo_key]
        meta = ALGORITHMS[algo_key]
        x, y_best = _get_convergence_xy(df, algo_key)
        if len(y_best) == 0:
            continue
        ax.plot(x, y_best, color=meta["color"], linewidth=1.8,
                label=meta["label"], alpha=0.9, zorder=4)

    # Build legend: highlighted algorithms + one entry for "Other algorithms"
    handles, labels = ax.get_legend_handles_labels()
    other_line = Line2D([0], [0], color="#b0b0b0", linewidth=0.7, alpha=0.5)
    n_other = len(ranked) - len(top_set)
    handles.append(other_line)
    labels.append(f"Other algorithms ({n_other})")

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best KGE")
    ax.set_ylim(bottom=0.3)
    ax.set_xlim(left=-50)
    ax.legend(handles=handles, labels=labels, loc="lower right",
              fontsize=7.5, framealpha=0.9)

    _save_figure(fig, output_path, fmt)


def fig_parameter_equifinality(
    all_params: Dict[str, Dict],
    metrics: Dict[str, Dict],
    output_path: Path,
    fmt: str = "pdf",
):
    """
    Fig. 3: Parameter equifinality heatmap.

    Algorithms are ordered by calibration KGE (descending, matching Fig. 1)
    so cross-referencing between figures is straightforward. Columns are
    normalized to [0, 1] within each parameter to highlight relative
    differences. Horizontal white lines separate algorithm families.
    """
    # Order algorithms by Cal KGE to match Fig. 1 ordering
    algos_with_params = [k for k in ALGORITHMS if k in all_params]
    algos = _sort_algos_by_cal_kge(algos_with_params, metrics)
    if len(algos) < 2:
        return

    params_present = [p for p in HBV_PARAMS
                      if any(p in all_params.get(a, {}) for a in algos)]
    n_params = len(params_present)
    n_algos = len(algos)

    # Size: generous width per column, generous height per row
    fig_w = max(7.5, n_params * 0.58 + 1.8)   # +1.8 for y-labels and colorbar
    fig_h = max(4.5, n_algos * 0.40 + 1.2)    # +1.2 for x-labels and padding
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Build data matrix
    data = np.full((n_algos, n_params), np.nan)
    for i, algo in enumerate(algos):
        for j, param in enumerate(params_present):
            v = all_params.get(algo, {}).get(param, np.nan)
            if isinstance(v, list):
                v = v[0]
            data[i, j] = v

    # Normalize columns to [0, 1]
    data_norm = np.copy(data)
    for j in range(n_params):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0 and valid.max() > valid.min():
            data_norm[:, j] = (col - valid.min()) / (valid.max() - valid.min())
        else:
            data_norm[:, j] = 0.5

    im = ax.imshow(data_norm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(n_algos):
        for j in range(n_params):
            val = data[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                tc = ("white" if data_norm[i, j] > 0.7
                      or data_norm[i, j] < 0.3 else "black")
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6, color=tc, fontweight="bold")

    # Family separator lines — group by family in the KGE-sorted order
    prev_fam = None
    for i, algo in enumerate(algos):
        fam = ALGORITHMS[algo]["family"]
        if prev_fam is not None and fam != prev_fam:
            ax.axhline(y=i - 0.5, color="white", linewidth=1.5)
        prev_fam = fam

    # X-axis: two-line labels — parameter name + unit, rotated to avoid overlap
    x_labels = []
    for p in params_present:
        name = HEATMAP_PARAM_LABELS.get(p, p)
        unit = HEATMAP_PARAM_UNITS.get(p, "")
        x_labels.append(f"{name}\n{unit}")

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(x_labels, fontsize=7, ha="center", linespacing=0.85)
    # Move x-tick labels to top so they don't collide with the figure edge
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Y-axis: algorithm name + Cal KGE for cross-reference
    ax.set_yticks(range(n_algos))
    ytick_labels = []
    for a in algos:
        lbl = ALGORITHMS[a]["label"]
        kge = metrics.get(a, {}).get("cal_kge", np.nan)
        if not np.isnan(kge):
            lbl = f"{lbl}  ({kge:.3f})"
        ytick_labels.append(lbl)
    ax.set_yticklabels(ytick_labels, fontsize=7.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    cbar.set_label("Normalized Value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.18, right=0.90, top=0.88, bottom=0.04)
    _save_figure(fig, output_path, fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Create publication figures for Calibration Ensemble Study"
    )
    parser.add_argument("--data-dir", "-d", type=Path, default=SYMFLUENCE_DATA_DIR)
    parser.add_argument("--output-dir", "-o", type=Path, default=RESULTS_DIR / "plots")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "svg", "eps"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Publication Figure Generation - Section 4.4")
    print("=" * 60)

    # Load all data
    metrics = {}
    convergence = {}
    all_params = {}

    for algo_key, meta in ALGORITHMS.items():
        opt_dir = find_optimization_dir(args.data_dir, algo_key)
        if opt_dir is None:
            continue

        final = load_final_metrics(opt_dir)
        if final:
            cm = final.get("calibration_metrics", {})
            em = final.get("evaluation_metrics", {})
            metrics[algo_key] = {
                "cal_kge": float(cm.get("Calib_KGE", cm.get("KGE", np.nan))),
                "eval_kge": float(em.get("Eval_KGE", em.get("KGE", np.nan))),
            }

        conv = load_convergence_data(opt_dir)
        if conv is not None:
            convergence[algo_key] = conv

        params = load_best_params(opt_dir)
        if params is not None:
            all_params[algo_key] = params

    print(f"\n  Loaded {len(metrics)} algorithm results")

    # --- Fallback: load from CSV if no optimization dirs found ---
    if not metrics:
        csv_path = RESULTS_DIR / "performance_summary.csv"
        if csv_path.exists():
            print(f"  Loading metrics from {csv_path.name}")
            df = pd.read_csv(csv_path)
            # Map algorithm labels back to keys
            label_to_key = {v["label"]: k for k, v in ALGORITHMS.items()}
            for _, row in df.iterrows():
                algo_key = label_to_key.get(row["Algorithm"])
                if algo_key is None:
                    continue
                metrics[algo_key] = {
                    "cal_kge": float(row.get("Cal_KGE", np.nan)),
                    "eval_kge": float(row.get("Eval_KGE", np.nan)),
                }

    if not all_params:
        csv_path = RESULTS_DIR / "parameter_comparison.csv"
        if csv_path.exists():
            print(f"  Loading parameters from {csv_path.name}")
            df = pd.read_csv(csv_path)
            label_to_key = {v["label"]: k for k, v in ALGORITHMS.items()}
            for _, row in df.iterrows():
                algo_key = label_to_key.get(row["Algorithm"])
                if algo_key is None:
                    continue
                params = {}
                for p in HBV_PARAMS:
                    if p in row and not pd.isna(row[p]):
                        params[p] = float(row[p])
                if params:
                    all_params[algo_key] = params

    print(f"  Metrics: {len(metrics)}, Convergence: {len(convergence)}, "
          f"Parameters: {len(all_params)}")

    # Generate publication figures
    print("\nGenerating publication figures...")

    if metrics:
        fig_performance_and_generalization(
            metrics, args.output_dir / "fig_performance", fmt=args.format)
    else:
        print("  SKIP fig_performance: no metrics loaded")

    if convergence and metrics:
        fig_convergence(
            convergence, metrics,
            args.output_dir / "fig_convergence", fmt=args.format)
    else:
        print("  SKIP fig_convergence: no convergence data loaded")

    if all_params and metrics:
        fig_parameter_equifinality(
            all_params, metrics,
            args.output_dir / "fig_equifinality", fmt=args.format)
    else:
        print("  SKIP fig_equifinality: no parameter data loaded")

    print("\n" + "=" * 60)
    print("Publication figures complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
