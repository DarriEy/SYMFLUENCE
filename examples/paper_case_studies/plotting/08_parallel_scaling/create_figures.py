#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Generate publication-quality figures for Section 4.13 Parallel Scaling.

Produces:
  - Fig 1: TauDEM delineation scaling (Iceland local, 2-panel)
  - Fig 2: SUMMA calibration scaling (local + HPC, 2x3 panel)
  - Fig 3: Combined efficiency comparison (all experiments)

Usage:
    python create_figures.py                    # All figures
    python create_figures.py --figure 1 2       # Specific figures
    python create_figures.py --format pdf       # PDF output
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")

# analysis/ JSONs are produced by analyze_scaling.py (next to this script)
# from the timing CSVs of the 08_parallel_scaling runs; figures go to
# plotting/output/.
_HERE = Path(__file__).resolve()
ANALYSIS_DIR = _HERE.parent / "analysis"
FIGURES_DIR = _HERE.parents[1] / "output"

# Publication style
COLORS = {
    "90m": "#1f77b4",
    "ProcessPool": "#1f77b4",
    "MPI": "#ff7f0e",
    "DDS": "#2ca02c",
    "DE": "#d62728",
    "ideal": "#cccccc",
}

MARKERS = {
    "90m": "o",
    "ProcessPool": "o",
    "MPI": "s",
    "DDS": "^",
    "DE": "D",
}


def load_analysis(name: str) -> Dict[str, Any]:
    """Load analysis JSON."""
    path = ANALYSIS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def setup_style():
    """Set publication figure style."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, name: str, fmt: str = "png"):
    """Save figure in specified format."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fmt, "pdf") if fmt != "pdf" else ("pdf",):
        path = FIGURES_DIR / f"{name}.{f}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.{fmt}")


def fig1_taudem_scaling(fmt: str = "png"):
    """Figure 1: TauDEM delineation scaling (Iceland local, 2-panel)."""
    if not HAS_MPL:
        return

    data = load_analysis("scaling_taudem")
    local = data.get("local", [])
    if not local:
        print("  Fig 1: No TauDEM data available")
        return

    # Filter to 90m only
    records = [r for r in local if r["group"] == "90m"]
    if not records:
        print("  Fig 1: No 90m TauDEM data available")
        return

    records.sort(key=lambda x: x["num_processes"])
    nps = [r["num_processes"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("TauDEM Delineation Scaling (Iceland 90 m, Local)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Wall-clock time
    ax = axes[0]
    times = [r["wall_clock_seconds"] for r in records]
    ax.plot(nps, times, marker="o", color=COLORS["90m"],
            linewidth=1.5, markersize=6, label="Iceland 90 m")
    ax.set_xlabel("Number of MPI Processes")
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_title("Execution Time")
    ax.set_xticks(nps)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Speedup
    ax = axes[1]
    speedups = [r["speedup"] for r in records]
    ax.plot(nps, speedups, marker="o", color=COLORS["90m"],
            linewidth=1.5, markersize=6, label="Measured")
    max_np = max(nps)
    ax.plot([1, max_np], [1, max_np], "--",
            color=COLORS["ideal"], label="Ideal", linewidth=1)
    ax.set_xlabel("Number of MPI Processes")
    ax.set_ylabel("Speedup")
    ax.set_title("Strong Scaling Speedup")
    ax.set_xticks(nps)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "fig1_taudem_scaling", fmt)


def fig2_calibration_scaling(fmt: str = "png"):
    """Figure 2: Calibration scaling (local + HPC, 2x3 panel)."""
    if not HAS_MPL:
        return

    data = load_analysis("scaling_calibration")
    if not data.get("local") and not data.get("hpc"):
        print("  Fig 2: No calibration data available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Calibration Scaling: Local (~15 s evals) vs HPC Fir (~60 s evals)",
                 fontsize=13, fontweight="bold")

    # Top row: Local (ProcessPool vs MPI)
    local_records = data.get("local", [])
    if local_records:
        groups = {}
        for r in local_records:
            groups.setdefault(r["group"], []).append(r)

        for metric_idx, metric in enumerate(["wall_clock_seconds", "speedup", "efficiency"]):
            ax = axes[0, metric_idx]
            for group_name, grecs in groups.items():
                grecs.sort(key=lambda x: x["num_processes"])
                nps = [r["num_processes"] for r in grecs]
                vals = [r[metric] for r in grecs]
                ax.plot(nps, vals, marker=MARKERS.get(group_name, "o"),
                        color=COLORS.get(group_name, "#333"), label=group_name,
                        linewidth=1.5, markersize=5)

            if metric == "speedup":
                max_np = max(r["num_processes"] for r in local_records)
                ax.plot([1, max_np], [1, max_np], "--",
                        color=COLORS["ideal"], label="Ideal", linewidth=1)
            elif metric == "efficiency":
                ax.axhline(y=1.0, linestyle="--", color=COLORS["ideal"],
                           label="Ideal", linewidth=1)
                ax.set_ylim(0, 1.2)

            ax.set_xlabel("Number of Processes")
            ylabels = {
                "wall_clock_seconds": "Wall-clock Time (s)",
                "speedup": "Speedup",
                "efficiency": "Parallel Efficiency",
            }
            ax.set_ylabel(ylabels[metric])
            ax.set_title(f"Local (~15 s evals) \u2014 {ylabels[metric]}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)

    # Bottom row: HPC (DDS vs DE)
    hpc_records = data.get("hpc", [])
    if hpc_records:
        groups = {}
        for r in hpc_records:
            groups.setdefault(r["group"], []).append(r)

        for metric_idx, metric in enumerate(["wall_clock_seconds", "speedup", "efficiency"]):
            ax = axes[1, metric_idx]

            # For wall-clock, use minutes for HPC
            use_minutes = (metric == "wall_clock_seconds")

            for group_name, grecs in groups.items():
                grecs.sort(key=lambda x: x["num_processes"])
                nps = [r["num_processes"] for r in grecs]
                if use_minutes:
                    vals = [r["wall_clock_minutes"] for r in grecs]
                else:
                    vals = [r[metric] for r in grecs]
                ax.plot(nps, vals, marker=MARKERS.get(group_name, "o"),
                        color=COLORS.get(group_name, "#333"), label=group_name,
                        linewidth=1.5, markersize=5)

            if metric == "speedup":
                min_np = min(r["num_processes"] for r in hpc_records)
                max_np = max(r["num_processes"] for r in hpc_records)
                ax.plot([min_np, max_np], [1, max_np / min_np], "--",
                        color=COLORS["ideal"], label="Ideal", linewidth=1)
            elif metric == "efficiency":
                ax.axhline(y=1.0, linestyle="--", color=COLORS["ideal"],
                           label="Ideal", linewidth=1)
                ax.set_ylim(0, 1.4)

            ax.set_xlabel("Number of Cores")
            ylabels = {
                "wall_clock_seconds": "Wall-clock Time (min)",
                "speedup": "Speedup",
                "efficiency": "Parallel Efficiency",
            }
            ax.set_ylabel(ylabels[metric])
            ax.set_title(f"HPC Fir (~60 s evals) \u2014 {ylabels[metric]}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)

    fig.tight_layout()
    save_figure(fig, "fig2_calibration_scaling", fmt)


def fig3_combined_efficiency(fmt: str = "png"):
    """Figure 3: Combined efficiency comparison across all experiments."""
    if not HAS_MPL:
        return

    taudem = load_analysis("scaling_taudem")
    calib = load_analysis("scaling_calibration")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Parallel Efficiency Across Workflow Stages",
                 fontsize=14, fontweight="bold")

    plot_data = []

    # TauDEM local 90m
    for r in taudem.get("local", []):
        if r["group"] == "90m":
            plot_data.append(("TauDEM (Iceland 90 m)", r["num_processes"], r["efficiency"]))

    # Calibration local ProcessPool
    for r in calib.get("local", []):
        if r["group"] == "ProcessPool":
            plot_data.append(("Calibration PP (local)", r["num_processes"], r["efficiency"]))

    # Calibration local MPI
    for r in calib.get("local", []):
        if r["group"] == "MPI":
            plot_data.append(("Calibration MPI (local)", r["num_processes"], r["efficiency"]))

    # Calibration HPC DDS
    for r in calib.get("hpc", []):
        if r["group"] == "DDS":
            plot_data.append(("Calibration DDS (HPC)", r["num_processes"], r["efficiency"]))

    # Calibration HPC DE
    for r in calib.get("hpc", []):
        if r["group"] == "DE":
            plot_data.append(("Calibration DE (HPC)", r["num_processes"], r["efficiency"]))

    # Group and plot
    groups = {}
    for label, np_val, eff in plot_data:
        groups.setdefault(label, ([], []))
        groups[label][0].append(np_val)
        groups[label][1].append(eff)

    colors_list = ["#1f77b4", "#ff7f0e", "#e377c2", "#2ca02c", "#d62728"]
    markers_list = ["o", "s", "^", "D", "v"]
    for idx, (label, (nps, effs)) in enumerate(groups.items()):
        order = sorted(range(len(nps)), key=lambda i: nps[i])
        nps_sorted = [nps[i] for i in order]
        effs_sorted = [effs[i] for i in order]
        ax.plot(nps_sorted, effs_sorted,
                marker=markers_list[idx % len(markers_list)],
                color=colors_list[idx % len(colors_list)],
                label=label, linewidth=1.5, markersize=6)

    ax.axhline(y=1.0, linestyle="--", color="#cccccc", linewidth=1, label="Ideal")
    ax.set_xlabel("Number of Processors")
    ax.set_ylabel("Parallel Efficiency")
    ax.set_ylim(0, 1.4)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "fig3_combined_efficiency", fmt)


FIGURES = {
    1: ("TauDEM Delineation Scaling", fig1_taudem_scaling),
    2: ("Calibration Scaling (Local + HPC)", fig2_calibration_scaling),
    3: ("Combined Efficiency", fig3_combined_efficiency),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Section 4.13 parallel scaling figures"
    )
    parser.add_argument(
        "--figure", type=int, nargs="*", default=None,
        help="Figure numbers to generate (default: all)"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png",
        help="Output format (default: png)"
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        return

    setup_style()
    figures = args.figure or list(FIGURES.keys())

    print("=" * 60)
    print("  Section 4.13 Parallel Scaling Figures")
    print("=" * 60)

    for fig_num in figures:
        if fig_num not in FIGURES:
            print(f"  WARNING: Unknown figure {fig_num}")
            continue
        name, func = FIGURES[fig_num]
        print(f"\n  Figure {fig_num}: {name}")
        func(fmt=args.format)

    print(f"\n{'=' * 60}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
