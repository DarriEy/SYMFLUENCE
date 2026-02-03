#!/usr/bin/env python3
"""
Create publication-ready figures for Section 4.11 Parallel Scaling.

Generates four composite figures suitable for journal submission:
  Fig A: Strong scaling (ProcessPool + MPI)
  Fig B: Async-DDS convergence and worker utilization
  Fig C: JAX acceleration and composability
  Fig D: Weak scaling and ensemble Gantt chart

Usage:
    python create_publication_figures.py                  # PNG + PDF
    python create_publication_figures.py --format pdf     # PDF only
    python create_publication_figures.py --dpi 600        # High resolution
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

# Paths
STUDY_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = STUDY_DIR / "analysis"
PLOTS_DIR = STUDY_DIR / "results" / "plots"

# Publication style
COLORS = {
    "processpool": "#2196F3",
    "mpi": "#FF5722",
    "ideal": "#9E9E9E",
    "numpy": "#9E9E9E",
    "jax_cpu": "#FF9800",
    "jax_jit": "#2196F3",
    "jax_gpu": "#4CAF50",
    "sync": "#9E9E9E",
    "async": "#E91E63",
    "hbv": "#2196F3",
    "gr4j": "#4CAF50",
    "fuse": "#FF9800",
    "jfuse": "#9C27B0",
}

FONT_SIZE = 10
TICK_SIZE = 9
LABEL_SIZE = 11


def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_analysis_data() -> Dict[str, Any]:
    """Load pre-computed analysis results."""
    summary_path = ANALYSIS_DIR / "scaling_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run analyze_scaling.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        return json.load(f)


# =============================================================================
# Figure A: Strong Scaling
# =============================================================================

def create_fig_strong_scaling(data: Dict, output_dir: Path, fmt: str, dpi: int):
    """
    Figure A: Strong scaling speedup and efficiency.
    Panel (a): Speedup vs workers for ProcessPool and MPI
    Panel (b): Parallel efficiency vs workers
    """
    pp = data.get("processpool_scaling", [])
    mpi = data.get("mpi_scaling", [])

    if not pp and not mpi:
        print("WARNING: No strong-scaling data; skipping Figure A")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Speedup
    ax = axes[0]
    if pp:
        nps = [m["num_processes"] for m in pp]
        ax.plot(nps, [m["speedup"] for m in pp], "o-",
                color=COLORS["processpool"], label="ProcessPool",
                linewidth=2, markersize=6, zorder=3)
    if mpi:
        nps_m = [m["num_processes"] for m in mpi]
        ax.plot(nps_m, [m["speedup"] for m in mpi], "s--",
                color=COLORS["mpi"], label="MPI",
                linewidth=2, markersize=6, zorder=3)

    all_nps = sorted(set(
        [m["num_processes"] for m in pp] + [m["num_processes"] for m in mpi]
    ))
    ax.plot(all_nps, all_nps, ":", color=COLORS["ideal"],
            alpha=0.7, label="Ideal", linewidth=1.5)

    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup ($T_1 / T_n$)")
    ax.set_title("(a) Speedup")
    ax.legend(loc="upper left")
    if max(all_nps) > 8:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    # Panel (b): Efficiency
    ax = axes[1]
    if pp:
        nps = [m["num_processes"] for m in pp]
        ax.plot(nps, [m["efficiency"] * 100 for m in pp], "o-",
                color=COLORS["processpool"], label="ProcessPool",
                linewidth=2, markersize=6)
    if mpi:
        nps_m = [m["num_processes"] for m in mpi]
        ax.plot(nps_m, [m["efficiency"] * 100 for m in mpi], "s--",
                color=COLORS["mpi"], label="MPI",
                linewidth=2, markersize=6)

    ax.axhline(y=100, color=COLORS["ideal"], linestyle=":",
               alpha=0.7, label="Ideal (100%)", linewidth=1.5)
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("(b) Efficiency")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 110)
    if max(all_nps) > 8:
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_strong_scaling", fmt, dpi)


# =============================================================================
# Figure B: Async DDS
# =============================================================================

def create_fig_async_dds(data: Dict, output_dir: Path, fmt: str, dpi: int):
    """
    Figure B: Async vs Sync DDS comparison.
    Panel (a): Wall-clock time comparison bar chart
    Panel (b): Placeholder for convergence trajectories (populated from logs)
    """
    # This figure requires async/sync comparison data
    # For now, create the figure template with placeholder annotations
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.set_title("(a) Wall-clock time: Sync vs Async DDS")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Wall-clock time (s)")
    ax.text(0.5, 0.5, "Populate after running\nexperiment 3",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey", style="italic")

    ax = axes[1]
    ax.set_title("(b) Convergence trajectories")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best KGE")
    ax.text(0.5, 0.5, "Populate from\ncalibration logs",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey", style="italic")

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_async_dds", fmt, dpi)


# =============================================================================
# Figure C: JAX Acceleration
# =============================================================================

def create_fig_jax(data: Dict, output_dir: Path, fmt: str, dpi: int):
    """
    Figure C: JAX backend acceleration.
    Panel (a): Backend comparison bar chart
    Panel (b): JAX + ProcessPool composability speedup
    """
    jax_metrics = data.get("jax_acceleration", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Backend comparison
    ax = axes[0]
    if jax_metrics:
        labels = [m["label"] for m in jax_metrics]
        times = [m["wall_clock_median_s"] for m in jax_metrics]
        speedups = [m["speedup_vs_numpy"] for m in jax_metrics]

        bar_colors = []
        for label in labels:
            if "gpu" in label:
                bar_colors.append(COLORS["jax_gpu"])
            elif "jit" in label:
                bar_colors.append(COLORS["jax_jit"])
            elif "nojit" in label:
                bar_colors.append(COLORS["jax_cpu"])
            else:
                bar_colors.append(COLORS["numpy"])

        bars = ax.bar(labels, times, color=bar_colors,
                      edgecolor="black", linewidth=0.5)

        for bar, spd in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(times) * 0.02,
                    f"{spd:.1f}x", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_ylabel("Wall-clock time (s)")
        plt.sca(ax)
        plt.xticks(rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "Populate after running\nexperiment 4",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="grey", style="italic")

    ax.set_title("(a) Backend comparison")

    # Panel (b): Composability placeholder
    ax = axes[1]
    ax.set_title("(b) JAX JIT + ProcessPool")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup")
    ax.text(0.5, 0.5, "Populate after running\nexperiment 4",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey", style="italic")

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_jax_acceleration", fmt, dpi)


# =============================================================================
# Figure D: Weak Scaling + Ensemble
# =============================================================================

def create_fig_weak_ensemble(data: Dict, output_dir: Path, fmt: str, dpi: int):
    """
    Figure D: Weak scaling and ensemble execution.
    Panel (a): Time per evaluation vs domain complexity
    Panel (b): Ensemble execution Gantt chart (sequential vs parallel)
    """
    weak_metrics = data.get("weak_scaling", [])
    ensemble = data.get("ensemble", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Weak scaling
    ax = axes[0]
    if weak_metrics:
        valid = [m for m in weak_metrics if m.get("n_hrus", 0) > 0]
        if valid:
            hrus = [m["n_hrus"] for m in valid]
            times = [m["time_per_eval_s"] for m in valid]
            labels = [m["label"] for m in valid]

            ax.plot(hrus, times, "o-", color="#673AB7",
                    linewidth=2, markersize=8)
            for x, y, label in zip(hrus, times, labels):
                ax.annotate(label, (x, y),
                            textcoords="offset points", xytext=(10, 5),
                            fontsize=8, style="italic")

            ax.set_xscale("log")
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Populate after running\nexperiment 5",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="grey", style="italic")

    ax.set_xlabel("Number of HRUs")
    ax.set_ylabel("Time per evaluation (s)")
    ax.set_title("(a) Weak scaling")

    # Panel (b): Ensemble Gantt chart
    ax = axes[1]
    models = ["HBV", "GR4J", "FUSE", "jFUSE"]
    model_colors = [COLORS["hbv"], COLORS["gr4j"], COLORS["fuse"], COLORS["jfuse"]]

    if ensemble:
        # Sequential execution (stacked)
        y_seq = 1
        cumulative = 0
        for i, model in enumerate(models):
            key = f"{model.lower()}_sequential_s"
            duration = ensemble.get(key, 100)  # placeholder
            ax.barh(y_seq, duration, left=cumulative, height=0.4,
                    color=model_colors[i], edgecolor="black", linewidth=0.5)
            ax.text(cumulative + duration / 2, y_seq, model,
                    ha="center", va="center", fontsize=8, fontweight="bold")
            cumulative += duration

        # Parallel execution (simultaneous)
        y_par = 0
        max_duration = max(
            ensemble.get(f"{m.lower()}_sequential_s", 100) for m in models
        )
        for i, model in enumerate(models):
            key = f"{model.lower()}_sequential_s"
            duration = ensemble.get(key, 100)
            y_offset = y_par - 0.15 + i * 0.1
            ax.barh(y_offset, duration, left=0, height=0.08,
                    color=model_colors[i], edgecolor="black", linewidth=0.5)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Parallel", "Sequential"])
        ax.set_xlabel("Wall-clock time (s)")

        # Add speedup annotation
        if "ensemble_speedup" in ensemble:
            spd = ensemble["ensemble_speedup"]
            ax.text(0.95, 0.95, f"Speedup: {spd:.1f}x",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="orange"))
    else:
        ax.text(0.5, 0.5, "Populate after running\nexperiment 6",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="grey", style="italic")

    ax.set_title("(b) Ensemble execution timeline")

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_weak_ensemble", fmt, dpi)


# =============================================================================
# Utilities
# =============================================================================

def _save_fig(fig, base_path: Path, fmt: str, dpi: int):
    """Save figure in requested format(s)."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        path = base_path.with_suffix(".png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
    if fmt in ("pdf", "both"):
        path = base_path.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create publication figures for Section 4.11"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for PNG output (default: 300)"
    )
    args = parser.parse_args()

    setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_analysis_data()

    print("Generating publication figures...")
    create_fig_strong_scaling(data, PLOTS_DIR, args.format, args.dpi)
    create_fig_async_dds(data, PLOTS_DIR, args.format, args.dpi)
    create_fig_jax(data, PLOTS_DIR, args.format, args.dpi)
    create_fig_weak_ensemble(data, PLOTS_DIR, args.format, args.dpi)

    print(f"\nAll figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
