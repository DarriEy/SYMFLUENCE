#!/usr/bin/env python3
"""
Analyze results from the Section 4.11 Parallel Scaling Study.

Computes scaling metrics (speedup, efficiency, utilization) from raw
timing data and generates summary tables and diagnostic plots.

Usage:
    python analyze_scaling.py                         # Default analysis
    python analyze_scaling.py --input timing_raw.csv  # Specific input
    python analyze_scaling.py --output-dir ../results  # Custom output
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Paths
STUDY_DIR = Path(__file__).parent.parent
RESULTS_DIR = STUDY_DIR / "results"
ANALYSIS_DIR = STUDY_DIR / "analysis"


# =============================================================================
# Data loading
# =============================================================================

def load_timing_data(filepath: Path) -> List[Dict[str, Any]]:
    """Load raw timing CSV into list of dicts."""
    records = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["wall_clock_seconds", "num_processes", "run_index"]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                        if key in ("num_processes", "run_index"):
                            row[key] = int(row[key])
                    except ValueError:
                        pass
            row["success"] = row.get("success", "").lower() == "true"
            records.append(row)
    return records


def filter_records(
    records: List[Dict], experiment: str, success_only: bool = True
) -> List[Dict]:
    """Filter records by experiment name."""
    filtered = [r for r in records if r.get("experiment") == experiment]
    if success_only:
        filtered = [r for r in filtered if r.get("success")]
    return filtered


# =============================================================================
# Scaling metrics computation
# =============================================================================

def compute_strong_scaling_metrics(
    records: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Compute speedup and parallel efficiency from strong-scaling records.

    Expects records with 'num_processes' and 'wall_clock_seconds' fields.
    Uses the median wall-clock time across repeats per worker count.
    """
    # Group by num_processes
    by_np: Dict[int, List[float]] = {}
    for r in records:
        np_val = r["num_processes"]
        wc = r["wall_clock_seconds"]
        by_np.setdefault(np_val, []).append(wc)

    # Compute medians
    medians = {}
    for np_val, times in sorted(by_np.items()):
        times_sorted = sorted(times)
        n = len(times_sorted)
        if n % 2 == 0:
            median = (times_sorted[n // 2 - 1] + times_sorted[n // 2]) / 2
        else:
            median = times_sorted[n // 2]
        medians[np_val] = median

    # Baseline (np=1)
    t1 = medians.get(1)
    if t1 is None:
        print("WARNING: No single-process baseline found; using smallest np")
        min_np = min(medians.keys())
        t1 = medians[min_np] * min_np  # Estimate T1

    # Compute metrics
    results = []
    for np_val in sorted(medians.keys()):
        tn = medians[np_val]
        speedup = t1 / tn if tn > 0 else 0
        efficiency = speedup / np_val if np_val > 0 else 0
        overhead = 1.0 - efficiency

        results.append({
            "num_processes": np_val,
            "wall_clock_median_s": round(tn, 2),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 4),
            "overhead_fraction": round(overhead, 4),
            "n_runs": len(by_np[np_val]),
            "wall_clock_min_s": round(min(by_np[np_val]), 2),
            "wall_clock_max_s": round(max(by_np[np_val]), 2),
        })

    return results


def compute_jax_speedup(records: List[Dict]) -> List[Dict[str, Any]]:
    """Compute speedup relative to NumPy baseline for JAX experiments."""
    # Group by config file
    by_config: Dict[str, List[float]] = {}
    for r in records:
        cfg = r["config_file"]
        wc = r["wall_clock_seconds"]
        by_config.setdefault(cfg, []).append(wc)

    # Median per config
    medians = {}
    for cfg, times in by_config.items():
        times_sorted = sorted(times)
        n = len(times_sorted)
        medians[cfg] = times_sorted[n // 2]

    # Baseline = numpy
    numpy_time = medians.get("jax_numpy.yaml")
    if numpy_time is None:
        print("WARNING: NumPy baseline not found in JAX results")
        return []

    results = []
    for cfg in sorted(medians.keys()):
        tn = medians[cfg]
        speedup = numpy_time / tn if tn > 0 else 0
        # Extract backend label from filename
        label = cfg.replace("jax_", "").replace(".yaml", "")
        results.append({
            "config": cfg,
            "label": label,
            "wall_clock_median_s": round(tn, 2),
            "speedup_vs_numpy": round(speedup, 3),
        })

    return results


def compute_weak_scaling(records: List[Dict]) -> List[Dict[str, Any]]:
    """Compute per-evaluation time for weak-scaling experiments."""
    domain_hrus = {
        "weak_lumped.yaml": 1,
        "weak_elevation.yaml": 12,
        "weak_semidist.yaml": 379,
        "weak_distributed.yaml": 2335,
    }

    by_config: Dict[str, List[float]] = {}
    for r in records:
        cfg = r["config_file"]
        wc = r["wall_clock_seconds"]
        by_config.setdefault(cfg, []).append(wc)

    results = []
    for cfg in sorted(by_config.keys()):
        times = by_config[cfg]
        median_t = sorted(times)[len(times) // 2]
        nhrus = domain_hrus.get(cfg, 0)
        iterations = 500  # Fixed in weak-scaling configs
        time_per_eval = median_t / iterations if iterations > 0 else 0

        label = cfg.replace("weak_", "").replace(".yaml", "")
        results.append({
            "config": cfg,
            "label": label,
            "n_hrus": nhrus,
            "wall_clock_median_s": round(median_t, 2),
            "time_per_eval_s": round(time_per_eval, 4),
            "iterations": iterations,
        })

    return results


def compute_ensemble_comparison(records: List[Dict]) -> Dict[str, Any]:
    """Compare sequential vs parallel ensemble execution."""
    seq_total = [
        r for r in records
        if r.get("experiment") == "exp6_sequential_total"
    ]
    par_total = [
        r for r in records
        if r.get("experiment") == "exp6_parallel_total"
    ]

    result = {}
    if seq_total:
        result["sequential_total_s"] = seq_total[0]["wall_clock_seconds"]
    if par_total:
        result["parallel_total_s"] = par_total[0]["wall_clock_seconds"]
    if "sequential_total_s" in result and "parallel_total_s" in result:
        st = result["sequential_total_s"]
        pt = result["parallel_total_s"]
        result["ensemble_speedup"] = round(st / pt, 3) if pt > 0 else 0

    # Per-model times (sequential)
    seq_models = [
        r for r in records if r.get("experiment") == "exp6_sequential"
    ]
    for r in seq_models:
        model = r["config_file"].replace("ensemble_", "").replace(".yaml", "")
        result[f"{model}_sequential_s"] = r["wall_clock_seconds"]

    return result


# =============================================================================
# Output
# =============================================================================

def save_csv(data: List[Dict], filepath: Path) -> None:
    """Save list of dicts to CSV."""
    if not data:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(data[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved: {filepath}")


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def print_table(data: List[Dict], title: str) -> None:
    """Print a formatted table to stdout."""
    if not data:
        print(f"\n{title}: No data available")
        return

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    keys = list(data[0].keys())
    widths = {k: max(len(k), max(len(str(row.get(k, ""))) for row in data)) for k in keys}

    header = " | ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in data:
        line = " | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys)
        print(line)


# =============================================================================
# Plotting (optional, requires matplotlib)
# =============================================================================

def plot_strong_scaling(
    processpool_metrics: List[Dict],
    mpi_metrics: List[Dict],
    output_path: Path,
) -> None:
    """Plot strong-scaling speedup curves."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Speedup
    ax = axes[0]
    if processpool_metrics:
        nps = [m["num_processes"] for m in processpool_metrics]
        speedups = [m["speedup"] for m in processpool_metrics]
        ax.plot(nps, speedups, "o-", color="#2196F3", label="ProcessPool", linewidth=2)

    if mpi_metrics:
        nps = [m["num_processes"] for m in mpi_metrics]
        speedups = [m["speedup"] for m in mpi_metrics]
        ax.plot(nps, speedups, "s--", color="#FF5722", label="MPI", linewidth=2)

    # Ideal scaling reference
    all_nps = sorted(set(
        [m["num_processes"] for m in processpool_metrics] +
        [m["num_processes"] for m in mpi_metrics]
    ))
    if all_nps:
        ax.plot(all_nps, all_nps, ":", color="grey", alpha=0.6, label="Ideal")

    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup (T$_1$ / T$_n$)")
    ax.set_title("(a) Strong scaling speedup")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # Panel (b): Parallel efficiency
    ax = axes[1]
    if processpool_metrics:
        nps = [m["num_processes"] for m in processpool_metrics]
        effs = [m["efficiency"] * 100 for m in processpool_metrics]
        ax.plot(nps, effs, "o-", color="#2196F3", label="ProcessPool", linewidth=2)

    if mpi_metrics:
        nps = [m["num_processes"] for m in mpi_metrics]
        effs = [m["efficiency"] * 100 for m in mpi_metrics]
        ax.plot(nps, effs, "s--", color="#FF5722", label="MPI", linewidth=2)

    ax.axhline(y=100, color="grey", linestyle=":", alpha=0.6, label="Ideal (100%)")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("(b) Parallel efficiency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_jax_acceleration(
    jax_metrics: List[Dict],
    output_path: Path,
) -> None:
    """Plot JAX backend comparison."""
    if not HAS_MATPLOTLIB or not jax_metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [m["label"] for m in jax_metrics]
    times = [m["wall_clock_median_s"] for m in jax_metrics]
    speedups = [m["speedup_vs_numpy"] for m in jax_metrics]

    colors = []
    for label in labels:
        if "gpu" in label:
            colors.append("#4CAF50")
        elif "jit" in label:
            colors.append("#2196F3")
        elif "nojit" in label:
            colors.append("#FF9800")
        else:
            colors.append("#9E9E9E")

    bars = ax.bar(labels, times, color=colors, edgecolor="black", linewidth=0.5)

    # Add speedup labels above bars
    for bar, spd in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{spd:.1f}x",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("JAX Backend Comparison (1,000 DDS iterations)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_weak_scaling(
    weak_metrics: List[Dict],
    output_path: Path,
) -> None:
    """Plot time per evaluation vs domain complexity."""
    if not HAS_MATPLOTLIB or not weak_metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    hrus = [m["n_hrus"] for m in weak_metrics if m["n_hrus"] > 0]
    times = [
        m["time_per_eval_s"]
        for m in weak_metrics
        if m["n_hrus"] > 0
    ]
    labels = [m["label"] for m in weak_metrics if m["n_hrus"] > 0]

    ax.plot(hrus, times, "o-", color="#673AB7", linewidth=2, markersize=8)

    for x, y, label in zip(hrus, times, labels):
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(10, 5),
            fontsize=9, style="italic",
        )

    ax.set_xlabel("Number of HRUs")
    ax.set_ylabel("Time per model evaluation (s)")
    ax.set_title("Weak Scaling: Domain Complexity vs Evaluation Cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main analysis pipeline
# =============================================================================

def run_analysis(input_path: Path, output_dir: Path) -> None:
    """Run the full analysis pipeline."""
    print(f"Loading timing data from: {input_path}")
    records = load_timing_data(input_path)
    print(f"Loaded {len(records)} records")

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- 4.11.1: ProcessPool strong scaling ---
    pp_records = filter_records(records, "exp1_processpool")
    pp_metrics = compute_strong_scaling_metrics(pp_records) if pp_records else []
    if pp_metrics:
        save_csv(pp_metrics, output_dir / "strong_scaling_processpool.csv")
        print_table(pp_metrics, "4.11.1 ProcessPool Strong Scaling")

    # --- 4.11.2: MPI strong scaling ---
    mpi_records = filter_records(records, "exp2_mpi")
    mpi_metrics = compute_strong_scaling_metrics(mpi_records) if mpi_records else []
    if mpi_metrics:
        save_csv(mpi_metrics, output_dir / "strong_scaling_mpi.csv")
        print_table(mpi_metrics, "4.11.2 MPI Strong Scaling")

    # Combined strong scaling plot
    if pp_metrics or mpi_metrics:
        plot_strong_scaling(
            pp_metrics, mpi_metrics,
            plots_dir / "fig_strong_scaling.png",
        )

    # --- 4.11.3: Async DDS ---
    sync_records = filter_records(records, "exp3_sync_dds")
    async_records = filter_records(records, "exp3_async_dds")
    all_dds = sync_records + async_records
    if all_dds:
        dds_summary = []
        for r in all_dds:
            dds_summary.append({
                "config": r["config_file"],
                "experiment": r["experiment"],
                "wall_clock_s": r["wall_clock_seconds"],
                "num_processes": r["num_processes"],
                "best_kge": r.get("best_kge", ""),
            })
        save_csv(dds_summary, output_dir / "async_vs_sync_dds.csv")
        print_table(dds_summary, "4.11.3 Async vs Sync DDS")

    # --- 4.11.4: JAX acceleration ---
    jax_backend_records = filter_records(records, "exp4_jax_backend")
    jax_metrics = compute_jax_speedup(jax_backend_records) if jax_backend_records else []
    if jax_metrics:
        save_csv(jax_metrics, output_dir / "jax_acceleration.csv")
        print_table(jax_metrics, "4.11.4 JAX Acceleration")
        plot_jax_acceleration(jax_metrics, plots_dir / "fig_jax_acceleration.png")

    # JAX composability
    jax_comp_records = filter_records(records, "exp4_jax_composability")
    if jax_comp_records:
        comp_metrics = compute_strong_scaling_metrics(jax_comp_records)
        save_csv(comp_metrics, output_dir / "jax_composability.csv")
        print_table(comp_metrics, "4.11.4 JAX + ProcessPool Composability")

    # --- 4.11.5: Weak scaling ---
    weak_records = filter_records(records, "exp5_weak_scaling")
    weak_metrics = compute_weak_scaling(weak_records) if weak_records else []
    if weak_metrics:
        save_csv(weak_metrics, output_dir / "weak_scaling.csv")
        print_table(weak_metrics, "4.11.5 Weak Scaling")
        plot_weak_scaling(weak_metrics, plots_dir / "fig_weak_scaling.png")

    # --- 4.11.6: Ensemble parallelism ---
    ensemble_comparison = compute_ensemble_comparison(records)
    if ensemble_comparison:
        save_json(ensemble_comparison, output_dir / "ensemble_comparison.json")
        print(f"\n{'=' * 70}")
        print("  4.11.6 Ensemble Parallelism")
        print(f"{'=' * 70}")
        for k, v in ensemble_comparison.items():
            print(f"  {k}: {v}")

    # --- Combined summary ---
    summary = {
        "processpool_scaling": pp_metrics,
        "mpi_scaling": mpi_metrics,
        "jax_acceleration": jax_metrics,
        "weak_scaling": weak_metrics,
        "ensemble": ensemble_comparison,
    }
    save_json(summary, output_dir / "scaling_summary.json")

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete. Results in: {output_dir}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Section 4.11 Parallel Scaling results"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input CSV path (default: results/timing_raw.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: analysis/)"
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / "timing_raw.csv"
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run run_scaling_study.py first to generate timing data.")
        sys.exit(1)

    run_analysis(input_path, output_dir)


if __name__ == "__main__":
    main()
