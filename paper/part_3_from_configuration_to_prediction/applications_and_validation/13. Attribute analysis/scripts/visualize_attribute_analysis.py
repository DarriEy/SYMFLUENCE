#!/usr/bin/env python3
"""
Attribute Analysis Visualization for SYMFLUENCE Paper Section 4.13

Generates publication-quality figures for the attribute analysis experiment:

  Figure 1: HRU discretization maps - visual comparison of HRU layouts per scenario
  Figure 2: Complexity-performance trade-off - HRU count vs KGE scatter
  Figure 3: Calibration convergence - KGE trajectories across scenarios
  Figure 4: Performance bar chart - KGE decomposition (r, alpha, beta) per scenario
  Figure 5: Attribute statistics - elevation/soil/land cover distributions

Usage:
    python visualize_attribute_analysis.py [--output-dir PATH] [--format png|pdf|svg]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = DATA_DIR / "domain_Bow_at_Banff_attribute_analysis"

# Scenario metadata (label, color, marker)
SCENARIO_META = {
    "lumped_baseline":  {"label": "Lumped",         "color": "#636363", "marker": "o", "cat": "baseline"},
    "elevation_200m":   {"label": "Elev 200m",      "color": "#1b9e77", "marker": "s", "cat": "single"},
    "elevation_400m":   {"label": "Elev 400m",      "color": "#66c2a5", "marker": "s", "cat": "single"},
    "landclass":        {"label": "Land Cover",     "color": "#d95f02", "marker": "^", "cat": "single"},
    "soilclass":        {"label": "Soil Type",      "color": "#7570b3", "marker": "D", "cat": "single"},
    "aspect":           {"label": "Aspect",         "color": "#e7298a", "marker": "v", "cat": "single"},
    "radiation":        {"label": "Radiation",      "color": "#e6ab02", "marker": "P", "cat": "single"},
    "elev_land":        {"label": "Elev+Land",      "color": "#a6761d", "marker": "X", "cat": "combined"},
    "elev_soil_land":   {"label": "Elev+Soil+Land", "color": "#1f78b4", "marker": "*", "cat": "combined"},
}

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_csv_data(csv_path: Path) -> List[Dict]:
    """Load CSV file as list of dictionaries."""
    import csv
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_discretization_data() -> Dict[str, Dict]:
    """Load discretization summary data."""
    rows = load_csv_data(ANALYSIS_DIR / "discretization_summary.csv")
    return {row["scenario"]: row for row in rows}


def load_performance_data() -> Dict[str, Dict]:
    """Load performance comparison data."""
    rows = load_csv_data(ANALYSIS_DIR / "performance_comparison.csv")
    return {row["scenario"]: row for row in rows}


def load_convergence_data() -> Dict[str, Dict]:
    """Load convergence summary data."""
    rows = load_csv_data(ANALYSIS_DIR / "convergence_summary.csv")
    return {row["scenario"]: row for row in rows}


def load_convergence_trajectories() -> Dict[str, np.ndarray]:
    """Load full convergence trajectories from optimization results."""
    trajectories = {}
    for scenario_name, meta in SCENARIO_META.items():
        exp_id = f"attr_{scenario_name}"
        opt_dir = DOMAIN_DIR / "optimization"
        if not opt_dir.exists():
            continue
        for csv_file in opt_dir.rglob(f"*{exp_id}*iteration_results*.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if "KGE" in df.columns:
                    trajectories[scenario_name] = np.maximum.accumulate(df["KGE"].values)
                break
            except Exception:
                continue
    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: HRU discretization maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_hru_maps(output_dir: Path, fmt: str = "png"):
    """Plot HRU boundary maps for each discretization scenario."""
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available, skipping HRU map figure")
        return

    hru_dir = DOMAIN_DIR / "shapefiles" / "catchment"
    if not hru_dir.exists():
        print("No shapefile directory found, skipping HRU map figure")
        return

    scenarios_with_shp = {}
    for scenario_name in SCENARIO_META:
        exp_id = f"attr_{scenario_name}"
        for pattern in [f"*{exp_id}*HRUs*.shp", f"*{exp_id}*HRUs*.gpkg"]:
            for shp in hru_dir.rglob(pattern):
                scenarios_with_shp[scenario_name] = shp
                break

    if not scenarios_with_shp:
        print("No HRU shapefiles found, skipping map figure")
        return

    n = len(scenarios_with_shp)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, shp_path) in enumerate(scenarios_with_shp.items()):
        ax = axes[i]
        meta = SCENARIO_META[name]
        try:
            gdf = gpd.read_file(shp_path)
            gdf.plot(ax=ax, edgecolor="black", linewidth=0.5, color=meta["color"], alpha=0.6)
            ax.set_title(f"{meta['label']} ({len(gdf)} HRUs)", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center")
            ax.set_title(meta["label"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attribute-Based Domain Discretization: Bow River at Banff", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"fig_hru_maps.{fmt}"
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Complexity-performance trade-off
# ─────────────────────────────────────────────────────────────────────────────

def plot_complexity_performance(output_dir: Path, fmt: str = "png"):
    """Scatter plot: HRU count vs calibrated KGE, coloured by category."""
    disc = load_discretization_data()
    perf = load_performance_data()

    if not disc or not perf:
        print("Insufficient data for complexity-performance plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for name in SCENARIO_META:
        if name not in disc or name not in perf:
            continue
        meta = SCENARIO_META[name]
        n_hrus = int(disc[name].get("n_hrus", 0))
        kge_str = perf[name].get("calibration_best_kge", "")
        if not kge_str:
            continue
        kge = float(kge_str)

        ax.scatter(n_hrus, kge, s=120, color=meta["color"], marker=meta["marker"],
                   edgecolors="black", linewidths=0.5, zorder=5, label=meta["label"])

    ax.set_xlabel("Number of HRUs")
    ax.set_ylabel("Calibration KGE")
    ax.set_title("Complexity-Performance Trade-off")
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    out_path = output_dir / f"fig_complexity_performance.{fmt}"
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Calibration convergence trajectories
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(output_dir: Path, fmt: str = "png"):
    """Plot cumulative best KGE trajectories for all scenarios."""
    trajectories = load_convergence_trajectories()

    if not trajectories:
        print("No convergence data available, skipping convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, traj in trajectories.items():
        meta = SCENARIO_META.get(name, {"label": name, "color": "gray"})
        ax.plot(range(1, len(traj) + 1), traj, color=meta["color"],
                linewidth=1.5, alpha=0.8, label=meta["label"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Best KGE")
    ax.set_title("Calibration Convergence Across Discretization Scenarios")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"fig_convergence.{fmt}"
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Performance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_bars(output_dir: Path, fmt: str = "png"):
    """Grouped bar chart: KGE (and components if available) per scenario."""
    perf = load_performance_data()

    if not perf:
        print("No performance data available, skipping bar chart")
        return

    ordered = [name for name in SCENARIO_META if name in perf]
    labels = [SCENARIO_META[name]["label"] for name in ordered]
    colors = [SCENARIO_META[name]["color"] for name in ordered]

    # Extract calibration KGE
    cal_kge = []
    eval_kge = []
    for name in ordered:
        ck = perf[name].get("calibration_best_kge", "")
        ek = perf[name].get("evaluation_kge", "")
        cal_kge.append(float(ck) if ck else 0)
        eval_kge.append(float(ek) if ek else 0)

    has_eval = any(v > 0 for v in eval_kge)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered))
    width = 0.35

    if has_eval:
        ax.bar(x - width / 2, cal_kge, width, color=colors, edgecolor="black",
               linewidth=0.5, alpha=0.8, label="Calibration")
        ax.bar(x + width / 2, eval_kge, width, color=colors, edgecolor="black",
               linewidth=0.5, alpha=0.5, hatch="//", label="Evaluation")
        ax.legend()
    else:
        ax.bar(x, cal_kge, width * 1.5, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("KGE")
    ax.set_title("Model Performance Across Discretization Scenarios")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=min(0, min(cal_kge) - 0.1))

    plt.tight_layout()
    out_path = output_dir / f"fig_performance_bars.{fmt}"
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Attribute statistics overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_attribute_overview(output_dir: Path, fmt: str = "png"):
    """Summary panel showing domain attribute distributions."""
    zonal_path = ANALYSIS_DIR / "zonal_statistics.json"
    if not zonal_path.exists():
        print("No zonal statistics data, skipping attribute overview")
        return

    with open(zonal_path) as f:
        zonal = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Elevation panel
    ax = axes[0]
    elev_vals = {k: zonal.get(k, 0) for k in ["elev_mean", "elev_min", "elev_max"]}
    if any(v > 0 for v in elev_vals.values()):
        bars = ax.bar(["Min", "Mean", "Max"],
                      [elev_vals["elev_min"], elev_vals["elev_mean"], elev_vals["elev_max"]],
                      color=["#1b9e77", "#d95f02", "#7570b3"], edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Elevation Statistics")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Elevation Statistics")

    # Soil classes panel
    ax = axes[1]
    n_soil = zonal.get("soil_classes", 0)
    dominant = zonal.get("dominant_soil", "N/A")
    ax.text(0.5, 0.6, f"{n_soil} classes", ha="center", va="center",
            transform=ax.transAxes, fontsize=24, fontweight="bold")
    ax.text(0.5, 0.3, f"Dominant: {dominant}", ha="center", va="center",
            transform=ax.transAxes, fontsize=11)
    ax.set_title("Soil Classes")
    ax.set_xticks([])
    ax.set_yticks([])

    # Land cover panel
    ax = axes[2]
    n_land = zonal.get("land_classes", 0)
    dominant_lc = zonal.get("dominant_landcover", "N/A")
    ax.text(0.5, 0.6, f"{n_land} classes", ha="center", va="center",
            transform=ax.transAxes, fontsize=24, fontweight="bold")
    ax.text(0.5, 0.3, f"Dominant: {dominant_lc}", ha="center", va="center",
            transform=ax.transAxes, fontsize=11)
    ax.set_title("Land Cover Classes")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Bow at Banff: Geospatial Attribute Summary", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / f"fig_attribute_overview.{fmt}"
    fig.savefig(out_path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for attribute analysis (Section 4.13)"
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating attribute analysis figures...")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}\n")

    plot_hru_maps(output_dir, args.format)
    plot_complexity_performance(output_dir, args.format)
    plot_convergence(output_dir, args.format)
    plot_performance_bars(output_dir, args.format)
    plot_attribute_overview(output_dir, args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
