#!/usr/bin/env python3
"""
Data Processing Pipeline Visualisation for SYMFLUENCE Paper Section 4.12

Creates 5 publication-quality figures tracing the pipeline across three
canonical scales: Paradise (point), Bow at Banff (watershed), Iceland (regional).

  Figure 1 -- Layered DAG with swim-lanes per category, curved edges
  Figure 2 -- Data flow Sankey (Raw Sources -> Remapping -> Standardisation -> Model-Ready)
  Figure 3 -- Multi-scale panel (3 rows x 4 cols: info, weights, compression, storage)
  Figure 4 -- Observation timeline (Gantt-style, 3 domain groups)
  Figure 5 -- Scaling law (log-log GRU count vs data volume and compression)

Usage:
    python visualize_pipeline.py [--analysis-dir DIR] [--format png|pdf|svg]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_visualization")

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Three canonical domains
# n_hrus = hydrological response units (forcing remapping target dimension)
# n_grus = grouped response units (sub-basins); GRUs contain one or more HRUs
DOMAINS = {
    "paradise": {
        "dir_name": "paradise",
        "dir_name_fallbacks": ["paradise_snotel_wa_era5", "paradise_multivar"],
        "label": "Paradise SNOTEL",
        "scale": "point",
        "n_hrus": 1,
        "n_grus": 1,
        "raw_grid_cells": 9,
        "area_km2": 0.01,
        "sections": ["4.3", "4.10"],
    },
    "bow": {
        "dir_name": "Bow_at_Banff_semi_distributed",
        "dir_name_fallbacks": ["Bow_at_Banff_lumped_era5", "Bow_at_Banff_multivar"],
        "label": "Bow at Banff",
        "scale": "watershed",
        "n_hrus": 49,
        "n_grus": 49,
        "raw_grid_cells": 42,
        "area_km2": 2210,
        "sections": ["4.2", "4.4-4.7"],
    },
    "iceland": {
        "dir_name": "Iceland",
        "dir_name_fallbacks": ["ellioaar_iceland", "Iceland_multivar"],
        "label": "Iceland",
        "scale": "regional",
        "n_hrus": 21474,
        "n_grus": 6600,
        "raw_grid_cells": 954,
        "area_km2": 103000,
        "sections": ["4.8", "4.9"],
    },
}


def resolve_domain_dir(domain_info: dict) -> Path:
    """Resolve domain directory with fallbacks."""
    primary = DATA_DIR / f"domain_{domain_info['dir_name']}"
    if primary.exists():
        return primary
    for fb in domain_info.get("dir_name_fallbacks", []):
        fallback = DATA_DIR / f"domain_{fb}"
        if fallback.exists():
            return fallback
    return primary


def resolve_observation_dir(domain_info: dict) -> Path:
    """Find best directory for observation data (richest obs)."""
    candidates = [domain_info["dir_name"]] + domain_info.get("dir_name_fallbacks", [])
    best_dir = None
    best_count = -1
    for name in candidates:
        obs_dir = DATA_DIR / f"domain_{name}" / "observations"
        if obs_dir.exists():
            count = sum(1 for d in obs_dir.iterdir()
                        if d.is_dir() and (any(d.rglob("*.csv")) or any(d.rglob("*.nc"))))
            if count > best_count:
                best_count = count
                best_dir = obs_dir.parent
    return best_dir if best_dir else DATA_DIR / f"domain_{domain_info['dir_name']}"

# Consistent domain colours across all figures
DOMAIN_COLORS = {
    "paradise": "#2ca02c",   # green
    "bow":      "#1f77b4",   # blue
    "iceland":  "#d62728",   # red
}

# Category colours for DAG
CATEGORY_COLORS = {
    "setup":       "#999999",
    "geospatial":  "#4C72B0",
    "forcing":     "#DD8452",
    "observation":  "#55A868",
    "model_setup": "#C44E52",
}

# Data product type colours for DAG edges
PRODUCT_COLORS = {
    "shapefile": "#4C72B0",
    "raster": "#8B4513",
    "netcdf": "#DD8452",
    "csv": "#55A868",
    "config": "#999999",
}

def _product_color(data_product: str) -> str:
    """Map data product name to edge colour."""
    dp = data_product.lower()
    if "shapefile" in dp or "shp" in dp:
        return PRODUCT_COLORS["shapefile"]
    if "raster" in dp or "dem" in dp or "landcover" in dp or "soilclass" in dp:
        return PRODUCT_COLORS["raster"]
    if "nc" in dp or "forcing" in dp or "weight" in dp or "cfif" in dp:
        return PRODUCT_COLORS["netcdf"]
    if "csv" in dp or "streamflow" in dp or "obs" in dp:
        return PRODUCT_COLORS["csv"]
    return PRODUCT_COLORS["config"]


# ---------------------------------------------------------------- helpers
def load_latest_analysis(analysis_dir: Path) -> Optional[Dict]:
    """Load the most recent pipeline_analysis JSON."""
    analyses = sorted(analysis_dir.glob("pipeline_analysis_*.json"), reverse=True)
    if not analyses:
        return None
    with open(analyses[0]) as f:
        return json.load(f)


# ============================================================
# Figure 1: Layered DAG with swim-lanes
# ============================================================
def fig_layered_dag(analysis: Dict, save_path: Path, fmt: str = "png"):
    """
    Clean top-to-bottom layered DAG with swim-lanes per category,
    curved edges coloured by data product type, and a banner showing
    which domains the pipeline is executed for.
    """
    dag = analysis.get("stage_dag", {})
    stages = dag.get("stages", [])
    edges = dag.get("edges", [])
    categories_info = dag.get("categories", {})

    if not stages:
        logger.warning("No DAG data found; skipping DAG figure")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Swim-lane columns (left to right)
    lane_order = ["setup", "geospatial", "forcing", "observation", "model_setup"]
    lane_x = {cat: i * 2.5 for i, cat in enumerate(lane_order)}
    lane_width = 2.0

    # Draw swim-lane backgrounds
    for cat in lane_order:
        x = lane_x[cat]
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        rect = mpatches.FancyBboxPatch(
            (x - lane_width / 2, -0.3), lane_width, 8.3,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.07, edgecolor=color,
            linewidth=1.0, linestyle="--",
        )
        ax.add_patch(rect)
        info = categories_info.get(cat, {})
        label = info.get("label", cat)
        ax.text(x, 8.2, label, ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=color, alpha=0.8)

    # Group stages by category, assign vertical positions
    category_stages = {}
    for s in stages:
        cat = s["category"]
        if cat not in category_stages:
            category_stages[cat] = []
        category_stages[cat].append(s)

    positions = {}
    for cat, cat_stages in category_stages.items():
        x = lane_x.get(cat, 5)
        n = len(cat_stages)
        for i, s in enumerate(cat_stages):
            y = 7.5 - (i * 7.5 / max(n, 1)) - (7.5 / max(n, 1)) / 2
            positions[s["id"]] = (x, y)

    # Draw edges (behind boxes) with curved connections coloured by product type
    for edge in edges:
        if edge["from"] in positions and edge["to"] in positions:
            x1, y1 = positions[edge["from"]]
            x2, y2 = positions[edge["to"]]
            ecolor = _product_color(edge.get("data_product", ""))
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ecolor,
                    lw=1.2,
                    connectionstyle="arc3,rad=0.15",
                    alpha=0.6,
                ),
            )

    # Draw stage boxes
    box_w, box_h = 1.8, 0.55
    for s in stages:
        if s["id"] not in positions:
            continue
        x, y = positions[s["id"]]
        cat = s["category"]
        color = CATEGORY_COLORS.get(cat, "#cccccc")

        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.88,
        )
        ax.add_patch(rect)

        label = s["label"]
        if len(label) > 20:
            words = label.split()
            mid = len(words) // 2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white",
                path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])

    # Banner: domains executed
    banner_y = -0.8
    ax.text(5.0, banner_y, "Executed for:", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333")
    for i, (dk, di) in enumerate(DOMAINS.items()):
        xb = 2.5 + i * 2.5
        color = DOMAIN_COLORS[dk]
        ax.add_patch(mpatches.FancyBboxPatch(
            (xb - 1.0, banner_y - 0.25), 2.0, 0.5,
            boxstyle="round,pad=0.08", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5,
        ))
        ax.text(xb, banner_y, f"{di['label']}\n({di['scale']}, {di['n_hrus']:,} HRUs)",
                ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    # Edge colour legend
    edge_handles = [
        mpatches.Patch(color=c, label=l, alpha=0.7)
        for l, c in [("Shapefile/vector", PRODUCT_COLORS["shapefile"]),
                      ("Raster", PRODUCT_COLORS["raster"]),
                      ("NetCDF", PRODUCT_COLORS["netcdf"]),
                      ("CSV/tabular", PRODUCT_COLORS["csv"]),
                      ("Configuration", PRODUCT_COLORS["config"])]
    ]
    ax.legend(handles=edge_handles, loc="lower left", fontsize=8, frameon=True,
              fancybox=True, framealpha=0.9, title="Data flow type", title_fontsize=9)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.5, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Data processing pipeline: stage dependency graph", fontsize=13, pad=15)

    fig.tight_layout()
    out = save_path / f"fig_layered_dag.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 2: Data Flow Sankey
# ============================================================
def fig_data_flow_sankey(analysis: Dict, save_path: Path, fmt: str = "png"):
    """
    Sankey/alluvial diagram showing data volumes flowing through 4 pipeline
    columns (Raw Sources -> Remapping -> Standardisation -> Model-Ready)
    with 3 colour-coded domain bands.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    columns = ["Raw\nSources", "Spatial\nRemapping", "Variable\nStandardisation", "Model-\nReady"]
    col_x = [0, 3, 6, 9]

    # Draw column headers
    for x, label in zip(col_x, columns):
        ax.text(x, 6.5, label, ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                          edgecolor="#cccccc", alpha=0.9))

    # Collect per-domain volumes at each stage
    domain_keys = ["paradise", "bow", "iceland"]
    stage_map = {
        0: "Forcing (raw)",
        1: "Remapping weights",
        2: "Forcing (basin-averaged)",
        3: "Forcing (merged)",
    }

    domain_volumes = {}  # domain_key -> [vol_raw, vol_remap, vol_ba, vol_merged]
    for dk in domain_keys:
        dr = analysis.get(dk, {})
        vols = dr.get("data_volumes", {})
        v = []
        for stage_idx in range(4):
            cat = stage_map[stage_idx]
            vol_bytes = vols.get(cat, {}).get("total_bytes", 0)
            v.append(vol_bytes)
        domain_volumes[dk] = v

    # Use log-scale normalisation so small domains remain visible
    # Map bytes to bar height via log10, with minimum visible height
    def _log_height(nbytes, max_h=1.2, min_h=0.15):
        if nbytes <= 0:
            return min_h * 0.5
        # log10 range: ~3 (1 KB) to ~10 (10 GB)
        log_val = np.log10(max(nbytes, 1))
        return max(min_h, min(max_h, log_val / 10.0 * max_h))

    for di, dk in enumerate(domain_keys):
        color = DOMAIN_COLORS[dk]
        vols = domain_volumes[dk]
        y_base = 4.5 - di * 2.0  # vertical offset per domain
        info = DOMAINS[dk]

        # Draw domain label
        ax.text(-1.5, y_base, f"{info['label']}\n({info['scale']})",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=color)

        for ci in range(4):
            x = col_x[ci]
            h = _log_height(vols[ci])
            bar = mpatches.FancyBboxPatch(
                (x - 0.4, y_base - h / 2), 0.8, h,
                boxstyle="round,pad=0.03",
                facecolor=color, alpha=0.7, edgecolor="white", linewidth=1,
            )
            ax.add_patch(bar)

            # Volume label
            vol_mb = vols[ci] / (1024 * 1024)
            if vol_mb >= 1:
                label = f"{vol_mb:.1f}\nMB"
            elif vols[ci] >= 1024:
                label = f"{vols[ci] / 1024:.0f}\nKB"
            elif vols[ci] > 0:
                label = f"{vols[ci]}\nB"
            else:
                label = "—"
            ax.text(x, y_base, label, ha="center", va="center", fontsize=7,
                    color="white" if h > 0.3 else color, fontweight="bold")

            # Flow arrow to next column
            if ci < 3:
                next_x = col_x[ci + 1]
                ax.annotate(
                    "",
                    xy=(next_x - 0.45, y_base),
                    xytext=(x + 0.45, y_base),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=max(1, h * 2),
                        alpha=0.4,
                        connectionstyle="arc3,rad=0.0",
                    ),
                )

    ax.set_xlim(-3, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")
    ax.set_title("Data volume flow through pipeline stages", fontsize=13, pad=15)

    fig.tight_layout()
    out = save_path / f"fig_data_flow_sankey.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 3: Multi-scale Panel (3 rows x 4 cols)
# ============================================================
def fig_multiscale_panel(analysis: Dict, save_path: Path, fmt: str = "png"):
    """
    3-row x 4-col panel: each row = one domain.
    Cols: (a) domain info card, (b) weight matrix structure,
          (c) compression butterfly chart, (d) storage footprint stacked bar.
    """
    domain_keys = ["paradise", "bow", "iceland"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for row, dk in enumerate(domain_keys):
        color = DOMAIN_COLORS[dk]
        info = DOMAINS[dk]
        dr = analysis.get(dk, {})

        # --- Col 0: Domain info card ---
        ax = axes[row, 0]
        ax.axis("off")
        card_text = (
            f"{info['label']}\n"
            f"Scale: {info['scale']}\n"
            f"HRUs: {info['n_hrus']:,}\n"
            f"GRUs: {info['n_grus']:,}\n"
            f"Area: {info['area_km2']:,.0f} km$^2$\n"
            f"Sections: {', '.join(info['sections'])}"
        )
        ax.text(0.5, 0.5, card_text, ha="center", va="center",
                fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color,
                          alpha=0.12, edgecolor=color, linewidth=2))
        if row == 0:
            ax.set_title("Domain", fontsize=11, fontweight="bold")

        # --- Col 1: Weight matrix structure (use real data where available) ---
        ax = axes[row, 1]
        domain_dir = resolve_domain_dir(info)
        intersect_dir = domain_dir / "shapefiles" / "catchment_intersection" / "with_forcing"
        # Prefer the intersected_shapefile CSV which has AP1N weights
        csv_files = sorted(intersect_dir.glob("*intersected_shapefile*.csv")) if intersect_dir.exists() else []
        if not csv_files:
            csv_files = sorted(intersect_dir.glob("*.csv")) if intersect_dir.exists() else []
        real_matrix_plotted = False

        if csv_files:
            try:
                idf = pd.read_csv(csv_files[0])
                if "S_1_HRU_ID" in idf.columns and "S_2_ID" in idf.columns and "AP1N" in idf.columns:
                    hru_ids = sorted(idf["S_1_HRU_ID"].unique())
                    grid_ids = sorted(idf["S_2_ID"].unique())
                    hru_map = {h: i for i, h in enumerate(hru_ids)}
                    grid_map = {g: i for i, g in enumerate(grid_ids)}

                    W = np.zeros((len(grid_ids), len(hru_ids)))
                    for _, r in idf.iterrows():
                        hi = hru_map.get(r["S_1_HRU_ID"])
                        gi = grid_map.get(r["S_2_ID"])
                        if hi is not None and gi is not None:
                            W[gi, hi] = r["AP1N"]

                    # Downsample for large matrices
                    if W.shape[0] > 80 or W.shape[1] > 80:
                        step_r = max(1, W.shape[0] // 80)
                        step_c = max(1, W.shape[1] // 80)
                        W_display = W[::step_r, ::step_c]
                    else:
                        W_display = W

                    ax.imshow(W_display, aspect="auto", cmap="YlOrRd",
                              interpolation="nearest", vmin=0, vmax=1)
                    ax.set_xlabel(f"Target HRUs (of {len(hru_ids):,})")
                    ax.set_ylabel(f"Source cells (of {len(grid_ids)})")
                    n_nz = np.count_nonzero(W)
                    sparsity = 1.0 - n_nz / max(W.size, 1)
                    ax.text(0.02, 0.98,
                            f"sparsity: {sparsity:.1%}\nnon-zero: {n_nz:,}",
                            transform=ax.transAxes, fontsize=8, va="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                    real_matrix_plotted = True
            except Exception as e:
                logger.warning(f"Could not load real weight matrix for {dk}: {e}")

        if not real_matrix_plotted:
            # Fallback to analysis JSON data
            weights = dr.get("remapping_weights", {})
            if weights.get("exists") and weights.get("files"):
                wf = weights["files"][0]
                sparsity = wf.get("sparsity", None)
                n_src = wf.get("n_source_cells", 0)
                n_total = wf.get("n_total_elements", 0)
                n_hrus = info["n_hrus"]
                n_grid = info["raw_grid_cells"]
                ax.text(0.5, 0.5,
                        f"Weight file: {wf.get('size', '?')}\n"
                        f"{n_grid} cells → {n_hrus:,} HRUs\n"
                        f"sparsity: {sparsity:.1%}" if sparsity else "",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color=color,
                        bbox=dict(boxstyle="round", facecolor=color, alpha=0.1))
            else:
                ax.text(0.5, 0.5, "Weight file: 98 B\n(trivial, point scale)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="#888888", style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Weight matrix", fontsize=11, fontweight="bold")

        # --- Col 2: Compression butterfly chart ---
        ax = axes[row, 2]
        comp = dr.get("compression", {})
        per_var = comp.get("per_variable", {})
        forcing_vars = {k: v for k, v in per_var.items()
                        if k not in ("latitude", "longitude", "hruId", "time")}

        if forcing_vars:
            var_names = sorted(forcing_vars.keys())
            raw_mb = [forcing_vars[v].get("raw_bytes", 0) / (1024 * 1024) for v in var_names]
            proc_mb = [forcing_vars[v].get("processed_bytes", 0) / (1024 * 1024) for v in var_names]

            y_pos = np.arange(len(var_names))
            # Butterfly: raw on left (negative), processed on right (positive)
            ax.barh(y_pos, [-r for r in raw_mb], height=0.6, color="#888888",
                    alpha=0.7, label="Raw (grid)")
            ax.barh(y_pos, proc_mb, height=0.6, color=color,
                    alpha=0.7, label="Basin-avg")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(var_names, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("MB", fontsize=9)
            ax.legend(fontsize=7, loc="lower right", frameon=True)

            overall = comp.get("overall_compression_ratio", 0)
            ax.text(0.02, 0.98, f"Overall: {overall:.1f}x",
                    transform=ax.transAxes, fontsize=8, va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No compression data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row == 0:
            ax.set_title("Compression", fontsize=11, fontweight="bold")

        # --- Col 3: Storage footprint stacked bar ---
        ax = axes[row, 3]
        vols = dr.get("data_volumes", {})
        storage_cats = [
            ("Forcing (raw)", "#DD8452"),
            ("Forcing (basin-averaged)", color),
            ("Remapping weights", "#4C72B0"),
            ("Catchment shapefiles", "#55A868"),
            ("Streamflow obs", "#9467bd"),
            ("Model settings", "#999999"),
            ("Elevation (DEM)", "#8c564b"),
        ]
        cat_names = []
        cat_vals = []
        cat_colors = []
        for cat_name, cat_color in storage_cats:
            val = vols.get(cat_name, {}).get("total_bytes", 0) / (1024 * 1024)
            if val > 0:
                cat_names.append(cat_name.replace(" obs", "").replace(" (", "\n("))
                cat_vals.append(val)
                cat_colors.append(cat_color)

        if cat_vals:
            bars = ax.barh(range(len(cat_names)), cat_vals, color=cat_colors,
                           alpha=0.8, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(cat_names)))
            ax.set_yticklabels(cat_names, fontsize=7)
            ax.set_xlabel("MB", fontsize=9)
            for i, (bar, v) in enumerate(zip(bars, cat_vals)):
                if v > 0.01:
                    ax.text(bar.get_width() + 0.01 * max(cat_vals),
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.1f}", va="center", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No volume data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row == 0:
            ax.set_title("Storage footprint", fontsize=11, fontweight="bold")

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    out = save_path / f"fig_multiscale_panel.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 4: Observation Timeline
# ============================================================
def fig_observation_timeline(analysis: Dict, save_path: Path, fmt: str = "png"):
    """
    Gantt-style timeline grouped by domain (3 groups), with experiment
    period overlays and section cross-references.
    """
    domain_keys = ["paradise", "bow", "iceland"]
    fig, ax = plt.subplots(figsize=(12, 6))

    obs_type_colors = {
        "streamflow": "#1f77b4",
        "snow": "#2ca02c",
        "et": "#ff7f0e",
        "grace": "#9467bd",
        "soil_moisture": "#8c564b",
    }

    y = 0
    yticks = []
    ylabels = []
    group_spans = []  # (y_start, y_end, domain_key)

    for dk in domain_keys:
        dr = analysis.get(dk, {})
        obs = dr.get("observation_coverage", {})
        info = DOMAINS[dk]
        color = DOMAIN_COLORS[dk]
        y_start = y

        if not obs:
            # Placeholder for domains without observation data
            yticks.append(y)
            ylabels.append("  (no obs data)")
            y += 1
        else:
            for obs_type, obs_info in sorted(obs.items()):
                rng = obs_info.get("date_range")
                if not rng:
                    continue
                try:
                    start = pd.Timestamp(rng[0])
                    end = pd.Timestamp(rng[1])
                except Exception:
                    continue

                oc = obs_type_colors.get(obs_type, "#7f7f7f")
                gap_frac = obs_info.get("gap_fraction", 0)
                n_records = obs_info.get("n_records", "?")

                start_mpl = mdates.date2num(start)
                end_mpl = mdates.date2num(end)
                ax.barh(y, end_mpl - start_mpl, left=start_mpl, height=0.6,
                        color=oc, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.text(end_mpl + 60, y,
                        f"gap: {gap_frac:.1%}, n={n_records}",
                        va="center", fontsize=7.5)
                yticks.append(y)
                ylabels.append(f"  {obs_type.replace('_', ' ').title()}")
                y += 1

        y_end = y - 1 if y > y_start else y_start
        group_spans.append((y_start, y_end, dk))

        # Group separator
        if dk != domain_keys[-1]:
            ax.axhline(y - 0.5, color="#cccccc", linewidth=0.5, linestyle="--")

        y += 0.5  # gap between groups

    # Draw domain group labels on the left
    for y_start, y_end, dk in group_spans:
        info = DOMAINS[dk]
        color = DOMAIN_COLORS[dk]
        y_mid = (y_start + y_end) / 2
        ax.text(-0.02, y_mid, f"{info['label']}\n({', '.join(info['sections'])})",
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=color, transform=ax.get_yaxis_transform())

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    # Experiment period overlays
    for period_name, dates, pcolor in [
        ("Spin-up", ("2002-01-01", "2003-12-31"), "#cccccc"),
        ("Calibration", ("2004-01-01", "2007-12-31"), "#e6f2ff"),
        ("Evaluation", ("2008-01-01", "2009-12-31"), "#fff2e6"),
    ]:
        s = mdates.date2num(pd.Timestamp(dates[0]))
        e = mdates.date2num(pd.Timestamp(dates[1]))
        ax.axvspan(s, e, alpha=0.12, color=pcolor, label=period_name)

    ax.set_xlabel("Date")
    ax.set_title("Observation data coverage across domains", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    out = save_path / f"fig_observation_timeline.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 5: Scaling Law
# ============================================================
def fig_scaling_law(analysis: Dict, save_path: Path, fmt: str = "png"):
    """
    2-panel figure:
    (a) log-log GRU count vs data volume by category
    (b) log-log GRU count vs compression ratio with break-even line
    """
    cs = analysis.get("cross_scale_summary", {})
    domains_cs = cs.get("domains", {})
    scaling = cs.get("scaling", {})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    domain_keys = ["paradise", "bow", "iceland"]

    # --- Panel (a): GRU count vs data volume by category ---
    volume_categories = [
        ("Forcing (raw)", "Forcing (raw)", "#DD8452"),
        ("Forcing (basin-averaged)", "Forcing (basin-avg)", "#1f77b4"),
        ("Remapping weights", "Remap weights", "#55A868"),
        ("Catchment shapefiles", "Shapefiles", "#9467bd"),
    ]

    for cat_key, cat_label, cat_color in volume_categories:
        hrus = []
        vols = []
        for dk in domain_keys:
            dr = analysis.get(dk, {})
            dv = dr.get("data_volumes", {})
            vol = dv.get(cat_key, {}).get("total_bytes", 0)
            if vol > 0:
                hrus.append(DOMAINS[dk]["n_hrus"])
                vols.append(vol / (1024 * 1024))  # MB

        if hrus:
            ax_a.plot(hrus, vols, "o-", color=cat_color, label=cat_label,
                      markersize=8, linewidth=2, alpha=0.8)

    # Mark domain names
    for dk in domain_keys:
        n_hrus = DOMAINS[dk]["n_hrus"]
        color = DOMAIN_COLORS[dk]
        ax_a.axvline(n_hrus, color=color, linewidth=0.8, linestyle=":", alpha=0.5)
        ax_a.text(n_hrus, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 0.001,
                  DOMAINS[dk]["label"], rotation=90, fontsize=8, color=color,
                  ha="right", va="bottom")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Number of HRUs")
    ax_a.set_ylabel("Data volume (MB)")
    ax_a.set_title("(a) Data volume vs spatial resolution")
    ax_a.legend(fontsize=8, frameon=True, loc="upper left")
    ax_a.grid(True, alpha=0.3, which="both")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # --- Panel (b): HRU count vs compression ratio ---
    hru_counts = []
    comp_ratios = []
    for dk in domain_keys:
        dr = analysis.get(dk, {})
        comp = dr.get("compression", {})
        cr = comp.get("overall_compression_ratio", None)
        if cr and cr > 0:
            hru_counts.append(DOMAINS[dk]["n_hrus"])
            comp_ratios.append(cr)

    if hru_counts:
        for i, dk in enumerate(domain_keys):
            dr = analysis.get(dk, {})
            comp = dr.get("compression", {})
            cr = comp.get("overall_compression_ratio", None)
            if cr and cr > 0:
                color = DOMAIN_COLORS[dk]
                ax_b.plot(DOMAINS[dk]["n_hrus"], cr, "o", color=color,
                          markersize=12, label=DOMAINS[dk]["label"], zorder=5)
                ax_b.annotate(f"{cr:.1f}x",
                              (DOMAINS[dk]["n_hrus"], cr),
                              textcoords="offset points", xytext=(10, 5),
                              fontsize=9, fontweight="bold", color=color)

        # Connect points
        if len(hru_counts) > 1:
            sorted_pairs = sorted(zip(hru_counts, comp_ratios))
            ax_b.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                      "--", color="#888888", linewidth=1.5, alpha=0.5, zorder=3)

    # Break-even line at compression ratio = 1.0
    ax_b.axhline(1.0, color="#d62728", linewidth=1.5, linestyle="-", alpha=0.7,
                 label="Break-even (1:1)")
    ax_b.fill_between([0.5, 10000], [1.0, 1.0], [0.01, 0.01],
                      color="#d62728", alpha=0.05)
    ax_b.text(0.98, 0.02, "Expansion\n(storage increases)",
              transform=ax_b.transAxes, fontsize=8, ha="right", va="bottom",
              color="#d62728", alpha=0.7, style="italic")
    ax_b.text(0.98, 0.98, "Compression\n(storage decreases)",
              transform=ax_b.transAxes, fontsize=8, ha="right", va="top",
              color="#2ca02c", alpha=0.7, style="italic")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Number of HRUs")
    ax_b.set_ylabel("Compression ratio (raw / basin-averaged)")
    ax_b.set_title("(b) Spatial compression vs resolution")
    ax_b.legend(fontsize=8, frameon=True, loc="upper right")
    ax_b.grid(True, alpha=0.3, which="both")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    fig.tight_layout()
    out = save_path / f"fig_scaling_law.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Create publication figures for Section 4.12 pipeline experiment"
    )
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR))
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    analysis = load_latest_analysis(analysis_dir)

    if not analysis:
        logger.error("No analysis data found. Run analyze_pipeline.py first.")
        return

    logger.info("Creating layered DAG figure ...")
    fig_layered_dag(analysis, FIGURES_DIR, args.format)

    logger.info("Creating data flow Sankey figure ...")
    fig_data_flow_sankey(analysis, FIGURES_DIR, args.format)

    logger.info("Creating multi-scale panel figure ...")
    fig_multiscale_panel(analysis, FIGURES_DIR, args.format)

    logger.info("Creating observation timeline figure ...")
    fig_observation_timeline(analysis, FIGURES_DIR, args.format)

    logger.info("Creating scaling law figure ...")
    fig_scaling_law(analysis, FIGURES_DIR, args.format)

    logger.info(f"All 5 figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
