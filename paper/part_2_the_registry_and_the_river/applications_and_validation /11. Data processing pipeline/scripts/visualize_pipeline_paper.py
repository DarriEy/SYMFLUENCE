#!/usr/bin/env python3
"""
Composite Publication Figures for SYMFLUENCE Paper Section 4.12

Three publication-ready composite figures:

  Figure 1 -- Pipeline Architecture and Scaling (DAG + volume scaling)
  Figure 2 -- Forcing Data Transformation (spatial remapping + variable transform)
  Figure 3 -- Observation Data Flow (workflow schematic + station + GRACE)

Usage:
    python visualize_pipeline_paper.py [--analysis-dir DIR] [--format png|pdf] [--dpi DPI]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_paper_figures")

# Publication style — 8 pt minimum for Nature-style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------- domains
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

# ---------------------------------------------------------------- colours
DOMAIN_COLORS = {
    "paradise": "#2ca02c",   # green
    "bow":      "#1f77b4",   # blue
    "iceland":  "#d62728",   # red
}

CATEGORY_COLORS = {
    "setup":       "#999999",
    "geospatial":  "#4C72B0",
    "forcing":     "#DD8452",
    "observation": "#55A868",
    "model_setup": "#C44E52",
}

PRODUCT_COLORS = {
    "shapefile": "#4C72B0",
    "raster":    "#8B4513",
    "netcdf":    "#DD8452",
    "csv":       "#55A868",
    "config":    "#999999",
}

# Category descriptions for DAG annotations
CATEGORY_DESCRIPTIONS = {
    "setup":       "Project scaffolding\n& config validation",
    "geospatial":  "Catchment delineation,\ndiscretisation & attributes",
    "forcing":     "ERA5 download, spatial\nremapping & corrections",
    "observation": "Station & remote-sensing\ndata acquisition & QC",
    "model_setup": "Format conversion\nfor SUMMA/mizuRoute",
}

# Shortened stage labels that fit inside DAG boxes
STAGE_SHORT_LABELS = {
    "setup_project":                "Project init",
    "create_pour_point":            "Pour-point\ncreation",
    "define_domain":                "Domain\ndelineation",
    "discretize_domain":            "Domain\ndiscretisation",
    "acquire_attributes":           "Attribute\nacquisition",
    "compute_zonal_statistics":     "Zonal\nstatistics",
    "acquire_forcings":             "Forcing\nacquisition",
    "generate_weights":             "Weight\ngeneration",
    "apply_weights":                "Weight\napplication",
    "standardize_variables":        "Variable\nstandardisation",
    "lapse_rate_correction":        "Lapse-rate\ncorrection",
    "merge_forcing":                "Forcing\nmerge",
    "process_streamflow":           "Streamflow\nprocessing",
    "process_snow":                 "Snow obs\nprocessing",
    "process_et_grace":             "ET / GRACE\nprocessing",
    "model_specific_preprocessing": "Model-format\nconversion",
}


# ---------------------------------------------------------------- helpers
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


def _panel_label(ax, label, x=-0.08, y=1.05):
    """Add Nature-style bold panel label (a), (b), etc."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


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


def load_latest_analysis(analysis_dir: Path) -> Optional[Dict]:
    """Load the most recent pipeline_analysis JSON."""
    analyses = sorted(analysis_dir.glob("pipeline_analysis_*.json"), reverse=True)
    if not analyses:
        return None
    with open(analyses[0]) as f:
        return json.load(f)


def _build_full_weight_matrix(csv_path: Path, lats: np.ndarray, lons: np.ndarray):
    """Build full weight matrix with all ERA5 grid cells as rows.

    Rows are ordered by latitude descending, longitude ascending (matching
    the spatial grid layout). Non-intersecting cells appear as zero rows.

    Returns W (n_grid_total x n_hrus), hru_ids, grid_labels.
    """
    idf = pd.read_csv(csv_path)
    if not {"S_1_HRU_ID", "S_2_lat", "S_2_lon", "AP1N"}.issubset(idf.columns):
        return None, None, None

    hru_ids = sorted(idf["S_1_HRU_ID"].unique())
    hru_map = {h: i for i, h in enumerate(hru_ids)}

    # Build ordered grid: lat descending, lon ascending
    all_grid = [(lat, lon) for lat in sorted(lats, reverse=True)
                for lon in sorted(lons)]
    grid_map = {(round(lat, 4), round(lon, 4)): i
                for i, (lat, lon) in enumerate(all_grid)}

    W = np.zeros((len(all_grid), len(hru_ids)))
    for _, r in idf.iterrows():
        hi = hru_map.get(r["S_1_HRU_ID"])
        key = (round(r["S_2_lat"], 4), round(r["S_2_lon"], 4))
        gi = grid_map.get(key)
        if hi is not None and gi is not None:
            W[gi, hi] += r["AP1N"]

    grid_labels = [f"{lat:.1f},{lon:.1f}" for lat, lon in all_grid]
    return W, hru_ids, grid_labels


# ============================================================
# Figure 1: Pipeline Architecture and Scaling
# ============================================================
def fig_paper_architecture(analysis: Dict, save_path: Path, fmt: str = "png",
                           dpi: int = 300):
    """
    2-row composite: (a) improved DAG, (b) volume scaling.
    Size: ~7x8 inches.
    """
    dag = analysis.get("stage_dag", {})
    stages = dag.get("stages", [])
    edges = dag.get("edges", [])
    categories_info = dag.get("categories", {})

    if not stages:
        logger.warning("No DAG data found; skipping Figure 1")
        return

    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.22)

    # ---- (a) Improved left-to-right DAG ----
    ax_dag = fig.add_subplot(gs[0])

    lane_order = ["setup", "geospatial", "forcing", "observation", "model_setup"]
    # Horizontal layout: each category is a column band
    lane_height = 9.0
    lane_spacing = 4.0
    lane_x = {cat: i * lane_spacing for i, cat in enumerate(lane_order)}
    lane_width = 3.4

    # Swim-lane backgrounds
    for cat in lane_order:
        x = lane_x[cat]
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        rect = mpatches.FancyBboxPatch(
            (x - lane_width / 2, -0.5), lane_width, lane_height,
            boxstyle="round,pad=0.15",
            facecolor=color, alpha=0.06, edgecolor=color,
            linewidth=0.8, linestyle="--",
        )
        ax_dag.add_patch(rect)
        # Category label at top (use short labels to avoid overlap)
        short_labels = {
            "setup": "Setup",
            "geospatial": "Geospatial",
            "forcing": "Forcing",
            "observation": "Observations",
            "model_setup": "Model setup",
        }
        label = short_labels.get(cat, cat)
        ax_dag.text(x, lane_height - 0.7, label, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color, alpha=0.9)
        # Annotation describing what the lane does
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        if desc:
            ax_dag.text(x, -0.8, desc, ha="center", va="top",
                        fontsize=5.5, color=color, alpha=0.7,
                        fontstyle="italic", linespacing=1.1)

    # Position stages within swim-lanes (vertical stacking within each lane)
    category_stages = {}
    for s in stages:
        cat = s["category"]
        if cat not in category_stages:
            category_stages[cat] = []
        category_stages[cat].append(s)

    positions = {}
    for cat, cat_stages in category_stages.items():
        x = lane_x.get(cat, 6)
        n = len(cat_stages)
        for i, s in enumerate(cat_stages):
            y = (lane_height - 1.5) - (i * (lane_height - 2.0) / max(n, 1)) - \
                ((lane_height - 2.0) / max(n, 1)) / 2
            positions[s["id"]] = (x, y)

    # Curved edges coloured by product type
    for edge in edges:
        if edge["from"] in positions and edge["to"] in positions:
            x1, y1 = positions[edge["from"]]
            x2, y2 = positions[edge["to"]]
            ecolor = _product_color(edge.get("data_product", ""))
            rad = 0.2 if abs(x2 - x1) > lane_spacing * 0.5 else 0.1
            ax_dag.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=ecolor, lw=0.9,
                    connectionstyle=f"arc3,rad={rad}", alpha=0.5,
                ),
            )

    # Stage boxes — sized to fit within lanes
    box_w, box_h = 3.0, 0.8
    for s in stages:
        if s["id"] not in positions:
            continue
        x, y = positions[s["id"]]
        cat = s["category"]
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.88,
        )
        ax_dag.add_patch(rect)
        label = STAGE_SHORT_LABELS.get(s["id"], s["label"])
        ax_dag.text(x, y, label, ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color="white", linespacing=1.1,
                    path_effects=[patheffects.withStroke(linewidth=0.5,
                                                        foreground="black")])

    # Edge colour legend
    edge_handles = [
        mpatches.Patch(color=c, label=l, alpha=0.7)
        for l, c in [("Shapefile", PRODUCT_COLORS["shapefile"]),
                      ("Raster", PRODUCT_COLORS["raster"]),
                      ("NetCDF", PRODUCT_COLORS["netcdf"]),
                      ("CSV", PRODUCT_COLORS["csv"]),
                      ("Config", PRODUCT_COLORS["config"])]
    ]
    ax_dag.legend(handles=edge_handles, loc="lower left", fontsize=6.5,
                  frameon=True, fancybox=True, framealpha=0.9,
                  title="Data flow type", title_fontsize=7, ncol=3,
                  bbox_to_anchor=(0.0, -0.01))

    total_w = (len(lane_order) - 1) * lane_spacing + lane_width
    ax_dag.set_xlim(-lane_width / 2 - 0.3, total_w - lane_width / 2 + 0.3)
    ax_dag.set_ylim(-2.0, lane_height + 0.3)
    ax_dag.axis("off")
    _panel_label(ax_dag, "a", x=0.0, y=1.02)

    # ---- (b) Volume scaling grouped by DAG category ----
    ax_vol = fig.add_subplot(gs[1])
    domain_keys = ["paradise", "bow", "iceland"]

    # Volume categories grouped to mirror the DAG swim lanes
    volume_groups = [
        ("Geospatial", CATEGORY_COLORS["geospatial"], [
            "Elevation (DEM)", "Soil class", "Land cover", "Aspect",
            "Catchment shapefiles", "River network",
        ]),
        ("Forcing", CATEGORY_COLORS["forcing"], [
            "Forcing (raw)", "Forcing (basin-averaged)", "Forcing (merged)",
            "Remapping weights",
        ]),
        ("Observations", CATEGORY_COLORS["observation"], [
            "Streamflow obs", "Snow obs", "ET obs", "GRACE obs",
            "Soil moisture obs",
        ]),
        ("Model setup", CATEGORY_COLORS["model_setup"], [
            "Model settings",
        ]),
    ]

    for grp_label, grp_color, vol_keys in volume_groups:
        hrus_list = []
        vols = []
        for dk in domain_keys:
            dr = analysis.get(dk, {})
            dv = dr.get("data_volumes", {})
            total = sum(dv.get(k, {}).get("total_bytes", 0) for k in vol_keys)
            if total > 0:
                hrus_list.append(DOMAINS[dk]["n_hrus"])
                vols.append(total / (1024 * 1024))
        if hrus_list:
            ax_vol.plot(hrus_list, vols, "o-", color=grp_color, label=grp_label,
                        markersize=6, linewidth=1.8, alpha=0.85)

    for dk in domain_keys:
        n_hrus = DOMAINS[dk]["n_hrus"]
        color = DOMAIN_COLORS[dk]
        ax_vol.axvline(n_hrus, color=color, linewidth=0.6, linestyle=":", alpha=0.5)
        ax_vol.text(n_hrus, ax_vol.get_ylim()[0] if ax_vol.get_ylim()[0] > 0 else 0.001,
                    DOMAINS[dk]["label"], rotation=90, fontsize=7, color=color,
                    ha="right", va="bottom")

    ax_vol.set_xscale("log")
    ax_vol.set_yscale("log")
    ax_vol.set_xlabel("Number of HRUs")
    ax_vol.set_ylabel("Data volume (MB)")
    ax_vol.legend(fontsize=7, frameon=True, loc="upper left")
    ax_vol.grid(True, alpha=0.3, which="both")
    ax_vol.spines["top"].set_visible(False)
    ax_vol.spines["right"].set_visible(False)
    _panel_label(ax_vol, "b")

    out = save_path / f"fig_paper_architecture.{fmt}"
    fig.savefig(out, dpi=dpi)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 2: Forcing Data Transformation
# ============================================================
def fig_paper_forcing(analysis: Dict, save_path: Path, fmt: str = "png",
                      dpi: int = 300):
    """
    2x3 composite (6 panels):
      Row 1 — Spatial remapping (Bow):
        (a) ERA5 grid + HRUs, (b) full weight matrix, (c) temperature with ERA5 contour underlay
      Row 2 — Variable transformation + lapse correction (Bow):
        (d) basin-avg temperature time series, (e) lapse-rate scatter, (f) lapse correction map

    Size: ~8x6 inches.
    """
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.40)

    bow_dir = DATA_DIR / "domain_Bow_at_Banff_semi_distributed"
    hru_shp = bow_dir / "shapefiles" / "catchment" / "Bow_at_Banff_semi_distributed_HRUs_GRUs.shp"
    grid_shp = bow_dir / "shapefiles" / "forcing" / "forcing_ERA5.shp"
    intersect_csv = (bow_dir / "shapefiles" / "catchment_intersection" / "with_forcing" /
                     "Bow_at_Banff_semi_distributed_ERA5_intersected_shapefile.csv")
    raw_dir = bow_dir / "forcing" / "raw_data"
    ba_dir = bow_dir / "forcing" / "basin_averaged_data"
    summa_dir = bow_dir / "forcing" / "SUMMA_input"

    # Load shapefiles
    hrus = gpd.read_file(hru_shp) if HAS_GEOPANDAS and hru_shp.exists() else None
    grid = gpd.read_file(grid_shp) if HAS_GEOPANDAS and grid_shp.exists() else None

    # Load forcing data
    raw_files = sorted(raw_dir.glob("*.nc")) if raw_dir.exists() else []
    ba_files = sorted(ba_dir.glob("*.nc")) if ba_dir.exists() else []
    summa_files = sorted(summa_dir.glob("*.nc")) if summa_dir.exists() else []
    ds_raw = xr.open_dataset(raw_files[0]) if raw_files else None
    ds_ba = xr.open_dataset(ba_files[0]) if ba_files else None
    ds_summa = xr.open_dataset(summa_files[0]) if summa_files else None

    # Common time index for spatial panels
    t_idx = min(14 * 24 + 12, len(ds_ba.time) - 1) if ds_ba is not None else 0

    # ---- ROW 1: Spatial remapping (Bow) ----

    # (a) ERA5 grid overlaid on HRU polygons — keep as-is
    ax_a = fig.add_subplot(gs[0, 0])
    if hrus is not None and grid is not None:
        hrus.plot(ax=ax_a, facecolor=DOMAIN_COLORS["bow"], alpha=0.3,
                  edgecolor=DOMAIN_COLORS["bow"], linewidth=0.6)
        grid.plot(ax=ax_a, facecolor="none", edgecolor="#DD8452",
                  linewidth=1.2, linestyle="--")
        for _, row in grid.iterrows():
            c = row.geometry.centroid
            ax_a.plot(c.x, c.y, "s", color="#DD8452", markersize=2.5, alpha=0.7)
        for _, row in hrus.iterrows():
            c = row.geometry.centroid
            ax_a.plot(c.x, c.y, ".", color=DOMAIN_COLORS["bow"], markersize=1.5)
        ax_a.set_xlabel("Longitude", fontsize=8)
        ax_a.set_ylabel("Latitude", fontsize=8)
        legend_elements = [
            mpatches.Patch(facecolor=DOMAIN_COLORS["bow"], alpha=0.3,
                           edgecolor=DOMAIN_COLORS["bow"],
                           label=f"HRUs (n={len(hrus)})"),
            mpatches.Patch(facecolor="none", edgecolor="#DD8452",
                           linestyle="--", linewidth=1.2,
                           label=f"ERA5 grid (n={len(grid)})"),
        ]
        ax_a.legend(handles=legend_elements, loc="upper left", fontsize=6, frameon=True)
    else:
        ax_a.text(0.5, 0.5, "Shapefiles\nnot available",
                  transform=ax_a.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_a, "a")

    # (b) Raw ERA5 gridded temperature snapshot
    ax_b = fig.add_subplot(gs[0, 1])
    if ds_raw is not None and "airtemp" in ds_raw.data_vars:
        lats_raw = ds_raw.latitude.values
        lons_raw = ds_raw.longitude.values
        raw_slice = ds_raw["airtemp"].isel(time=t_idx)
        im = ax_b.pcolormesh(lons_raw, lats_raw, raw_slice.values,
                             cmap="RdYlBu_r", shading="nearest")
        cb = fig.colorbar(im, ax=ax_b, shrink=0.75, pad=0.02)
        cb.set_label("T (K)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        # Overlay HRU outlines for context
        if hrus is not None:
            hrus.boundary.plot(ax=ax_b, edgecolor="white",
                               linewidth=0.8, alpha=0.9)
        ax_b.set_xlabel("Longitude", fontsize=8)
        ax_b.set_ylabel("Latitude", fontsize=8)
        ax_b.text(0.02, 0.98, f"ERA5 raw ({len(lats_raw)}\u00d7{len(lons_raw)} grid)",
                  transform=ax_b.transAxes, fontsize=7, va="top",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax_b.text(0.5, 0.5, "Raw data\nnot available",
                  transform=ax_b.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_b, "b")

    # (c) Temperature mapped to HRUs with ERA5 contour underlay, zoomed to basin
    ax_c = fig.add_subplot(gs[0, 2])
    if ds_raw is not None and ds_ba is not None and hrus is not None:
        temp_vals = ds_ba["airtemp"].isel(time=t_idx).values
        if len(temp_vals) == len(hrus):
            # Compute HRU bounds with buffer for zoom
            bounds = hrus.total_bounds  # minx, miny, maxx, maxy
            buf = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            xlim = (bounds[0] - buf, bounds[2] + buf)
            ylim = (bounds[1] - buf, bounds[3] + buf)

            # ERA5 temperature contour underlay
            lats_raw = ds_raw.latitude.values
            lons_raw = ds_raw.longitude.values
            raw_temp = ds_raw["airtemp"].isel(time=t_idx).values
            lon_grid, lat_grid = np.meshgrid(lons_raw, lats_raw)
            # Interpolate to finer grid for smooth contours
            fine_lon = np.linspace(xlim[0], xlim[1], 80)
            fine_lat = np.linspace(ylim[0], ylim[1], 80)
            fine_lon_g, fine_lat_g = np.meshgrid(fine_lon, fine_lat)
            points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
            values = raw_temp.ravel()
            fine_temp = griddata(points, values, (fine_lon_g, fine_lat_g),
                                 method="cubic")
            ax_c.contourf(fine_lon_g, fine_lat_g, fine_temp, levels=15,
                          cmap="RdYlBu_r", alpha=0.4)

            # HRU choropleth on top
            hrus_plot = hrus.copy()
            hrus_plot["airtemp"] = temp_vals
            hrus_plot.plot(column="airtemp", ax=ax_c, cmap="RdYlBu_r",
                           edgecolor="white", linewidth=0.3, alpha=0.7,
                           legend=True,
                           legend_kwds={"label": "T (K)", "shrink": 0.65})

            # ERA5 grid overlay (dotted)
            if grid is not None:
                grid.plot(ax=ax_c, facecolor="none", edgecolor="#DD8452",
                          linewidth=0.5, linestyle=":", alpha=0.4)

            ax_c.set_xlim(xlim)
            ax_c.set_ylim(ylim)
            ax_c.set_xlabel("Longitude", fontsize=8)
            ax_c.set_ylabel("Latitude", fontsize=8)
        else:
            ax_c.text(0.5, 0.5, f"Shape mismatch:\n{len(temp_vals)} vs {len(hrus)}",
                      transform=ax_c.transAxes, ha="center", va="center", fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "Data not\navailable",
                  transform=ax_c.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_c, "c")

    # ---- ROW 2: Variable transformation + lapse correction ----

    # (d) Basin-averaged temperature time series (raw spatial mean vs basin-avg vs SUMMA-ready)
    ax_d = fig.add_subplot(gs[1, 0])
    if ds_raw is not None and ds_ba is not None and "airtemp" in ds_raw.data_vars:
        raw_ts = ds_raw["airtemp"].mean(dim=["latitude", "longitude"])
        ba_ts = ds_ba["airtemp"].mean(dim="hru")
        n_show = min(14 * 24, len(raw_ts))
        hours = np.arange(n_show)
        ax_d.plot(hours, raw_ts.values[:n_show], color="#888888",
                  linewidth=0.6, alpha=0.7, label="Raw (spatial mean)")
        ax_d.plot(hours, ba_ts.values[:n_show], color=DOMAIN_COLORS["bow"],
                  linewidth=0.9, label="Basin-avg (HRU mean)")
        if ds_summa is not None and "airtemp" in ds_summa.data_vars:
            summa_ts = ds_summa["airtemp"].mean(dim="hru")
            ax_d.plot(hours, summa_ts.values[:n_show], color="#d62728",
                      linewidth=0.8, linestyle="--", alpha=0.8, label="SUMMA-ready")
        ax_d.set_xlabel("Hour of month", fontsize=8)
        ax_d.set_ylabel("Temperature (K)", fontsize=8)
        ax_d.legend(fontsize=6, loc="best", frameon=True)
        diff = np.nanmean(np.abs(raw_ts.values[:n_show] - ba_ts.values[:n_show]))
        ax_d.text(0.02, 0.02, f"MAD: {diff:.2f} K",
                  transform=ax_d.transAxes, fontsize=7, va="bottom",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)
    else:
        ax_d.text(0.5, 0.5, "Forcing data\nnot available",
                  transform=ax_d.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_d, "d")

    # (e) Lapse-rate scatter: temperature vs elevation
    ax_e = fig.add_subplot(gs[1, 1])
    if ds_ba is not None and ds_summa is not None and hrus is not None and HAS_GEOPANDAS:
        elevations = hrus["elev_mean"].values if "elev_mean" in hrus.columns else None
        if elevations is not None:
            ba_temp = ds_ba["airtemp"].isel(time=t_idx).values
            summa_temp = ds_summa["airtemp"].isel(time=t_idx).values
            ax_e.scatter(elevations, ba_temp, color="#888888", s=15, alpha=0.6,
                         edgecolors="white", linewidth=0.3,
                         label="Before correction", zorder=3)
            ax_e.scatter(elevations, summa_temp, color=DOMAIN_COLORS["bow"],
                         s=15, alpha=0.6, edgecolors="white", linewidth=0.3,
                         label="After correction", zorder=4)
            # Theoretical lapse rate line
            elev_range = np.linspace(elevations.min(), elevations.max(), 100)
            lapse_rate = 0.0065
            ref_temp = np.nanmean(ba_temp)
            ref_elev = np.nanmean(elevations)
            theoretical = ref_temp - lapse_rate * (elev_range - ref_elev)
            ax_e.plot(elev_range, theoretical, "--", color="#d62728",
                      linewidth=1.0, alpha=0.6, label="\u22126.5 K/km")
            ax_e.set_xlabel("Elevation (m)", fontsize=8)
            ax_e.set_ylabel("Temperature (K)", fontsize=8)
            ax_e.legend(fontsize=6, frameon=True)
            ax_e.grid(True, alpha=0.2)
            ax_e.spines["top"].set_visible(False)
            ax_e.spines["right"].set_visible(False)
        else:
            ax_e.text(0.5, 0.5, "No elevation\ndata in shapefile",
                      transform=ax_e.transAxes, ha="center", va="center", fontsize=8)
    else:
        ax_e.text(0.5, 0.5, "Lapse-rate data\nnot available",
                  transform=ax_e.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_e, "e")

    # (f) Spatial map of lapse-rate correction difference (delta T = after - before)
    ax_f = fig.add_subplot(gs[1, 2])
    if ds_ba is not None and ds_summa is not None and hrus is not None and HAS_GEOPANDAS:
        ba_temp_f = ds_ba["airtemp"].isel(time=t_idx).values
        summa_temp_f = ds_summa["airtemp"].isel(time=t_idx).values
        if len(ba_temp_f) == len(hrus) and len(summa_temp_f) == len(hrus):
            delta_t = summa_temp_f - ba_temp_f
            hrus_dt = hrus.copy()
            hrus_dt["delta_T"] = delta_t
            # Symmetric colorbar centered at 0
            vmax = max(abs(np.nanmin(delta_t)), abs(np.nanmax(delta_t)))
            if vmax < 0.01:
                vmax = 1.0
            hrus_dt.plot(column="delta_T", ax=ax_f, cmap="RdBu_r",
                         edgecolor="white", linewidth=0.3,
                         legend=True, vmin=-vmax, vmax=vmax,
                         legend_kwds={"label": "\u0394T (K)", "shrink": 0.65})
            # Grid overlay dotted
            if grid is not None:
                grid.plot(ax=ax_f, facecolor="none", edgecolor="#DD8452",
                          linewidth=0.5, linestyle=":", alpha=0.4)
            # Zoom to basin extent
            bounds = hrus.total_bounds
            buf = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            ax_f.set_xlim(bounds[0] - buf, bounds[2] + buf)
            ax_f.set_ylim(bounds[1] - buf, bounds[3] + buf)
            ax_f.set_xlabel("Longitude", fontsize=8)
            ax_f.set_ylabel("Latitude", fontsize=8)
        else:
            ax_f.text(0.5, 0.5, "Shape mismatch",
                      transform=ax_f.transAxes, ha="center", va="center", fontsize=8)
    else:
        ax_f.text(0.5, 0.5, "Lapse data\nnot available",
                  transform=ax_f.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_f, "f")

    out = save_path / f"fig_paper_forcing.{fmt}"
    fig.savefig(out, dpi=dpi)
    logger.info(f"Saved: {out}")
    plt.close(fig)

    # Close datasets
    for ds in [ds_raw, ds_ba, ds_summa]:
        if ds is not None:
            ds.close()


# ============================================================
# Figure 3: Observation Data Flow
# ============================================================
def _draw_workflow_schematic(ax, analysis: Dict):
    """Draw observation processing workflow as a structured table-style diagram.

    Central pipeline stages as a shared backbone, with input sources feeding in
    from the left and output products exiting to the right.
    """
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    obs_color = CATEGORY_COLORS["observation"]
    station_color = "#1f77b4"
    remote_color = "#9467bd"
    input_color = "#78909C"
    output_color = "#455A64"

    def _rounded_box(xy, w, h, text, facecolor, fontsize=7, textcolor="white",
                     edgecolor="white", alpha=0.9):
        x, y = xy
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor,
                path_effects=[patheffects.withStroke(linewidth=0.3,
                                                    foreground="black"
                                                    if textcolor == "white"
                                                    else "none")],
                linespacing=1.15)

    def _arrow(xy_from, xy_to, color="#666666", lw=1.0):
        ax.annotate("", xy=xy_to, xytext=xy_from,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   shrinkA=2, shrinkB=2))

    # ----- Central pipeline stages (horizontal backbone) -----
    stages = ["Configure", "Acquire", "QC / Process", "Align &\nStandardise", "Output"]
    stage_colors = [input_color, obs_color, obs_color, obs_color, output_color]
    n_stages = len(stages)
    sw, sh = 1.8, 0.75  # stage box size
    x_start = 1.5
    x_spacing = 2.5
    y_mid = 3.0
    stage_xs = [x_start + i * x_spacing for i in range(n_stages)]

    # Background band for pipeline
    band = mpatches.FancyBboxPatch(
        (x_start - 0.3, y_mid - sh / 2 - 0.15), (n_stages - 1) * x_spacing + sw + 0.6,
        sh + 0.3, boxstyle="round,pad=0.15",
        facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.5, alpha=0.6,
    )
    ax.add_patch(band)

    # Draw stage boxes and connecting arrows
    for i, (sx, stage, sc) in enumerate(zip(stage_xs, stages, stage_colors)):
        _rounded_box((sx, y_mid - sh / 2), sw, sh, stage, sc, fontsize=7)
        if i > 0:
            _arrow((stage_xs[i - 1] + sw, y_mid),
                   (sx, y_mid), color="#888888", lw=1.2)

    # ----- Top row: Station / point data examples -----
    y_top = 5.0
    ax.text(0.2, y_top + 0.15, "Station / point data", fontsize=8,
            fontweight="bold", color=station_color, va="bottom")

    station_items = [
        ("WSC 05BB001\n(streamflow)", station_color),
        ("SNOTEL 1056\n(SWE, snow depth)", station_color),
    ]
    item_w, item_h = 2.2, 0.55
    for j, (label, color) in enumerate(station_items):
        ix = 0.2 + j * (item_w + 0.4)
        _rounded_box((ix, y_top - item_h), item_w, item_h, label, color,
                     fontsize=6, edgecolor=color)
        # Arrow down to "Acquire" stage
        _arrow((ix + item_w / 2, y_top - item_h),
               (stage_xs[1] + sw / 2 - 0.3 + j * 0.6, y_mid + sh / 2),
               color=station_color, lw=0.8)

    # Station outputs (right side, top)
    out_x = stage_xs[-1] + sw + 0.5
    _rounded_box((out_x, y_top - item_h), 2.0, item_h,
                 "Hourly CSV\n(394k records)", station_color,
                 fontsize=6, edgecolor=station_color)
    _arrow((stage_xs[-1] + sw, y_mid + 0.1),
           (out_x, y_top - item_h / 2), color=station_color, lw=0.8)

    # ----- Bottom row: Remote sensing / gridded data examples -----
    y_bot = 0.5
    ax.text(0.2, y_bot + item_h + 0.15, "Remote sensing / gridded", fontsize=8,
            fontweight="bold", color=remote_color, va="bottom")

    remote_items = [
        ("GRACE Mascon\n(\u00d73 solutions)", remote_color),
        ("MODIS ET\n(gridded)", remote_color),
    ]
    for j, (label, color) in enumerate(remote_items):
        ix = 0.2 + j * (item_w + 0.4)
        _rounded_box((ix, y_bot), item_w, item_h, label, color,
                     fontsize=6, edgecolor=color)
        # Arrow up to "Acquire" stage
        _arrow((ix + item_w / 2, y_bot + item_h),
               (stage_xs[1] + sw / 2 - 0.3 + j * 0.6, y_mid - sh / 2),
               color=remote_color, lw=0.8)

    # Remote outputs (right side, bottom)
    _rounded_box((out_x, y_bot), 2.0, item_h,
                 "Monthly TWS\n(283 records)", remote_color,
                 fontsize=6, edgecolor=remote_color)
    _arrow((stage_xs[-1] + sw, y_mid - 0.1),
           (out_x, y_bot + item_h / 2), color=remote_color, lw=0.8)

    # ----- Data volume annotations -----
    vol_lines = []
    for dk in ["bow", "iceland"]:
        dr = analysis.get(dk, {})
        dv = dr.get("data_volumes", {})
        info = DOMAINS[dk]
        obs_bytes = 0
        for obs_key in ["Streamflow obs", "Snow obs", "ET obs", "GRACE obs",
                        "Soil moisture obs"]:
            obs_bytes += dv.get(obs_key, {}).get("total_bytes", 0)
        if obs_bytes > 0:
            obs_mb = obs_bytes / (1024 * 1024)
            vol_lines.append(f"{info['label']}: {obs_mb:.0f} MB" if obs_mb >= 1
                             else f"{info['label']}: {obs_mb:.1f} MB")
    if vol_lines:
        vol_text = "Obs volumes: " + " | ".join(vol_lines)
        ax.text(7.0, 0.0, vol_text, fontsize=6, color="#888888",
                ha="center", va="bottom", fontstyle="italic")


def fig_paper_observations(analysis: Dict, save_path: Path, fmt: str = "png",
                           dpi: int = 300):
    """
    3-row composite:
      (a) Observation processing workflow schematic (top, full width)
      (b) Bow streamflow time series (middle-left)
      (c) GRACE TWS time series for Iceland (bottom, full width)

    Size: ~7x8 inches.
    """
    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.8, 1, 1],
                           hspace=0.35, wspace=0.30)

    # ---- (a) Workflow schematic (full width) ----
    ax_wf = fig.add_subplot(gs[0, :])
    _draw_workflow_schematic(ax_wf, analysis)
    _panel_label(ax_wf, "a", x=0.0, y=1.02)

    # ---- (b) Bow streamflow — full record with detail inset ----
    ax_sf = fig.add_subplot(gs[1, :])
    streamflow_csv = (DATA_DIR / "domain_Bow_at_Banff_semi_distributed" /
                      "observations" / "streamflow" / "preprocessed" /
                      "Bow_at_Banff_semi_distributed_streamflow_processed.csv")

    if streamflow_csv.exists():
        df_sf = pd.read_csv(streamflow_csv, parse_dates=["datetime"],
                            index_col="datetime")
        col = "discharge_cms"

        # Daily mean for the full record (cleaner than hourly)
        df_daily = df_sf[col].resample("D").mean()

        ax_sf.fill_between(df_daily.index, 0, df_daily.values,
                           color=DOMAIN_COLORS["bow"], alpha=0.25, linewidth=0)
        ax_sf.plot(df_daily.index, df_daily.values,
                   color=DOMAIN_COLORS["bow"], linewidth=0.35, alpha=0.8)

        # Experiment period overlays
        for period_name, dates, pcolor in [
            ("Calibration", ("2004-01-01", "2007-12-31"), "#aed6f1"),
            ("Evaluation", ("2008-01-01", "2009-12-31"), "#f9e79f"),
        ]:
            s, e = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
            if s >= df_daily.index[0] and s <= df_daily.index[-1]:
                ax_sf.axvspan(s, e, alpha=0.20, color=pcolor, zorder=0,
                              label=period_name)

        # Summary statistics annotation box
        n_records = len(df_sf)
        n_years = (df_sf.index[-1] - df_sf.index[0]).days / 365.25
        peak = df_sf[col].max()
        mean_q = df_sf[col].mean()
        gap_frac = df_sf[col].isna().sum() / len(df_sf) * 100
        stats_text = (f"Records: {n_records:,} (hourly)\n"
                      f"Period: {df_sf.index[0].year}\u2013{df_sf.index[-1].year} "
                      f"({n_years:.0f} yr)\n"
                      f"Peak: {peak:.0f} m\u00b3/s | Mean: {mean_q:.1f} m\u00b3/s\n"
                      f"Gaps: {gap_frac:.1f}%")
        ax_sf.text(0.98, 0.97, stats_text, transform=ax_sf.transAxes,
                   fontsize=6.5, va="top", ha="right", family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             edgecolor="#cccccc", alpha=0.9))

        ax_sf.set_xlabel("Date", fontsize=8)
        ax_sf.set_ylabel("Discharge (m$^3$/s)", fontsize=8)
        ax_sf.set_title("Station data: WSC 05BB001 — Bow at Banff (processed hourly \u2192 daily mean)",
                        fontsize=8, loc="left")
        ax_sf.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_sf.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_sf.legend(fontsize=6, loc="upper left", frameon=True)
        ax_sf.grid(True, alpha=0.2)
        ax_sf.spines["top"].set_visible(False)
        ax_sf.spines["right"].set_visible(False)
        ax_sf.set_ylim(bottom=0)
    else:
        ax_sf.text(0.5, 0.5, "Streamflow data\nnot available",
                   transform=ax_sf.transAxes, ha="center", va="center", fontsize=8)
    _panel_label(ax_sf, "b")

    # ---- (c) GRACE TWS time series (bottom, full width) ----
    ax_grace = fig.add_subplot(gs[2, :])

    grace_paths = [
        ("Iceland (Ellidaar)",
         DATA_DIR / "domain_ellioaar_iceland" / "observations" / "grace" /
         "preprocessed" / "ellioaar_iceland_grace_tws_processed.csv"),
        ("Gulkana",
         DATA_DIR / "domain_Gulkana" / "observations" / "grace" /
         "preprocessed" / "Gulkana_grace_tws_processed.csv"),
    ]

    grace_csv = None
    domain_label = None
    for label, path in grace_paths:
        if path.exists():
            grace_csv = path
            domain_label = label
            break

    solution_cols = {
        "grace_jpl_anomaly": ("JPL Mascon", "#1f77b4"),
        "grace_csr_anomaly": ("CSR Mascon", "#ff7f0e"),
        "grace_gsfc_anomaly": ("GSFC Mascon", "#2ca02c"),
    }

    if grace_csv is not None:
        df = pd.read_csv(grace_csv, parse_dates=True, index_col=0)

        # Plot each solution with markers at data points
        for col, (lbl, color) in solution_cols.items():
            if col in df.columns:
                valid = df[col].dropna()
                ax_grace.plot(valid.index, valid.values, color=color, linewidth=0.9,
                              alpha=0.85, label=lbl, zorder=3)
                ax_grace.scatter(valid.index, valid.values, color=color, s=4,
                                 alpha=0.4, zorder=2, edgecolors="none")
                # Gap shading (grey to distinguish from data)
                gap_mask = df[col].isna()
                if gap_mask.any():
                    gap_starts = df.index[gap_mask & ~gap_mask.shift(1, fill_value=False)]
                    gap_ends = df.index[gap_mask & ~gap_mask.shift(-1, fill_value=False)]
                    for gs_t, ge_t in zip(gap_starts, gap_ends):
                        ax_grace.axvspan(gs_t, ge_t, alpha=0.05, color="#888888",
                                         zorder=0)

        # GRACE / GRACE-FO transition line
        grace_fo_start = pd.Timestamp("2018-06-01")
        if df.index[-1] > grace_fo_start:
            ax_grace.axvline(grace_fo_start, color="#888888", linewidth=0.8,
                             linestyle="--", alpha=0.5, zorder=1)
            ylims = ax_grace.get_ylim()
            ax_grace.text(grace_fo_start, ylims[1] * 0.92,
                          "  GRACE \u2192 GRACE-FO", fontsize=6, color="#888888",
                          va="top")

        # Experiment period overlays
        for period_name, dates, pcolor in [
            ("Calibration", ("2004-01-01", "2007-12-31"), "#aed6f1"),
            ("Evaluation", ("2008-01-01", "2009-12-31"), "#f9e79f"),
        ]:
            s, e = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
            if s >= df.index[0]:
                ax_grace.axvspan(s, e, alpha=0.15, color=pcolor, zorder=0,
                                 label=period_name)

        # Summary statistics annotation
        jpl_col = "grace_jpl_anomaly"
        if jpl_col in df.columns:
            n_total = len(df)
            n_valid = df[jpl_col].notna().sum()
            n_gaps = n_total - n_valid
            date_range = f"{df.index[0].strftime('%Y-%m')}\u2013{df.index[-1].strftime('%Y-%m')}"
            stats_text = (f"Records: {n_total} monthly ({date_range})\n"
                          f"Valid: {n_valid} | Gaps: {n_gaps}\n"
                          f"3 Mascon solutions (JPL, CSR, GSFC)")
            ax_grace.text(0.98, 0.97, stats_text, transform=ax_grace.transAxes,
                          fontsize=6.5, va="top", ha="right", family="monospace",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                    edgecolor="#cccccc", alpha=0.9))

        ax_grace.set_xlabel("Date", fontsize=8)
        ax_grace.set_ylabel("TWS anomaly (mm w.e.)", fontsize=8)
        ax_grace.legend(loc="lower left", fontsize=6, frameon=True, ncol=3)
        ax_grace.grid(True, alpha=0.2)
        ax_grace.spines["top"].set_visible(False)
        ax_grace.spines["right"].set_visible(False)
        ax_grace.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_grace.xaxis.set_major_locator(mdates.YearLocator(3))
        ax_grace.set_title(f"Remote sensing data: GRACE TWS \u2014 {domain_label}",
                           fontsize=8, loc="left")
    else:
        ax_grace.text(0.5, 0.5, "GRACE data\nnot available",
                      transform=ax_grace.transAxes, ha="center", va="center",
                      fontsize=8)
    _panel_label(ax_grace, "c", x=0.0, y=1.02)

    out = save_path / f"fig_paper_observations.{fmt}"
    fig.savefig(out, dpi=dpi)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Create composite publication figures for Section 4.12"
    )
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR))
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    analysis = load_latest_analysis(analysis_dir)
    if not analysis:
        logger.error("No analysis data found. Run analyze_pipeline.py first.")
        return

    logger.info("=" * 60)
    logger.info("Creating composite publication figures for Section 4.12")
    logger.info("=" * 60)

    logger.info("\n[1/3] Pipeline Architecture and Scaling ...")
    fig_paper_architecture(analysis, FIGURES_DIR, args.format, args.dpi)

    logger.info("\n[2/3] Forcing Data Transformation ...")
    fig_paper_forcing(analysis, FIGURES_DIR, args.format, args.dpi)

    logger.info("\n[3/3] Observation Data Flow ...")
    fig_paper_observations(analysis, FIGURES_DIR, args.format, args.dpi)

    logger.info(f"\nAll 3 composite figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
