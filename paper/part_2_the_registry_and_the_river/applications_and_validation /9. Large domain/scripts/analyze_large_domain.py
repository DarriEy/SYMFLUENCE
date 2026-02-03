#!/usr/bin/env python3
"""Generate overview figure of the distributed Iceland domain for Section 4.9.

Creates a multi-panel figure:
  (a) Map of Iceland showing GRU mesh coloured by elevation, with river network.
  (b) Histogram of GRU areas.
  (c) Histogram of GRU mean elevations.

Modelled on Section 4.8's fig_study_basins_overview.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
ANALYSIS_DIR = BASE_DIR / "analysis"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = SYMFLUENCE_DATA / "domain_Iceland_Multivar"
CATCHMENT_SHP_DIR = DOMAIN_DIR / "shapefiles" / "catchment" / "semidistributed" / "large_domain"
RIVER_SHP_DIR = DOMAIN_DIR / "shapefiles" / "river_network"

# ── Publication style (matching Section 4.8) ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Iceland bounding box (from config)
LON_MIN, LON_MAX = -25.0, -13.0
LAT_MIN, LAT_MAX = 63.0, 66.5
ASPECT_CORRECTION = 1.0 / np.cos(np.radians(65.0))


def find_shapefile(directory: Path, pattern: str = "*.shp") -> "Path | None":
    """Return the first shapefile found in *directory*, or None."""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern)) + sorted(directory.glob("*.gpkg"))
    return candidates[0] if candidates else None


def load_gru_shapefile() -> "gpd.GeoDataFrame | None":
    """Load GRU/HRU catchment shapefile for the Iceland distributed domain."""
    if not HAS_GEOPANDAS:
        print("Warning: geopandas not available — skipping spatial plots.")
        return None

    shp = find_shapefile(CATCHMENT_SHP_DIR)
    if shp is None:
        print(f"Warning: No catchment shapefile found in {CATCHMENT_SHP_DIR}")
        return None

    print(f"  Loading: {shp.name}")
    return gpd.read_file(shp)


def load_river_network() -> "gpd.GeoDataFrame | None":
    """Load river network shapefile."""
    if not HAS_GEOPANDAS:
        return None

    shp = find_shapefile(RIVER_SHP_DIR)
    if shp is None:
        print(f"Warning: No river network shapefile found in {RIVER_SHP_DIR}")
        return None

    print(f"  Loading: {shp.name}")
    return gpd.read_file(shp)


def load_routing_output() -> "xr.Dataset | None":
    """Load mizuRoute output for optional flow statistics."""
    if not HAS_XARRAY:
        return None

    routing_dir = DOMAIN_DIR / "simulations" / "mizuRoute"
    nc_files = sorted(routing_dir.glob("*.nc")) if routing_dir.exists() else []
    if not nc_files:
        return None

    print(f"  Loading {len(nc_files)} mizuRoute file(s)...")
    return xr.open_mfdataset(nc_files, combine="by_coords")


def resolve_elevation_column(gdf: "gpd.GeoDataFrame") -> str:
    """Identify the elevation column in the GRU shapefile."""
    for candidate in ("elev_mean", "HRU_elev", "ELEV_MEAN", "elevation", "MEAN_ELEV"):
        if candidate in gdf.columns:
            return candidate
    return ""


def resolve_area_column(gdf: "gpd.GeoDataFrame") -> str:
    """Identify the area column in the GRU shapefile."""
    for candidate in ("HRU_area", "GRU_area", "area_km2", "AREA", "Shape_Area"):
        if candidate in gdf.columns:
            return candidate
    return ""


def _detect_area_unit(values) -> float:
    """Return divisor to convert area values to km².  Heuristic: if median > 1e5 assume m²."""
    median = np.nanmedian(values)
    if median > 1e5:
        return 1e6   # m² → km²
    return 1.0        # already km²


def compute_gru_stats(gdf: "gpd.GeoDataFrame", elev_col: str, area_col: str) -> pd.DataFrame:
    """Compute per-GRU summary statistics and return as a DataFrame."""
    area_divisor = _detect_area_unit(gdf[area_col].values) if area_col else 1.0
    records = []
    gru_col = "GRU_ID" if "GRU_ID" in gdf.columns else None

    if gru_col:
        for gru_id, grp in gdf.groupby(gru_col):
            rec = {"GRU_ID": gru_id}
            if area_col:
                rec["area_km2"] = grp[area_col].sum() / area_divisor
            if elev_col:
                rec["elev_mean"] = grp[elev_col].mean()
            records.append(rec)
    else:
        for idx, row in gdf.iterrows():
            rec = {"GRU_ID": idx}
            if area_col:
                rec["area_km2"] = row[area_col] / area_divisor
            if elev_col:
                rec["elev_mean"] = row[elev_col]
            records.append(rec)

    return pd.DataFrame(records)


def plot_placeholder(figures_dir: Path):
    """Create a placeholder figure when shapefiles are not available."""
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1],
                           height_ratios=[1, 1], wspace=0.30, hspace=0.40)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_elev = fig.add_subplot(gs[1, 1])

    # Placeholder map — Iceland bounding box
    from matplotlib.patches import Rectangle
    ax_map.add_patch(Rectangle((LON_MIN, LAT_MIN), LON_MAX - LON_MIN,
                               LAT_MAX - LAT_MIN, fill=False, edgecolor="#999",
                               linewidth=1.0, linestyle="--"))
    ax_map.set_xlim(LON_MIN - 1, LON_MAX + 1)
    ax_map.set_ylim(LAT_MIN - 0.5, LAT_MAX + 0.5)
    ax_map.set_aspect(ASPECT_CORRECTION)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.text(0.5, 0.5, "GRU shapefile not yet generated",
                transform=ax_map.transAxes, ha="center", va="center",
                fontsize=10, color="#888888")
    ax_map.text(0.015, 0.97, "(a)", transform=ax_map.transAxes,
                fontsize=11, fontweight="bold", va="top",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    for ax, label, xlabel in [(ax_area, "(b)", "GRU area (km²)"),
                               (ax_elev, "(c)", "Elevation (m a.s.l.)")]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.text(0.04, 0.94, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888888")

    ax_map.grid(True, linewidth=0.3, alpha=0.4, color="#999999")

    for ext in ("png", "pdf"):
        out = figures_dir / f"fig_large_domain_overview.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved placeholder: {out}")
    plt.close(fig)


def plot_overview(gdf, rivers, gru_stats, figures_dir: Path):
    """Create the publication-quality multi-panel overview figure."""

    elev_col = resolve_elevation_column(gdf)
    area_col = resolve_area_column(gdf)

    # ── Figure layout ──
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1],
                           height_ratios=[1, 1], wspace=0.30, hspace=0.40)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_elev = fig.add_subplot(gs[1, 1])

    # ── Panel (a): Map ──
    if elev_col:
        vmin = gdf[elev_col].quantile(0.02)
        vmax = gdf[elev_col].quantile(0.98)
        gdf.plot(ax=ax_map, column=elev_col, cmap="terrain",
                 edgecolor="#444444", linewidth=0.08, alpha=0.9,
                 vmin=vmin, vmax=vmax, legend=False, zorder=2)
    else:
        gdf.plot(ax=ax_map, color="#b8d4a8", edgecolor="#444444",
                 linewidth=0.08, alpha=0.9, zorder=2)

    # River network overlay
    if rivers is not None:
        rivers.plot(ax=ax_map, color="#2166ac", linewidth=0.25,
                    alpha=0.7, zorder=3)

    # Map extent and aspect
    ax_map.set_xlim(LON_MIN, LON_MAX)
    ax_map.set_ylim(LAT_MIN, LAT_MAX)
    ax_map.set_aspect(ASPECT_CORRECTION)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(True, linewidth=0.3, alpha=0.4, color="#999999")

    # Colourbar — horizontal below map
    if elev_col:
        sm = plt.cm.ScalarMappable(cmap="terrain",
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_map, orientation="horizontal",
                            fraction=0.035, pad=0.12, shrink=0.5)
        cbar.set_label("Elevation (m a.s.l.)", fontsize=8)

    # Panel label
    ax_map.text(0.015, 0.97, "(a)", transform=ax_map.transAxes,
                fontsize=11, fontweight="bold", va="top",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Summary stats box — positioned in ocean SW of Iceland
    n_gru = gru_stats["GRU_ID"].nunique() if "GRU_ID" in gru_stats.columns else len(gru_stats)
    n_hru = len(gdf)
    parts = [f"n = {n_gru} GRUs, {n_hru} HRUs"]
    if area_col and "area_km2" in gru_stats.columns:
        total_area = gru_stats["area_km2"].sum()
        parts.append(f"Total area: {total_area / 1000:.0f} \u00d7 10\u00b3 km\u00b2")
    if elev_col and "elev_mean" in gru_stats.columns:
        emin = gru_stats["elev_mean"].min()
        emax = gru_stats["elev_mean"].max()
        parts.append(f"Elev: {emin:.0f}\u2013{emax:.0f} m")
    if rivers is not None:
        parts.append(f"River segments: {len(rivers)}")
    summary = "\n".join(parts)

    ax_map.text(
        -24.8, 63.15, summary, fontsize=7.5,
        va="bottom", ha="left", family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # ── Panel (b): GRU area histogram ──
    if "area_km2" in gru_stats.columns:
        areas = gru_stats["area_km2"].dropna()
        if areas.min() > 0:
            bins = np.logspace(np.log10(areas.min()), np.log10(areas.max()), 30)
            ax_area.hist(areas, bins=bins, color="#4a90d9", edgecolor="#333333",
                         linewidth=0.4, alpha=0.85)
            ax_area.set_xscale("log")
        else:
            ax_area.hist(areas, bins=30, color="#4a90d9", edgecolor="#333333",
                         linewidth=0.4, alpha=0.85)
        median_area = areas.median()
        ax_area.axvline(median_area, color="#b03a2e", linewidth=0.8, linestyle="--")
        ax_area.text(0.95, 0.85, f"median\n{median_area:.1f} km\u00b2",
                     transform=ax_area.transAxes, fontsize=7, color="#b03a2e",
                     ha="right", va="top")
    ax_area.set_xlabel("GRU area (km\u00b2)")
    ax_area.set_ylabel("Count")
    ax_area.text(0.04, 0.94, "(b)", transform=ax_area.transAxes,
                 fontsize=10, fontweight="bold", va="top")

    # ── Panel (c): Elevation histogram ──
    if "elev_mean" in gru_stats.columns:
        elevs = gru_stats["elev_mean"].dropna()
        ax_elev.hist(elevs, bins=30, color="#6ab04c", edgecolor="#333333",
                     linewidth=0.4, alpha=0.85)
        median_elev = elevs.median()
        ax_elev.axvline(median_elev, color="#b03a2e", linewidth=0.8, linestyle="--")
        ax_elev.text(0.95, 0.85, f"median\n{median_elev:.0f} m",
                     transform=ax_elev.transAxes, fontsize=7, color="#b03a2e",
                     ha="right", va="top")
    ax_elev.set_xlabel("Elevation (m a.s.l.)")
    ax_elev.set_ylabel("Count")
    ax_elev.text(0.04, 0.94, "(c)", transform=ax_elev.transAxes,
                 fontsize=10, fontweight="bold", va="top")

    # ── Save ──
    for ext in ("png", "pdf"):
        out = figures_dir / f"fig_large_domain_overview.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate overview figure for Section 4.9 (Iceland distributed domain)."
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory for figures")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR,
                        help="Output directory for CSV statistics")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Section 4.9 — Large-Domain Overview Figure")
    print("=" * 55)

    # Load spatial data
    print("\nLoading GRU shapefile...")
    gdf = load_gru_shapefile()

    print("Loading river network...")
    rivers = load_river_network()

    if gdf is None:
        print("\nNo GRU shapefile available — generating placeholder figure.")
        plot_placeholder(args.figures_dir)
        print("\nDone (placeholder mode).")
        return

    # Resolve attribute columns
    elev_col = resolve_elevation_column(gdf)
    area_col = resolve_area_column(gdf)
    print(f"  Elevation column: {elev_col or '(not found)'}")
    print(f"  Area column:      {area_col or '(not found)'}")

    # Compute GRU-level statistics
    print("Computing GRU statistics...")
    gru_stats = compute_gru_stats(gdf, elev_col, area_col)

    # Save statistics CSV
    if not gru_stats.empty:
        stats_path = args.analysis_dir / "iceland_gru_statistics.csv"
        gru_stats.to_csv(stats_path, index=False)
        print(f"  Saved: {stats_path}  ({len(gru_stats)} GRUs)")

    # Optionally load mizuRoute output for river flow stats
    print("Checking for mizuRoute output...")
    ds = load_routing_output()
    if ds is not None:
        print(f"  Found routing output with {ds.dims.get('time', '?')} time steps.")
        ds.close()
    else:
        print("  No routing output found (this is expected before simulation).")

    # Generate figure
    print("\nGenerating overview figure...")
    plot_overview(gdf, rivers, gru_stats, args.figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
