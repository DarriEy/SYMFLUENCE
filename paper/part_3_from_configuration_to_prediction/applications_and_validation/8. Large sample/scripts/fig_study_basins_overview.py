#!/usr/bin/env python3
"""Generate overview figure of LamaH-Ice study basins for Section 4.8.

Creates a multi-panel figure:
  (a) Map of Iceland showing all 111 catchment boundaries, coloured by area,
      with the national GRU mesh as context.
  (b) Histogram of catchment area (log scale).
  (c) Histogram of streamflow record length.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
STATS_CSV = BASE_DIR / "analysis" / "catchment_stats.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ICELAND_GRU_SHP = (
    BASE_DIR.parent / "1. Domain definition" / "shapefiles" / "Iceland"
    / "catchment" / "semidistributed" / "regional_tutorial" / "Iceland_HRUs_GRUs.shp"
)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_catchment_boundaries():
    """Load all catchment boundary polygons."""
    gdfs = []
    stats = pd.read_csv(STATS_CSV)
    for _, row in stats.iterrows():
        did = int(row["domain_id"])
        shp = DATA_DIR / f"domain_{did}/shapefiles/catchment/{did}_HRUs_GRUs_wgs84.shp"
        if shp.exists():
            gdf = gpd.read_file(shp)
            gdf["domain_id"] = did
            gdf["area_km2"] = row["area_km2"]
            gdf["elev_mean_m"] = row["elev_mean_m"]
            gdf["glac_fra"] = row["glac_fra"]
            gdf["record_years"] = row["record_years"]
            gdfs.append(gdf)
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))


def main():
    stats = pd.read_csv(STATS_CSV)

    # Load catchment polygons
    print("Loading catchment boundaries...")
    catchments = load_catchment_boundaries()

    # Load Iceland national GRU shapefile for context
    print("Loading Iceland GRU context layer...")
    iceland_grus = gpd.read_file(ICELAND_GRU_SHP)

    # ── Figure layout ──
    # Use gridspec: map on left (wide), two histograms stacked on right
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[2.2, 1],
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.40,
    )

    ax_map = fig.add_subplot(gs[:, 0])   # map spans both rows
    ax_area = fig.add_subplot(gs[0, 1])  # top-right
    ax_rec = fig.add_subplot(gs[1, 1])   # bottom-right

    # ── Panel (a): Map ──
    # Iceland national GRU context layer
    iceland_grus.plot(ax=ax_map, color="#f0ece3", edgecolor="#d5d0c4",
                      linewidth=0.08, zorder=1)

    # Catchment polygons coloured by log(area)
    log_area = np.log10(catchments["area_km2"].clip(lower=1))
    vmin, vmax = log_area.min(), log_area.max()
    catchments.plot(
        ax=ax_map,
        column=log_area,
        cmap="YlGnBu",
        edgecolor="#444444",
        linewidth=0.15,
        alpha=0.85,
        zorder=2,
        legend=False,
    )

    # Highlight glacierized catchments (>30%)
    glac = catchments[catchments["glac_fra"] > 0.3]
    if not glac.empty:
        glac.plot(ax=ax_map, facecolor="none", edgecolor="#b03a2e",
                  linewidth=0.4, linestyle="--", zorder=3)

    # Map extent — geographic aspect correction for ~65°N
    ax_map.set_xlim(-25.0, -13.0)
    ax_map.set_ylim(63.0, 66.8)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    # Correct aspect: 1/cos(65°) ≈ 2.37 to avoid stretching Iceland E–W
    ax_map.set_aspect(1.0 / np.cos(np.radians(65.0)))

    # Grid
    ax_map.grid(True, linewidth=0.3, alpha=0.4, color="#999999")

    # Colourbar — horizontal below the map, with space below xlabel
    sm = plt.cm.ScalarMappable(
        cmap="YlGnBu",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, orientation="horizontal",
                        fraction=0.035, pad=0.12, shrink=0.5)
    tick_vals = [1, 2, 3, 3.87]  # log10 of 10, 100, 1000, ~7500
    tick_labels = ["10", "100", "1 000", "7 500"]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Catchment area (km²)", fontsize=8)

    # Panel label — upper left, in ocean
    ax_map.text(0.015, 0.97, "(a)", transform=ax_map.transAxes,
                fontsize=11, fontweight="bold", va="top",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Summary stats annotation — positioned in ocean SW of Iceland
    n_total = len(stats)
    n_glac = (stats["glac_fra"] > 0).sum()
    total_area = stats["area_km2"].sum()
    summary = (
        f"n = {n_total} catchments\n"
        f"Total area: {total_area/1000:.0f} \u00d7 10\u00b3 km\u00b2\n"
        f"Glacierized: {n_glac} ({100*n_glac/n_total:.0f}%)\n"
        f"Elev: {stats['elev_mean_m'].min():.0f}\u2013{stats['elev_mean_m'].max():.0f} m"
    )
    ax_map.text(
        -24.8, 63.15, summary, fontsize=7.5,
        va="bottom", ha="left", family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # Legend — positioned in ocean SE of Iceland
    legend_elements = [
        Line2D([0], [0], color="#b03a2e", linewidth=0.8, linestyle="--",
               label="Glacier fraction > 30%"),
    ]
    ax_map.legend(handles=legend_elements, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor="#cccccc", fontsize=7)

    # ── Panel (b): Area histogram ──
    areas = stats["area_km2"]
    ax_area.hist(areas, bins=np.logspace(np.log10(1), np.log10(10000), 25),
                 color="#4a90d9", edgecolor="#333333", linewidth=0.4, alpha=0.85)
    ax_area.set_xscale("log")
    ax_area.set_xlabel("Area (km²)")
    ax_area.set_ylabel("Count")
    ax_area.text(0.04, 0.94, "(b)", transform=ax_area.transAxes,
                 fontsize=10, fontweight="bold", va="top")
    median_area = areas.median()
    ax_area.axvline(median_area, color="#b03a2e", linewidth=0.8, linestyle="--")
    ax_area.text(median_area * 1.3, ax_area.get_ylim()[1] * 0.85,
                 f"median\n{median_area:.0f} km²", fontsize=7, color="#b03a2e")

    # ── Panel (c): Record length histogram ──
    rec = stats["record_years"]
    ax_rec.hist(rec, bins=20, color="#6ab04c", edgecolor="#333333",
                linewidth=0.4, alpha=0.85)
    ax_rec.set_xlabel("Record length (years)")
    ax_rec.set_ylabel("Count")
    ax_rec.text(0.04, 0.94, "(c)", transform=ax_rec.transAxes,
                fontsize=10, fontweight="bold", va="top")
    median_rec = rec.median()
    ax_rec.axvline(median_rec, color="#b03a2e", linewidth=0.8, linestyle="--")
    ax_rec.text(median_rec + 2, ax_rec.get_ylim()[1] * 0.85,
                f"median\n{median_rec:.0f} yr", fontsize=7, color="#b03a2e")

    # Save
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"fig_large_sample_overview.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
