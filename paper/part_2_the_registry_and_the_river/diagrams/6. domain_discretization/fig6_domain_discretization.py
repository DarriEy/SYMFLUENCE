"""
SYMFLUENCE GRU → HRU spatial hierarchy diagram (Section 3.4).

Uses real Bow-at-Banff domain data:
  - DEM with hillshade, 49 GRU subcatchments, river network
  - Zoom into GRU 38: elevation-only (11 HRUs) vs elevation×aspect (78 HRUs)

Layout:
  (a) Left — full catchment DEM + GRU boundaries + river network + inset
  (b)+(c) Top-right — GRU 38 zoom: elevation bands vs elevation × aspect
  (d) Bottom-right — Conceptual hierarchy schematic (3 levels)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.colors import LightSource, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import mapping

# ── paths ────────────────────────────────────────────────────────────
BASE_SD = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_elevation"
DEM_PATH = f"{BASE_SD}/attributes/elevation/dem/domain_Bow_at_Banff_elevation_elv.tif"
GRU_PATH = f"{BASE_SD}/shapefiles/river_basins/Bow_at_Banff_elevation_riverBasins_delineate.shp"
RIV_PATH = f"{BASE_SD}/shapefiles/river_network/Bow_at_Banff_elevation_riverNetwork_delineate.shp"

BASE_HRU = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/Applications and validation /1. Domain definition/shapefiles/bow/catchment/semidistributed/run_1"
ELEV_HRU_PATH   = f"{BASE_HRU}/Bow_at_Banff_lumped_era5_HRUs_elevation.shp"
ASPECT_HRU_PATH = f"{BASE_HRU}/Bow_at_Banff_lumped_era5_HRUs_elevation_aspect.shp"

NE_LAND   = "/Users/darrieythorsson/.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_land.shp"
NE_BORDER = "/Users/darrieythorsson/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_50m_admin_0_boundary_lines_land.shp"

OUT_DIR = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/6. domain_discretization"

# ── style ────────────────────────────────────────────────────────────
TEXT_DARK  = "#2D2D2D"
TEXT_GREY  = "#555555"
TEXT_WHITE = "#FFFFFF"
RIVER_CLR  = "#2166AC"
GRU_EDGE   = "#2D2D2D"
HIGHLIGHT  = "#D62728"
CATCH_EDGE = "#1a1a1a"

ELEV_CMAP = plt.colormaps["RdYlGn_r"]

ASPECT_COLORS = {
    0: "#B0B0B0", 1: "#1B3A8C", 2: "#5B72C4", 3: "#F5D300", 4: "#FF8C00",
    5: "#D62728", 6: "#FF5CAD", 7: "#8B2FC9", 8: "#3AAFCF",
}
ASPECT_LABELS = {
    0: "Flat", 1: "N", 2: "NE", 3: "E", 4: "SE",
    5: "S", 6: "SW", 7: "W", 8: "NW",
}

ZOOM_GRU_ID = 38

# ── load data ────────────────────────────────────────────────────────
gru = gpd.read_file(GRU_PATH)
riv = gpd.read_file(RIV_PATH)
catchment_dissolved = gru.dissolve()

all_elev_hrus = gpd.read_file(ELEV_HRU_PATH)
all_aspect_hrus = gpd.read_file(ASPECT_HRU_PATH)
all_aspect_hrus["aspect_class"] = all_aspect_hrus["combined_e"].str.split("_").str[0].astype(int)

zoom_gru_geom = gru[gru["GRU_ID"] == ZOOM_GRU_ID]
zoom_elev_hrus = all_elev_hrus[all_elev_hrus["GRU_ID"] == ZOOM_GRU_ID].sort_values("elev_mean")
zoom_asp_hrus = all_aspect_hrus[all_aspect_hrus["GRU_ID"] == ZOOM_GRU_ID].copy()

zoom_seg_id = zoom_gru_geom["gru_to_seg"].iloc[0]
zoom_river_seg = riv[riv["LINKNO"] == zoom_seg_id]

ne_land = gpd.read_file(NE_LAND)
ne_border = gpd.read_file(NE_BORDER)

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(float)
    dem[dem == src.nodata] = np.nan
    dem_extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

# ── hillshade ────────────────────────────────────────────────────────
ls = LightSource(azdeg=315, altdeg=45)
dem_filled = np.nan_to_num(dem, nan=float(np.nanmin(dem)))
terrain_cmap = plt.cm.terrain

elev_norm = Normalize(vmin=np.nanmin(dem), vmax=np.nanmax(dem))
terrain_rgb = terrain_cmap(elev_norm(dem_filled))
blended = ls.shade_rgb(terrain_rgb[:, :, :3], dem_filled, blend_mode="hsv")

# ── clip DEM for zoom ────────────────────────────────────────────────
with rasterio.open(DEM_PATH) as src:
    zoom_geom_json = [mapping(zoom_gru_geom.geometry.iloc[0])]
    zoom_dem, zoom_transform = rio_mask(src, zoom_geom_json, crop=True, nodata=np.nan)
    zoom_dem = zoom_dem[0].astype(float)

zoom_bounds = zoom_gru_geom.total_bounds
zoom_dem_filled = np.nan_to_num(zoom_dem, nan=float(np.nanmin(np.nan_to_num(zoom_dem, nan=0))))
zoom_rgb = terrain_cmap(Normalize(vmin=np.nanmin(zoom_dem), vmax=np.nanmax(zoom_dem))(zoom_dem_filled))
zoom_blended = ls.shade_rgb(zoom_rgb[:, :, :3], zoom_dem_filled, blend_mode="hsv")

zoom_pixel_extent = [
    zoom_transform.c,
    zoom_transform.c + zoom_transform.a * zoom_dem.shape[1],
    zoom_transform.f + zoom_transform.e * zoom_dem.shape[0],
    zoom_transform.f,
]

# ── mask paths ───────────────────────────────────────────────────────
full_bounds = gru.total_bounds
cat_geom = catchment_dissolved.geometry.iloc[0]
x0, x1 = dem_extent[0], dem_extent[1]
y0, y1 = dem_extent[2], dem_extent[3]
outer_rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])

cat_ext = np.array(list(
    cat_geom.geoms[0].exterior.coords if cat_geom.geom_type == "MultiPolygon"
    else cat_geom.exterior.coords
))
outer_codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer_rect) - 2) + [MplPath.CLOSEPOLY]
inner_codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(cat_ext) - 2) + [MplPath.CLOSEPOLY]
catch_mask_verts = np.concatenate([outer_rect, cat_ext[::-1]])
catch_mask_codes = outer_codes + inner_codes

gru_geom = zoom_gru_geom.geometry.iloc[0]
gru_ext = np.array(list(
    gru_geom.geoms[0].exterior.coords if gru_geom.geom_type == "MultiPolygon"
    else gru_geom.exterior.coords
))


def make_gru_mask_patch():
    x0z = zoom_pixel_extent[0] - 0.1
    x1z = zoom_pixel_extent[1] + 0.1
    y0z = zoom_pixel_extent[2] - 0.1
    y1z = zoom_pixel_extent[3] + 0.1
    outer_z = np.array([[x0z, y0z], [x1z, y0z], [x1z, y1z], [x0z, y1z], [x0z, y0z]])
    oc = [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer_z) - 2) + [MplPath.CLOSEPOLY]
    ic = [MplPath.MOVETO] + [MplPath.LINETO] * (len(gru_ext) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(np.concatenate([outer_z, gru_ext[::-1]]), oc + ic),
                     facecolor="white", edgecolor="none", alpha=0.85, zorder=1.5)


# ══════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 9.5), facecolor="white")

# Outer 1×2: left = (a), right = stacked (b+c) / (d)
gs_outer = fig.add_gridspec(1, 2, width_ratios=[0.48, 0.52],
                            wspace=0.08,
                            left=0.03, right=0.97, bottom=0.03, top=0.92)

ax_catch = fig.add_subplot(gs_outer[0, 0])

# Right column: top = zoom panels, bottom = hierarchy
gs_right = gs_outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.95],
                                       hspace=0.04)

# Zoom area: (b) and (c) equal width
gs_zoom = gs_right[0].subgridspec(1, 2, wspace=0.14)
ax_elev = fig.add_subplot(gs_zoom[0, 0])
ax_asp  = fig.add_subplot(gs_zoom[0, 1])

ax_hier = fig.add_subplot(gs_right[1])

# ════════════════════════════════════════════════════════════════════
# Panel (a): Full catchment
# ════════════════════════════════════════════════════════════════════

ax_catch.imshow(blended, extent=dem_extent, aspect="auto", zorder=0)
ax_catch.add_patch(PathPatch(MplPath(catch_mask_verts, catch_mask_codes),
                              facecolor="white", edgecolor="none", alpha=0.65, zorder=1))

catchment_dissolved.boundary.plot(ax=ax_catch, color=CATCH_EDGE, linewidth=2.5, zorder=5)
gru.boundary.plot(ax=ax_catch, color=GRU_EDGE, linewidth=0.8, alpha=0.75, zorder=2)
riv.plot(ax=ax_catch, color=RIVER_CLR, linewidth=1.4, alpha=0.9, zorder=3)
zoom_gru_geom.boundary.plot(ax=ax_catch, color=HIGHLIGHT, linewidth=2.5, zorder=4)

ax_catch.set_xlim(full_bounds[0] - 0.02, full_bounds[2] + 0.02)
ax_catch.set_ylim(full_bounds[1] - 0.02, full_bounds[3] + 0.02)

zc = zoom_gru_geom.geometry.iloc[0].centroid
ax_catch.annotate(
    f"GRU {ZOOM_GRU_ID}", xy=(zc.x, zc.y),
    xytext=(zc.x + 0.15, zc.y + 0.10),
    fontsize=8, fontweight="bold", color=HIGHLIGHT,
    arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=1.2),
    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    zorder=6,
)

ax_catch.set_xlabel("Longitude", fontsize=8, color=TEXT_GREY)
ax_catch.set_ylabel("Latitude", fontsize=8, color=TEXT_GREY)
ax_catch.tick_params(labelsize=7, colors=TEXT_GREY)
ax_catch.set_title("(a)  Catchment: 49 GRUs + river network",
                    fontsize=10.5, fontweight="bold", color=TEXT_DARK, pad=10)

leg_items = [
    mpatches.Patch(edgecolor=CATCH_EDGE, facecolor="none", linewidth=2.0,
                   label="Catchment boundary"),
    mpatches.Patch(edgecolor=GRU_EDGE, facecolor="none", linewidth=0.8,
                   label="GRU boundaries"),
    Line2D([0], [0], color=RIVER_CLR, linewidth=1.4, label="River network"),
    mpatches.Patch(edgecolor=HIGHLIGHT, facecolor="#FFCCCC", linewidth=2.0,
                   alpha=0.5, label=f"GRU {ZOOM_GRU_ID} (zoom)"),
]
ax_catch.legend(handles=leg_items, loc="upper right", fontsize=6.5,
                framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
                handlelength=1.5, handletextpad=0.5)

# Natural Earth inset — positioned flush in lower-left corner
ax_inset = ax_catch.inset_axes([-0.02, -0.02, 0.50, 0.40])
ax_inset.set_xlim(-145, -50)
ax_inset.set_ylim(22, 72)
ax_inset.set_facecolor("#D6EAF8")
ne_land.plot(ax=ax_inset, facecolor="#E8E4D8", edgecolor="#AAAAAA",
             linewidth=0.3, zorder=1)
ne_border.plot(ax=ax_inset, color="#999999", linewidth=0.3, zorder=2)

cat_cx = (full_bounds[0] + full_bounds[2]) / 2
cat_cy = (full_bounds[1] + full_bounds[3]) / 2
ax_inset.plot(cat_cx, cat_cy, "*", color=HIGHLIGHT, markersize=14,
              markeredgecolor="white", markeredgewidth=0.8, zorder=4)
ax_inset.text(cat_cx + 4, cat_cy - 5, "Bow at\nBanff",
              fontsize=6, color=TEXT_DARK, fontweight="bold",
              path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
              zorder=5, linespacing=1.1)
ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.8)

# ════════════════════════════════════════════════════════════════════
# Helper: plot zoom GRU panel
# ════════════════════════════════════════════════════════════════════

def plot_zoom_panel(ax, hru_gdf, cmap_func, title, show_labels=True):
    ax.imshow(zoom_blended, extent=zoom_pixel_extent, aspect="auto", zorder=0)
    ax.add_patch(make_gru_mask_patch())

    for _, row in hru_gdf.iterrows():
        color = cmap_func(row)
        geom = row.geometry
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for part in polys:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, facecolor=color, edgecolor="white",
                    linewidth=0.5, alpha=0.55, zorder=2)

    hru_gdf.boundary.plot(ax=ax, color="white", linewidth=0.4, alpha=0.8, zorder=3)

    zoom_river_seg.plot(ax=ax, color="white", linewidth=4.0, alpha=0.4, zorder=3.5)
    zoom_river_seg.plot(ax=ax, color=RIVER_CLR, linewidth=2.0, alpha=0.9, zorder=4)

    zoom_gru_geom.boundary.plot(ax=ax, color=HIGHLIGHT, linewidth=2.5, zorder=5)

    # Elevation labels — greedy vertical de-overlap
    if show_labels and len(hru_gdf) <= 15:
        sorted_hrus = hru_gdf.sort_values("elev_mean")
        entries = [(row.geometry.centroid.x, row.geometry.centroid.y,
                    int(row["elev_mean"]), row.geometry.area)
                   for _, row in sorted_hrus.iterrows()]

        pad_lbl = 0.008
        y_lo = zoom_bounds[1] + pad_lbl
        y_hi = zoom_bounds[3] - pad_lbl
        x_lo = zoom_bounds[0] + pad_lbl
        x_hi = zoom_bounds[2] - pad_lbl

        # Sort bottom to top
        entries.sort(key=lambda e: e[1])
        placed = []  # (y, x) of placed labels
        min_y_gap = 0.008
        min_x_gap = 0.020

        for cx, cy, elev_val, area in entries:
            proposed_y = max(y_lo, min(cy, y_hi))
            proposed_x = max(x_lo, min(cx, x_hi))

            # Check for overlap with all previously placed labels
            for py, px in placed:
                if abs(proposed_y - py) < min_y_gap and abs(proposed_x - px) < min_x_gap:
                    proposed_y = py + min_y_gap

            proposed_y = min(proposed_y, y_hi)

            # Skip if still overlapping (too crowded)
            skip = False
            for py, px in placed:
                if abs(proposed_y - py) < min_y_gap * 0.6 and abs(proposed_x - px) < min_x_gap * 0.6:
                    skip = True
                    break
            if skip:
                continue

            ax.text(proposed_x, proposed_y, f"{elev_val} m",
                    ha="center", va="center", fontsize=5.5,
                    fontweight="bold", color=TEXT_DARK,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=6)
            placed.append((proposed_y, proposed_x))

    pad = 0.005
    ax.set_xlim(zoom_bounds[0] - pad, zoom_bounds[2] + pad)
    ax.set_ylim(zoom_bounds[1] - pad, zoom_bounds[3] + pad)
    ax.set_xlabel("Longitude", fontsize=7, color=TEXT_GREY)
    ax.tick_params(labelsize=6, colors=TEXT_GREY)
    ax.set_title(title, fontsize=9, fontweight="bold", color=TEXT_DARK, pad=6)


# ════════════════════════════════════════════════════════════════════
# Panel (b): GRU 38 — elevation bands
# ════════════════════════════════════════════════════════════════════

elev_min = zoom_elev_hrus["elev_mean"].min()
elev_max = zoom_elev_hrus["elev_mean"].max()

def elev_color(row):
    t = (row["elev_mean"] - elev_min) / max(elev_max - elev_min, 1)
    return ELEV_CMAP(t)

plot_zoom_panel(ax_elev, zoom_elev_hrus, elev_color,
                f"(b)  Elevation bands — {len(zoom_elev_hrus)} HRUs")

# Small inset colorbar in upper-left white area (outside basin)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax_elev, width="5%", height="40%", loc="upper left",
                 bbox_to_anchor=(0.02, -0.04, 1, 1), bbox_transform=ax_elev.transAxes,
                 borderpad=0)
sm = plt.cm.ScalarMappable(cmap=ELEV_CMAP,
                            norm=Normalize(vmin=elev_min, vmax=elev_max))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Elev. (m)", fontsize=5.5, color=TEXT_GREY, labelpad=3)
cbar.ax.tick_params(labelsize=4.5)

# ════════════════════════════════════════════════════════════════════
# Panel (c): GRU 38 — elevation × aspect
# ════════════════════════════════════════════════════════════════════

def aspect_color(row):
    return ASPECT_COLORS.get(row["aspect_class"], "#B0B0B0")

plot_zoom_panel(ax_asp, zoom_asp_hrus, aspect_color,
                f"(c)  Elev. × aspect — {len(zoom_asp_hrus)} HRUs",
                show_labels=False)

asp_patches = [mpatches.Patch(facecolor=ASPECT_COLORS[k], edgecolor="white",
                               label=ASPECT_LABELS[k])
               for k in sorted(ASPECT_COLORS.keys())]
asp_patches.append(Line2D([0], [0], color=RIVER_CLR, linewidth=2.0,
                           label="River segment"))
ax_asp.legend(handles=asp_patches, loc="lower right", fontsize=5,
              framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
              title="Aspect class", title_fontsize=6, ncol=2,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

# ════════════════════════════════════════════════════════════════════
# Bracket around (b) + colorbar + (c)
# ════════════════════════════════════════════════════════════════════

fig.canvas.draw()
pos_elev = ax_elev.get_position()
pos_asp  = ax_asp.get_position()

bracket_left  = pos_elev.x0 - 0.008
bracket_right = pos_asp.x1 + 0.008
bracket_top   = max(pos_elev.y1, pos_asp.y1) + 0.025
bracket_bot   = min(pos_elev.y0, pos_asp.y0) - 0.032  # more room for axis labels

bracket_color = "#888888"
bracket_lw = 1.5

for xs, ys in [
    ([bracket_left, bracket_right], [bracket_top, bracket_top]),
    ([bracket_left, bracket_right], [bracket_bot, bracket_bot]),
    ([bracket_left, bracket_left], [bracket_bot, bracket_top]),
    ([bracket_right, bracket_right], [bracket_bot, bracket_top]),
]:
    fig.add_artist(Line2D(xs, ys, transform=fig.transFigure,
                          color=bracket_color, lw=bracket_lw,
                          clip_on=False, zorder=10))

fig.text((bracket_left + bracket_right) / 2, bracket_top + 0.008,
         f"GRU {ZOOM_GRU_ID}: HRU discretization comparison",
         ha="center", va="bottom", fontsize=10, fontweight="bold",
         color=TEXT_DARK)

# ════════════════════════════════════════════════════════════════════
# Connector lines: panel (a) GRU 38 corners → bracket
# ════════════════════════════════════════════════════════════════════

# Connect from GRU 38 top/bottom edges to bracket top-left/bottom-left
# Use the midpoint of the top and bottom edges of the bounding box
zoom_xmid = (zoom_bounds[0] + zoom_bounds[2]) / 2
zoom_top = (zoom_xmid, zoom_bounds[3])   # top-center of GRU 38
zoom_bot = (zoom_xmid, zoom_bounds[1])   # bottom-center of GRU 38

for (x_from, y_from), (x_to, y_to) in [
    (zoom_top, (bracket_left, bracket_top)),
    (zoom_bot, (bracket_left, bracket_bot)),
]:
    con = mpatches.ConnectionPatch(
        xyA=(x_from, y_from), coordsA=ax_catch.transData,
        xyB=(x_to, y_to), coordsB=fig.transFigure,
        arrowstyle="-", color=HIGHLIGHT, lw=1.2,
        linestyle="--", alpha=0.7, zorder=10,
    )
    fig.add_artist(con)

# ════════════════════════════════════════════════════════════════════
# Panel (d): Conceptual hierarchy — 3 levels (compact for half-width)
# ════════════════════════════════════════════════════════════════════

# Use full axis width; centre content at midpoint
ax_hier.set_xlim(-3.5, 24)
ax_hier.set_ylim(-0.5, 7.6)
ax_hier.axis("off")
ax_hier.set_title("(d)  Spatial hierarchy: multi-level discretization",
                   fontsize=10, fontweight="bold", color=TEXT_DARK, pad=6)

MID = 13.1

# ── Level 1: Catchment → GRUs ──
L1_BOT = 5.6
L1_H = 1.4
gru_w = 2.3

catch_box = FancyBboxPatch((1.8, L1_BOT), 3.8, L1_H,
                            boxstyle="round,pad=0.12",
                            facecolor="#E8E8E8", edgecolor=GRU_EDGE,
                            linewidth=1.5, zorder=2)
ax_hier.add_patch(catch_box)
ax_hier.text(3.7, L1_BOT + L1_H / 2, "Catchment", ha="center", va="center",
             fontsize=9.5, fontweight="bold", color=TEXT_DARK, zorder=3)

ax_hier.annotate("", xy=(6.4, L1_BOT + L1_H / 2), xytext=(5.6, L1_BOT + L1_H / 2),
                 arrowprops=dict(arrowstyle="-|>", color=TEXT_GREY, lw=1.5))

gru_labels = ["GRU 1", "GRU 2", f"GRU {ZOOM_GRU_ID}", "···", "GRU n−1", "GRU n"]
gru_xs     = [6.6,      9.4,      12.2,                 15.0,  15.8,      18.6]
gru_fcs    = ["#D5CCE0", "#D5CCE0", "#FFCCCC",          None,  "#D5CCE0", "#D5CCE0"]
gru_ecs    = [GRU_EDGE,  GRU_EDGE,  HIGHLIGHT,          None,  GRU_EDGE,  GRU_EDGE]
gru_lws    = [1.2,       1.2,       2.2,                 None,  1.2,       1.2]
gru_tcs    = [TEXT_DARK,  TEXT_DARK,  HIGHLIGHT,          TEXT_GREY, TEXT_DARK, TEXT_DARK]

for gx, gl, fc, ec, lw, tc in zip(gru_xs, gru_labels, gru_fcs, gru_ecs, gru_lws, gru_tcs):
    if fc is None:
        ax_hier.text(gx, L1_BOT + L1_H / 2, gl, ha="center", va="center",
                     fontsize=13, color=tc, zorder=3)
        continue
    box = FancyBboxPatch((gx, L1_BOT), gru_w, L1_H, boxstyle="round,pad=0.1",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax_hier.add_patch(box)
    ax_hier.text(gx + gru_w / 2, L1_BOT + L1_H / 2, gl, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=tc, zorder=3)

gru38_cx = 12.2 + gru_w / 2  # 13.35

ax_hier.text(MID, L1_BOT + L1_H + 0.15,
             "Routing structure  (drainage topology & lateral connectivity)",
             ha="center", va="bottom", fontsize=7.5, color=TEXT_GREY, fontstyle="italic")

# ── Level 2: Single-attribute HRUs ──
L2_BOT = 3.0
L2_H = 1.35
ax_hier.annotate("", xy=(gru38_cx, L2_BOT + L2_H), xytext=(gru38_cx, L1_BOT),
                 arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=1.5))

level2_options = [
    ("Elev.\nband 1", ELEV_CMAP(0.1)),
    ("Elev.\nband 2", ELEV_CMAP(0.35)),
    ("Elev.\nband n", ELEV_CMAP(0.7)),
    ("Land\ncover 1", "#7CB342"),
    ("Land\ncover m", "#558B2F"),
    ("Soil\ntype 1", "#C9943A"),
    ("Soil\ntype p", "#A67B2E"),
]
n_l2 = len(level2_options)
l2_w, l2_gap = 2.20, 0.16
total_l2 = n_l2 * l2_w + (n_l2 - 1) * l2_gap
l2_start = gru38_cx - total_l2 / 2

for i, (label, color) in enumerate(level2_options):
    hx = l2_start + i * (l2_w + l2_gap)
    box = FancyBboxPatch((hx, L2_BOT), l2_w, L2_H, boxstyle="round,pad=0.07",
                          facecolor=color, edgecolor="white",
                          linewidth=1.0, alpha=0.85, zorder=2)
    ax_hier.add_patch(box)
    ax_hier.text(hx + l2_w / 2, L2_BOT + L2_H / 2, label, ha="center", va="center",
                 fontsize=6.8, fontweight="bold", color=TEXT_WHITE, zorder=3,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#00000055")],
                 linespacing=1.15)
    ax_hier.annotate("", xy=(hx + l2_w / 2, L2_BOT + L2_H), xytext=(gru38_cx, L1_BOT),
                     arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT,
                                    lw=0.5, alpha=0.2))

# Separators between attribute groups
for idx in [3, 5]:
    sx = l2_start + idx * (l2_w + l2_gap) - l2_gap / 2
    ax_hier.text(sx, L2_BOT + L2_H / 2, "|", ha="center", va="center",
                 fontsize=12, color="#BBBBBB", zorder=1)

ax_hier.text(gru38_cx, L2_BOT + L2_H + 0.15,
             "Single-attribute discretization  (one attribute at a time)",
             ha="center", va="bottom", fontsize=7.5, color=TEXT_GREY, fontstyle="italic")

# ── Level 3: Combined-attribute HRUs ──
L3_BOT = 0.1
L3_H = 1.35
ax_hier.annotate("", xy=(gru38_cx, L3_BOT + L3_H), xytext=(gru38_cx, L2_BOT),
                 arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3))
ax_hier.text(gru38_cx + 0.25, (L2_BOT + L3_BOT + L3_H) / 2, "combine",
             ha="left", va="center",
             fontsize=6.5, color=TEXT_GREY, fontstyle="italic", rotation=90)

level3_options = [
    ("Elev. × Aspect\n(N, 1800 m)", "#1B3A8C"),
    ("Elev. × Aspect\n(S, 2200 m)", "#D62728"),
    ("Elev. × Land\n(Forest, 1800 m)", "#2E7D32"),
    ("Elev. × Land\n× Soil", "#795548"),
    ("···", None),
]
n_l3 = len(level3_options) - 1
l3_w, l3_gap = 3.7, 0.20
total_l3 = n_l3 * l3_w + (n_l3 - 1) * l3_gap
l3_start = gru38_cx - total_l3 / 2

for i, (label, color) in enumerate(level3_options):
    if color is None:
        hx = l3_start + n_l3 * (l3_w + l3_gap)
        ax_hier.text(hx, L3_BOT + L3_H / 2, label, ha="center", va="center",
                     fontsize=12, color=TEXT_GREY, zorder=3)
        continue
    hx = l3_start + i * (l3_w + l3_gap)
    box = FancyBboxPatch((hx, L3_BOT), l3_w, L3_H, boxstyle="round,pad=0.07",
                          facecolor=color, edgecolor="white",
                          linewidth=1.0, alpha=0.85, zorder=2)
    ax_hier.add_patch(box)
    ax_hier.text(hx + l3_w / 2, L3_BOT + L3_H / 2, label, ha="center", va="center",
                 fontsize=6.5, fontweight="bold", color=TEXT_WHITE, zorder=3,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#00000055")],
                 linespacing=1.15)

ax_hier.text(gru38_cx, L3_BOT - 0.12,
             "Combined-attribute discretization  (intersection of multiple attributes — extensible)",
             ha="center", va="top", fontsize=7.5, color=TEXT_GREY, fontstyle="italic")

for y, label in [
    (L1_BOT + L1_H / 2, "Level 1:\nGrouped Response\nUnits (GRUs)"),
    (L2_BOT + L2_H / 2, "Level 2:\nSingle-attribute\nHRUs"),
    (L3_BOT + L3_H / 2, "Level 3:\nCombined-attribute\nHRUs"),
]:
    ax_hier.text(-0.2, y, label, ha="right", va="center",
                 fontsize=7, color=TEXT_GREY, fontstyle="italic", linespacing=1.3)

# ── suptitle ─────────────────────────────────────────────────────────
fig.suptitle("Spatial Discretization Hierarchy:  Catchment → GRU → HRU",
             fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.97)

# ── save ─────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/fig6_domain_discretization.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    print(f"Saved fig6_domain_discretization.{fmt}")
plt.close(fig)
