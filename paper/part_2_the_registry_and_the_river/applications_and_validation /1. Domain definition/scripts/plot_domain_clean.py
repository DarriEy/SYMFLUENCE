#!/usr/bin/env python3
"""
SYMFLUENCE Paper Figure: Domain Definition - Clean Version
Section 4.1
"""

import matplotlib
matplotlib.use('Agg')

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import contextily as cx
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/Applications and validation /1. Domain definition")
SHP_DIR = BASE_DIR / "shapefiles"
OUTPUT_DIR = BASE_DIR / "figures"

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

# Iceland color palette
ICELAND_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
]


def create_main_figure():
    """Create main 2-row figure."""
    print("Creating main figure...")

    # Load all data fresh
    para_pp = gpd.read_file(SHP_DIR / "paradise/pour_point/paradise_pourPoint.shp").to_crs(epsg=3857)
    para_forcing = gpd.read_file(SHP_DIR / "paradise/forcing/forcing_ERA5.shp").to_crs(epsg=3857)

    ice_basins = gpd.read_file(SHP_DIR / "Iceland/river_basins/Iceland_riverBasins_with_coastal.shp").to_crs(epsg=3857)
    ice_rivers = gpd.read_file(SHP_DIR / "Iceland/river_network/Iceland_riverNetwork_semidistributed.shp").to_crs(epsg=3857)

    bow_pp = gpd.read_file(SHP_DIR / "bow/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp").to_crs(epsg=3857)
    bow_lumped = gpd.read_file(SHP_DIR / "bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp").to_crs(epsg=3857)
    bow_grus = gpd.read_file(SHP_DIR / "bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp").to_crs(epsg=3857)
    bow_elev = gpd.read_file(SHP_DIR / "bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp").to_crs(epsg=3857)
    bow_r1 = gpd.read_file(SHP_DIR / "bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_lumped.shp").to_crs(epsg=3857)
    bow_rfull = gpd.read_file(SHP_DIR / "bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_distributed_1000m.shp").to_crs(epsg=3857)

    # Create figure
    fig = plt.figure(figsize=(16, 9))

    # Create axes manually
    ax_para = fig.add_axes([0.02, 0.52, 0.46, 0.42])  # [left, bottom, width, height]
    ax_ice = fig.add_axes([0.52, 0.52, 0.46, 0.42])
    ax_bow = [
        fig.add_axes([0.02, 0.08, 0.22, 0.38]),
        fig.add_axes([0.26, 0.08, 0.22, 0.38]),
        fig.add_axes([0.50, 0.08, 0.22, 0.38]),
        fig.add_axes([0.74, 0.08, 0.22, 0.38]),
    ]

    # ===== PARADISE =====
    print("  Paradise...")
    para_forcing.plot(ax=ax_para, facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.5, alpha=0.95)
    for idx, row in para_forcing.iterrows():
        c = row.geometry.centroid
        ax_para.text(c.x, c.y, str(idx+1), ha='center', va='center', fontsize=14, fontweight='bold', color='#BF360C')
    pp_geom = para_pp.iloc[0].geometry
    ax_para.scatter(pp_geom.x, pp_geom.y, c='#D32F2F', s=400, marker='^', edgecolor='white', linewidth=2.5, zorder=10)
    ax_para.annotate('Paradise\nSNOTEL', (pp_geom.x, pp_geom.y), xytext=(15, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#B71C1C',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    try:
        cx.add_basemap(ax_para, source=cx.providers.CartoDB.Positron, zoom=11)
    except:
        pass
    ax_para.set_title('(a) Point-scale: Paradise SNOTEL, Washington', fontweight='bold', fontsize=11, pad=8)
    ax_para.text(0.02, 0.98, 'Single location\n9 ERA5 grid cells\nNo routing required',
                transform=ax_para.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='#666'))
    ax_para.set_xticks([]); ax_para.set_yticks([])

    # ===== ICELAND =====
    print("  Iceland...")
    np.random.seed(42)
    ice_basins['cid'] = np.random.permutation(len(ice_basins)) % len(ICELAND_COLORS)
    ice_basins.plot(ax=ax_ice, column='cid', cmap=ListedColormap(ICELAND_COLORS),
                   edgecolor='white', linewidth=0.4, alpha=0.9)
    ice_rivers.plot(ax=ax_ice, color='#0D47A1', linewidth=0.5, alpha=0.8)
    try:
        cx.add_basemap(ax_ice, source=cx.providers.CartoDB.Positron, zoom=6)
    except:
        pass
    area = ice_basins.to_crs(epsg=32627).area.sum() / 1e6
    ax_ice.set_title('(b) Regional-scale: Iceland', fontweight='bold', fontsize=11, pad=8)
    ax_ice.text(0.02, 0.98, f'{len(ice_basins)} GRUs\n{area:,.0f} km²\nFull river network',
               transform=ax_ice.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='#666'))
    ax_ice.set_xticks([]); ax_ice.set_yticks([])

    # ===== BOW RIVER PANELS =====
    print("  Bow River...")
    pp_pt = bow_pp.iloc[0].geometry
    bounds = bow_lumped.total_bounds
    pad = 2000
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)

    # Elevation range
    vmin = min(bow_lumped['elev_mean'].min(), bow_grus['elev_mean'].min(), bow_elev['elev_mean'].min())
    vmax = max(bow_lumped['elev_mean'].max(), bow_grus['elev_mean'].max(), bow_elev['elev_mean'].max())

    configs = [
        (ax_bow[0], bow_lumped, bow_r1, '(c) Lumped', 1, 1, 3),
        (ax_bow[1], bow_lumped, bow_rfull, '(d) Lumped + Dist.', 1, len(bow_rfull), 3),
        (ax_bow[2], bow_grus, bow_rfull, '(e) Sub-basin GRUs', len(bow_grus), len(bow_rfull), 1.8),
        (ax_bow[3], bow_elev, bow_rfull, '(f) Elevation HRUs', len(bow_elev), len(bow_rfull), 0.5),
    ]

    for ax, catch, rivers, title, n_catch, n_seg, lw in configs:
        # Plot data first
        catch.plot(ax=ax, column='elev_mean', cmap='terrain',
                  edgecolor='#333' if n_catch > 1 else '#01579B',
                  linewidth=lw, alpha=0.9,
                  vmin=vmin, vmax=vmax)
        rivers.plot(ax=ax, color='#0D47A1', linewidth=5 if n_seg == 1 else 1.2, alpha=0.9)
        ax.scatter(pp_pt.x, pp_pt.y, c='#D32F2F', s=120, marker='^', edgecolor='white', linewidth=2, zorder=10)

        # Set limits
        ax.set_xlim(xlim); ax.set_ylim(ylim)

        # Add basemap
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=10)
        except:
            pass

        ax.set_title(title, fontweight='bold', fontsize=10, pad=6)
        unit = 'GRU' if n_catch == 1 else ('GRUs' if n_catch < 100 else 'HRUs')
        seg = 'seg' if n_seg == 1 else 'segs'
        ax.text(0.03, 0.97, f'{n_catch} {unit}\n{n_seg} {seg}', transform=ax.transAxes,
               fontsize=8, va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95))
        ax.set_xticks([]); ax.set_yticks([])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.96, 0.12, 0.015, 0.30])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Elevation (m)', fontsize=9)

    # Section title
    fig.text(0.5, 0.48, 'Watershed Discretization: Bow River at Banff (2,210 km²)',
            ha='center', fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#8BC34A', edgecolor='#01579B', linewidth=2, label='Catchment'),
        Line2D([0], [0], color='#0D47A1', linewidth=3, label='River network'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#D32F2F',
               markeredgecolor='white', markersize=10, label='Gauge'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5, label='ERA5 grid'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, 0.01), fontsize=9)

    fig.savefig(OUTPUT_DIR / "figure_4_1_domain_definition.png", dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / "figure_4_1_domain_definition.pdf", facecolor='white')
    print("  Saved main figure")
    plt.close(fig)


def create_gru_comparison():
    """Create focused 1x3 GRU comparison."""
    print("\nCreating GRU comparison...")

    # Load fresh
    bow_lumped = gpd.read_file(SHP_DIR / "bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp").to_crs(epsg=3857)
    bow_grus = gpd.read_file(SHP_DIR / "bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp").to_crs(epsg=3857)
    bow_elev = gpd.read_file(SHP_DIR / "bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp").to_crs(epsg=3857)
    bow_rfull = gpd.read_file(SHP_DIR / "bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_distributed_1000m.shp").to_crs(epsg=3857)
    bow_pp = gpd.read_file(SHP_DIR / "bow/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp").to_crs(epsg=3857)

    pp_pt = bow_pp.iloc[0].geometry
    bounds = bow_lumped.total_bounds
    pad = 2000
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)

    vmin = min(bow_lumped['elev_mean'].min(), bow_grus['elev_mean'].min(), bow_elev['elev_mean'].min())
    vmax = max(bow_lumped['elev_mean'].max(), bow_grus['elev_mean'].max(), bow_elev['elev_mean'].max())

    # Create figure - simple approach without basemaps
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    panels = [
        (axes[0], bow_lumped, '(a) Lumped: 1 GRU', 'Single computational unit\nBasin-average elevation', 4),
        (axes[1], bow_grus, f'(b) Sub-basins: {len(bow_grus)} GRUs', 'Drainage-based subdivision\nMean elevation per GRU', 1.8),
        (axes[2], bow_elev, f'(c) Elevation bands: {len(bow_elev)} HRUs', 'Elevation-based subdivision\nCaptures lapse rate effects', 0.4),
    ]

    for ax, data, title, desc, lw in panels:
        # Plot without basemap for clean colors
        data.plot(ax=ax, column='elev_mean', cmap='terrain',
                 edgecolor='#333333', linewidth=lw, alpha=0.95,
                 vmin=vmin, vmax=vmax)
        bow_rfull.plot(ax=ax, color='#0D47A1', linewidth=1.2, alpha=0.9)
        ax.scatter(pp_pt.x, pp_pt.y, c='#D32F2F', s=180, marker='^', edgecolor='white', linewidth=2.5, zorder=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.text(0.03, 0.97, desc, transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='#666'))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2], shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('Elevation (m)', fontsize=9)

    fig.suptitle('Spatial Discretization Progression: Bow River at Banff (2,210 km²)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "figure_4_1_gru_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / "figure_4_1_gru_comparison.pdf", bbox_inches='tight', facecolor='white')
    print("  Saved GRU comparison")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("SYMFLUENCE Paper - Figure 4.1: Domain Definition")
    print("=" * 60)
    create_main_figure()
    create_gru_comparison()
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
