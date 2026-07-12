#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
SYMFLUENCE Paper Figures: Domain Definition (Section 4.1)
Three clean, focused figures showing spatial discretization capabilities.

Figure 1: Paradise - Point-scale forcing
Figure 2: Iceland - Regional scale (3 panels)
Figure 3: Bow River - Watershed discretization (3 columns)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from pathlib import Path

# Paths: data root comes from SYMFLUENCE_DATA_DIR (default: sibling
# SYMFLUENCE_data of the repo root). Figures go to plotting/output/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(os.environ.get('SYMFLUENCE_DATA_DIR', _REPO_ROOT.parent / 'SYMFLUENCE_data'))
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output'

# The original paper script read from a hand-staged "shapefiles/" folder. Each
# staged input maps to a file produced by the 01_domain_definition configs
# (one reproduction domain per discretization variant).
_SHP_MAP = {
    # Paradise (config_paradise_point.yaml -> domain paradise_snotel_wa)
    'paradise/pour_point/paradise_pourPoint.shp':
        'domain_paradise_snotel_wa/shapefiles/pour_point/paradise_snotel_wa_pourPoint.shp',
    # ERA5 forcing cell polygons: written by the ERA5 acquisition/remap step
    # (03_forcing_ensemble config_paradise_era5.yaml). Absent until that runs.
    'paradise/forcing/forcing_ERA5.shp':
        'domain_paradise_snotel_wa/shapefiles/forcing/forcing_ERA5.shp',
    # Iceland (config_iceland_regional / _coastal / _coastal_elev)
    'Iceland/river_basins/Iceland_riverBasins_semidistributed.shp':
        'domain_Iceland_regional/shapefiles/river_basins/Iceland_regional_riverBasins_semidistributed.shp',
    'Iceland/river_basins/Iceland_riverBasins_with_coastal.shp':
        'domain_Iceland_coastal/shapefiles/river_basins/Iceland_coastal_riverBasins_with_coastal.shp',
    'Iceland/catchment/semidistributed/regional_tutorial/Iceland_HRUs_elevation.shp':
        'domain_Iceland_coastal_elev/shapefiles/catchment/semidistributed/Iceland_coastal_elev/'
        'Iceland_coastal_elev_HRUs_elevation.shp',
    'Iceland/river_network/Iceland_riverNetwork_semidistributed.shp':
        'domain_Iceland_regional/shapefiles/river_network/Iceland_regional_riverNetwork_semidistributed.shp',
    # Bow at Banff (config_bow_* -> one domain per discretization)
    'bow/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp':
        'domain_Bow_at_Banff_lumped_era5/shapefiles/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp',
    'bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp':
        'domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp',
    'bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp':
        'domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/'
        'Bow_at_Banff_lumped_era5_HRUs_elevation.shp',
    'bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_landclass.shp':
        'domain_Bow_at_Banff_lumped_land_classes/shapefiles/catchment/lumped/Bow_at_Banff_lumped_land_classes/'
        'Bow_at_Banff_lumped_land_classes_HRUs_landclass.shp',
    'bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp':
        'domain_Bow_at_Banff_semidistributed_era5/shapefiles/catchment/semidistributed/'
        'Bow_at_Banff_semidistributed_era5/Bow_at_Banff_semidistributed_era5_HRUs_GRUs.shp',
    'bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp':
        'domain_Bow_at_Banff_semidistributed_elev/shapefiles/catchment/semidistributed/'
        'Bow_at_Banff_semidistributed_elev/Bow_at_Banff_semidistributed_elev_HRUs_elevation.shp',
    'bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation_aspect.shp':
        'domain_Bow_at_Banff_semidistributed_elev_aspect/shapefiles/catchment/semidistributed/'
        'Bow_at_Banff_semidistributed_elev_aspect/Bow_at_Banff_semidistributed_elev_aspect_HRUs_elevation_aspect.shp',
    'bow/catchment/distributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp':
        'domain_Bow_at_Banff_distributed/shapefiles/catchment/distributed/Bow_at_Banff_distributed/'
        'Bow_at_Banff_distributed_HRUs_GRUs.shp',
    'bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_lumped.shp':
        'domain_Bow_at_Banff_lumped_era5/shapefiles/river_network/Bow_at_Banff_lumped_era5_riverNetwork_lumped.shp',
    'bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_semidistributed.shp':
        'domain_Bow_at_Banff_semidistributed_era5/shapefiles/river_network/'
        'Bow_at_Banff_semidistributed_era5_riverNetwork_semidistributed.shp',
    'bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_distributed_1000m.shp':
        'domain_Bow_at_Banff_distributed/shapefiles/river_network/'
        'Bow_at_Banff_distributed_riverNetwork_distributed_1000m.shp',
}

# Full-coverage Paradise DEM: the paper used a raster clipped to the ERA5 cell
# extent; the reproduction uses the domain DEM produced by acquire_attributes.
PARADISE_DEM = (DATA_DIR / 'domain_paradise_snotel_wa/data/attributes/elevation/dem/'
                'domain_paradise_snotel_wa_elv.tif')


def _shp(rel):
    """Resolve an original staged-shapefile relative path to the reproduction dataset."""
    if rel not in _SHP_MAP:
        raise KeyError(f'No reproduction mapping for staged shapefile "{rel}"')
    return DATA_DIR / _SHP_MAP[rel]

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})

# Color palettes
ICELAND_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
]

IGBP_CLASSES = {
    1: ('Evergreen Needleleaf', '#05450a'),
    8: ('Woody Savannas', '#dade48'),
    9: ('Savannas', '#fbff13'),
    10: ('Grasslands', '#b6ff05'),
    11: ('Permanent Wetlands', '#27ff87'),
    13: ('Urban', '#a5a5a5'),
    15: ('Snow and Ice', '#69fff8'),
    16: ('Barren', '#f9ffa4'),
    17: ('Water Bodies', '#1c0dff'),
}


def plot_with_fill(ax, gdf, col_name, cmap, norm, edgecolor='#bbbbbb', linewidth=0.5, alpha=0.9, zorder=2):
    """Plot polygons with fill colors based on attribute."""
    for idx, row in gdf.iterrows():
        geom = row.geometry
        val = float(row[col_name])
        color = cmap(norm(val))
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)


def plot_with_patches(ax, gdf, col_name, cmap, norm, zorder=2, edgecolor='none', linewidth=0):
    """Plot polygons using PatchCollection for efficient rendering of dense data.

    edgecolor/linewidth draw thin HRU outlines so the discretization stays legible
    even where fill colours are similar (per Knoben review of Fig. 2).
    """
    patches = []
    colors = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        val = float(row[col_name])
        colors.append(cmap(norm(val)))
        if geom.geom_type == 'Polygon':
            patches.append(MplPolygon(np.array(geom.exterior.coords), closed=True))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                patches.append(MplPolygon(np.array(poly.exterior.coords), closed=True))
                # Repeat the color for each sub-polygon
                if len(patches) > len(colors):
                    colors.append(colors[-1])

    pc = PatchCollection(patches, facecolors=colors, edgecolors=edgecolor,
                         linewidths=linewidth, zorder=zorder)
    ax.add_collection(pc)


def plot_landclass(ax, gdf, edgecolor='#bbbbbb', linewidth=0.5, alpha=0.9, zorder=2):
    """Plot land cover classes using IGBP colormap."""
    for idx, row in gdf.iterrows():
        geom = row.geometry
        lc = int(row['landClass'])
        color = IGBP_CLASSES.get(lc, ('Unknown', '#888888'))[1]
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)


def get_stats(gdf, elev_col='elev_mean'):
    """Calculate statistics for a GeoDataFrame."""
    area_km2 = gdf.to_crs(epsg=32612).area.sum() / 1e6  # UTM zone 12 for Bow
    n_features = len(gdf)
    if elev_col in gdf.columns:
        elev_min = gdf[elev_col].min()
        elev_max = gdf[elev_col].max()
        elev_mean = gdf[elev_col].mean()
        return {'n': n_features, 'area': area_km2, 'elev_min': elev_min, 'elev_max': elev_max, 'elev_mean': elev_mean}
    return {'n': n_features, 'area': area_km2}


def create_era5_grid(gdf, target_crs=3857):
    """Create ERA5 grid cells (0.25° resolution) covering the given GeoDataFrame extent."""
    from shapely.geometry import box

    # Get bounds in WGS84 (EPSG:4326)
    gdf_4326 = gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_4326.total_bounds

    # ERA5 resolution is 0.25 degrees
    res = 0.25

    # Expand bounds to align with ERA5 grid
    minx = np.floor(minx / res) * res
    miny = np.floor(miny / res) * res
    maxx = np.ceil(maxx / res) * res
    maxy = np.ceil(maxy / res) * res

    # Create grid cells
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + res, y + res))
            y += res
        x += res

    # Create GeoDataFrame and reproject
    grid = gpd.GeoDataFrame(geometry=cells, crs='EPSG:4326')
    return grid.to_crs(epsg=target_crs), len(cells)


def plot_era5_grid(ax, grid, style='overlay', subtle=False):
    """Plot ERA5 grid as stylish overlay or underlay."""
    if subtle:
        # Subtle but still dashed orange for consistency
        grid.boundary.plot(ax=ax, color='#E65100', linewidth=0.8, linestyle='--', alpha=0.5, zorder=8)
    elif style == 'overlay':
        # Dashed lines on top - unified style
        grid.boundary.plot(ax=ax, color='#E65100', linewidth=1.0, linestyle='--', alpha=0.6, zorder=8)
    else:
        # Subtle underlay
        grid.plot(ax=ax, facecolor='none', edgecolor='#E65100', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)


# =============================================================================
# FIGURE 1: PARADISE
# =============================================================================
def compute_hillshade(dem, azimuth=315, altitude=45):
    """Compute hillshade from a DEM array."""
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hillshade = (np.sin(alt_rad) * np.cos(slope) +
                 np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hillshade, 0, 1)


def create_paradise_figure():
    """Create Paradise point-scale figure with DEM under ERA5 grid and location inset."""
    print("Creating Paradise figure...")

    # Load shapefiles
    para_pp = gpd.read_file(_shp("paradise/pour_point/paradise_pourPoint.shp"))
    para_forcing = gpd.read_file(_shp("paradise/forcing/forcing_ERA5.shp"))

    # Get Paradise coordinates in WGS84 for inset
    paradise_lon, paradise_lat = para_pp.iloc[0].geometry.x, para_pp.iloc[0].geometry.y

    # Load full-coverage DEM (EPSG:5070 -> reproject to WGS84 for plotting)
    dem_path = PARADISE_DEM
    with rasterio.open(dem_path) as src:
        # Reproject to WGS84
        dst_crs = 'EPSG:4326'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        dem_data = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear)
        # Compute WGS84 bounds from transform
        dem_left = transform.c
        dem_top = transform.f
        dem_right = dem_left + transform.a * width
        dem_bottom = dem_top + transform.e * height

    # Mask nodata
    dem_data[dem_data <= 0] = np.nan

    # Compute hillshade for terrain texture
    hillshade = compute_hillshade(np.nan_to_num(dem_data, nan=0))

    # DEM extent in WGS84 for imshow
    dem_extent = [dem_left, dem_right, dem_bottom, dem_top]

    # ERA5 grid extent
    forcing_bounds = para_forcing.total_bounds  # minx, miny, maxx, maxy

    # Color palette
    cell_edge = '#E65100'   # Muted burnt orange for ERA5 grid
    marker_color = '#c0392b'  # Red
    marker_size = 120       # Consistent marker size across map, inset, legend

    # Elevation colormap
    cmap_elev = plt.colormaps['terrain']
    dem_valid = dem_data[~np.isnan(dem_data)]
    norm_elev = mcolors.Normalize(vmin=dem_valid.min(), vmax=dem_valid.max())

    # --- Single-panel figure with DEM + ERA5 grid overlay ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot DEM elevation
    im = ax.imshow(dem_data, cmap=cmap_elev, norm=norm_elev,
                   extent=dem_extent, origin='upper', zorder=1)

    # Overlay hillshade for terrain texture
    ax.imshow(hillshade, cmap='gray', alpha=0.25,
              extent=dem_extent, origin='upper', zorder=2)

    # Generate AORC grid (0.01° resolution) covering ERA5 extent as subtle underlay
    aorc_color = '#c850c0'  # Magenta-purple tint (contrasts with green terrain + orange ERA5)
    aorc_res = 0.01
    # Dash period must stay short relative to the ~0.01° cell pitch, otherwise a
    # single cell edge is one dash and the grid reads as solid.
    aorc_dashes = (0, (2.2, 2.2))
    fb = forcing_bounds  # minx, miny, maxx, maxy
    aorc_x = np.arange(np.floor(fb[0] / aorc_res) * aorc_res,
                        np.ceil(fb[2] / aorc_res) * aorc_res + aorc_res, aorc_res)
    aorc_y = np.arange(np.floor(fb[1] / aorc_res) * aorc_res,
                        np.ceil(fb[3] / aorc_res) * aorc_res + aorc_res, aorc_res)
    for x in aorc_x:
        ax.axvline(x, color=aorc_color, linewidth=0.35, alpha=0.7, zorder=3,
                   linestyle=aorc_dashes)
    for y in aorc_y:
        ax.axhline(y, color=aorc_color, linewidth=0.35, alpha=0.7, zorder=3,
                   linestyle=aorc_dashes)

    # Plot ERA5 grid cells - semi-transparent fill with styled dashed edges
    para_forcing.plot(ax=ax, facecolor='none', edgecolor=cell_edge,
                      linewidth=1.8, linestyle='--', alpha=0.85, zorder=5)

    # Number each ERA5 cell
    for idx, row in para_forcing.iterrows():
        c = row.geometry.centroid
        ax.text(c.x, c.y, str(idx + 1), ha='center', va='center',
                fontsize=16, fontweight='bold', color='white',
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=3, foreground=cell_edge)
                ],
                zorder=6)

    # Plot station - styled marker with shadow
    ax.scatter(paradise_lon, paradise_lat, c='black', s=marker_size + 40, marker='^',
               edgecolor='none', alpha=0.3, zorder=9)  # drop shadow
    ax.scatter(paradise_lon, paradise_lat, c=marker_color, s=marker_size, marker='^',
               edgecolor='white', linewidth=1.5, zorder=10)

    # Set extent to ERA5 grid bounds with small padding
    pad_x = (forcing_bounds[2] - forcing_bounds[0]) * 0.03
    pad_y = (forcing_bounds[3] - forcing_bounds[1]) * 0.03
    ax.set_xlim(forcing_bounds[0] - pad_x, forcing_bounds[2] + pad_x)
    ax.set_ylim(forcing_bounds[1] - pad_y, forcing_bounds[3] + pad_y)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#555')

    # --- Inset location map (bottom-left corner) with terrain ---
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='lower left',
                          borderpad=0.5)

    # Natural Earth state/province and country borders for the inset.
    # cartopy is preferred; without it we read the identical Natural Earth
    # shapefiles from cartopy's on-disk cache. A missing layer must raise:
    # swallowing it silently renders the inset as an empty blue rectangle.
    def _natural_earth(name):
        try:
            from cartopy.io import shapereader
            return gpd.read_file(shapereader.natural_earth(
                resolution='50m', category='cultural', name=name))
        except ImportError:
            cached = (Path.home() / '.local/share/cartopy/shapefiles'
                      / 'natural_earth' / 'cultural' / f'ne_50m_{name}.shp')
            if not cached.exists():
                raise FileNotFoundError(
                    f'Natural Earth layer "{name}" needs cartopy or {cached}')
            return gpd.read_file(cached)

    # State/province boundaries, filtered to the PNW
    pnw_states = _natural_earth('admin_1_states_provinces_lakes').cx[-135:-105, 35:55]
    pnw_states.plot(ax=ax_inset, facecolor='#e8e8e8', edgecolor='#999',
                    linewidth=0.3, zorder=1)

    # Country borders (thicker)
    countries = _natural_earth('admin_0_countries')
    na_countries = countries[countries['CONTINENT'] == 'North America']
    na_countries.boundary.plot(ax=ax_inset, edgecolor='#555', linewidth=0.8, zorder=2)

    # Use py3dep for a small terrain overview in the inset
    try:
        import py3dep
        inset_dem = py3dep.get_dem((-135, 35, -105, 55), resolution=10000, crs='EPSG:4326')
        inset_extent = [float(inset_dem.x.min()), float(inset_dem.x.max()),
                        float(inset_dem.y.min()), float(inset_dem.y.max())]
        inset_hs = compute_hillshade(np.nan_to_num(inset_dem.values, nan=0))
        ax_inset.imshow(inset_dem.values, cmap='terrain', alpha=0.35,
                        extent=inset_extent, origin='upper', zorder=0)
        ax_inset.imshow(inset_hs, cmap='gray', alpha=0.2,
                        extent=inset_extent, origin='upper', zorder=0)
    except Exception:
        pass

    ax_inset.scatter(paradise_lon, paradise_lat, c='black', s=marker_size + 40, marker='^',
                     edgecolor='none', alpha=0.3, zorder=9)
    ax_inset.scatter(paradise_lon, paradise_lat, c=marker_color, s=marker_size, marker='^',
                     edgecolor='white', linewidth=1.5, zorder=10)
    ax_inset.annotate('Paradise', (paradise_lon, paradise_lat),
                      xytext=(4, -10), textcoords='offset points',
                      fontsize=6, fontweight='bold', color=marker_color)
    ax_inset.set_xlim(-130, -110)
    ax_inset.set_ylim(40, 55)
    ax_inset.set_facecolor('#dbe9f4')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1)
        spine.set_color('#555')

    # Colorbar for elevation - more prominent
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03,
                        shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    fig.subplots_adjust(top=0.95, bottom=0.10)

    # Legend in upper-right - marker size matched to map (s=120 -> markersize~11)
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=cell_edge, linewidth=1.8,
                       linestyle='--', label='ERA5 (0.25°)'),
        mpatches.Patch(facecolor='none', edgecolor=aorc_color, linewidth=0.9,
                       linestyle='--', label='AORC (0.01°)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=marker_color,
               markeredgecolor='white', markersize=11, label='Station'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13,
              framealpha=0.95, edgecolor='#ccc')

    fig.savefig(OUTPUT_DIR / "figure_4_1a_paradise.png", dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure_4_1a_paradise.pdf", facecolor='white', bbox_inches='tight')
    print("  Saved Paradise figure")
    plt.close(fig)


# =============================================================================
# FIGURE 2: ICELAND
# =============================================================================
def create_iceland_figure():
    """Create Iceland 3-panel figure with polished design."""
    print("Creating Iceland figure...")

    # Load data
    ice_no_coastal = gpd.read_file(_shp("Iceland/river_basins/Iceland_riverBasins_semidistributed.shp")).to_crs(epsg=3857)
    ice_with_coastal = gpd.read_file(_shp("Iceland/river_basins/Iceland_riverBasins_with_coastal.shp")).to_crs(epsg=3857)
    ice_elev = gpd.read_file(_shp("Iceland/catchment/semidistributed/regional_tutorial/Iceland_HRUs_elevation.shp")).to_crs(epsg=3857)
    ice_rivers = gpd.read_file(_shp("Iceland/river_network/Iceland_riverNetwork_semidistributed.shp")).to_crs(epsg=3857)

    # Filter elevation HRUs for non-coastal
    ice_elev_no_coastal = ice_elev[~ice_elev['GRU_ID'].isin(
        ice_with_coastal[ice_with_coastal['is_coastal'] == 1]['GRU_ID'].values
    )]

    # Create ERA5 grid
    era5_grid, n_era5_cells = create_era5_grid(ice_with_coastal)

    # Calculate areas
    area_with_coastal = ice_with_coastal.to_crs(epsg=32627).area.sum() / 1e6
    n_coastal = len(ice_with_coastal) - len(ice_no_coastal)

    # Common bounds - tight to Iceland
    bounds = ice_with_coastal.total_bounds
    pad = 15000
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)

    # Elevation colormap
    cmap = plt.colormaps['terrain']
    elev_min = max(0, ice_elev['elev_mean'].min())
    elev_max = ice_elev['elev_mean'].max()
    norm_elev = mcolors.Normalize(vmin=elev_min, vmax=elev_max)

    # Colors
    river_color = '#b30000'   # Dark red -- contrasts with terrain greens/blues
    era5_color = '#E65100'

    # Figure sizing: match Iceland's aspect to eliminate dead space
    # Each panel is ~1.44:1 (w:h). 3 panels with gaps.
    map_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])  # w/h per panel
    fig_w = 16.0
    # Layout fractions
    left, right = 0.02, 0.98
    wspace_frac = 0.015  # gap between panels as fraction of fig width
    panel_w_frac = (right - left - 2 * wspace_frac) / 3  # width of one panel

    # We want: panel_h_pixels / panel_w_pixels = 1/map_aspect
    # panel_h_frac * fig_h = panel_w_frac * fig_w / map_aspect
    # We also need space for title (~0.08 of fig_h) and colorbar+legend (~0.08)
    title_frac = 0.14   # fraction of fig height for title+subtitle+panel titles above maps
    bottom_frac = 0.12  # fraction of fig height for colorbar+legend below maps

    panel_h_frac = 1.0 - title_frac - bottom_frac  # fraction available for maps
    fig_h = (panel_w_frac * fig_w) / (map_aspect * panel_h_frac)

    bottom = bottom_frac
    top = 1.0 - title_frac

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, 3, figure=fig,
                  left=left, right=right, top=top, bottom=bottom,
                  wspace=wspace_frac / panel_w_frac)

    def style_ax(ax):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_facecolor('#dbe9f4')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#888')

    # --- Panel (a): River Basin GRUs ---
    ax1 = fig.add_subplot(gs[0, 0])
    era5_grid.boundary.plot(ax=ax1, color=era5_color, linewidth=1.0,
                            linestyle='--', alpha=0.6, zorder=1)
    ice_elev_no_coastal.plot(ax=ax1, column='elev_mean', cmap=cmap, norm=norm_elev,
                              edgecolor='none', linewidth=0, zorder=2)
    # Rivers with white halo
    ice_rivers.plot(ax=ax1, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax1, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ax(ax1)
    ax1.set_title('(a) River Basin GRUs', fontsize=16, fontweight='bold', pad=6)
    ax1.text(0.03, 0.03, f'{len(ice_no_coastal):,} GRUs\n{len(ice_rivers):,} seg',
             transform=ax1.transAxes, fontsize=13, ha='left', va='bottom',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # --- Panel (b): + Coastal GRUs ---
    ax2 = fig.add_subplot(gs[0, 1])
    era5_grid.boundary.plot(ax=ax2, color=era5_color, linewidth=1.0,
                            linestyle='--', alpha=0.6, zorder=1)
    ice_elev.plot(ax=ax2, column='elev_mean', cmap=cmap, norm=norm_elev,
                  edgecolor='none', linewidth=0, zorder=2)
    ice_rivers.plot(ax=ax2, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax2, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ax(ax2)
    ax2.set_title('(b) + Coastal GRUs', fontsize=16, fontweight='bold', pad=6)
    ax2.text(0.03, 0.03, f'{len(ice_with_coastal):,} GRUs\n(+{n_coastal:,} coastal)',
             transform=ax2.transAxes, fontsize=13, ha='left', va='bottom',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # --- Panel (c): + Elevation HRUs ---
    ax3 = fig.add_subplot(gs[0, 2])
    era5_grid.boundary.plot(ax=ax3, color=era5_color, linewidth=1.0,
                            linestyle='--', alpha=0.6, zorder=1)
    ice_elev.plot(ax=ax3, column='elev_mean', cmap=cmap, norm=norm_elev,
                  edgecolor='none', linewidth=0, zorder=2)
    ice_rivers.plot(ax=ax3, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax3, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ax(ax3)
    ax3.set_title('(c) + Elevation HRUs', fontsize=16, fontweight='bold', pad=6)
    ax3.text(0.03, 0.03, f'{len(ice_elev):,} HRUs\n{elev_min:.0f}\u2013{elev_max:.0f} m',
             transform=ax3.transAxes, fontsize=13, ha='left', va='bottom',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # --- Colorbar: centered below maps ---
    cbar_y = bottom * 0.38
    cbar_ax = fig.add_axes([0.25, cbar_y, 0.50, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_elev)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mean Elevation (m)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    # --- Legend: bottom-right, horizontal ---
    legend_elements = [
        Line2D([0], [0], color=river_color, linewidth=2, label='River network'),
        Line2D([0], [0], color=era5_color, linewidth=1.5, linestyle='--',
               alpha=0.6, label='ERA5 grid'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(right, 0.002), ncol=2, fontsize=13,
               framealpha=0.95, edgecolor='#ccc', handlelength=1.5,
               columnspacing=1.2)

    fig.savefig(OUTPUT_DIR / "figure_4_1b_iceland.png", dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure_4_1b_iceland.pdf", facecolor='white', bbox_inches='tight')
    print("  Saved Iceland figure")
    plt.close(fig)


# =============================================================================
# FIGURE 3: BOW RIVER
# =============================================================================
def create_bow_figure(shp_subdir='bow', out_stem='figure_4_1c_bow'):
    """Create Bow River 3-column figure with all discretization options.

    shp_subdir selects the staged shapefile set (see _SHP_MAP). Use 'bow' for the
    Copernicus-DEM delineation (49 sub-basins) or 'bow_merit29' for the original
    MERIT-DEM delineation (29 sub-basins) reported in the paper.
    out_stem sets the output filename stem (png/pdf) under OUTPUT_DIR.
    """
    print(f"Creating Bow River figure from '{shp_subdir}'...")

    # Load all shapefiles
    bow_pp = gpd.read_file(_shp(shp_subdir + "/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp")).to_crs(epsg=3857)

    # Lumped configurations
    bow_l_gru = gpd.read_file(_shp(shp_subdir + "/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp")).to_crs(epsg=3857)
    bow_l_elev = gpd.read_file(_shp(shp_subdir + "/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp")).to_crs(epsg=3857)
    bow_l_land = gpd.read_file(_shp(shp_subdir + "/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_landclass.shp")).to_crs(epsg=3857)

    # Semi-distributed configurations
    bow_sd_gru = gpd.read_file(_shp(shp_subdir + "/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp")).to_crs(epsg=3857)
    bow_sd_elev = gpd.read_file(_shp(shp_subdir + "/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp")).to_crs(epsg=3857)
    bow_sd_asp = gpd.read_file(_shp(shp_subdir + "/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation_aspect.shp")).to_crs(epsg=3857)

    # Distributed
    bow_dist = gpd.read_file(_shp(shp_subdir + "/catchment/distributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp")).to_crs(epsg=3857)

    # River networks
    bow_r_lump = gpd.read_file(_shp(shp_subdir + "/river_network/Bow_at_Banff_lumped_era5_riverNetwork_lumped.shp")).to_crs(epsg=3857)
    bow_r_semi = gpd.read_file(_shp(shp_subdir + "/river_network/Bow_at_Banff_lumped_era5_riverNetwork_semidistributed.shp")).to_crs(epsg=3857)
    # Distributed routing = one reach per grid cell, so segment count == cell count.
    # The explicit distributed river-network shapefile is optional (not drawn in
    # panel (c)); fall back to the cell count when it is absent.
    _dist_rn = _shp(shp_subdir + "/river_network/Bow_at_Banff_lumped_era5_riverNetwork_distributed_1000m.shp")
    bow_r_dist = gpd.read_file(_dist_rn).to_crs(epsg=3857) if _dist_rn.exists() else bow_dist

    # Common settings
    pp_pt = bow_pp.iloc[0].geometry
    bounds = bow_l_gru.total_bounds
    pad = 3000
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)

    # Global elevation range for consistent coloring
    all_elevs = np.concatenate([
        bow_l_elev['elev_mean'].values,
        bow_sd_elev['elev_mean'].values,
        bow_dist['elev_mean'].values
    ])
    vmin_global, vmax_global = all_elevs.min(), all_elevs.max()
    cmap = plt.colormaps['terrain']
    norm_global = mcolors.Normalize(vmin=vmin_global, vmax=vmax_global)

    # Create ERA5 grid for Bow
    era5_grid, n_era5_cells = create_era5_grid(bow_l_gru)

    # Figure sizing: maps have aspect ~1.2 (taller than wide)
    # 3 cols of maps + bracket + colorbar; 3 rows + title + legend
    # Target: map cells match natural aspect so set_aspect('equal') wastes no space
    map_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])  # ~1.204

    # Layout fractions (of figure width/height)
    left, right = 0.02, 0.84   # map area (bracket+cbar go in 0.84-1.0)
    bottom, top = 0.055, 0.90  # map area (legend below, title above)
    col_gap = 0.015             # horizontal gap between columns (fraction of fig width)
    row_gap = 0.035             # vertical gap between rows (fraction of fig height)
    title_space = 0.06          # above top for title + column headers

    map_width_frac = (right - left - 2 * col_gap) / 3   # width of one map cell
    map_height_frac = (top - bottom - 2 * row_gap) / 3  # height of one map cell

    # Choose fig dimensions so cell aspect matches map aspect
    # map_height_frac * fig_h / (map_width_frac * fig_w) = map_aspect
    fig_w = 14
    fig_h = fig_w * (map_width_frac / map_height_frac) * map_aspect * \
            (top - bottom + title_space + 0.065) / (right - left)
    # Clamp to reasonable range
    fig_h = max(13, min(18, fig_h))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(3, 3, figure=fig,
                  left=left, right=right, top=top, bottom=bottom,
                  wspace=col_gap / map_width_frac,   # GridSpec wants ratio to subplot size
                  hspace=row_gap / map_height_frac)

    # Helper function for plotting panels - directly plots to axis
    def plot_panel(ax, data, rivers, use_landclass=False, edge_lw=1, edge_color='#bbbbbb', no_edges=False, show_rivers=True, show_era5=True, hru_outline_lw=0.0, gru_boundary=None):
        # Set limits first
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Plot data
        if use_landclass:
            plot_landclass(ax, data, edgecolor=edge_color, linewidth=edge_lw, zorder=2)
        elif no_edges:
            # Use PatchCollection for efficient rendering of dense data.
            # hru_outline_lw>0 draws thin HRU boundaries so the subdivision is
            # legible even when neighbouring fills are similar (Knoben review).
            hru_edge = (0.2, 0.2, 0.2, 0.55) if hru_outline_lw > 0 else 'none'
            plot_with_patches(ax, data, 'elev_mean', cmap, norm_global, zorder=2,
                              edgecolor=hru_edge, linewidth=hru_outline_lw)
        else:
            plot_with_fill(ax, data, 'elev_mean', cmap, norm_global,
                          edgecolor=edge_color, linewidth=edge_lw, zorder=2)

        # GRU / sub-basin boundary overlay (Knoben review): drawn crisply on top of
        # the fine HRU outlines so the two-tier discretization — GRUs subdivided into
        # HRUs — reads clearly. A white halo keeps it legible over the terrain fills.
        if gru_boundary is not None:
            gru_boundary.boundary.plot(ax=ax, color='white', linewidth=1.4, alpha=0.7, zorder=3)
            gru_boundary.boundary.plot(ax=ax, color='#111111', linewidth=0.7, alpha=0.95, zorder=4)

        # Rivers - white outline + dark red for visibility against terrain colormap
        if show_rivers:
            if len(rivers) == 1:
                river_lw = 2.5
            elif len(rivers) < 100:
                river_lw = 1.2
            else:
                river_lw = 0.5
            # White halo for contrast
            rivers.plot(ax=ax, color='white', linewidth=river_lw + 1.0, alpha=0.6, zorder=5)
            rivers.plot(ax=ax, color='#b30000', linewidth=river_lw, alpha=0.95, zorder=6)

        # ERA5 grid overlay - slightly stronger for Bow (fewer cells)
        if show_era5:
            plot_era5_grid(ax, era5_grid, style='overlay')

        # Pour point always visible
        ax.scatter(pp_pt.x, pp_pt.y, c='#D32F2F', s=70, marker='^', edgecolor='white', linewidth=1.5, zorder=10)

        # Styling
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#888')

    # Create all axes first, then compute column centers for headers
    # Explicit, self-contained panel titles (Knoben review: subplot titles must
    # stand alone without relying on shared column headers). Each names the
    # GRU-level spatial mode, the HRU-level subdivision, and the routing variant.
    # textwrap keeps the longer captions to two lines without overrunning a panel.
    import textwrap
    tprops = dict(fontsize=12.5, fontweight='bold', pad=6)

    def set_title(ax, text, width=32):
        ax.set_title('\n'.join(textwrap.wrap(text, width=width)), **tprops)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel(ax_a, bow_l_gru, bow_r_lump, edge_lw=1.5, edge_color='#bbbbbb')
    set_title(ax_a, '(a) Lumped — single GRU')
    ax_a.text(0.03, 0.03, '1 GRU', transform=ax_a.transAxes, fontsize=14, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (b) Semi-distributed GRUs - thin lines
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel(ax_b, bow_sd_gru, bow_r_semi, edge_lw=0.3, edge_color='#bbbbbb')
    set_title(ax_b, '(b) Semi-distributed — sub-basin GRUs')
    ax_b.text(0.03, 0.03, f'{len(bow_sd_gru)} GRUs\n{len(bow_r_semi)} seg', transform=ax_b.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (c) Distributed grid - no edges, no rivers (too dense)
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel(ax_c, bow_dist, bow_r_dist, no_edges=True, show_rivers=False)
    set_title(ax_c, '(c) Distributed — 1 km grid cells')
    ax_c.text(0.03, 0.03, f'{len(bow_dist):,} cells\n{len(bow_r_dist):,} seg', transform=ax_c.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ROW 2: Elevation subdivision. Thin HRU outlines added (Knoben review).
    # (d) Single GRU + elevation-band HRUs
    ax_d = fig.add_subplot(gs[1, 0])
    plot_panel(ax_d, bow_l_elev, bow_r_lump, no_edges=True, hru_outline_lw=0.4, gru_boundary=bow_l_gru)
    set_title(ax_d, '(d) Single GRU + elevation-band HRUs')
    ax_d.text(0.03, 0.03, f'{len(bow_l_elev)} HRUs', transform=ax_d.transAxes, fontsize=14, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (e) Sub-basin GRUs + elevation-band HRUs
    ax_e = fig.add_subplot(gs[1, 1])
    plot_panel(ax_e, bow_sd_elev, bow_r_semi, no_edges=True, hru_outline_lw=0.15, gru_boundary=bow_sd_gru)
    set_title(ax_e, '(e) Sub-basin GRUs + elevation-band HRUs')
    ax_e.text(0.03, 0.03, f'{len(bow_sd_elev)} HRUs\n{len(bow_r_semi)} seg', transform=ax_e.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (f) Single-GRU elevation bands + semi-distributed routing.
    # (Knoben review: previously mislabelled "Lumped" despite 12 HRUs + routing.)
    ax_f = fig.add_subplot(gs[1, 2])
    plot_panel(ax_f, bow_l_elev, bow_r_semi, no_edges=True, hru_outline_lw=0.4, gru_boundary=bow_l_gru)
    set_title(ax_f, '(f) Elevation bands + semi-distributed routing')
    ax_f.text(0.03, 0.03, f'{len(bow_l_elev)} HRUs\n{len(bow_r_semi)} seg', transform=ax_f.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ROW 3: Additional subdivisions
    # (g) Single GRU + land-cover HRUs
    ax_g = fig.add_subplot(gs[2, 0])
    plot_panel(ax_g, bow_l_land, bow_r_lump, use_landclass=True, edge_lw=0.5, edge_color='#bbbbbb')
    set_title(ax_g, '(g) Single GRU + land-cover HRUs (IGBP)')
    ax_g.text(0.03, 0.03, f'{len(bow_l_land)} HRUs', transform=ax_g.transAxes, fontsize=14, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (h) Sub-basin GRUs + elevation & aspect HRUs - thin outlines despite density
    ax_h = fig.add_subplot(gs[2, 1])
    plot_panel(ax_h, bow_sd_asp, bow_r_semi, no_edges=True, hru_outline_lw=0.08, gru_boundary=bow_sd_gru)
    set_title(ax_h, '(h) Sub-basin GRUs + elevation & aspect HRUs')
    ax_h.text(0.03, 0.03, f'{len(bow_sd_asp):,} HRUs\n{len(bow_r_semi)} seg', transform=ax_h.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (i) Single GRU + semi-distributed routing - single GRU boundary
    ax_i = fig.add_subplot(gs[2, 2])
    plot_panel(ax_i, bow_l_gru, bow_r_semi, edge_lw=1.5, edge_color='#bbbbbb')
    set_title(ax_i, '(i) Single GRU + semi-distributed routing')
    ax_i.text(0.03, 0.03, f'1 GRU\n{len(bow_r_semi)} seg', transform=ax_i.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Column headers (LUMPED / SEMI-DISTRIBUTED / DISTRIBUTED) and the COMBINED
    # bracket were removed per Knoben review: they did not apply cleanly to every
    # panel in their column (e.g. (f) is single-GRU, not distributed). The mode is
    # now carried in each panel's self-contained title instead.
    fig.canvas.draw()

    # --- Global elevation colorbar aligned to map area (applies to all panels but g) ---
    cbar_ax = fig.add_axes([0.91, bottom + 0.02, 0.012, top - bottom - 0.04])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_global)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Elevation (m)  —  all panels except (g)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    # --- Two separate legend boxes (Knoben review) ---
    # 1) "Map symbols" — river network, gauge, ERA5 grid: these apply to every panel.
    # 2) "Land cover classes shown in (g)" — the IGBP classes, which appear only in (g).
    present_lc = sorted(bow_l_land['landClass'].unique())
    short_labels = {1: 'Forest', 8: 'Woody Sav.', 9: 'Savanna', 10: 'Grass',
                    11: 'Wetland', 13: 'Urban', 15: 'Snow/Ice', 16: 'Barren', 17: 'Water'}

    map_symbols = [
        Line2D([0], [0], color='#b30000', linewidth=2.5, label='River network'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#D32F2F',
               markeredgecolor='white', markersize=10, label='Gauge'),
        Line2D([0], [0], color='#E65100', linewidth=1.5, linestyle='--',
               alpha=0.7, label='ERA5 grid (0.25°)'),
        Line2D([0], [0], color='#111111', linewidth=0.9, label='GRU / sub-basin boundary'),
        Line2D([0], [0], color=(0.2, 0.2, 0.2, 0.55), linewidth=0.5, label='HRU boundary'),
    ]
    landcover = [
        mpatches.Patch(facecolor=IGBP_CLASSES[lc][1], edgecolor='#555',
                       linewidth=0.5, label=short_labels.get(lc, str(lc)))
        for lc in present_lc if lc in IGBP_CLASSES
    ]

    # Land cover classes — bottom-left, titled and scoped to panel (g)
    leg_lc = fig.legend(handles=landcover, loc='upper left',
                        bbox_to_anchor=(left, bottom - 0.008),
                        title='Land cover classes shown in (g)',
                        ncol=5, fontsize=11.5, title_fontsize=12.5,
                        framealpha=0.95, edgecolor='#ccc', columnspacing=1.0,
                        handletextpad=0.4, handlelength=1.4)
    leg_lc.get_title().set_fontweight('bold')
    fig.add_artist(leg_lc)

    # Map symbols — bottom-right, applies to all panels
    leg_ms = fig.legend(handles=map_symbols, loc='upper right',
                        bbox_to_anchor=(right, bottom - 0.008),
                        title='Map symbols (all panels)',
                        ncol=3, fontsize=11.5, title_fontsize=12.5,
                        framealpha=0.95, edgecolor='#ccc', columnspacing=1.0,
                        handletextpad=0.4, handlelength=1.4)
    leg_ms.get_title().set_fontweight('bold')
    fig.add_artist(leg_ms)

    fig.savefig(OUTPUT_DIR / f"{out_stem}.png", dpi=400, facecolor='white', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{out_stem}.pdf", facecolor='white', bbox_inches='tight')
    print(f"  Saved Bow River figure -> {out_stem}")
    plt.close(fig)


# =============================================================================
# FIGURE 4: COMBINED STACKED FIGURE
# =============================================================================
def create_combined_figure():
    """Create a combined stacked figure with Paradise, Iceland, and Bow River panels.

    Layout:
      Top row: (a) Paradise | (b)-(d) Iceland 3 panels  [4 columns]
      Below:   (e)-(m) Bow River 3x3 grid               [3 columns spanning width]
    Single shared legend at bottom, single shared elevation colorbar.
    """
    print("Creating combined figure...")

    # =========================================================================
    # Load all data (reuse logic from individual functions)
    # =========================================================================

    # --- Paradise data ---
    para_pp = gpd.read_file(_shp("paradise/pour_point/paradise_pourPoint.shp"))
    para_forcing = gpd.read_file(_shp("paradise/forcing/forcing_ERA5.shp"))
    paradise_lon, paradise_lat = para_pp.iloc[0].geometry.x, para_pp.iloc[0].geometry.y

    dem_path = PARADISE_DEM
    with rasterio.open(dem_path) as src:
        dst_crs = 'EPSG:4326'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        dem_data = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear)
        dem_left = transform.c
        dem_top = transform.f
        dem_right = dem_left + transform.a * width
        dem_bottom = dem_top + transform.e * height

    dem_data[dem_data <= 0] = np.nan
    hillshade = compute_hillshade(np.nan_to_num(dem_data, nan=0))
    dem_extent = [dem_left, dem_right, dem_bottom, dem_top]
    forcing_bounds = para_forcing.total_bounds

    # --- Iceland data ---
    ice_no_coastal = gpd.read_file(_shp("Iceland/river_basins/Iceland_riverBasins_semidistributed.shp")).to_crs(epsg=3857)
    ice_with_coastal = gpd.read_file(_shp("Iceland/river_basins/Iceland_riverBasins_with_coastal.shp")).to_crs(epsg=3857)
    ice_elev = gpd.read_file(_shp("Iceland/catchment/semidistributed/regional_tutorial/Iceland_HRUs_elevation.shp")).to_crs(epsg=3857)
    ice_rivers = gpd.read_file(_shp("Iceland/river_network/Iceland_riverNetwork_semidistributed.shp")).to_crs(epsg=3857)
    ice_elev_no_coastal = ice_elev[~ice_elev['GRU_ID'].isin(
        ice_with_coastal[ice_with_coastal['is_coastal'] == 1]['GRU_ID'].values
    )]
    era5_grid_ice, _ = create_era5_grid(ice_with_coastal)
    ice_bounds = ice_with_coastal.total_bounds
    ice_pad = 15000
    ice_xlim = (ice_bounds[0] - ice_pad, ice_bounds[2] + ice_pad)
    ice_ylim = (ice_bounds[1] - ice_pad, ice_bounds[3] + ice_pad)
    n_coastal = len(ice_with_coastal) - len(ice_no_coastal)

    # --- Bow data ---
    bow_pp = gpd.read_file(_shp("bow/pour_point/Bow_at_Banff_lumped_era5_pourPoint.shp")).to_crs(epsg=3857)
    bow_l_gru = gpd.read_file(_shp("bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp")).to_crs(epsg=3857)
    bow_l_elev = gpd.read_file(_shp("bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp")).to_crs(epsg=3857)
    bow_l_land = gpd.read_file(_shp("bow/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_landclass.shp")).to_crs(epsg=3857)
    bow_sd_gru = gpd.read_file(_shp("bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUs.shp")).to_crs(epsg=3857)
    bow_sd_elev = gpd.read_file(_shp("bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation.shp")).to_crs(epsg=3857)
    bow_sd_asp = gpd.read_file(_shp("bow/catchment/semidistributed/run_1/Bow_at_Banff_lumped_era5_HRUs_elevation_aspect.shp")).to_crs(epsg=3857)
    bow_dist = gpd.read_file(_shp("bow/catchment/distributed/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp")).to_crs(epsg=3857)
    bow_r_lump = gpd.read_file(_shp("bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_lumped.shp")).to_crs(epsg=3857)
    bow_r_semi = gpd.read_file(_shp("bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_semidistributed.shp")).to_crs(epsg=3857)
    bow_r_dist = gpd.read_file(_shp("bow/river_network/Bow_at_Banff_lumped_era5_riverNetwork_distributed_1000m.shp")).to_crs(epsg=3857)
    pp_pt = bow_pp.iloc[0].geometry
    bow_bounds = bow_l_gru.total_bounds
    bow_pad = 3000
    bow_xlim = (bow_bounds[0] - bow_pad, bow_bounds[2] + bow_pad)
    bow_ylim = (bow_bounds[1] - bow_pad, bow_bounds[3] + bow_pad)
    era5_grid_bow, _ = create_era5_grid(bow_l_gru)

    # =========================================================================
    # Global elevation colormap (unified across all panels)
    # =========================================================================
    dem_valid = dem_data[~np.isnan(dem_data)]
    ice_elev_min = max(0, ice_elev['elev_mean'].min())
    ice_elev_max = ice_elev['elev_mean'].max()
    bow_all_elevs = np.concatenate([bow_l_elev['elev_mean'].values, bow_sd_elev['elev_mean'].values, bow_dist['elev_mean'].values])

    global_vmin = min(dem_valid.min(), ice_elev_min, bow_all_elevs.min())
    global_vmax = max(dem_valid.max(), ice_elev_max, bow_all_elevs.max())
    cmap = plt.colormaps['terrain']
    norm_global = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

    # Colors
    cell_edge = '#E65100'
    river_color = '#b30000'
    marker_color = '#c0392b'
    aorc_color = '#c850c0'

    # =========================================================================
    # Figure layout
    # =========================================================================
    # Top row: 4 columns (1 Paradise + 3 Iceland)
    # Bottom 3 rows: 3 columns (Bow 3x3)
    # Plus space for colorbar and legend at bottom

    fig = plt.figure(figsize=(20, 28))

    # Use nested GridSpecs for the two sections
    # Overall: top section ~25% height, bottom section ~65% height, legend/cbar ~10%
    gs_outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 2.8, 0.25],
                        left=0.03, right=0.95, top=0.97, bottom=0.02,
                        hspace=0.08)

    # --- Top row: 4 columns (Paradise + 3 Iceland) ---
    gs_top = gs_outer[0].subgridspec(1, 4, wspace=0.04)

    # --- Bottom section: Bow 3x3 ---
    gs_bow = gs_outer[1].subgridspec(3, 3, wspace=0.04, hspace=0.08)

    # --- Bottom bar: colorbar + legend ---
    gs_bottom = gs_outer[2].subgridspec(1, 1)

    # =========================================================================
    # Helper: style Iceland/Bow axes
    # =========================================================================
    def style_ice_ax(ax):
        ax.set_xlim(ice_xlim)
        ax.set_ylim(ice_ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_facecolor('#dbe9f4')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#888')

    def style_bow_ax(ax):
        ax.set_xlim(bow_xlim)
        ax.set_ylim(bow_ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#888')

    def plot_bow_panel(ax, data, rivers, use_landclass=False, edge_lw=1,
                       edge_color='#bbbbbb', no_edges=False, show_rivers=True, show_era5=True):
        ax.set_xlim(bow_xlim)
        ax.set_ylim(bow_ylim)
        if use_landclass:
            plot_landclass(ax, data, edgecolor=edge_color, linewidth=edge_lw, zorder=2)
        elif no_edges:
            plot_with_patches(ax, data, 'elev_mean', cmap, norm_global, zorder=2)
        else:
            plot_with_fill(ax, data, 'elev_mean', cmap, norm_global,
                           edgecolor=edge_color, linewidth=edge_lw, zorder=2)
        if show_rivers:
            if len(rivers) == 1:
                river_lw = 2.5
            elif len(rivers) < 100:
                river_lw = 1.2
            else:
                river_lw = 0.5
            rivers.plot(ax=ax, color='white', linewidth=river_lw + 1.0, alpha=0.6, zorder=5)
            rivers.plot(ax=ax, color=river_color, linewidth=river_lw, alpha=0.95, zorder=6)
        if show_era5:
            plot_era5_grid(ax, era5_grid_bow, style='overlay')
        ax.scatter(pp_pt.x, pp_pt.y, c='#D32F2F', s=70, marker='^',
                   edgecolor='white', linewidth=1.5, zorder=10)
        style_bow_ax(ax)

    # =========================================================================
    # Panel (a): Paradise
    # =========================================================================
    ax_para = fig.add_subplot(gs_top[0, 0])
    im_para = ax_para.imshow(dem_data, cmap=cmap, norm=norm_global,
                              extent=dem_extent, origin='upper', zorder=1)
    ax_para.imshow(hillshade, cmap='gray', alpha=0.25,
                   extent=dem_extent, origin='upper', zorder=2)

    # AORC grid
    fb = forcing_bounds
    aorc_res = 0.01
    aorc_x = np.arange(np.floor(fb[0] / aorc_res) * aorc_res,
                        np.ceil(fb[2] / aorc_res) * aorc_res + aorc_res, aorc_res)
    aorc_y = np.arange(np.floor(fb[1] / aorc_res) * aorc_res,
                        np.ceil(fb[3] / aorc_res) * aorc_res + aorc_res, aorc_res)
    for x in aorc_x:
        ax_para.axvline(x, color=aorc_color, linewidth=0.3, alpha=0.7, zorder=3)
    for y in aorc_y:
        ax_para.axhline(y, color=aorc_color, linewidth=0.3, alpha=0.7, zorder=3)

    # ERA5 grid
    para_forcing.plot(ax=ax_para, facecolor='none', edgecolor=cell_edge,
                      linewidth=1.8, linestyle='--', alpha=0.85, zorder=5)

    # Number ERA5 cells
    for idx, row in para_forcing.iterrows():
        c = row.geometry.centroid
        ax_para.text(c.x, c.y, str(idx + 1), ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white',
                     path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground=cell_edge)],
                     zorder=6)

    # Station marker
    ax_para.scatter(paradise_lon, paradise_lat, c='black', s=120 + 40, marker='^',
                    edgecolor='none', alpha=0.3, zorder=9)
    ax_para.scatter(paradise_lon, paradise_lat, c=marker_color, s=120, marker='^',
                    edgecolor='white', linewidth=1.5, zorder=10)

    pad_x = (forcing_bounds[2] - forcing_bounds[0]) * 0.03
    pad_y = (forcing_bounds[3] - forcing_bounds[1]) * 0.03
    ax_para.set_xlim(forcing_bounds[0] - pad_x, forcing_bounds[2] + pad_x)
    ax_para.set_ylim(forcing_bounds[1] - pad_y, forcing_bounds[3] + pad_y)
    ax_para.set_aspect('equal')
    ax_para.set_xticks([])
    ax_para.set_yticks([])
    for spine in ax_para.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#555')
    ax_para.set_title('(a) Paradise SNOTEL', fontsize=16, fontweight='bold', pad=6)

    # =========================================================================
    # Panels (b)-(d): Iceland
    # =========================================================================
    # (b) River Basin GRUs
    ax_ice1 = fig.add_subplot(gs_top[0, 1])
    era5_grid_ice.boundary.plot(ax=ax_ice1, color=cell_edge, linewidth=1.0,
                                linestyle='--', alpha=0.6, zorder=1)
    ice_elev_no_coastal.plot(ax=ax_ice1, column='elev_mean', cmap=cmap, norm=norm_global,
                              edgecolor='none', linewidth=0, zorder=2)
    ice_rivers.plot(ax=ax_ice1, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax_ice1, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ice_ax(ax_ice1)
    ax_ice1.set_title('(b) River Basin GRUs', fontsize=16, fontweight='bold', pad=6)
    ax_ice1.text(0.03, 0.03, f'{len(ice_no_coastal):,} GRUs\n{len(ice_rivers):,} seg',
                 transform=ax_ice1.transAxes, fontsize=12, ha='left', va='bottom',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (c) + Coastal GRUs
    ax_ice2 = fig.add_subplot(gs_top[0, 2])
    era5_grid_ice.boundary.plot(ax=ax_ice2, color=cell_edge, linewidth=1.0,
                                linestyle='--', alpha=0.6, zorder=1)
    ice_elev.plot(ax=ax_ice2, column='elev_mean', cmap=cmap, norm=norm_global,
                  edgecolor='none', linewidth=0, zorder=2)
    ice_rivers.plot(ax=ax_ice2, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax_ice2, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ice_ax(ax_ice2)
    ax_ice2.set_title('(c) + Coastal GRUs', fontsize=16, fontweight='bold', pad=6)
    ax_ice2.text(0.03, 0.03, f'{len(ice_with_coastal):,} GRUs\n(+{n_coastal:,} coastal)',
                 transform=ax_ice2.transAxes, fontsize=12, ha='left', va='bottom',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (d) + Elevation HRUs
    ax_ice3 = fig.add_subplot(gs_top[0, 3])
    era5_grid_ice.boundary.plot(ax=ax_ice3, color=cell_edge, linewidth=1.0,
                                linestyle='--', alpha=0.6, zorder=1)
    ice_elev.plot(ax=ax_ice3, column='elev_mean', cmap=cmap, norm=norm_global,
                  edgecolor='none', linewidth=0, zorder=2)
    ice_rivers.plot(ax=ax_ice3, color='white', linewidth=0.7, alpha=0.5, zorder=4)
    ice_rivers.plot(ax=ax_ice3, color=river_color, linewidth=0.35, alpha=0.9, zorder=5)
    style_ice_ax(ax_ice3)
    ice_elev_min_val = max(0, ice_elev['elev_mean'].min())
    ice_elev_max_val = ice_elev['elev_mean'].max()
    ax_ice3.set_title('(d) + Elevation HRUs', fontsize=16, fontweight='bold', pad=6)
    ax_ice3.text(0.03, 0.03, f'{len(ice_elev):,} HRUs\n{ice_elev_min_val:.0f}\u2013{ice_elev_max_val:.0f} m',
                 transform=ax_ice3.transAxes, fontsize=12, ha='left', va='bottom',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # =========================================================================
    # Panels (e)-(m): Bow River 3x3
    # =========================================================================
    # Row 1
    ax_e = fig.add_subplot(gs_bow[0, 0])
    plot_bow_panel(ax_e, bow_l_gru, bow_r_lump, edge_lw=1.5, edge_color='#bbbbbb')
    ax_e.set_title('(e) Single GRU', fontsize=16, fontweight='bold', pad=6)
    ax_e.text(0.03, 0.03, '1 GRU', transform=ax_e.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_f = fig.add_subplot(gs_bow[0, 1])
    plot_bow_panel(ax_f, bow_sd_gru, bow_r_semi, edge_lw=0.3, edge_color='#bbbbbb')
    ax_f.set_title('(f) Sub-basin GRUs', fontsize=16, fontweight='bold', pad=6)
    ax_f.text(0.03, 0.03, f'{len(bow_sd_gru)} GRUs\n{len(bow_r_semi)} seg', transform=ax_f.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_g = fig.add_subplot(gs_bow[0, 2])
    plot_bow_panel(ax_g, bow_dist, bow_r_dist, no_edges=True, show_rivers=False)
    ax_g.set_title('(g) Grid cells (1 km)', fontsize=16, fontweight='bold', pad=6)
    ax_g.text(0.03, 0.03, f'{len(bow_dist):,} cells\n{len(bow_r_dist):,} seg', transform=ax_g.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Row 2
    ax_h = fig.add_subplot(gs_bow[1, 0])
    plot_bow_panel(ax_h, bow_l_elev, bow_r_lump, no_edges=True)
    ax_h.set_title('(h) + Elevation bands', fontsize=16, fontweight='bold', pad=6)
    ax_h.text(0.03, 0.03, f'{len(bow_l_elev)} HRUs', transform=ax_h.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_i = fig.add_subplot(gs_bow[1, 1])
    plot_bow_panel(ax_i, bow_sd_elev, bow_r_semi, no_edges=True)
    ax_i.set_title('(i) + Elevation bands', fontsize=16, fontweight='bold', pad=6)
    ax_i.text(0.03, 0.03, f'{len(bow_sd_elev)} HRUs\n{len(bow_r_semi)} seg', transform=ax_i.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_j = fig.add_subplot(gs_bow[1, 2])
    plot_bow_panel(ax_j, bow_l_elev, bow_r_semi, no_edges=True)
    ax_j.set_title('(j) Lumped + semi-dist. routing', fontsize=16, fontweight='bold', pad=6)
    ax_j.text(0.03, 0.03, f'{len(bow_l_elev)} HRUs\n{len(bow_r_semi)} seg', transform=ax_j.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Row 3
    ax_k = fig.add_subplot(gs_bow[2, 0])
    plot_bow_panel(ax_k, bow_l_land, bow_r_lump, use_landclass=True, edge_lw=0.5, edge_color='#bbbbbb')
    ax_k.set_title('(k) + Land cover (IGBP)', fontsize=16, fontweight='bold', pad=6)
    ax_k.text(0.03, 0.03, f'{len(bow_l_land)} HRUs', transform=ax_k.transAxes, fontsize=13, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_l = fig.add_subplot(gs_bow[2, 1])
    plot_bow_panel(ax_l, bow_sd_asp, bow_r_semi, no_edges=True)
    ax_l.set_title('(l) + Elevation + Aspect', fontsize=16, fontweight='bold', pad=6)
    ax_l.text(0.03, 0.03, f'{len(bow_sd_asp):,} HRUs\n{len(bow_r_semi)} seg', transform=ax_l.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax_m = fig.add_subplot(gs_bow[2, 2])
    plot_bow_panel(ax_m, bow_l_gru, bow_r_semi, edge_lw=1.5, edge_color='#bbbbbb')
    ax_m.set_title('(m) Lumped + semi-dist. routing', fontsize=16, fontweight='bold', pad=6)
    ax_m.text(0.03, 0.03, f'1 GRU\n{len(bow_r_semi)} seg', transform=ax_m.transAxes, fontsize=12, ha='left', va='bottom',
              fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # =========================================================================
    # Shared colorbar
    # =========================================================================
    cbar_ax = fig.add_axes([0.15, 0.035, 0.45, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_global)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elevation (m)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # =========================================================================
    # Shared legend
    # =========================================================================
    present_lc = sorted(bow_l_land['landClass'].unique())
    short_labels = {1: 'Forest', 8: 'Woody Sav.', 9: 'Savanna', 10: 'Grass',
                    11: 'Wetland', 13: 'Urban', 15: 'Snow/Ice', 16: 'Barren', 17: 'Water'}

    legend_elements = [
        Line2D([0], [0], color=river_color, linewidth=2.5, label='River network'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#D32F2F',
               markeredgecolor='white', markersize=10, label='Gauge/Station'),
        Line2D([0], [0], color=cell_edge, linewidth=1.5, linestyle='--',
               alpha=0.7, label='ERA5 grid (0.25\u00b0)'),
        mpatches.Patch(facecolor='none', edgecolor=aorc_color, linewidth=0.6,
                       label='AORC grid (0.01\u00b0)'),
    ]
    for lc in present_lc:
        if lc in IGBP_CLASSES:
            legend_elements.append(
                mpatches.Patch(facecolor=IGBP_CLASSES[lc][1], edgecolor='#555',
                               linewidth=0.5, label=short_labels.get(lc, str(lc))))

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(legend_elements), 6), fontsize=12,
               framealpha=0.95, edgecolor='#ccc', columnspacing=1.0,
               handletextpad=0.4, handlelength=1.5)

    fig.savefig(OUTPUT_DIR / "figure_4_1_combined.png", dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure_4_1_combined.pdf", facecolor='white', bbox_inches='tight')
    print("  Saved combined figure")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("SYMFLUENCE Paper - Section 4.1: Domain Definition")
    print("=" * 60)

    # Each figure only needs its own reproduction domains; render what is
    # available and report the rest (e.g. Paradise needs the ERA5 forcing
    # shapefile produced by the 03_forcing_ensemble ERA5 config).
    failures = []
    for fn in (create_paradise_figure, create_iceland_figure,
               create_bow_figure, create_combined_figure):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - keep rendering the other figures
            failures.append((fn.__name__, e))
            print(f"  SKIPPED {fn.__name__}: {e}")

    print("\n" + "=" * 60)
    print("Generated figures (unless skipped above):")
    print("  - figure_4_1a_paradise.png/pdf")
    print("  - figure_4_1b_iceland.png/pdf")
    print("  - figure_4_1c_bow.png/pdf")
    print("  - figure_4_1_combined.png/pdf")
    print("=" * 60)
    if failures:
        raise SystemExit(1)
