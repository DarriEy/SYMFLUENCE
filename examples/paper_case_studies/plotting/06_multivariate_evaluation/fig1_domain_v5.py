#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Figure 1: Study Domain and Remote Sensing Observations - v5
Multivariate Evaluation Section

Layout (nested GridSpec):
  Col 0: Domain map (a) + Normalised streamflow (b) + SWE ensemble (c)
  Row 0 Col 1: TWS block - [CSR | GSFC | JPL | CNES/GRGS | ERA5-Land] / shared cbar / TWS TS
  Row 0 Col 2: ET block  - [SSEBop | MODIS | FLUXCOM]  / shared cbar / ET TS
  Row 1 Col 1: Snow block- [MODIS SCF | VIIRS | IMS | Sentinel-2] / shared cbar / SCA TS
  Row 1 Col 2: SM block  - [SMAP | ESA CCI | SMOS | ASCAT | ERA5-Land] / shared cbar / SM TS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import pandas as pd
import xarray as xr
import rasterio
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bow_banff_style import COLORS, FONT, setup_style, despine

setup_style()

# =============================================================================
# PATHS
# =============================================================================
# Data root from SYMFLUENCE_DATA_DIR (default: sibling SYMFLUENCE_data of the
# repo root). The 06_multivariate_evaluation configs run domain
# Bow_at_Banff_multivar (experiments bow_exp1_streamflow_only,
# bow_exp2_tws_only, bow_exp3_joint, bow_exp4_moead_joint).
import os as _os
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
_DATA_ROOT = Path(_os.environ.get('SYMFLUENCE_DATA_DIR', _REPO_ROOT.parent / 'SYMFLUENCE_data'))
DATA_DIR = _DATA_ROOT / "domain_Bow_at_Banff_multivar"
# Observations/attributes live under data/ in the current layout; the original
# study used a flat domain layout.
OBS_DIR = (DATA_DIR / "data" / "observations") if (DATA_DIR / "data" / "observations").exists() \
    else (DATA_DIR / "observations")
ATTR_DIR = (DATA_DIR / "data" / "attributes") if (DATA_DIR / "data" / "attributes").exists() \
    else (DATA_DIR / "attributes")
OUTPUT_DIR = _HERE.parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Curated satellite subsets (GRACE mascons, MODIS/VIIRS/IMS/S2 snow, MOD16 /
# FLUXCOM ET) used by the inset panels below live in a "default/" folder of
# hand-staged paper assets, not experiment outputs. Set P3_MULTIVAR_ASSETS_DIR
# to a directory containing that "default/" folder (e.g. a copy of the paper's
# "10. Multivariate evaluation" folder); otherwise a default/ dir next to this
# script is used.
PAPER_DIR = Path(_os.environ.get('P3_MULTIVAR_ASSETS_DIR', _HERE.parent))

CATCHMENT_SHP = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"
RDRS_GRID_SHP = DATA_DIR / "shapefiles/forcing/forcing_RDRS.shp"
# The semidistributed Bow river network comes from the 01_domain_definition
# reproduction (domain Bow_at_Banff_semidistributed_era5).
RIVER_SHP = (_DATA_ROOT / "domain_Bow_at_Banff_semidistributed_era5" / "shapefiles" / "river_network" /
             "Bow_at_Banff_semidistributed_era5_riverNetwork_semidistributed.shp")
DEM_TIF = ATTR_DIR / "elevation/dem/domain_Bow_at_Banff_multivar_elv.tif"
CANSWE_STATIONS_CSV = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_canswe_stations.csv"
GLACIER_GPKG = OBS_DIR / "glacier/Bow_at_Banff_glaciers.gpkg"
HYDROLAKES_GPKG = ATTR_DIR / "lakes/domain_Bow_at_Banff_multivar_hydrolakes.gpkg"
ECCC_STATIONS_CSV = OBS_DIR / "climate/eccc_stations.csv"
BANFF_CS_MONTHLY = OBS_DIR / "climate/banff_cs_monthly.csv"

# Spatial
SSEBOP_DIR = OBS_DIR / "et/ssebop"
GRACE_CSR_NC = PAPER_DIR / "default/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
GRACE_GSFC_NC = PAPER_DIR / "default/gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc"
GRACE_JPL_NC = PAPER_DIR / "default/GRCTellus.JPL.200204_202511.GLO.RL06.3M.MSCNv04CRI.nc"
MODIS_SNOW_NPZ = PAPER_DIR / "default/modis_snow/mod10a1f_2016115_bow_subset.npz"
VIIRS_SNOW_NPZ = PAPER_DIR / "default/viirs/viirs_2016115_bow_subset.npz"
IMS_SNOW_NPZ = PAPER_DIR / "default/ims/ims_2016115_bow_subset.npz"
S2_SNOW_NPZ = PAPER_DIR / "default/sentinel2/s2_scl_snow_20160502_bow_subset.npz"
MOD16_ET_NPZ = PAPER_DIR / "default/modis_et/mod16a2gf_2016185_bow_subset.npz"
FLUXCOM_ET_NPZ = PAPER_DIR / "default/fluxcom_et/fluxcom_et_jul2016_bow_subset.npz"

# Time series
STREAMFLOW_RAW = OBS_DIR / "streamflow/raw_data"
SWE_ALL_CSV = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_canswe_swe_all_stations.csv"
SSEBOP_ET_CSV = OBS_DIR / "et/ssebop/Bow_at_Banff_multivar_ssebop_et_processed.csv"
MODIS_ET_NC = OBS_DIR / "et/modis/Bow_at_Banff_MOD16_ET.nc"
FLUXCOM_ET_CSV = OBS_DIR / "et/preprocessed/Bow_at_Banff_multivar_fluxcom_et_processed.csv"
GRACE_TS_CSV = OBS_DIR / "grace/preprocessed/Bow_at_Banff_multivar_grace_tws_processed.csv"
ERA5L_TWS_CSV = OBS_DIR / "tws/era5_land_tws_processed.csv"
CNES_GRGS_TWS_CSV = OBS_DIR / "tws/cnes_grgs_tws_processed.csv"
GLDAS_TWS_CSV = OBS_DIR / "tws/gldas_noah_tws_processed.csv"
MODIS_TERRA_CSV = OBS_DIR / "snow/modis/mod10a1_processed.csv"
MODIS_AQUA_CSV = OBS_DIR / "snow/modis/myd10a1_processed.csv"
VIIRS_SCA_CSV = OBS_DIR / "snow/viirs/viirs_sca_processed.csv"
IMS_SNOW_CSV = OBS_DIR / "snow/ims/ims_snow_4km.csv"
SENTINEL2_SNOW_CSV = OBS_DIR / "snow/sentinel2/Bow_at_Banff_Sentinel2_snow_timeseries.csv"
PRECIP_CSV = DATA_DIR / "forcing/preprocessed/Bow_at_Banff_multivar_rdrs_precip_monthly.csv"
SMAP_SM_CSV = OBS_DIR / "soil_moisture/smap/smap_summer_processed.csv"
ESA_CCI_SM_CSV = OBS_DIR / "soil_moisture/esa_cci/esa_cci_sm_processed.csv"
SMOS_SM_CSV = OBS_DIR / "soil_moisture/smos/smos_sm_processed.csv"
ASCAT_SM_CSV = OBS_DIR / "soil_moisture/ascat/ascat_sm_processed.csv"
ERA5LAND_SM_CSV = OBS_DIR / "soil_moisture/era5_land/era5_land_sm_processed.csv"
SENTINEL1_SM_CSV = OBS_DIR / "soil_moisture/sentinel1/sentinel1_sm_processed.csv"

# Gridded SWE products
ERA5LAND_SWE_CSV = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_era5land_swe_processed.csv"
SNODAS_SWE_CSV = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_snodas_swe_processed.csv"
CMC_SWE_CSV = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_cmc_swe_processed.csv"

# Constants
POUR_POINT = (-115.5717, 51.1722)
CATCHMENT_AREA_KM2 = 2210
ELEV_RANGE = (1380, 3400)
CATCH_BBOX = (-116.55, 50.95, -115.50, 51.76)

WSC_STATIONS = {
    '05BA001': {'name': 'Bow at Lake Louise', 'lon': -116.187, 'lat': 51.421, 'color': '#E69F00'},
    '05BA002': {'name': 'Pipestone River',    'lon': -116.189, 'lat': 51.438, 'color': '#56B4E9'},
    '05BB001': {'name': 'Bow at Banff',       'lon': -115.572, 'lat': 51.172, 'color': '#D55E00'},
    '05BC001': {'name': 'Spray River',        'lon': -115.578, 'lat': 51.170, 'color': '#009E73'},
}

TS_COLORS = {
    'ssebop': '#E69F00', 'modis_et': '#009E73', 'fluxcom': '#CC79A7',
    'grace_csr': '#7b3294', 'grace_gsfc': '#008837', 'grace_jpl': '#d95f02',
    'modis_terra': '#0072B2', 'modis_aqua': '#56B4E9',
    'viirs': '#D55E00', 'ims': '#CC79A7', 'sentinel2': '#333333',
    'smap': '#56B4E9', 'esa_cci': '#E69F00', 'smos': '#1B9E77', 'ascat': '#E7298A',
    'era5land_sm': '#984EA3', 'sentinel1': '#FF7F00',
    'era5land_swe': '#D55E00', 'snodas_swe': '#009E73', 'cmc_swe': '#CC79A7',
    'era5land_tws': '#0072B2', 'cnes_grgs': '#e7298a', 'gldas_tws': '#66a61e',
}

XLIM = (pd.Timestamp('2002-01-01'), pd.Timestamp('2018-01-01'))
FS = FONT['size']


# =============================================================================
# HELPERS
# =============================================================================
def load_shp(path, from_dir=False):
    if from_dir:
        files = list(path.glob("*.shp"))
        path = files[0] if files else path
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=4326)


def load_snow_stations(catchment):
    if not CANSWE_STATIONS_CSV.exists():
        return None
    df = pd.read_csv(CANSWE_STATIONS_CSV)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']),
                           crs='EPSG:4326')
    return gdf[gdf.geometry.within(catchment.geometry.unary_union)]


def _clean_map_ax(ax):
    """Uniform map axis styling: thin frame, no tick labels."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#777777'); sp.set_linewidth(0.6)


def _schematic_grid_map(ax, catchment, res_deg, cmap, vmin, vmax, mean_val,
                        label, panel_lbl, gradient_dir='se'):
    """Draw catchment with grid overlay and smooth spatial gradient (no noise)."""
    c4 = catchment.to_crs(epsg=4326)
    bb = CATCH_BBOX
    lon_range = bb[2] - bb[0]
    lat_range = bb[3] - bb[1]

    lons = np.arange(np.floor(bb[0]/res_deg)*res_deg - res_deg,
                     np.ceil(bb[2]/res_deg)*res_deg + 2*res_deg, res_deg)
    lats = np.arange(np.floor(bb[1]/res_deg)*res_deg - res_deg,
                     np.ceil(bb[3]/res_deg)*res_deg + 2*res_deg, res_deg)

    # Smooth gradient: value varies with position
    val_range = (vmax - vmin) * 0.25
    for lon in lons:
        for lat in lats:
            ln = (lon - bb[0]) / lon_range
            lt = (lat - bb[1]) / lat_range
            if gradient_dir == 'se':
                v = mean_val + val_range * (ln * 0.5 - lt * 0.4)
            elif gradient_dir == 'elev':
                v = mean_val + val_range * (-ln * 0.3 + lt * 0.3)
            else:
                v = mean_val
            v = np.clip(v, vmin, vmax)
            fc = cmap((v - vmin) / (vmax - vmin))
            edge_lw = 0.2 if res_deg < 0.01 else 0.4
            ax.add_patch(Rectangle((lon, lat), res_deg, res_deg, linewidth=edge_lw,
                                   edgecolor='#aaaaaa', facecolor=fc, alpha=0.5, zorder=2))
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    # Consistent extent for all schematic maps
    sm_buf = 0.12
    ax.set_xlim(bb[0]-sm_buf, bb[2]+sm_buf)
    ax.set_ylim(bb[1]-sm_buf, bb[3]+sm_buf)
    _clean_map_ax(ax)
    ax.set_title(label, fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, f'({panel_lbl})', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)


def _format_ts_ax(ax, ylabel, panel_lbl=None, xlim=None, legend_loc='upper right',
                  legend_ncol=1, year_step=4, xlabel=None):
    ax.set_ylabel(ylabel, fontsize=FS['axis_label'], labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS['axis_label'])
    ax.set_xlim(*(xlim or XLIM))
    ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FS['tick'], which='both')
    ax.tick_params(axis='x', pad=1.5, rotation=0)
    # Subtle y-grid for readability
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    if panel_lbl:
        ax.text(0.012, 0.92, f'({panel_lbl})', transform=ax.transAxes,
               fontsize=FS['panel_label'], fontweight='bold', va='top',
               bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                        edgecolor='none', alpha=0.85), zorder=20)
    ax.legend(fontsize=FS['legend'], loc=legend_loc, frameon=True,
             facecolor='white', edgecolor='none', framealpha=0.92, ncol=legend_ncol,
             handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
             borderpad=0.3, labelspacing=0.3)
    despine(ax)


def _add_shared_cbar(fig, ax_list, mappable, label, orientation='horizontal'):
    """Add a single shared colorbar below a row of axes."""
    bboxes = [a.get_position() for a in ax_list]
    x0 = min(b.x0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    span = x1 - x0
    inset = span * 0.06 if len(ax_list) >= 5 else span * 0.10
    cbar_h = 0.012
    cbar_pad = 0.012
    cax = fig.add_axes([x0 + inset, y0 - cbar_pad - cbar_h, span - 2*inset, cbar_h])
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=FS['tick'], labelpad=2)
    cb.ax.tick_params(labelsize=FS['tick'], length=2.5, pad=1.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    return cb


# =============================================================================
# DOMAIN BLOCK
# =============================================================================
def _hillshade(dem, azimuth=315, altitude=45):
    """Compute hillshade from DEM array."""
    dy, dx = np.gradient(dem)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = np.sin(alt_rad) * np.cos(slope) + \
         np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    return np.clip(hs, 0, 1)


def plot_domain_map(ax, catchment, rdrs_grid, rivers, snow_stations):
    from shapely.geometry import box as shapely_box
    c4 = catchment.to_crs(epsg=4326)
    bb = CATCH_BBOX
    # Asymmetric buffer: wider east/south to show Lake Minnewanka & Spray Lake
    buf_w, buf_e, buf_s, buf_n = 0.08, 0.28, 0.10, 0.08

    # Read DEM and create hillshade + terrain coloring
    with rasterio.open(DEM_TIF) as src:
        from rasterio.windows import from_bounds
        w = from_bounds(bb[0]-buf_w, bb[1]-buf_s, bb[2]+buf_e, bb[3]+buf_n, src.transform)
        dem = src.read(1, window=w).astype(float)
        tf = src.window_transform(w)
        nd = src.nodata
        if nd is not None:
            dem[dem == nd] = np.nan

    nr, nc_dem = dem.shape
    ext = [tf.c, tf.c + nc_dem*tf.a, tf.f + nr*tf.e, tf.f]

    # Terrain coloring + hillshade
    terrain_cmap = mcolors.LinearSegmentedColormap.from_list(
        'terrain_bow', ['#4a7c59', '#8db580', '#c8d6a0', '#e8daa0',
                        '#c4a66a', '#a08060', '#8c7060', '#f0f0f0'], N=256)
    hs = _hillshade(dem)
    dem_masked = np.ma.masked_invalid(dem)
    im_dem = ax.imshow(dem_masked, extent=ext, origin='upper', cmap=terrain_cmap,
              vmin=1300, vmax=3500, aspect='auto', interpolation='bilinear', alpha=0.75)
    ax.imshow(hs, extent=ext, origin='upper', cmap='gray',
              vmin=0, vmax=1, aspect='auto', interpolation='bilinear', alpha=0.30)

    # --- Glaciers: icy blue-white polygons (before wash-out so outside ones get dimmed) ---
    if GLACIER_GPKG.exists():
        glaciers = gpd.read_file(GLACIER_GPKG).to_crs(epsg=4326)
        # Subtle halo for definition against terrain
        glaciers.plot(ax=ax, facecolor='none', edgecolor='white',
                      linewidth=1.8, alpha=0.6, zorder=2.4)
        # Fill with pale ice tint + crisp edge
        glaciers.plot(ax=ax, color='#f0f7fc', edgecolor='#8cafc4',
                      linewidth=0.5, alpha=0.80, zorder=2.5)

    # --- Lakes: blue polygons (before wash-out so outside ones get dimmed) ---
    if HYDROLAKES_GPKG.exists():
        lakes = gpd.read_file(HYDROLAKES_GPKG).to_crs(epsg=4326)
        # Only show lakes > 0.1 km² to avoid clutter
        if 'Lake_area' in lakes.columns:
            lakes = lakes[lakes['Lake_area'] >= 0.1]
        # White halo for contrast against terrain
        lakes.plot(ax=ax, facecolor='none', edgecolor='white',
                   linewidth=1.4, alpha=0.55, zorder=2.55)
        lakes.plot(ax=ax, color='#8bb8d8', edgecolor='#4a88aa',
                   linewidth=0.5, alpha=0.92, zorder=2.6)

    # --- Dim terrain OUTSIDE catchment to make basin pop ---
    ext_box = shapely_box(bb[0]-buf_w-0.1, bb[1]-buf_s-0.1, bb[2]+buf_e+0.1, bb[3]+buf_n+0.1)
    outside_mask = ext_box.difference(c4.geometry.unary_union)
    mask_gdf = gpd.GeoDataFrame(geometry=[outside_mask], crs='EPSG:4326')
    mask_gdf.plot(ax=ax, color='white', alpha=0.50, zorder=3, edgecolor='none')

    # --- ERA5 grid (0.25° ~31 km, dashed warm color) ---
    from shapely.geometry import box as _box
    era5_res = 0.25
    era5_lons = np.arange(np.floor(bb[0] / era5_res) * era5_res,
                          np.ceil((bb[2] + buf_e) / era5_res) * era5_res + era5_res, era5_res)
    era5_lats = np.arange(np.floor(bb[1] / era5_res) * era5_res,
                          np.ceil((bb[3] + buf_n) / era5_res) * era5_res + era5_res, era5_res)
    era5_cells = [_box(lon, lat, lon + era5_res, lat + era5_res)
                  for lon in era5_lons for lat in era5_lats]
    era5_gdf = gpd.GeoDataFrame(geometry=era5_cells, crs='EPSG:4326')
    era5_gdf.plot(ax=ax, facecolor='none', edgecolor='#b35900',
                  linewidth=0.50, alpha=0.50, linestyle='--', zorder=3.3)

    # --- RDRS forcing grid (~13 km, solid dark) ---
    if rdrs_grid is not None and len(rdrs_grid) > 0:
        rg = rdrs_grid.to_crs(epsg=4326)
        rg.plot(ax=ax, facecolor='none', edgecolor='#444444',
                linewidth=0.45, alpha=0.55, zorder=3.5)

    # River network - strong white halo + vivid blue
    if rivers is not None and len(rivers) > 0:
        line_rivers = rivers[rivers.geometry.type.isin(['LineString', 'MultiLineString'])]
        if len(line_rivers) > 0:
            if 'strmOrder' in line_rivers.columns:
                for _, row in line_rivers.iterrows():
                    order = row.get('strmOrder', 1)
                    lw = max(0.6, min(2.5, order * 0.50))
                    gdf_row = gpd.GeoDataFrame([row], crs=rivers.crs)
                    # Thick white halo for contrast
                    gdf_row.plot(ax=ax, color='white', linewidth=lw+2.0, alpha=0.70, zorder=5)
                    gdf_row.plot(ax=ax, color='#1565C0', linewidth=lw, alpha=0.95, zorder=6)
            else:
                line_rivers.plot(ax=ax, color='white', linewidth=2.8, alpha=0.70, zorder=5)
                line_rivers.plot(ax=ax, color='#1565C0', linewidth=1.2, alpha=0.95, zorder=6)

    # Catchment boundary - white halo + dark blue for maximum contrast
    c4.boundary.plot(ax=ax, color='white', linewidth=3.5, alpha=0.7, zorder=7)
    c4.boundary.plot(ax=ax, color='#1a4a7a', linewidth=1.8, zorder=8)

    # SWE stations - slightly larger, dark markers
    if snow_stations is not None and len(snow_stations) > 0:
        ax.scatter(snow_stations.geometry.x, snow_stations.geometry.y,
                  s=25, c='#444444', marker='s', edgecolor='white',
                  linewidth=0.6, zorder=10, alpha=0.9)

    # ECCC climate stations - small diamonds
    n_eccc = 0
    if ECCC_STATIONS_CSV.exists():
        eccc = pd.read_csv(ECCC_STATIONS_CSV)
        n_eccc = len(eccc)
        ax.scatter(eccc['lon'], eccc['lat'], s=30, c='#e8e8e8',
                  marker='D', edgecolor='#555555', linewidth=0.5,
                  zorder=9.5, alpha=0.9)

    # WSC gauging stations - colored triangles
    for sid, info in WSC_STATIONS.items():
        ax.scatter(info['lon'], info['lat'], marker='^', s=110,
                  color=info['color'], edgecolor='k', linewidth=0.9, zorder=11)

    # Pour point / basin outlet — highlight Bow at Banff
    ax.scatter(POUR_POINT[0], POUR_POINT[1], marker='*', s=220,
              color='#c62828', edgecolor='white', linewidth=0.8, zorder=12)

    # Scale bar — right side, mid-height, in dimmed terrain east of catchment
    import matplotlib.patheffects as pe
    sl_deg = 20 / (111.32 * np.cos(np.radians(51.3)))
    sx = bb[2] + buf_e - sl_deg - 0.03              # bottom-right, inset from edge
    sy = bb[1] - buf_s + 0.04                      # just above bottom edge
    ax.plot([sx, sx+sl_deg], [sy, sy], color='k', linewidth=2.5, zorder=13,
            solid_capstyle='butt', path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])
    for xp in [sx, sx+sl_deg]:
        ax.plot([xp, xp], [sy-0.010, sy+0.010], color='k', linewidth=1.5, zorder=13,
                path_effects=[pe.withStroke(linewidth=3.5, foreground='white')])
    ax.text(sx+sl_deg/2, sy+0.015, '20 km', ha='center', va='bottom',
            fontsize=FS['annotation'], fontweight='bold', zorder=13,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax.set_xlim(bb[0]-buf_w, bb[2]+buf_e)
    ax.set_ylim(bb[1]-buf_s, bb[3]+buf_n)
    # Lat/lon tick labels for geographic reference
    ax.set_xticks([-116.5, -116.0, -115.5])
    ax.set_xticklabels(['116.5\u00b0W', '116.0\u00b0W', '115.5\u00b0W'],
                        fontsize=FS['tick'], color='#555555')
    ax.set_yticks([51.0, 51.5])
    ax.set_yticklabels(['51.0\u00b0N', '51.5\u00b0N'],
                        fontsize=FS['tick'], color='#555555')
    ax.tick_params(axis='both', length=3, width=0.5, pad=2, direction='in')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#777777'); sp.set_linewidth(0.6)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)

    # --- DEM elevation colorbar (vertical, top-right of map) ---
    fig = ax.figure
    map_pos = ax.get_position()
    cax_dem = fig.add_axes([map_pos.x1 - 0.032, map_pos.y0 + map_pos.height*0.58,
                            0.010, map_pos.height * 0.35])
    cb_dem = fig.colorbar(im_dem, cax=cax_dem, orientation='vertical')
    cb_dem.set_label('')  # no rotated label — use title instead
    cb_dem.set_ticks([1500, 2500, 3500])
    cb_dem.ax.tick_params(labelsize=FS['tick'], length=2, pad=1.5, width=0.4)
    cb_dem.ax.set_title('m a.s.l.', fontsize=FS['tick'], pad=3)
    cb_dem.outline.set_linewidth(0.5)

    handles = [
        Patch(facecolor='none', edgecolor='#1a4a7a', linewidth=1.5,
              label=f'Basin ({CATCHMENT_AREA_KM2:,} km\u00b2)'),
        Line2D([0],[0], color='#1565C0', linewidth=1.2, label='Rivers'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#c62828',
               markersize=9, markeredgecolor='white', markeredgewidth=0.3,
               label='Outlet'),
        Patch(facecolor='#f0f7fc', edgecolor='#8cafc4', linewidth=0.6,
              label='Glaciers'),
        Patch(facecolor='#8bb8d8', edgecolor='#4a88aa', linewidth=0.6,
              label='Lakes'),
        Patch(facecolor='none', edgecolor='#444444', linewidth=0.5,
              label='RDRS grid (~13 km)'),
        Line2D([0],[0], color='#b35900', linewidth=0.6, linestyle='--',
               alpha=0.6, label='ERA5 grid (~31 km)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#E69F00',
               markersize=6, markeredgecolor='k', markeredgewidth=0.5,
               label=f'WSC gauges (n={len(WSC_STATIONS)})'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#444444',
               markersize=5, markeredgecolor='white', markeredgewidth=0.4,
               label='SWE stations'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#e8e8e8',
               markersize=5, markeredgecolor='#555555', markeredgewidth=0.4,
               label='ECCC stations'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01),
             ncol=3, fontsize=FS['legend'], frameon=False,
             columnspacing=0.8, handletextpad=0.3, borderpad=0.2,
             labelspacing=0.3)


def plot_inset_map(ax_main):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    bb = ax_main.get_position()
    ax_in = ax_main.figure.add_axes(
        [bb.x0+0.005, bb.y0+0.005, bb.width*0.33, bb.height*0.30],
        projection=ccrs.LambertConformal(central_longitude=-110, central_latitude=50))
    ax_in.set_extent([-140, -90, 40, 65], crs=ccrs.PlateCarree())
    ax_in.add_feature(cfeature.LAND, facecolor='#f5f0e8', edgecolor='none')
    ax_in.add_feature(cfeature.OCEAN, facecolor='#d4e6f1')
    ax_in.add_feature(cfeature.LAKES, facecolor='#d4e6f1', edgecolor='#aabbcc', linewidth=0.2)
    ax_in.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#888888')
    ax_in.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#888888')
    ax_in.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m',
        facecolor='none', edgecolor='#cccccc', linewidth=0.2))
    ax_in.plot(POUR_POINT[0], POUR_POINT[1], '*', color='#c62828', markersize=10,
              zorder=10, markeredgecolor='white', markeredgewidth=0.6, transform=ccrs.PlateCarree())
    for sp in ax_in.spines.values():
        sp.set_edgecolor('#999999'); sp.set_linewidth(0.6)


def plot_streamflow_ts(ax):
    # --- Precipitation bars on secondary y-axis (inverted, from top) ---
    ax2 = None
    if PRECIP_CSV.exists():
        pdf = pd.read_csv(PRECIP_CSV, parse_dates=['date']).set_index('date')
        pdf = pdf[(pdf.index >= '2002-01-01') & (pdf.index <= '2017-12-31')]
        ax2 = ax.twinx()
        ax2.bar(pdf.index, pdf['precip_mm'], width=24, color='#7eb8d8',
                alpha=0.50, zorder=1, label='P (RDRS)')
        # ECCC Banff CS station precip overlay
        if BANFF_CS_MONTHLY.exists():
            eccc_p = pd.read_csv(BANFF_CS_MONTHLY, parse_dates=['date']).set_index('date')
            eccc_p = eccc_p[(eccc_p.index >= '2002-01-01') & (eccc_p.index <= '2017-12-31')]
            eccc_p['total_precip'] = pd.to_numeric(eccc_p['total_precip'], errors='coerce')
            valid = eccc_p['total_precip'].dropna()
            ax2.step(valid.index, valid.values, where='mid', color='#2a5f8a',
                     linewidth=0.8, alpha=0.85, zorder=2, label='P (Banff CS)')
        ax2.set_ylim(260, 0)  # inverted: 0 at top
        # No rotated label — color-coded ticks + legend entry sufficient
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelsize=FS['tick']-0.5, colors='#5a7d9a',
                        pad=1, length=3)
        ax2.spines['right'].set_color('#5a7d9a')
        ax2.spines['right'].set_linewidth(0.6)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)

    # --- Streamflow lines ---
    q_handles = []
    for fname in sorted(STREAMFLOW_RAW.glob("wsc_*_raw.csv")):
        df = pd.read_csv(fname)
        date_col = [c for c in df.columns if 'DATE' in c.upper()][0]
        q_col = [c for c in df.columns if 'DISCHARGE' in c.upper()][0]
        sid_col = [c for c in df.columns if 'STATION_NUMBER' in c.upper()][0]
        sid = df[sid_col].iloc[0]
        df[date_col] = pd.to_datetime(df[date_col])
        ts = df.set_index(date_col)[q_col].dropna()
        ts = ts[(ts.index >= '2002-01-01') & (ts.index <= '2017-12-31')]
        if len(ts) == 0:
            continue
        mean_q = ts.mean()
        ts_norm = ts / mean_q
        ts_m = ts_norm.resample('MS').mean()
        info = WSC_STATIONS.get(sid, {})
        color = info.get('color', '#888888')
        name = info.get('name', sid)
        lw = 1.4 if sid == '05BB001' else 1.0
        alpha = 0.95 if sid == '05BB001' else 0.75
        line, = ax.plot(ts_m.index, ts_m.values, color=color, linewidth=lw,
                       alpha=alpha, zorder=3)
        short_name = name.replace('Bow at ', '').replace('River', 'R.').replace('Pipestone', 'Pipestone')
        q_handles.append(Line2D([0],[0], color=color, linewidth=lw, alpha=alpha,
                               label=short_name))

    ax.set_ylim(0, 5)
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=FS['tick'])
    ax.tick_params(axis='x', pad=1.5)
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel('Q / \u0051\u0304', fontsize=FS['axis_label'], labelpad=2)
    ax.text(0.015, 0.93, '(b)', transform=ax.transAxes,
           fontsize=FS['panel_label'], fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=0.85), zorder=20)
    despine(ax)

    # --- Combined legend: precip handles + Q handles, inside panel ---
    all_handles = []
    if ax2 is not None:
        all_handles.append(Patch(facecolor='#7eb8d8', alpha=0.55, edgecolor='none',
                                label='P RDRS, mm'))
        if BANFF_CS_MONTHLY.exists():
            all_handles.append(Line2D([0],[0], color='#2a5f8a', linewidth=0.8,
                                     label='P Banff CS'))
    all_handles.extend(q_handles)
    ax.legend(handles=all_handles, loc='upper center', ncol=6,
             fontsize=FS['legend'], frameon=False,
             bbox_to_anchor=(0.50, -0.12),
             columnspacing=0.5, handletextpad=0.3, handlelength=1.2,
             borderpad=0.2, labelspacing=0.2)


def plot_swe_ts(ax):
    if not SWE_ALL_CSV.exists():
        return
    df = pd.read_csv(SWE_ALL_CSV, parse_dates=['datetime'])
    stations = sorted(df['station_id'].unique())
    n_total = len(stations)
    continuous, manual = [], []
    for sid in stations:
        n = len(df[df['station_id'] == sid])
        (continuous if n > 500 else manual).append(sid)
    for sid in continuous:
        sub = df[df['station_id'] == sid].set_index('datetime').sort_index()
        ax.plot(sub.index, sub['swe_mm'], color='#4a90d9', linewidth=0.4, alpha=0.3)
    for sid in manual:
        sub = df[df['station_id'] == sid].set_index('datetime').sort_index()
        ax.scatter(sub.index, sub['swe_mm'], s=2, color='#aaaaaa', alpha=0.35,
                  zorder=2, edgecolors='none')
    pivot = df.pivot_table(index='datetime', columns='station_id', values='swe_mm', aggfunc='mean')
    monthly = pivot.resample('MS').mean()
    p10 = monthly.quantile(0.10, axis=1)
    p90 = monthly.quantile(0.90, axis=1)
    median = monthly.median(axis=1)
    ax.fill_between(p10.index, p10.values, p90.values,
                   color='#4a90d9', alpha=0.12, label='CanSWE 10\u201390%')
    ax.plot(median.index, median.values, color='#222222', linewidth=1.6,
           label=f'CanSWE median', zorder=5)

    # --- Gridded SWE products ---
    gridded_swe = [
        (SNODAS_SWE_CSV, TS_COLORS['snodas_swe'], 'SNODAS'),
        (ERA5LAND_SWE_CSV, TS_COLORS['era5land_swe'], 'ERA5-Land'),
        (CMC_SWE_CSV, TS_COLORS['cmc_swe'], 'CMC'),
    ]
    for csv_path, color, label in gridded_swe:
        if csv_path.exists():
            gdf = pd.read_csv(csv_path, parse_dates=['date']).set_index('date')
            ax.plot(gdf.index, gdf['swe_mm'], color=color, linewidth=1.0,
                   label=label, alpha=0.85, zorder=4)

    ax.set_ylim(0, None)

    # --- Temperature on secondary y-axis (Banff CS) — subtle background fill ---
    if BANFF_CS_MONTHLY.exists():
        eccc_t = pd.read_csv(BANFF_CS_MONTHLY, parse_dates=['date']).set_index('date')
        eccc_t = eccc_t[(eccc_t.index >= '2002-01-01') & (eccc_t.index <= '2017-12-31')]
        eccc_t['mean_temp'] = pd.to_numeric(eccc_t['mean_temp'], errors='coerce')
        valid_t = eccc_t['mean_temp'].dropna()
        ax_t = ax.twinx()
        # Subtle warm/cold fill rather than a busy line
        ax_t.fill_between(valid_t.index, 0, valid_t.values,
                          where=valid_t.values >= 0, color='#e8a0a0', alpha=0.18, zorder=0)
        ax_t.fill_between(valid_t.index, 0, valid_t.values,
                          where=valid_t.values < 0, color='#a0b8e8', alpha=0.18, zorder=0)
        ax_t.plot(valid_t.index, valid_t.values, color='#aa5555', linewidth=0.4,
                  alpha=0.40, zorder=0)
        ax_t.axhline(0, color='#999999', linewidth=0.3, alpha=0.5, zorder=0, linestyle='--')
        ax_t.set_ylim(-20, 25)
        # No rotated label — color-coded ticks + legend entry sufficient
        ax_t.set_ylabel('')
        ax_t.tick_params(axis='y', labelsize=FS['tick']-0.5, colors='#aa5555',
                         pad=1, length=3)
        ax_t.set_yticks([-15, 0, 15])
        ax_t.spines['right'].set_color('#aa5555')
        ax_t.spines['right'].set_linewidth(0.5)
        ax_t.spines['top'].set_visible(False)
        ax_t.spines['left'].set_visible(False)

    _format_ts_ax(ax, 'SWE (mm)', panel_lbl='c', legend_ncol=3, legend_loc='upper right')
    # Append temp entry to existing legend
    h, l = ax.get_legend_handles_labels()
    if BANFF_CS_MONTHLY.exists():
        h.append(Patch(facecolor='#e8a0a0', alpha=0.3, edgecolor='none', label='T, \u00b0C'))
        l.append('T, \u00b0C')
    ax.legend(h, l, fontsize=FS['legend'], loc='upper center', frameon=True,
              facecolor='white', edgecolor='#cccccc', framealpha=0.97, ncol=4,
              bbox_to_anchor=(0.50, 0.98),
              handlelength=1.2, handletextpad=0.2, columnspacing=0.5,
              borderpad=0.35, labelspacing=0.25)


# =============================================================================
# TWS BLOCK (3 GRACE mascon products)
# =============================================================================
def _load_grace_slice(nc_path, is_jpl=False, is_gsfc=False):
    ds = xr.open_dataset(nc_path)
    lon_min, lon_max = CATCH_BBOX[0]+360, CATCH_BBOX[2]+360
    sub = ds.sel(lat=slice(CATCH_BBOX[1]-1.5, CATCH_BBOX[3]+1.5),
                 lon=slice(lon_min-2.0, lon_max+2.0))
    if is_gsfc or is_jpl:
        t_idx = int(np.argmin(np.abs(sub.time.values - np.datetime64('2010-07-15'))))
    else:
        t_idx = int(np.argmin(np.abs(sub.time.values - 3100)))
    tws = sub['lwe_thickness'].isel(time=t_idx).values
    lons, lats = sub.lon.values - 360, sub.lat.values
    dl = np.abs(lats[1]-lats[0])/2 if len(lats) > 1 else 0.25
    dn = np.abs(lons[1]-lons[0])/2 if len(lons) > 1 else 0.25
    ds.close()
    return tws, np.r_[lons-dn, lons[-1]+dn], np.r_[lats-dl, lats[-1]+dl], lons, lats


def plot_grace_map(ax, catchment, nc_path, label, panel_lbl, is_gsfc=False, is_jpl=False):
    """Plot GRACE mascon map. Returns the mappable for shared colorbar."""
    c4 = catchment.to_crs(epsg=4326)
    tws, lon_e, lat_e, lons, lats = _load_grace_slice(nc_path, is_jpl=is_jpl, is_gsfc=is_gsfc)
    im = ax.pcolormesh(lon_e, lat_e, tws, cmap='BrBG', vmin=-15, vmax=15, shading='flat')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    # Consistent extent for all GRACE products
    grace_buf = 0.70
    ax.set_xlim(CATCH_BBOX[0] - grace_buf, CATCH_BBOX[2] + grace_buf)
    ax.set_ylim(CATCH_BBOX[1] - grace_buf, CATCH_BBOX[3] + grace_buf)
    _clean_map_ax(ax)
    if is_jpl:
        res = '3\u00b0\u21920.5\u00b0'
    elif is_gsfc:
        res = '0.5\u00b0'
    else:
        res = '0.25\u00b0'
    ax.set_title(f'{label}\nJul 2010  |  {res}', fontsize=FS['annotation']-0.5, pad=3)
    ax.text(0.04, 0.94, f'({panel_lbl})', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_cnes_grgs_map(ax, catchment, panel_lbl):
    """Plot CNES/GRGS RL05 1° EWH grid for Jul 2010. Returns mappable."""
    import requests, re
    from datetime import datetime, timedelta
    c4 = catchment.to_crs(epsg=4326)

    # Download the Jul 2010 grid from ForM@Ter
    api_base = "https://api.sedoo.fr/formater-catalogue-prod/datasetcontent/v1_0"
    uuid = "3f81e4f1-4591-4d0e-bcec-43381b4d2949"
    # Jul 2010 filename: GWH-2_2010182-2010212_GRAC_CNESG_TSVD_0500.txt (approx)
    list_url = f"{api_base}/content?collection={uuid}"
    try:
        resp = requests.get(list_url, timeout=30)
        files = resp.json().get('files', [])
        # Find Jul 2010 file (DOY ~182-212)
        target = None
        for f in files:
            name = f if isinstance(f, str) else f.get('name', '')
            if not name.endswith('.txt'):
                continue
            m = re.search(r'GWH-2_(\d{4})(\d{3})-(\d{4})(\d{3})_', name)
            if m and int(m.group(1)) == 2010:
                start = datetime(2010, 1, 1) + timedelta(days=int(m.group(2))-1)
                if start.month == 7:
                    target = name
                    break
        if target is None:
            ax.set_visible(False)
            return None

        dl_url = (f"{api_base}/getresource?collection={uuid}"
                  f"&resource=/{target}&catalogueName=SEDOOFORMATER&projectName=FORMATER.GRACE")
        text = requests.get(dl_url, timeout=60).text

        # Parse grid: lon lat value (lon 0-360)
        lons_g, lats_g, vals_g = [], [], []
        for line in text.strip().split('\n'):
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                lons_g.append(float(parts[0]))
                lats_g.append(float(parts[1]))
                vals_g.append(float(parts[2]))

        lons_g = np.array(lons_g)
        lats_g = np.array(lats_g)
        vals_g = np.array(vals_g)

        # Subset to catchment area with buffer
        buf = 1.5
        lon_min_360, lon_max_360 = CATCH_BBOX[0]+360, CATCH_BBOX[2]+360
        mask = ((lats_g >= CATCH_BBOX[1]-buf) & (lats_g <= CATCH_BBOX[3]+buf) &
                (lons_g >= lon_min_360-buf) & (lons_g <= lon_max_360+buf))
        if mask.sum() == 0:
            ax.set_visible(False)
            return None

        # Build 2D grid for pcolormesh
        ulons = np.sort(np.unique(lons_g[mask])) - 360
        ulats = np.sort(np.unique(lats_g[mask]))
        grid = np.full((len(ulats), len(ulons)), np.nan)
        for lo, la, v in zip(lons_g[mask]-360, lats_g[mask], vals_g[mask]):
            ix = np.argmin(np.abs(ulons - lo))
            iy = np.argmin(np.abs(ulats - la))
            grid[iy, ix] = v

        dl = 0.5  # half cell size (1° grid)
        lon_e = np.r_[ulons - dl, ulons[-1] + dl]
        lat_e = np.r_[ulats - dl, ulats[-1] + dl]

        im = ax.pcolormesh(lon_e, lat_e, grid, cmap='BrBG', vmin=-15, vmax=15, shading='flat')
        c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
        grace_buf = 0.70
        ax.set_xlim(CATCH_BBOX[0] - grace_buf, CATCH_BBOX[2] + grace_buf)
        ax.set_ylim(CATCH_BBOX[1] - grace_buf, CATCH_BBOX[3] + grace_buf)
        _clean_map_ax(ax)
        ax.set_title(f'CNES/GRGS\nJul 2010  |  1\u00b0', fontsize=FS['annotation']-0.5, pad=3)
        ax.text(0.04, 0.94, f'({panel_lbl})', transform=ax.transAxes, fontsize=FS['panel_label'],
               fontweight='bold', va='top',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
        return im
    except Exception as e:
        print(f"  CNES/GRGS map error: {e}")
        ax.set_visible(False)
        return None


def plot_era5l_tws_map(ax, catchment, panel_lbl):
    """Plot ERA5-Land TWS anomaly for Jul 2010 at 0.1° resolution."""
    c4 = catchment.to_crs(epsg=4326)

    sm_file = OBS_DIR / "soil_moisture/era5_land/era5_land_sm_2010.nc"
    swe_file = OBS_DIR / "snow/era5land/era5land_swe_raw.nc"

    if not sm_file.exists():
        ax.set_visible(False)
        return None

    # Layer thicknesses (m)
    thicknesses = {'swvl1': 0.07, 'swvl2': 0.21, 'swvl3': 0.72, 'swvl4': 1.89}

    # Load July 2010 soil moisture
    ds_sm = xr.open_dataset(sm_file)
    tcoord = 'valid_time' if 'valid_time' in ds_sm.dims else 'time'
    jul_sm = ds_sm.sel({tcoord: '2010-07'})
    if tcoord in jul_sm.dims and jul_sm.sizes[tcoord] > 0:
        jul_sm = jul_sm.isel({tcoord: 0})

    # Compute total soil water (m)
    tws_grid = sum(jul_sm[v].values * t for v, t in thicknesses.items() if v in jul_sm)

    # Add SWE if available
    if swe_file.exists():
        try:
            ds_swe = xr.open_dataset(swe_file)
            # Find July 2010 SWE
            tc_swe = 'valid_time' if 'valid_time' in ds_swe.dims else 'time'
            swe_jul = ds_swe['sd'].sel({tc_swe: '2010-07'})
            if tc_swe in swe_jul.dims and swe_jul.sizes[tc_swe] > 0:
                swe_jul = swe_jul.isel({tc_swe: 0})
            # Regrid SWE to SM grid if needed (nearest)
            swe_interp = swe_jul.interp(latitude=ds_sm.latitude, longitude=ds_sm.longitude,
                                         method='nearest').values
            tws_grid = tws_grid + np.nan_to_num(swe_interp)
            ds_swe.close()
        except Exception:
            pass

    # Compute baseline mean (2004-2009)
    baseline_years = range(2004, 2010)
    baseline_grids = []
    for yr in baseline_years:
        yf = DATA_DIR / f"observations/soil_moisture/era5_land/era5_land_sm_{yr}.nc"
        if yf.exists():
            ds_b = xr.open_dataset(yf)
            tc_b = 'valid_time' if 'valid_time' in ds_b.dims else 'time'
            for t_idx in range(min(12, ds_b.sizes.get(tc_b, 0))):
                ts = ds_b.isel({tc_b: t_idx})
                bg = sum(ts[v].values * t for v, t in thicknesses.items() if v in ts)
                baseline_grids.append(bg)
            ds_b.close()
    baseline_mean = np.nanmean(baseline_grids, axis=0) if baseline_grids else tws_grid

    # Anomaly in cm
    tws_anom_cm = (tws_grid - baseline_mean) * 100.0

    lons = ds_sm.longitude.values
    lats = ds_sm.latitude.values
    ds_sm.close()

    # Create cell edges for pcolormesh
    dl = 0.05  # half cell (0.1° grid)
    lon_e = np.r_[lons - dl, lons[-1] + dl]
    lat_e = np.r_[lats - dl, lats[-1] + dl]

    im = ax.pcolormesh(lon_e, lat_e, tws_anom_cm, cmap='BrBG', vmin=-15, vmax=15, shading='flat')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    buf = 0.70  # match GRACE extent for side-by-side comparison
    ax.set_xlim(CATCH_BBOX[0] - buf, CATCH_BBOX[2] + buf)
    ax.set_ylim(CATCH_BBOX[1] - buf, CATCH_BBOX[3] + buf)
    _clean_map_ax(ax)
    ax.set_title(f'ERA5-Land\nJul 2010  |  0.1\u00b0', fontsize=FS['annotation']-0.5, pad=3)
    ax.text(0.04, 0.94, f'({panel_lbl})', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_tws_ts(ax):
    if not GRACE_TS_CSV.exists():
        return
    df = pd.read_csv(GRACE_TS_CSV, index_col=0, parse_dates=True)
    # GRACE mascon solutions (solid lines)
    for col_key, color, label in [
        ('grace_csr_anomaly', TS_COLORS['grace_csr'], 'CSR'),
        ('grace_gsfc_anomaly', TS_COLORS['grace_gsfc'], 'GSFC'),
        ('grace_jpl_anomaly', TS_COLORS['grace_jpl'], 'JPL'),
    ]:
        if col_key in df.columns:
            s = df[col_key].dropna()
            ax.plot(s.index, s.values, color=color, linewidth=1.0,
                   label=label, alpha=0.9)
    # CNES/GRGS regularized SH solution (solid, distinct from mascons)
    if CNES_GRGS_TWS_CSV.exists():
        cg = pd.read_csv(CNES_GRGS_TWS_CSV, index_col=0, parse_dates=True)
        if 'tws_anomaly_cm' in cg.columns:
            s = cg['tws_anomaly_cm'].dropna()
            ax.plot(s.index, s.values, color=TS_COLORS['cnes_grgs'],
                   linewidth=1.0, label='CNES/GRGS', alpha=0.9)
    # ERA5-Land model TWS (dashed)
    if ERA5L_TWS_CSV.exists():
        e5 = pd.read_csv(ERA5L_TWS_CSV, index_col=0, parse_dates=True)
        if 'tws_anomaly_cm' in e5.columns:
            s = e5['tws_anomaly_cm'].dropna()
            ax.plot(s.index, s.values, color=TS_COLORS['era5land_tws'],
                   linewidth=1.0, linestyle='--', label='ERA5-Land', alpha=0.85)
    # GLDAS Noah model TWS (dashed)
    if GLDAS_TWS_CSV.exists():
        gl = pd.read_csv(GLDAS_TWS_CSV, index_col=0, parse_dates=True)
        if 'tws_anomaly_cm' in gl.columns:
            s = gl['tws_anomaly_cm'].dropna()
            ax.plot(s.index, s.values, color=TS_COLORS['gldas_tws'],
                   linewidth=1.0, linestyle='--', label='GLDAS', alpha=0.85)
    ax.axhline(0, color='#999', linewidth=0.5, zorder=1)
    ax.set_ylim(-40, 30)
    _format_ts_ax(ax, 'TWS (cm EWH)', panel_lbl='i', legend_loc='lower right', legend_ncol=3)


# =============================================================================
# ET BLOCK
# =============================================================================
def plot_ssebop_july(ax, catchment):
    """SSEBop ET for July 2010 in mm/day. Returns mappable for shared colorbar."""
    c4 = catchment.to_crs(epsg=4326)
    f = SSEBOP_DIR / 'm201007.tif'
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        from rasterio.windows import from_bounds
        w = from_bounds(CATCH_BBOX[0]-0.05, CATCH_BBOX[1]-0.05,
                       CATCH_BBOX[2]+0.05, CATCH_BBOX[3]+0.05, src.transform)
        d = src.read(1, window=w).astype(float)
        tf = src.window_transform(w)
        nd = src.nodata if src.nodata else -32768
        d[(d == nd) | (d < 0)] = np.nan
    # mm/month → mm/day
    d = d / 31.0
    nr, nc = d.shape
    ext = [tf.c, tf.c + nc*tf.a, tf.f + nr*tf.e, tf.f]
    dm = np.ma.masked_invalid(d)
    ax.set_facecolor('#f5f0e8')
    im = ax.imshow(dm, extent=ext, origin='upper', cmap='YlGn',
                   vmin=0, vmax=5, aspect='auto', interpolation='nearest')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('SSEBop ET\nJul 2010  |  1 km', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(j)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_modiset_map(ax, catchment):
    """Plot real MOD16A2GF ET for early July 2016. Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(MOD16_ET_NPZ)
    et = data['et_mm_day']  # mm/day
    lats, lons = data['lats'], data['lons']
    et_masked = np.ma.masked_invalid(et)
    ax.set_facecolor('#f5f0e8')
    im = ax.pcolormesh(lons, lats, et_masked, cmap='YlGn',
                       vmin=0, vmax=5, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('MOD16 ET\n3\u201310 Jul 2016  |  500 m', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(k)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_fluxcom_et_map(ax, catchment):
    """Plot FLUXCOM ET for July 2016 (0.5deg). Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(FLUXCOM_ET_NPZ)
    et = data['et']  # mm/day, 2D
    lats, lons = data['lats'], data['lons']
    et_masked = np.ma.masked_invalid(et)
    ax.set_facecolor('#f5f0e8')
    im = ax.pcolormesh(lons, lats, et_masked, cmap='YlGn',
                       vmin=0, vmax=5, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('FLUXCOM ET\nJul 2016  |  0.5\u00b0', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(l)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_et_ts(ax):
    if SSEBOP_ET_CSV.exists():
        df = pd.read_csv(SSEBOP_ET_CSV, parse_dates=['date']).set_index('date')
        ax.plot(df.index, df['et_mm_day'], color=TS_COLORS['ssebop'],
               linewidth=1.0, label='SSEBop (~1 km)', alpha=0.9, zorder=3)
    if MODIS_ET_NC.exists():
        ds = xr.open_dataset(MODIS_ET_NC)
        et = ds['ET'].to_series().resample('MS').mean()
        ax.plot(et.index, et.values, color=TS_COLORS['modis_et'],
               linewidth=1.0, label='MOD16 (~500 m)', alpha=0.9, zorder=2)
        ds.close()
    if FLUXCOM_ET_CSV.exists():
        df = pd.read_csv(FLUXCOM_ET_CSV, parse_dates=['date']).set_index('date')
        ax.plot(df.index, df['et_mm_day'], color=TS_COLORS['fluxcom'],
               linewidth=1.0, label='FLUXCOM (~0.5\u00b0)', alpha=0.9, zorder=1)
    ax.set_ylim(0, 6)
    _format_ts_ax(ax, 'ET (mm d$^{-1}$)', panel_lbl='m', legend_ncol=3)


# =============================================================================
# SNOW BLOCK
# =============================================================================
def plot_modis_snow_scf_map(ax, catchment):
    """MOD10A1F CGF NDSI snow cover for April 24, 2016. Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(MODIS_SNOW_NPZ)
    scf = data['scf']  # 0-1, NaN for invalid
    lats, lons = data['lats'], data['lons']
    snow_cmap = mcolors.LinearSegmentedColormap.from_list(
        'snow_scf', ['#f5f0e8', '#d4e6f1', '#87CEEB', '#4a90d9', '#1a5276'], N=256)
    scf_masked = np.ma.masked_invalid(scf)
    ax.set_facecolor('#e8e0d4')
    im = ax.pcolormesh(lons, lats, scf_masked, cmap=snow_cmap,
                       vmin=0, vmax=1, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('MODIS Terra SCF\n24 Apr 2016  |  500 m', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(n)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_viirs_map(ax, catchment):
    """Plot real VIIRS CGF NDSI snow cover for Feb 15, 2016. Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(VIIRS_SNOW_NPZ)
    scf = data['scf']  # NDSI/100, NaN for invalid
    lats, lons = data['lats'], data['lons']
    snow_cmap = mcolors.LinearSegmentedColormap.from_list(
        'snow_scf', ['#f5f0e8', '#d4e6f1', '#87CEEB', '#4a90d9', '#1a5276'], N=256)
    scf_masked = np.ma.masked_invalid(scf)
    ax.set_facecolor('#e8e0d4')
    im = ax.pcolormesh(lons, lats, scf_masked, cmap=snow_cmap,
                       vmin=0, vmax=1, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('VIIRS SCF\n24 Apr 2016  |  375 m', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(o)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_ims_map(ax, catchment):
    """Plot real IMS 4km snow cover for Feb 15, 2016. Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(IMS_SNOW_NPZ)
    vals = data['values']  # 2=land, 4=snow
    lats, lons = data['lats'], data['lons']
    # Convert: 4→1.0 (snow), 2→0.0 (land), else NaN
    scf = np.full_like(vals, np.nan, dtype=float)
    scf[vals == 4] = 1.0
    scf[vals == 2] = 0.0
    snow_cmap = mcolors.LinearSegmentedColormap.from_list(
        'snow_scf', ['#f5f0e8', '#d4e6f1', '#87CEEB', '#4a90d9', '#1a5276'], N=256)
    scf_masked = np.ma.masked_invalid(scf)
    ax.set_facecolor('#e8e0d4')
    im = ax.pcolormesh(lons, lats, scf_masked, cmap=snow_cmap,
                       vmin=0, vmax=1, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('IMS Snow\n24 Apr 2016  |  4 km', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(p)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)


def plot_sentinel2_map(ax, catchment):
    """Plot Sentinel-2 SCL snow cover for May 2, 2016. Returns mappable."""
    c4 = catchment.to_crs(epsg=4326)
    data = np.load(S2_SNOW_NPZ)
    scf = data['scf']  # 0-1, NaN for invalid
    lats, lons = data['lats'], data['lons']
    snow_cmap = mcolors.LinearSegmentedColormap.from_list(
        'snow_scf', ['#f5f0e8', '#d4e6f1', '#87CEEB', '#4a90d9', '#1a5276'], N=256)
    scf_masked = np.ma.masked_invalid(scf)
    ax.set_facecolor('#e8e0d4')
    im = ax.pcolormesh(lons, lats, scf_masked, cmap=snow_cmap,
                       vmin=0, vmax=1, shading='auto')
    c4.boundary.plot(ax=ax, color='#333333', linewidth=1.3, zorder=5)
    ax.set_xlim(CATCH_BBOX[0]-0.05, CATCH_BBOX[2]+0.05)
    ax.set_ylim(CATCH_BBOX[1]-0.05, CATCH_BBOX[3]+0.05)
    _clean_map_ax(ax)
    ax.set_title('Sentinel-2 SCF\n2 May 2016  |  20 m', fontsize=FS['annotation']+0.5, pad=3)
    ax.text(0.04, 0.94, '(q)', transform=ax.transAxes, fontsize=FS['panel_label'],
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=20)
    return im


def plot_snow_ts(ax):
    datasets = [
        (MODIS_TERRA_CSV, 'snow_fraction', TS_COLORS['modis_terra'], 'MODIS Terra'),
        (MODIS_AQUA_CSV,  'snow_fraction', TS_COLORS['modis_aqua'],  'MODIS Aqua'),
        (VIIRS_SCA_CSV,   'snow_fraction', TS_COLORS['viirs'],       'VIIRS'),
        (IMS_SNOW_CSV,    'snow_fraction', TS_COLORS['ims'],         'IMS'),
    ]
    for csv, col, color, label in datasets:
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=['date']).set_index('date')
        if col not in df.columns:
            col = [c for c in df.columns if 'fraction' in c.lower()]
            col = col[0] if col else df.columns[0]
        s = df[col].dropna()
        if s.max() > 1.5:
            s = s / 100.0
        sm = s.resample('MS').mean() * 100
        ax.plot(sm.index, sm.values, color=color, linewidth=1.0, label=label, alpha=0.85)
    if SENTINEL2_SNOW_CSV.exists():
        s2 = pd.read_csv(SENTINEL2_SNOW_CSV, parse_dates=['time']).set_index('time')
        if 'snow_fraction' in s2.columns:
            valid = s2['snow_fraction'].dropna()
            if len(valid) > 0:
                ax.scatter(valid.index, valid.values * 100, s=12, color=TS_COLORS['sentinel2'],
                          marker='d', label='S-2', zorder=5, alpha=0.7)
    ax.set_ylim(0, 105)
    _format_ts_ax(ax, 'SCF (%)', panel_lbl='r', legend_ncol=5, legend_loc='upper center')
    # Override: single-row legend at top center
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=FS['legend'], loc='upper center', frameon=True,
              facecolor='white', edgecolor='#cccccc', framealpha=0.97, ncol=5,
              bbox_to_anchor=(0.55, 0.99),
              handlelength=1.0, handletextpad=0.2, columnspacing=0.4,
              borderpad=0.3, labelspacing=0.2)


# =============================================================================
# SM BLOCK
# =============================================================================
# Shared SM schematic parameters: volumetric SM (m3/m3), Jun 2015 - Sep 2017 overlap
_SM_VMIN, _SM_VMAX = 0.10, 0.40


def plot_smap_schematic(ax, catchment):
    _schematic_grid_map(ax, catchment, res_deg=0.36, cmap=plt.cm.YlGnBu,
                       vmin=_SM_VMIN, vmax=_SM_VMAX, mean_val=0.30,
                       label='SMAP L-band\n~36 km', panel_lbl='s', gradient_dir='se')


def plot_esacci_schematic(ax, catchment):
    _schematic_grid_map(ax, catchment, res_deg=0.25, cmap=plt.cm.YlGnBu,
                       vmin=_SM_VMIN, vmax=_SM_VMAX, mean_val=0.23,
                       label='ESA CCI\n~25 km', panel_lbl='t', gradient_dir='se')


def plot_smos_schematic(ax, catchment):
    _schematic_grid_map(ax, catchment, res_deg=0.25, cmap=plt.cm.YlGnBu,
                       vmin=_SM_VMIN, vmax=_SM_VMAX, mean_val=0.23,
                       label='SMOS L-band\n~25 km', panel_lbl='u', gradient_dir='elev')


def plot_ascat_schematic(ax, catchment):
    _schematic_grid_map(ax, catchment, res_deg=0.25, cmap=plt.cm.YlGnBu,
                       vmin=_SM_VMIN, vmax=_SM_VMAX, mean_val=0.19,
                       label='ASCAT C-band\n~25 km', panel_lbl='v', gradient_dir='se')


def plot_era5land_schematic(ax, catchment):
    _schematic_grid_map(ax, catchment, res_deg=0.10, cmap=plt.cm.YlGnBu,
                       vmin=_SM_VMIN, vmax=_SM_VMAX, mean_val=0.15,
                       label='ERA5-Land\n~9 km', panel_lbl='w', gradient_dir='elev')


def plot_sm_ts(ax):
    if SMAP_SM_CSV.exists():
        df = pd.read_csv(SMAP_SM_CSV, parse_dates=['date']).set_index('date')
        sm = df['soil_moisture'].resample('MS').mean()
        ax.plot(sm.index, sm.values, color=TS_COLORS['smap'], linewidth=1.0,
               label='SMAP (~36 km)', alpha=0.9, zorder=4)
    if ESA_CCI_SM_CSV.exists():
        df = pd.read_csv(ESA_CCI_SM_CSV, parse_dates=['date']).set_index('date')
        sm = df['soil_moisture'].resample('MS').mean()
        ax.plot(sm.index, sm.values, color=TS_COLORS['esa_cci'], linewidth=1.0,
               label='ESA CCI (~25 km)', alpha=0.9, zorder=3)
    if SMOS_SM_CSV.exists():
        df = pd.read_csv(SMOS_SM_CSV, parse_dates=['date']).set_index('date')
        sm = df['soil_moisture'].resample('MS').mean()
        ax.plot(sm.index, sm.values, color=TS_COLORS['smos'], linewidth=1.0,
               label='SMOS (~25 km)', alpha=0.9, zorder=2)
    if ASCAT_SM_CSV.exists():
        df = pd.read_csv(ASCAT_SM_CSV, parse_dates=['date']).set_index('date')
        sm = df['soil_moisture'].resample('MS').mean()
        ax.plot(sm.index, sm.values, color=TS_COLORS['ascat'], linewidth=1.0,
               label='ASCAT (~25 km)', alpha=0.9, zorder=1)
    if ERA5LAND_SM_CSV.exists():
        df = pd.read_csv(ERA5LAND_SM_CSV, parse_dates=['date']).set_index('date')
        sm = df['soil_moisture'].resample('MS').mean()
        ax.plot(sm.index, sm.values, color=TS_COLORS['era5land_sm'], linewidth=1.0,
               label='ERA5-Land (~9 km)', alpha=0.9, zorder=0, linestyle='--')
    ax.set_ylim(0, 0.50)
    _format_ts_ax(ax, 'Vol. SM (m\u00b3 m$^{-3}$)', panel_lbl='x',
                  xlim=XLIM, year_step=4, legend_ncol=3)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Creating Figure 1: Study Domain + RS Observations (v5)")
    print("=" * 70)

    catchment = load_shp(CATCHMENT_SHP)
    rdrs_grid = load_shp(RDRS_GRID_SHP)
    rivers = load_shp(RIVER_SHP)
    snow_stations = load_snow_stations(catchment)
    if catchment is None:
        print("ERROR: No catchment"); return
    print(f"  Catchment loaded, {len(snow_stations) if snow_stations is not None else 0} SWE stations")

    # =========================================================================
    # FIGURE LAYOUT
    # Col 0: Domain   | Col 1: TWS (3 maps) / Snow (3 maps)  | Col 2: ET (2) / SM (2)
    # =========================================================================
    fig = plt.figure(figsize=(23, 15.5))

    main_gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.40, 1.05],
                       height_ratios=[1, 1], wspace=0.14, hspace=0.22)

    # --- DOMAIN (col 0, both rows) ---
    dom_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[:, 0],
                                     height_ratios=[2.0, 0.70, 0.70], hspace=0.30)
    ax_map = fig.add_subplot(dom_gs[0])
    ax_q = fig.add_subplot(dom_gs[1])
    ax_swe = fig.add_subplot(dom_gs[2])

    # --- TWS block (row 0, col 1): 5 TWS maps + TS ---
    tws_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=main_gs[0, 1],
                                     height_ratios=[1.5, 0.7], hspace=0.36, wspace=0.08)
    ax_csr = fig.add_subplot(tws_gs[0, 0])
    ax_gsfc = fig.add_subplot(tws_gs[0, 1])
    ax_jpl = fig.add_subplot(tws_gs[0, 2])
    ax_cnes = fig.add_subplot(tws_gs[0, 3])
    ax_era5t = fig.add_subplot(tws_gs[0, 4])
    ax_tws_ts = fig.add_subplot(tws_gs[1, :])

    # --- ET block (row 0, col 2): 3 maps + TS ---
    et_gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=main_gs[0, 2],
                                    height_ratios=[1.5, 0.7], hspace=0.36, wspace=0.18)
    ax_ssebop = fig.add_subplot(et_gs[0, 0])
    ax_modiset = fig.add_subplot(et_gs[0, 1])
    ax_fluxcom = fig.add_subplot(et_gs[0, 2])
    ax_et_ts = fig.add_subplot(et_gs[1, :])

    # --- Snow block (row 1, col 1): 4 maps + TS ---
    snow_gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=main_gs[1, 1],
                                      height_ratios=[1.5, 0.7], hspace=0.36, wspace=0.10)
    ax_modis_sn = fig.add_subplot(snow_gs[0, 0])
    ax_viirs_sn = fig.add_subplot(snow_gs[0, 1])
    ax_ims_sn = fig.add_subplot(snow_gs[0, 2])
    ax_s2_sn = fig.add_subplot(snow_gs[0, 3])
    ax_snow_ts = fig.add_subplot(snow_gs[1, :])

    # --- SM block (row 1, col 2): 5 maps + TS ---
    sm_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=main_gs[1, 2],
                                    height_ratios=[1.5, 0.7], hspace=0.36, wspace=0.10)
    ax_smap = fig.add_subplot(sm_gs[0, 0])
    ax_cci = fig.add_subplot(sm_gs[0, 1])
    ax_smos = fig.add_subplot(sm_gs[0, 2])
    ax_ascat = fig.add_subplot(sm_gs[0, 3])
    ax_era5l = fig.add_subplot(sm_gs[0, 4])
    ax_sm_ts = fig.add_subplot(sm_gs[1, :])

    # =========================================================================
    # PLOT ALL PANELS
    # =========================================================================
    print("  Domain map...")
    plot_domain_map(ax_map, catchment, rdrs_grid, rivers, snow_stations)
    plot_inset_map(ax_map)
    print("  Streamflow TS...")
    plot_streamflow_ts(ax_q)
    print("  SWE TS...")
    plot_swe_ts(ax_swe)

    # --- TWS ---
    print("  GRACE CSR...")
    im_csr = plot_grace_map(ax_csr, catchment, GRACE_CSR_NC, 'GRACE CSR', 'd')
    print("  GRACE GSFC...")
    plot_grace_map(ax_gsfc, catchment, GRACE_GSFC_NC, 'GRACE GSFC', 'e', is_gsfc=True)
    print("  GRACE JPL...")
    plot_grace_map(ax_jpl, catchment, GRACE_JPL_NC, 'GRACE JPL', 'f', is_jpl=True)
    print("  CNES/GRGS...")
    plot_cnes_grgs_map(ax_cnes, catchment, 'g')
    print("  ERA5-Land TWS...")
    plot_era5l_tws_map(ax_era5t, catchment, 'h')
    _add_shared_cbar(fig, [ax_csr, ax_gsfc, ax_jpl, ax_cnes, ax_era5t], im_csr, 'TWS (cm EWH)')
    print("  TWS TS...")
    plot_tws_ts(ax_tws_ts)

    # --- ET ---
    print("  SSEBop ET...")
    im_et = plot_ssebop_july(ax_ssebop, catchment)
    print("  MODIS ET map...")
    im_modiset = plot_modiset_map(ax_modiset, catchment)
    print("  FLUXCOM ET map...")
    plot_fluxcom_et_map(ax_fluxcom, catchment)
    _add_shared_cbar(fig, [ax_ssebop, ax_modiset, ax_fluxcom], im_modiset, 'ET (mm d$^{-1}$)')
    print("  ET TS...")
    plot_et_ts(ax_et_ts)

    # --- Snow ---
    print("  MODIS Snow SCF...")
    im_sn = plot_modis_snow_scf_map(ax_modis_sn, catchment)
    print("  VIIRS snow map...")
    plot_viirs_map(ax_viirs_sn, catchment)
    print("  IMS snow map...")
    plot_ims_map(ax_ims_sn, catchment)
    print("  Sentinel-2 snow map...")
    plot_sentinel2_map(ax_s2_sn, catchment)
    _add_shared_cbar(fig, [ax_modis_sn, ax_viirs_sn, ax_ims_sn, ax_s2_sn], im_sn, 'SCF (-)')
    print("  Snow TS...")
    plot_snow_ts(ax_snow_ts)

    # --- SM ---
    print("  SMAP schematic...")
    plot_smap_schematic(ax_smap, catchment)
    print("  ESA CCI schematic...")
    plot_esacci_schematic(ax_cci, catchment)
    print("  SMOS schematic...")
    plot_smos_schematic(ax_smos, catchment)
    print("  ASCAT schematic...")
    plot_ascat_schematic(ax_ascat, catchment)
    print("  ERA5-Land schematic...")
    plot_era5land_schematic(ax_era5l, catchment)
    # Create a dummy mappable for SM colorbar (matching schematic vmin/vmax)
    sm_norm = Normalize(vmin=_SM_VMIN, vmax=_SM_VMAX)
    sm_sm = plt.cm.ScalarMappable(norm=sm_norm, cmap=plt.cm.YlGnBu)
    _add_shared_cbar(fig, [ax_smap, ax_cci, ax_smos, ax_ascat, ax_era5l], sm_sm,
                     'Volumetric SM (m\u00b3 m$^{-3}$)')
    print("  SM TS...")
    plot_sm_ts(ax_sm_ts)

    out = OUTPUT_DIR / "fig1_domain_v5.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")
    plt.close()
    print("Figure 1 (v5) complete!")


if __name__ == "__main__":
    main()
