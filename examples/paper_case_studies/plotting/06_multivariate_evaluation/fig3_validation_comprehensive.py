#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Figure 3: Comprehensive Independent Validation
Multivariate Evaluation Section

Multi-panel figure showing validation against all available observation data:

Panel (a): Snow Cover Area - MODIS, VIIRS, IMS, Sentinel-2 vs simulated SCF
Panel (b): Soil Moisture - SMAP, ESA CCI, SMOS, ASCAT, ERA5-Land vs simulated SM
Panel (c): Snow Water Equivalent - CanSWE, SNODAS, CMC vs simulated SWE
Panel (d): Evapotranspiration - MODIS ET, SSEBop, FLUXCOM vs simulated ET
Panel (e): Summary metrics table (expanded to all products)

Compares all 3 calibration experiments:
  - Experiment 1: Streamflow-only (DDS)
  - Experiment 2: TWS-only (DDS)
  - Experiment 3: Joint (NSGA-II)

Design: Demonstrates multi-metric consistency of joint solution
        using full observation product coverage from fig1 domain figure
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bow_banff_style import COLORS, FONT, LINES, setup_style, despine, add_grid

setup_style()

# =============================================================================
# PATHS AND CONSTANTS
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

# Experiment directories
# Shipped exp1 eid is bow_exp1_streamflow_only (original study used an
# earlier run named bow_tws_calibrated).
EXP1_DIR = DATA_DIR / "optimization/SUMMA/dds_bow_exp1_streamflow_only"
EXP2_DIR = DATA_DIR / "optimization/SUMMA/dds_bow_exp2_tws_only"
EXP3_DIR = DATA_DIR / "optimization/SUMMA/nsga-ii_bow_exp3_joint"

# Observation data
MODIS_SCA = OBS_DIR / "snow/modis/modis_sca_merged.csv"
VIIRS_SCA = OBS_DIR / "snow/viirs/viirs_sca_processed.csv"
IMS_SNOW = OBS_DIR / "snow/ims/ims_snow_4km.csv"
SMAP_SM = OBS_DIR / "soil_moisture/smap/smap_summer_processed.csv"
ESA_CCI_SM = OBS_DIR / "soil_moisture/esa_cci/esa_cci_sm_processed.csv"
CANSWE_SWE = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_swe_processed.csv"
CANSWE_STATIONS = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_canswe_swe_all_stations.csv"
MODIS_ET = OBS_DIR / "et/modis/Bow_at_Banff_MOD16_ET_timeseries.csv"
SSEBOP_ET = OBS_DIR / "et/ssebop/Bow_at_Banff_multivar_ssebop_et_processed.csv"
SNOW_COMPARISON = OBS_DIR / "snow/preprocessed/snow_cover_comparison.csv"

# Additional observation products (matching fig1 domain coverage)
SMOS_SM = OBS_DIR / "soil_moisture/smos/smos_sm_processed.csv"
ASCAT_SM = OBS_DIR / "soil_moisture/ascat/ascat_sm_processed.csv"
ERA5LAND_SM = OBS_DIR / "soil_moisture/era5_land/era5_land_sm_processed.csv"
FLUXCOM_ET = OBS_DIR / "et/preprocessed/Bow_at_Banff_multivar_fluxcom_et_processed.csv"
SNODAS_SWE = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_snodas_swe_processed.csv"
CMC_SWE = OBS_DIR / "snow/preprocessed/Bow_at_Banff_multivar_cmc_swe_processed.csv"
SENTINEL2_SNOW = OBS_DIR / "snow/sentinel2/Bow_at_Banff_Sentinel2_snow_timeseries.csv"

EVALUATION_START = '2011-01-01'
EVALUATION_END = '2017-12-31'

CATCHMENT_AREA_KM2 = 2210

# Experiment colors
EXP_COLORS = {
    'exp1': '#0072B2',   # Blue - Streamflow-only
    'exp2': '#D55E00',   # Vermillion - TWS-only
    'exp3': '#009E73',   # Green - Joint
    'obs': '#E69F00',    # Orange - Observations
}

EXP_LABELS = {
    'exp1': 'Q-only',
    'exp2': 'TWS-only',
    'exp3': 'Joint',
}

# Observation product colors (matching fig1 domain figure)
OBS_COLORS = {
    'modis_sca': '#E69F00',
    'viirs': '#8B4513',
    'ims': '#4B0082',
    'sentinel2': '#333333',
    'smap': '#E69F00',
    'esa_cci': '#006400',
    'smos': '#2ca02c',
    'ascat': '#E7298A',
    'era5land': '#984EA3',
    'modis_et': '#E69F00',
    'ssebop': '#8B0000',
    'fluxcom': '#CC79A7',
    'canswe': '#E69F00',
    'snodas': '#17becf',
    'cmc': '#7f7f7f',
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_summa_output(exp_dir):
    """Load SUMMA simulation output from final evaluation."""
    daily_path = exp_dir / "final_evaluation" / "bow_tws_uncalibrated_day.nc"
    timestep_path = exp_dir / "final_evaluation" / "bow_tws_uncalibrated_timestep.nc"

    if not daily_path.exists():
        daily_files = list((exp_dir / "final_evaluation").glob("*_day.nc"))
        timestep_files = list((exp_dir / "final_evaluation").glob("*_timestep.nc"))
        if daily_files:
            daily_path = daily_files[0]
        if timestep_files:
            timestep_path = timestep_files[0]

    if not daily_path.exists():
        print(f"  WARNING: No daily output found in {exp_dir}")
        return None

    ds_day = xr.open_dataset(daily_path)
    times = pd.to_datetime(ds_day.time.values)

    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,
    }
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    # Calculate snow cover fraction (binary: SWE > threshold)
    data['SCF'] = (data['SWE'] > 1.0).astype(float)

    # Extract near-surface soil moisture for comparison with satellite products
    # mLayerVolFracLiq uses dynamic layer indexing; -9999 = missing/fill
    # When no snow: layers 0-2 are soil; when snow: layers 5-7 are soil
    if 'mLayerVolFracLiq' in ds_day:
        vol_frac = ds_day['mLayerVolFracLiq'].values  # (time, midToto, hru)
        # Mask fill values
        vol_frac_masked = np.where(vol_frac < -100, np.nan, vol_frac)
        # Take mean of all valid values per timestep (represents avg soil moisture)
        data['surf_SM'] = np.nanmean(vol_frac_masked[:, :, 0], axis=1)  # m³/m³

    # Load ET from daily file (scalarTotalET in kg/m2/s)
    # Note: SUMMA uses sign convention where negative = upward flux (evaporation)
    if 'scalarTotalET' in ds_day:
        et_kg_m2_s = ds_day['scalarTotalET'].values.flatten()
        data['sim_ET'] = np.abs(et_kg_m2_s) * 86400  # Convert to mm/day (abs for sign convention)
        print(f"    Loaded ET from daily file: mean={np.nanmean(data['sim_ET']):.2f} mm/day")

    df = pd.DataFrame(data).set_index('time')
    ds_day.close()

    # Load streamflow from timestep data
    if timestep_path.exists():
        ds_ts = xr.open_dataset(timestep_path)
        ts_times = pd.to_datetime(ds_ts.time.values)

        if 'averageRoutedRunoff' in ds_ts:
            runoff_ms = ds_ts['averageRoutedRunoff'].values.flatten()
            runoff_m3s = runoff_ms * CATCHMENT_AREA_KM2 * 1e6
            df_runoff = pd.DataFrame({'sim_Q': runoff_m3s}, index=ts_times)
            df = df.join(df_runoff.resample('D').mean(), how='left')

        ds_ts.close()

    return df


def load_modis_sca():
    """Load MODIS SCA observations."""
    if not MODIS_SCA.exists():
        return None
    df = pd.read_csv(MODIS_SCA, parse_dates=['date'], index_col='date')
    return df


def load_smap_sm():
    """Load SMAP soil moisture observations."""
    if not SMAP_SM.exists():
        return None
    df = pd.read_csv(SMAP_SM, parse_dates=['date'], index_col='date')
    return df


def load_esa_cci_sm():
    """Load ESA CCI soil moisture observations."""
    if not ESA_CCI_SM.exists():
        return None
    df = pd.read_csv(ESA_CCI_SM, parse_dates=['date'], index_col='date')
    return df


def load_canswe_swe():
    """Load CanSWE SWE observations (mean)."""
    if not CANSWE_SWE.exists():
        return None
    df = pd.read_csv(CANSWE_SWE, parse_dates=['datetime'], index_col='datetime')
    return df


def load_canswe_stations():
    """Load CanSWE SWE station-level data for ensemble display."""
    if not CANSWE_STATIONS.exists():
        return None
    df = pd.read_csv(CANSWE_STATIONS, parse_dates=['datetime'])
    # Pivot to get stations as columns
    df_pivot = df.pivot_table(index='datetime', columns='station_id', values='swe_mm')
    return df_pivot


def load_modis_et():
    """Load MODIS ET observations."""
    if not MODIS_ET.exists():
        return None
    df = pd.read_csv(MODIS_ET, parse_dates=['date'], index_col='date')
    return df


def load_viirs_sca():
    """Load VIIRS SCA observations."""
    if not VIIRS_SCA.exists():
        return None
    df = pd.read_csv(VIIRS_SCA, parse_dates=['date'], index_col='date')
    return df


def load_ims_snow():
    """Load IMS snow observations."""
    if not IMS_SNOW.exists():
        return None
    df = pd.read_csv(IMS_SNOW, parse_dates=['date'], index_col='date')
    return df


def load_ssebop_et():
    """Load SSEBop ET observations."""
    if not SSEBOP_ET.exists():
        return None
    df = pd.read_csv(SSEBOP_ET, parse_dates=['date'], index_col='date')
    return df


def load_smos_sm():
    """Load SMOS soil moisture observations."""
    if not SMOS_SM.exists():
        return None
    df = pd.read_csv(SMOS_SM, parse_dates=['date'], index_col='date')
    return df


def load_ascat_sm():
    """Load ASCAT soil moisture observations."""
    if not ASCAT_SM.exists():
        return None
    df = pd.read_csv(ASCAT_SM, parse_dates=['date'], index_col='date')
    return df


def load_era5land_sm():
    """Load ERA5-Land soil moisture observations."""
    if not ERA5LAND_SM.exists():
        return None
    df = pd.read_csv(ERA5LAND_SM, parse_dates=['date'], index_col='date')
    return df


def load_fluxcom_et():
    """Load FLUXCOM ET observations."""
    if not FLUXCOM_ET.exists():
        return None
    df = pd.read_csv(FLUXCOM_ET, parse_dates=['date'], index_col='date')
    return df


def load_snodas_swe():
    """Load SNODAS SWE observations."""
    if not SNODAS_SWE.exists():
        return None
    df = pd.read_csv(SNODAS_SWE, parse_dates=['date'], index_col='date')
    return df


def load_cmc_swe():
    """Load CMC SWE observations."""
    if not CMC_SWE.exists():
        return None
    df = pd.read_csv(CMC_SWE, parse_dates=['date'], index_col='date')
    return df


def load_sentinel2_snow():
    """Load Sentinel-2 snow cover fraction observations."""
    if not SENTINEL2_SNOW.exists():
        return None
    df = pd.read_csv(SENTINEL2_SNOW, parse_dates=['time'], index_col='time')
    return df


def calculate_metrics(sim, obs):
    """Calculate validation metrics."""
    common = sim.dropna().index.intersection(obs.dropna().index)
    if len(common) < 10:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    s = sim.loc[common].values
    o = obs.loc[common].values

    valid = ~(np.isnan(s) | np.isnan(o))
    s, o = s[valid], o[valid]

    if len(s) < 10:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    r = np.corrcoef(s, o)[0, 1]
    rmse = np.sqrt(np.mean((s - o)**2))
    bias = np.mean(s - o)

    return {'r': r, 'RMSE': rmse, 'bias': bias, 'n': len(s)}


def calculate_sca_metrics(sim_scf, obs_scf):
    """Calculate snow cover accuracy metrics."""
    common = sim_scf.dropna().index.intersection(obs_scf.dropna().index)
    if len(common) < 10:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'n': 0}

    s = sim_scf.loc[common].values
    o = obs_scf.loc[common].values

    # Binary classification (threshold at 0.5)
    s_bin = s > 0.5
    o_bin = o > 0.5

    # Confusion matrix elements
    TP = np.sum(s_bin & o_bin)
    TN = np.sum(~s_bin & ~o_bin)
    FP = np.sum(s_bin & ~o_bin)
    FN = np.sum(~s_bin & o_bin)

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan
    POD = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # Probability of detection
    FAR = FP / (TP + FP) if (TP + FP) > 0 else np.nan  # False alarm ratio

    # Also calculate correlation on continuous values
    r = np.corrcoef(s, o)[0, 1] if len(s) > 2 else np.nan

    return {'accuracy': accuracy, 'POD': POD, 'FAR': FAR, 'r': r, 'n': len(common)}


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sca_comparison(ax, obs_sca, sim_dfs, obs_viirs=None, obs_ims=None,
                        obs_s2=None):
    """Plot snow cover area comparison with MODIS, VIIRS, IMS, and Sentinel-2."""
    # Filter to evaluation period
    start = pd.Timestamp(EVALUATION_START)
    end = pd.Timestamp(EVALUATION_END)

    # Monthly means for cleaner visualization
    obs_modis_eval = obs_sca[(obs_sca.index >= start) & (obs_sca.index <= end)]['snow_fraction']
    obs_monthly = obs_modis_eval.resample('M').mean()

    # Plot MODIS
    ax.plot(obs_monthly.index, obs_monthly.values, color=OBS_COLORS['modis_sca'],
           linewidth=1.3, alpha=1.0, label='MODIS', zorder=5)

    # Prepare additional product data for metrics
    viirs_eval = None
    ims_eval = None
    s2_eval = None

    # Plot VIIRS if available
    if obs_viirs is not None:
        viirs_eval = obs_viirs[(obs_viirs.index >= start) & (obs_viirs.index <= end)]['snow_fraction']
        viirs_monthly = viirs_eval.resample('M').mean()
        ax.plot(viirs_monthly.index, viirs_monthly.values, color=OBS_COLORS['viirs'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='VIIRS', zorder=4)

    # Plot IMS if available
    if obs_ims is not None:
        ims_eval = obs_ims[(obs_ims.index >= start) & (obs_ims.index <= end)]['snow_fraction']
        ims_monthly = ims_eval.resample('M').mean()
        ax.plot(ims_monthly.index, ims_monthly.values, color=OBS_COLORS['ims'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='IMS', zorder=4)

    # Plot Sentinel-2 if available (sparse observations as scatter)
    if obs_s2 is not None and 'snow_fraction' in obs_s2.columns:
        s2_valid = obs_s2[(obs_s2.index >= start) & (obs_s2.index <= end)]['snow_fraction'].dropna()
        if len(s2_valid) > 0:
            s2_eval = s2_valid
            ax.scatter(s2_valid.index, s2_valid.values, s=14, color=OBS_COLORS['sentinel2'],
                      marker='d', label='S-2', zorder=6, alpha=0.7, edgecolors='none')

    # Plot simulations and calculate metrics per product
    # All sims use dashed patterns (obs are solid) for clear visual separation
    sim_line_styles = {'exp1': (0, (5, 2)), 'exp2': (0, (1.5, 1)), 'exp3': (0, (3, 1))}
    sim_line_widths = {'exp1': 0.8, 'exp2': 0.8, 'exp3': 1.0}
    metrics = {'MODIS SCA': {}, 'VIIRS SCA': {}, 'IMS': {}}
    if obs_s2 is not None:
        metrics['Sentinel-2'] = {}

    for exp, df in sim_dfs.items():
        if df is not None and 'SCF' in df.columns:
            df_eval = df[(df.index >= start) & (df.index <= end)]
            sim_monthly = df_eval['SCF'].resample('M').mean()

            ax.plot(sim_monthly.index, sim_monthly.values, color=EXP_COLORS[exp],
                   linewidth=sim_line_widths[exp], linestyle=sim_line_styles[exp],
                   alpha=0.7, label=EXP_LABELS[exp], zorder=3)

            # Metrics vs each product
            metrics['MODIS SCA'][exp] = calculate_sca_metrics(df_eval['SCF'], obs_modis_eval)
            if viirs_eval is not None:
                metrics['VIIRS SCA'][exp] = calculate_sca_metrics(df_eval['SCF'], viirs_eval)
            if ims_eval is not None:
                metrics['IMS'][exp] = calculate_sca_metrics(df_eval['SCF'], ims_eval)
            if s2_eval is not None:
                metrics['Sentinel-2'][exp] = calculate_sca_metrics(df_eval['SCF'], s2_eval)

    ax.set_ylabel('Snow Cover Fraction', fontsize=FONT['size']['axis_label'])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(start, end)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Y-grid
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(loc='lower left', fontsize=FONT['size']['legend']-2, ncol=4,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
             handlelength=1.5, handletextpad=0.3, columnspacing=0.5,
             borderpad=0.3, labelspacing=0.3)

    despine(ax)
    ax.set_title('(a) Snow Cover', loc='left', fontsize=FONT['size']['panel_label'],
                fontweight='bold', pad=8)

    return metrics


def plot_sm_comparison(ax, obs_sm, sim_dfs, obs_cci=None, obs_smos=None,
                       obs_ascat=None, obs_era5land=None):
    """Plot soil moisture comparison with SMAP, ESA CCI, SMOS, ASCAT, ERA5-Land."""
    start = pd.Timestamp('2011-01-01')  # Full evaluation period for room
    end = pd.Timestamp('2017-12-31')

    obs_eval = obs_sm[(obs_sm.index >= start) & (obs_sm.index <= end)]['soil_moisture']

    # Monthly resampling for all SM products (consistent at this wider scale)
    smap_monthly = obs_eval.resample('M').mean()
    ax.plot(smap_monthly.index, smap_monthly.values, color=OBS_COLORS['smap'],
           linewidth=1.4, alpha=1.0, label='SMAP', zorder=6)

    # Plot ESA CCI if available
    cci_eval = None
    if obs_cci is not None:
        cci_eval = obs_cci[(obs_cci.index >= start) & (obs_cci.index <= end)]['soil_moisture']
        cci_monthly = cci_eval.resample('M').mean()
        ax.plot(cci_monthly.index, cci_monthly.values, color=OBS_COLORS['esa_cci'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='ESA CCI', zorder=5)

    # Plot SMOS if available
    smos_eval = None
    if obs_smos is not None:
        smos_eval = obs_smos[(obs_smos.index >= start) & (obs_smos.index <= end)]['soil_moisture']
        smos_monthly = smos_eval.resample('M').mean()
        ax.plot(smos_monthly.index, smos_monthly.values, color=OBS_COLORS['smos'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='SMOS', zorder=4)

    # Plot ASCAT if available
    ascat_eval = None
    if obs_ascat is not None:
        ascat_eval = obs_ascat[(obs_ascat.index >= start) & (obs_ascat.index <= end)]['soil_moisture']
        ascat_monthly = ascat_eval.resample('M').mean()
        ax.plot(ascat_monthly.index, ascat_monthly.values, color=OBS_COLORS['ascat'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='ASCAT', zorder=3)

    # Plot ERA5-Land if available (reanalysis, still solid as obs reference)
    era5l_eval = None
    if obs_era5land is not None:
        era5l_eval = obs_era5land[(obs_era5land.index >= start) & (obs_era5land.index <= end)]['soil_moisture']
        era5l_monthly = era5l_eval.resample('M').mean()
        ax.plot(era5l_monthly.index, era5l_monthly.values, color=OBS_COLORS['era5land'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='ERA5-Land', zorder=2)

    # Plot simulations - all dashed (obs are solid) for clear visual separation
    sim_line_styles = {'exp1': (0, (5, 2)), 'exp2': (0, (1.5, 1)), 'exp3': (0, (3, 1))}
    sim_line_widths = {'exp1': 0.8, 'exp2': 0.8, 'exp3': 1.0}
    metrics = {'SMAP': {}, 'ESA CCI': {}}
    if obs_smos is not None:
        metrics['SMOS'] = {}
    if obs_ascat is not None:
        metrics['ASCAT'] = {}
    if obs_era5land is not None:
        metrics['ERA5-Land'] = {}

    for exp, df in sim_dfs.items():
        # Use surface layer volumetric SM for fair comparison with satellite products
        if df is not None and 'surf_SM' in df.columns:
            df_eval = df[(df.index >= start) & (df.index <= end)]
            sim_sm = df_eval['surf_SM']  # Already in m³/m³
            sim_monthly = sim_sm.resample('M').mean()

            ax.plot(sim_monthly.index, sim_monthly.values, color=EXP_COLORS[exp],
                   linewidth=sim_line_widths[exp], linestyle=sim_line_styles[exp],
                   alpha=0.7, label=EXP_LABELS[exp], zorder=7)

            # Calculate metrics vs each product
            metrics['SMAP'][exp] = calculate_metrics(sim_sm, obs_eval)
            if cci_eval is not None:
                metrics['ESA CCI'][exp] = calculate_metrics(sim_sm, cci_eval)
            if smos_eval is not None:
                metrics['SMOS'][exp] = calculate_metrics(sim_sm, smos_eval)
            if ascat_eval is not None:
                metrics['ASCAT'][exp] = calculate_metrics(sim_sm, ascat_eval)
            if era5l_eval is not None:
                metrics['ERA5-Land'][exp] = calculate_metrics(sim_sm, era5l_eval)
        elif df is not None and 'soil_water' in df.columns:
            # Fallback to total soil water if surf_SM not available
            df_eval = df[(df.index >= start) & (df.index <= end)]
            sim_sm = df_eval['soil_water'] / 1000
            sim_monthly = sim_sm.resample('M').mean()
            ax.plot(sim_monthly.index, sim_monthly.values, color=EXP_COLORS[exp],
                   linewidth=sim_line_widths[exp], linestyle=sim_line_styles[exp],
                   alpha=0.7, label=EXP_LABELS[exp], zorder=7)
            metrics['SMAP'][exp] = calculate_metrics(sim_sm, obs_eval)
            if cci_eval is not None:
                metrics['ESA CCI'][exp] = calculate_metrics(sim_sm, cci_eval)
            if smos_eval is not None:
                metrics['SMOS'][exp] = calculate_metrics(sim_sm, smos_eval)
            if ascat_eval is not None:
                metrics['ASCAT'][exp] = calculate_metrics(sim_sm, ascat_eval)
            if era5l_eval is not None:
                metrics['ERA5-Land'][exp] = calculate_metrics(sim_sm, era5l_eval)

    ax.set_ylabel('Soil Moisture (m³/m³)', fontsize=FONT['size']['axis_label'])
    ax.set_xlim(start, end)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Y-grid
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(loc='lower right', fontsize=FONT['size']['legend']-2, ncol=4,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
             handlelength=1.5, handletextpad=0.3, columnspacing=0.5,
             borderpad=0.3, labelspacing=0.3)

    despine(ax)
    ax.set_title('(b) Soil Moisture', loc='left', fontsize=FONT['size']['panel_label'],
                fontweight='bold', pad=8)

    return metrics


def plot_swe_comparison(ax, obs_swe, sim_dfs, obs_stations=None,
                        obs_snodas=None, obs_cmc=None):
    """Plot SWE comparison with station ensemble range, SNODAS, and CMC."""
    start = pd.Timestamp(EVALUATION_START)
    end = pd.Timestamp(EVALUATION_END)

    obs_eval = obs_swe[(obs_swe.index >= start) & (obs_swe.index <= end)]['swe_mm']

    # Weekly means for cleaner viz
    obs_weekly = obs_eval.resample('W').mean()

    # Plot station ensemble IQR if available (just IQR to keep y-axis reasonable)
    if obs_stations is not None:
        stations_eval = obs_stations[(obs_stations.index >= start) & (obs_stations.index <= end)]
        stations_weekly_p25 = stations_eval.resample('W').mean().quantile(0.25, axis=1)
        stations_weekly_p75 = stations_eval.resample('W').mean().quantile(0.75, axis=1)

        ax.fill_between(stations_weekly_p25.index, stations_weekly_p25.values,
                       stations_weekly_p75.values, color=OBS_COLORS['canswe'],
                       alpha=0.15, label='Station IQR', zorder=1)

    # Plot observed mean
    ax.plot(obs_weekly.index, obs_weekly.values, color=OBS_COLORS['canswe'],
           linewidth=1.3, alpha=1.0, label='CanSWE mean', zorder=4)

    # Plot SNODAS gridded SWE if available (monthly data - plot at native resolution)
    snodas_eval = None
    if obs_snodas is not None:
        snodas_eval = obs_snodas[(obs_snodas.index >= start) & (obs_snodas.index <= end)]['swe_mm']
        ax.plot(snodas_eval.index, snodas_eval.values, color=OBS_COLORS['snodas'],
               linewidth=1.4, alpha=1.0, linestyle='-', label='SNODAS', zorder=5)

    # Plot CMC gridded SWE if available (monthly data - plot at native resolution)
    cmc_eval = None
    if obs_cmc is not None:
        cmc_eval = obs_cmc[(obs_cmc.index >= start) & (obs_cmc.index <= end)]['swe_mm']
        ax.plot(cmc_eval.index, cmc_eval.values, color=OBS_COLORS['cmc'],
               linewidth=1.4, alpha=1.0, linestyle='-', label='CMC', zorder=5)

    # Plot simulations - all dashed (obs are solid) for clear visual separation
    sim_line_styles = {'exp1': (0, (5, 2)), 'exp2': (0, (1.5, 1)), 'exp3': (0, (3, 1))}
    sim_line_widths = {'exp1': 0.8, 'exp2': 0.8, 'exp3': 1.0}
    metrics = {'CanSWE': {}}
    if obs_snodas is not None:
        metrics['SNODAS'] = {}
    if obs_cmc is not None:
        metrics['CMC'] = {}

    for exp, df in sim_dfs.items():
        if df is not None and 'SWE' in df.columns:
            df_eval = df[(df.index >= start) & (df.index <= end)]
            sim_weekly = df_eval['SWE'].resample('W').mean()

            ax.plot(sim_weekly.index, sim_weekly.values, color=EXP_COLORS[exp],
                   linewidth=sim_line_widths[exp], linestyle=sim_line_styles[exp],
                   alpha=0.7, label=EXP_LABELS[exp], zorder=6)

            metrics['CanSWE'][exp] = calculate_metrics(df_eval['SWE'], obs_eval)
            if snodas_eval is not None:
                metrics['SNODAS'][exp] = calculate_metrics(df_eval['SWE'], snodas_eval)
            if cmc_eval is not None:
                metrics['CMC'][exp] = calculate_metrics(df_eval['SWE'], cmc_eval)

    ax.set_ylabel('SWE (mm)', fontsize=FONT['size']['axis_label'])
    ax.set_xlim(start, end)
    ax.set_ylim(0, None)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Y-grid
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(loc='lower right', fontsize=FONT['size']['legend']-2, ncol=4,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
             handlelength=1.5, handletextpad=0.3, columnspacing=0.5,
             borderpad=0.3, labelspacing=0.3)

    despine(ax)
    ax.set_title('(c) Snow Water Equivalent', loc='left', fontsize=FONT['size']['panel_label'],
                fontweight='bold', pad=8)

    return metrics


def plot_et_comparison(ax, obs_et, sim_dfs, obs_ssebop=None, obs_fluxcom=None):
    """Plot ET comparison with MODIS, SSEBop, FLUXCOM, and simulated ET."""
    start = pd.Timestamp(EVALUATION_START)
    end = pd.Timestamp(EVALUATION_END)

    obs_eval = obs_et[(obs_et.index >= start) & (obs_et.index <= end)]['et_mm_day']

    # Monthly means for clarity
    obs_monthly = obs_eval.resample('M').mean()

    # Plot MODIS ET
    ax.plot(obs_monthly.index, obs_monthly.values, color=OBS_COLORS['modis_et'],
           linewidth=1.3, alpha=1.0, label='MODIS ET', zorder=5)

    # Prepare SSEBop data
    ssebop_eval = None
    if obs_ssebop is not None:
        ssebop_eval = obs_ssebop[(obs_ssebop.index >= start) & (obs_ssebop.index <= end)]['et_mm_day']
        ssebop_monthly = ssebop_eval.resample('M').mean()
        ax.plot(ssebop_monthly.index, ssebop_monthly.values, color=OBS_COLORS['ssebop'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='SSEBop', zorder=4)

    # Plot FLUXCOM if available (monthly data - normalize timestamps)
    fluxcom_eval = None
    if obs_fluxcom is not None:
        fluxcom_tmp = obs_fluxcom.copy()
        fluxcom_tmp.index = fluxcom_tmp.index.normalize()  # Strip time component
        fluxcom_eval = fluxcom_tmp[(fluxcom_tmp.index >= start) & (fluxcom_tmp.index <= end)]['et_mm_day']
        fluxcom_monthly = fluxcom_eval.resample('M').mean()
        ax.plot(fluxcom_monthly.index, fluxcom_monthly.values, color=OBS_COLORS['fluxcom'],
               linewidth=1.2, alpha=1.0, linestyle='-', label='FLUXCOM', zorder=4)

    # Plot simulations - all dashed (obs are solid) for clear visual separation
    sim_line_styles = {'exp1': (0, (5, 2)), 'exp2': (0, (1.5, 1)), 'exp3': (0, (3, 1))}
    sim_line_widths = {'exp1': 0.8, 'exp2': 0.8, 'exp3': 1.0}
    metrics = {'MODIS ET': {}, 'SSEBop': {}}
    if obs_fluxcom is not None:
        metrics['FLUXCOM'] = {}

    for exp, df in sim_dfs.items():
        if df is not None and 'sim_ET' in df.columns:
            df_eval = df[(df.index >= start) & (df.index <= end)]
            sim_monthly = df_eval['sim_ET'].resample('M').mean()

            # Debug print
            if sim_monthly.notna().sum() > 0:
                print(f"    {exp} sim_ET: mean={sim_monthly.mean():.3f}, n={sim_monthly.notna().sum()}")

            ax.plot(sim_monthly.index, sim_monthly.values, color=EXP_COLORS[exp],
                   linewidth=sim_line_widths[exp], linestyle=sim_line_styles[exp],
                   alpha=0.7, label=EXP_LABELS[exp], zorder=3)

            # Metrics vs each product
            metrics['MODIS ET'][exp] = calculate_metrics(df_eval['sim_ET'], obs_eval)
            if ssebop_eval is not None:
                metrics['SSEBop'][exp] = calculate_metrics(df_eval['sim_ET'], ssebop_eval)
            if fluxcom_eval is not None:
                metrics['FLUXCOM'][exp] = calculate_metrics(df_eval['sim_ET'], fluxcom_eval)
        else:
            print(f"    WARNING: No sim_ET in {exp}")
            metrics['MODIS ET'][exp] = {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    ax.set_ylabel('ET (mm/day)', fontsize=FONT['size']['axis_label'])
    ax.set_xlim(start, end)
    ax.set_ylim(0, None)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Y-grid
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', fontsize=FONT['size']['legend']-2, ncol=3,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
             handlelength=1.5, handletextpad=0.3, columnspacing=0.5,
             borderpad=0.3, labelspacing=0.3)

    despine(ax)
    ax.set_title('(d) Evapotranspiration', loc='left', fontsize=FONT['size']['panel_label'],
                fontweight='bold', pad=8)

    return metrics


def plot_metrics_summary(ax, all_metrics):
    """Plot summary metrics by observation product - correlation and bias."""
    ax.axis('off')

    # Define category groupings in display order
    category_products = {
        'Snow Cover': ['MODIS SCA', 'VIIRS SCA', 'IMS', 'Sentinel-2'],
        'Soil Moisture': ['SMAP', 'ESA CCI', 'SMOS', 'ASCAT', 'ERA5-Land'],
        'SWE': ['CanSWE', 'SNODAS', 'CMC'],
        'ET': ['MODIS ET', 'SSEBop', 'FLUXCOM'],
    }

    # Flatten metrics by product (preserving order)
    product_metrics = {}
    for var_key, var_metrics in all_metrics.items():
        if isinstance(var_metrics, dict):
            for product, exp_metrics in var_metrics.items():
                if isinstance(exp_metrics, dict) and 'exp1' in exp_metrics:
                    product_metrics[product] = exp_metrics

    # Build table data with category separator rows
    experiments = ['exp1', 'exp2', 'exp3']
    data = []
    cell_colors = []
    is_category_row = []  # Track which rows are category headers

    def get_color(val):
        if val == '—':
            return '#F5F5F5'
        try:
            r = float(val.split()[0])
            if r > 0.8:
                return '#C8E6C9'  # Green
            elif r > 0.6:
                return '#FFF9C4'  # Yellow
            elif r > 0.4:
                return '#FFE0B2'  # Orange
            else:
                return '#FFCDD2'  # Red
        except:
            return '#F5F5F5'

    for category, prod_list in category_products.items():
        # Category separator row
        data.append([category, '', '', ''])
        cell_colors.append(['#ECECEC', '#ECECEC', '#ECECEC', '#ECECEC'])
        is_category_row.append(True)

        # Product rows within this category
        for product in prod_list:
            if product in product_metrics:
                exp_metrics = product_metrics[product]
                row = [product]
                r_values = []
                for exp in experiments:
                    if exp in exp_metrics:
                        r = exp_metrics[exp].get('r', np.nan)
                        if not np.isnan(r):
                            row.append(f'{r:.2f}')
                            r_values.append(r)
                        else:
                            row.append('—')
                            r_values.append(np.nan)
                    else:
                        row.append('—')
                        r_values.append(np.nan)

                # Bold the best r value in each row
                valid_r = [(i, v) for i, v in enumerate(r_values) if not np.isnan(v)]
                best_idx = None
                if valid_r:
                    best_idx = max(valid_r, key=lambda x: x[1])[0]
                    # Mark best with bold formatting (handled below in styling)

                data.append(row)
                row_colors = ['#FAFAFA'] + [get_color(v) for v in row[1:]]
                cell_colors.append(row_colors)
                is_category_row.append(False)

    if not data:
        return

    # Create table
    col_labels = ['Product', 'Q-only (r)', 'TWS-only (r)', 'Joint (r)']

    table = ax.table(cellText=data, colLabels=col_labels, cellLoc='center',
                    loc='upper center', colWidths=[0.22, 0.14, 0.14, 0.14],
                    cellColours=cell_colors)

    table.auto_set_font_size(False)
    table.set_fontsize(FONT['size']['table'] - 1)
    table.scale(1.0, 1.2)

    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#CCCCCC')

    # Style data cells - category rows and product rows
    product_row_data = []  # Track (table_row, r_values) for bold-best
    for i in range(1, len(data) + 1):
        is_cat = is_category_row[i - 1]
        for j in range(4):
            cell = table[(i, j)]
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.5)

            if is_cat:
                # Category separator: bold italic, light gray bg, left-aligned
                cell.set_facecolor('#ECECEC')
                cell.set_text_props(fontweight='bold', fontstyle='italic',
                                    ha='left')
            elif j == 0:
                # Left-align product names
                cell.set_text_props(ha='left')
        if not is_cat:
            # Collect r values for bold-best detection
            r_vals = []
            for j in range(1, 4):
                val_text = data[i - 1][j]
                try:
                    r_vals.append(float(val_text))
                except:
                    r_vals.append(np.nan)
            product_row_data.append((i, r_vals))

    # Bold the best r in each product row
    for table_row, r_vals in product_row_data:
        valid = [(j, v) for j, v in enumerate(r_vals) if not np.isnan(v)]
        if valid:
            best_j = max(valid, key=lambda x: x[1])[0]
            cell = table[(table_row, best_j + 1)]  # +1 because col 0 is Product
            cell.set_text_props(fontweight='bold')

    ax.set_title('(e) Correlation by Observation Product', loc='left',
                fontsize=FONT['size']['panel_label'], fontweight='bold', pad=12)

    # Add note about correlations
    ax.text(0.5, -0.01, 'Note: DSS correlations of sim. time series to magnitude-matched observations (see text)',
           transform=ax.transAxes, fontsize=7, ha='center', va='top',
           color='#888888', style='italic')


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print("=" * 70)
    print("Creating Figure 3: Comprehensive Independent Validation")
    print("=" * 70)

    # Load simulation outputs
    print("\nLoading simulation outputs...")
    sim_dfs = {
        'exp1': load_summa_output(EXP1_DIR),
        'exp2': load_summa_output(EXP2_DIR),
        'exp3': load_summa_output(EXP3_DIR),
    }

    for exp, df in sim_dfs.items():
        if df is not None:
            print(f"  {exp}: {len(df)} days, columns: {list(df.columns)}")

    # Load observations
    print("\nLoading observations...")
    obs_sca = load_modis_sca()
    obs_viirs = load_viirs_sca()
    obs_ims = load_ims_snow()
    obs_s2 = load_sentinel2_snow()
    obs_sm = load_smap_sm()
    obs_cci = load_esa_cci_sm()
    obs_smos = load_smos_sm()
    obs_ascat = load_ascat_sm()
    obs_era5land = load_era5land_sm()
    obs_swe = load_canswe_swe()
    obs_swe_stations = load_canswe_stations()
    obs_snodas = load_snodas_swe()
    obs_cmc = load_cmc_swe()
    obs_et = load_modis_et()
    obs_ssebop = load_ssebop_et()
    obs_fluxcom = load_fluxcom_et()

    print(f"  MODIS SCA: {len(obs_sca) if obs_sca is not None else 0} days")
    print(f"  VIIRS SCA: {len(obs_viirs) if obs_viirs is not None else 0} days")
    print(f"  IMS Snow: {len(obs_ims) if obs_ims is not None else 0} days")
    print(f"  Sentinel-2: {len(obs_s2) if obs_s2 is not None else 0} scenes")
    print(f"  SMAP SM: {len(obs_sm) if obs_sm is not None else 0} days")
    print(f"  ESA CCI SM: {len(obs_cci) if obs_cci is not None else 0} days")
    print(f"  SMOS SM: {len(obs_smos) if obs_smos is not None else 0} days")
    print(f"  ASCAT SM: {len(obs_ascat) if obs_ascat is not None else 0} days")
    print(f"  ERA5-Land SM: {len(obs_era5land) if obs_era5land is not None else 0} months")
    print(f"  CanSWE SWE: {len(obs_swe) if obs_swe is not None else 0} days")
    print(f"  CanSWE stations: {obs_swe_stations.shape[1] if obs_swe_stations is not None else 0} stations")
    print(f"  SNODAS SWE: {len(obs_snodas) if obs_snodas is not None else 0} months")
    print(f"  CMC SWE: {len(obs_cmc) if obs_cmc is not None else 0} months")
    print(f"  MODIS ET: {len(obs_et) if obs_et is not None else 0} days")
    print(f"  SSEBop ET: {len(obs_ssebop) if obs_ssebop is not None else 0} days")
    print(f"  FLUXCOM ET: {len(obs_fluxcom) if obs_fluxcom is not None else 0} months")

    # Create figure (taller to accommodate expanded metrics table)
    print("\nCreating figure...")
    fig = plt.figure(figsize=(13, 12.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.75],
                  wspace=0.28, hspace=0.38,
                  left=0.08, right=0.97, top=0.96, bottom=0.04)

    all_metrics = {}

    # Panel (a): SCA (with MODIS, VIIRS, IMS, Sentinel-2)
    ax_sca = fig.add_subplot(gs[0, 0])
    if obs_sca is not None:
        all_metrics['sca'] = plot_sca_comparison(ax_sca, obs_sca, sim_dfs,
                                                 obs_viirs=obs_viirs, obs_ims=obs_ims,
                                                 obs_s2=obs_s2)
    else:
        ax_sca.text(0.5, 0.5, 'MODIS SCA data not available',
                   transform=ax_sca.transAxes, ha='center', va='center')
        ax_sca.axis('off')

    # Panel (b): Soil Moisture (with SMAP, ESA CCI, SMOS, ASCAT, ERA5-Land)
    ax_sm = fig.add_subplot(gs[0, 1])
    if obs_sm is not None:
        all_metrics['sm'] = plot_sm_comparison(ax_sm, obs_sm, sim_dfs, obs_cci=obs_cci,
                                               obs_smos=obs_smos, obs_ascat=obs_ascat,
                                               obs_era5land=obs_era5land)
    else:
        ax_sm.text(0.5, 0.5, 'SMAP SM data not available',
                  transform=ax_sm.transAxes, ha='center', va='center')
        ax_sm.axis('off')

    # Panel (c): SWE (with CanSWE, SNODAS, CMC)
    ax_swe = fig.add_subplot(gs[1, 0])
    if obs_swe is not None:
        all_metrics['swe'] = plot_swe_comparison(ax_swe, obs_swe, sim_dfs,
                                                  obs_stations=obs_swe_stations,
                                                  obs_snodas=obs_snodas, obs_cmc=obs_cmc)
    else:
        ax_swe.text(0.5, 0.5, 'CanSWE SWE data not available',
                   transform=ax_swe.transAxes, ha='center', va='center')
        ax_swe.axis('off')

    # Panel (d): ET (with MODIS, SSEBop, FLUXCOM)
    ax_et = fig.add_subplot(gs[1, 1])
    if obs_et is not None:
        all_metrics['et'] = plot_et_comparison(ax_et, obs_et, sim_dfs,
                                               obs_ssebop=obs_ssebop,
                                               obs_fluxcom=obs_fluxcom)
    else:
        ax_et.text(0.5, 0.5, 'MODIS ET data not available',
                  transform=ax_et.transAxes, ha='center', va='center')
        ax_et.axis('off')

    # Panel (e): Summary metrics
    ax_metrics = fig.add_subplot(gs[2, :])
    plot_metrics_summary(ax_metrics, all_metrics)

    # Save
    output_path = OUTPUT_DIR / "fig3_validation_comprehensive.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    # Print metrics summary
    print("\n" + "=" * 50)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 50)
    for var, var_metrics in all_metrics.items():
        print(f"\n{var.upper()}:")
        if isinstance(var_metrics, dict):
            for product, exp_metrics in var_metrics.items():
                if isinstance(exp_metrics, dict) and 'exp1' in exp_metrics:
                    print(f"  {product}:")
                    for exp, m in exp_metrics.items():
                        r = m.get('r', np.nan)
                        n = m.get('n', 0)
                        print(f"    {EXP_LABELS.get(exp, exp)}: r = {r:.3f} (n={n})")

    plt.close()
    print("\nFigure 3 complete!")


if __name__ == "__main__":
    main()
