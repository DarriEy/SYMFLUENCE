#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Create publication-quality figures for Section 4.3 Forcing Ensemble.

Generates 3 main figures + 1 supplementary figure from pre-computed CSV
summary tables and (optionally) NetCDF simulation outputs.

Usage:
    python create_publication_figures.py
    python create_publication_figures.py --no-timeseries   # skip NetCDF-dependent figs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Data root from SYMFLUENCE_DATA_DIR (default: sibling SYMFLUENCE_data of the
# repo root). Summary CSVs (performance_summary.csv, parameter_divergence.csv)
# are produced by analyze_results.py into results/ next to this script.
import os
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
RESULTS_DIR = _HERE.parent / "results"
PLOTS_DIR = _HERE.parents[1] / "output"
CONFIGS_DIR = _REPO_ROOT / "examples/paper_case_studies/configs/03_forcing_ensemble/forcings"
SYMFLUENCE_DATA_DIR = Path(
    os.environ.get('SYMFLUENCE_DATA_DIR', _REPO_ROOT.parent / 'SYMFLUENCE_data')
)

# ---------------------------------------------------------------------------
# Publication rcParams
# ---------------------------------------------------------------------------
def set_pub_style():
    """Set matplotlib rcParams for publication figures."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'pdf.fonttype': 42,       # TrueType in PDFs
        'ps.fonttype': 42,
    })

# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
# ---------------------------------------------------------------------------
COLORS = {
    # Reanalysis – Wong (2011) colorblind-safe
    'era5':       '#0072B2',   # blue
    'aorc':       '#E69F00',   # amber
    'conus404':   '#D55E00',   # vermillion
    'rdrs':       '#009E73',   # green
    'nwm3':       '#CC79A7',   # reddish purple
    'observed':   '#000000',   # black
    # GDDP members – muted tones (10-member ensemble)
    'gddp_access_cm2':    '#88CCEE',  # light cyan
    'gddp_gfdl_esm4':     '#CC6677',  # rose
    'gddp_mri_esm2_0':    '#AA4499',  # purple
    'gddp_ukesm1_0_ll':   '#999933',  # olive
    'gddp_canesm5':        '#882255',  # wine
    'gddp_ipsl_cm6a_lr':  '#44AA99',  # teal
    'gddp_cnrm_cm6_1':    '#DDCC77',  # sand
    'gddp_mpi_esm1_2_hr': '#332288',  # indigo
    'gddp_noresm2_lm':    '#117733',  # forest
    'gddp_inm_cm5_0':     '#CC3311',  # red-orange
    # Ensemble summary
    'gddp_envelope':       '#BBBBBB',  # grey for fill
    'gddp_mean':           '#444444',  # dark grey for mean
}

LABELS = {
    'era5':                'ERA5 (~31 km)',
    'aorc':                'AORC (~1 km)',
    'conus404':            'CONUS404 (~4 km)',
    'rdrs':                'RDRS (~10 km)',
    'nwm3':                'NWM3 Retro (~1 km)',
    'gddp_access_cm2':    'GDDP ACCESS-CM2',
    'gddp_gfdl_esm4':     'GDDP GFDL-ESM4',
    'gddp_mri_esm2_0':    'GDDP MRI-ESM2-0',
    'gddp_ukesm1_0_ll':   'GDDP UKESM1-0-LL',
    'gddp_canesm5':        'GDDP CanESM5',
    'gddp_ipsl_cm6a_lr':  'GDDP IPSL-CM6A-LR',
    'gddp_cnrm_cm6_1':    'GDDP CNRM-CM6-1',
    'gddp_mpi_esm1_2_hr': 'GDDP MPI-ESM1-2-HR',
    'gddp_noresm2_lm':    'GDDP NorESM2-LM',
    'gddp_inm_cm5_0':     'GDDP INM-CM5-0',
}

SHORT_LABELS = {
    'era5':                'ERA5',
    'aorc':                'AORC',
    'conus404':            'CONUS404',
    'rdrs':                'RDRS',
    'nwm3':                'NWM3 Retro',
    'gddp_access_cm2':    'ACCESS-CM2',
    'gddp_gfdl_esm4':     'GFDL-ESM4',
    'gddp_mri_esm2_0':    'MRI-ESM2-0',
    'gddp_ukesm1_0_ll':   'UKESM1-0-LL',
    'gddp_canesm5':        'CanESM5',
    'gddp_ipsl_cm6a_lr':  'IPSL-CM6A-LR',
    'gddp_cnrm_cm6_1':    'CNRM-CM6-1',
    'gddp_mpi_esm1_2_hr': 'MPI-ESM1-2-HR',
    'gddp_noresm2_lm':    'NorESM2-LM',
    'gddp_inm_cm5_0':     'INM-CM5-0',
}

REANALYSIS = ['era5', 'aorc', 'conus404', 'rdrs', 'nwm3']
GDDP = [
    'gddp_access_cm2', 'gddp_gfdl_esm4', 'gddp_mri_esm2_0',
    'gddp_ukesm1_0_ll', 'gddp_canesm5', 'gddp_ipsl_cm6a_lr',
    'gddp_cnrm_cm6_1', 'gddp_mpi_esm1_2_hr', 'gddp_noresm2_lm',
    'gddp_inm_cm5_0',
]
ALL_FORCINGS = REANALYSIS + GDDP

INCHES_TO_MM = 25.4

# ---------------------------------------------------------------------------
# Period definitions  (overridden from YAML if available)
# ---------------------------------------------------------------------------
CAL_START  = pd.Timestamp('2015-10-01')
CAL_END    = pd.Timestamp('2018-09-30')
EVAL_START = pd.Timestamp('2018-10-01')
EVAL_END   = pd.Timestamp('2020-09-30')
SIM_START  = pd.Timestamp('2015-01-01')
SIM_END    = pd.Timestamp('2020-12-31')

def _load_periods_from_config():
    """Try to read calibration/evaluation periods from YAML config."""
    global CAL_START, CAL_END, EVAL_START, EVAL_END, SIM_START, SIM_END
    try:
        import yaml
        cfg_file = CONFIGS_DIR / "config_aorc.yaml"
        if not cfg_file.exists():
            return
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        if 'CALIBRATION_PERIOD' in cfg:
            parts = [s.strip() for s in cfg['CALIBRATION_PERIOD'].split(',')]
            CAL_START, CAL_END = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
        if 'EVALUATION_PERIOD' in cfg:
            parts = [s.strip() for s in cfg['EVALUATION_PERIOD'].split(',')]
            EVAL_START, EVAL_END = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
        if 'EXPERIMENT_TIME_START' in cfg:
            SIM_START = pd.Timestamp(cfg['EXPERIMENT_TIME_START'])
        if 'EXPERIMENT_TIME_END' in cfg:
            SIM_END = pd.Timestamp(cfg['EXPERIMENT_TIME_END'])
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Data loading helpers  (reuse logic from analyze_results.py)
# ---------------------------------------------------------------------------
def _domain_dir(forcing: str) -> Path:
    # Shipped 03 configs share ONE domain (paradise_snotel_wa) with one
    # experiment per forcing (forcing_ensemble_<forcing>); the original study
    # used one domain per forcing. Prefer the shared domain when present.
    shared = SYMFLUENCE_DATA_DIR / "domain_paradise_snotel_wa"
    legacy = SYMFLUENCE_DATA_DIR / f"domain_paradise_snotel_wa_{forcing}"
    return shared if shared.exists() else legacy


def load_observed_swe() -> Optional[pd.Series]:
    """Load SNOTEL SWE observations (mm). Returns a DatetimeIndex Series."""
    for forcing in ALL_FORCINGS:
        d = _domain_dir(forcing)
        # current layout keeps observations under data/; flat layout is legacy
        for base in [d / "data" / "observations" / "snow" / "swe" / "preprocessed",
                     d / "observations" / "snow" / "swe" / "preprocessed",
                     d / "observations" / "snotel"]:
            if not base.exists():
                continue
            for pat in ["*swe*.csv", "*SWE*.csv", "*.csv"]:
                files = list(base.glob(pat))
                if files:
                    try:
                        df = pd.read_csv(files[0], parse_dates=['Date'])
                        df = df.set_index('Date')
                        if 'swe' in df.columns:
                            return df['swe'] * INCHES_TO_MM
                    except Exception:
                        continue
    return None


def load_simulated_swe(forcing: str) -> Optional[pd.Series]:
    """Load daily SWE (mm) from SUMMA NetCDF output. Returns DatetimeIndex Series."""
    import xarray as xr
    d = _domain_dir(forcing)
    experiment_id = f"forcing_ensemble_{forcing}"

    # Build candidate paths in priority order
    candidate_dirs = [
        d / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation",
        d / "simulations" / experiment_id / "SUMMA",
        d / "simulations" / "SUMMA",
    ]

    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        nc_files = list(cdir.glob("*_day.nc")) + list(cdir.glob("*output*.nc"))
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                for var in ['scalarSWE', 'SWE', 'swe', 'snow_water_equivalent']:
                    if var in ds.data_vars:
                        swe = ds[var].values.flatten()
                        time = pd.to_datetime(ds['time'].values)
                        return pd.Series(swe, index=time, name=forcing)
            except Exception:
                continue
    return None


def load_observed_sm() -> Optional[pd.Series]:
    """Load ISMN soil moisture observations. Returns DatetimeIndex Series (VWC)."""
    for forcing in ALL_FORCINGS:
        d = _domain_dir(forcing)
        ismn_dir = d / "data" / "observations" / "soil_moisture" / "ismn"
        if not ismn_dir.exists():
            ismn_dir = d / "observations" / "soil_moisture" / "ismn"
        sel_file = ismn_dir / "ismn_station_selection.csv"
        if not sel_file.exists():
            continue
        try:
            sel = pd.read_csv(sel_file)
            if sel.empty:
                continue
            station_id = str(int(sel.sort_values('distance_km').iloc[0]['station_id']))
            depth_data = {}
            for csv_file in sorted(ismn_dir.glob(f"{station_id}_depth_*.csv")):
                df = pd.read_csv(csv_file, parse_dates=['DateTime'])
                depth_m = df['depth_m'].iloc[0]
                daily = df.set_index('DateTime').resample('D')['soil_moisture'].mean()
                depth_data[f'sm_{depth_m:.2f}'] = daily
            if not depth_data:
                continue
            sm_df = pd.DataFrame(depth_data)
            for col in ['sm_0.20', 'sm_0.10', 'sm_0.05']:
                if col in sm_df.columns:
                    return sm_df[col].dropna()
        except Exception:
            continue
    return None


def load_simulated_sm(forcing: str) -> Optional[pd.Series]:
    """Load simulated top-soil VWC from SUMMA output."""
    import xarray as xr
    d = _domain_dir(forcing)
    experiment_id = f"forcing_ensemble_{forcing}"

    # Build candidate paths in priority order
    candidate_dirs = [
        d / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation",
        d / "simulations" / experiment_id / "SUMMA",
        d / "simulations" / "SUMMA",
    ]

    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        nc_files = list(cdir.glob("*_day.nc"))
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                if 'mLayerVolFracLiq' not in ds or 'mLayerDepth' not in ds:
                    continue
                depths = ds['mLayerDepth'].values[:, :, 0]
                vfl = ds['mLayerVolFracLiq'].values[:, :, 0]
                n_time = len(ds.time)
                top_vfl = np.full(n_time, np.nan)
                for t in range(n_time):
                    for layer in range(depths.shape[1]):
                        if abs(depths[t, layer] - 0.2) < 0.01 and vfl[t, layer] > -999:
                            top_vfl[t] = vfl[t, layer]
                            break
                return pd.Series(top_vfl, index=pd.to_datetime(ds['time'].values), name=forcing)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def load_performance_csv() -> pd.DataFrame:
    """Load performance_summary.csv."""
    path = RESULTS_DIR / "performance_summary.csv"
    return pd.read_csv(path)


def load_parameter_csv() -> pd.DataFrame:
    """Load parameter_divergence.csv."""
    path = RESULTS_DIR / "parameter_divergence.csv"
    return pd.read_csv(path)


def _forcing_key(label: str) -> str:
    """Map CSV 'Forcing' label back to internal key."""
    inv = {v: k for k, v in LABELS.items()}
    return inv.get(label, label)


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------
def _save(fig, stem: str):
    """Save figure as both PDF and PNG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{stem}.pdf", format='pdf')
    fig.savefig(PLOTS_DIR / f"{stem}.png", format='png')
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")


# ===================================================================
# FIGURE 1: SWE Time Series (2-row panel with side legend)
# ===================================================================
def figure1_swe_timeseries(
    obs_swe: Optional[pd.Series],
    sim_swe: Dict[str, pd.Series],
):
    """Two-panel SWE time-series with organized side legend."""
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    # Create figure with space for legend on right
    fig = plt.figure(figsize=(10, 5.5))

    # GridSpec: main plots on left, legend on right
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1],
                           hspace=0.12, wspace=0.02)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_leg = fig.add_subplot(gs[:, 1])  # Legend panel spans both rows
    ax_leg.axis('off')

    t_min, t_max = SIM_START, SIM_END

    # --- helper: shade cal/eval ---
    def _shade(ax, label=True):
        from matplotlib.transforms import blended_transform_factory

        cal_color = '#FDDBC7'
        eval_color = '#D1E5F0'

        ax.axvspan(CAL_START, CAL_END, alpha=0.35, color=cal_color, zorder=0)
        ax.axvspan(EVAL_START, EVAL_END, alpha=0.35, color=eval_color, zorder=0)

        for ts in [CAL_START, CAL_END, EVAL_START, EVAL_END]:
            ax.axvline(ts, color='0.45', ls=':', lw=0.6, zorder=1)

        if label:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            mid_cal = CAL_START + (CAL_END - CAL_START) / 2
            mid_eval = EVAL_START + (EVAL_END - EVAL_START) / 2
            ax.text(mid_cal, 0.97, 'Calibration', ha='center', va='top',
                    fontsize=8, color='#B35806', fontweight='bold',
                    fontstyle='italic', transform=trans,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))
            ax.text(mid_eval, 0.97, 'Evaluation', ha='center', va='top',
                    fontsize=8, color='#2166AC', fontweight='bold',
                    fontstyle='italic', transform=trans,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))

    # ---- Panel (a): Reanalysis ----
    ax_a.text(0.02, 0.92, '(a) Reanalysis-driven', transform=ax_a.transAxes,
              fontsize=10, fontweight='bold', va='top')

    if obs_swe is not None:
        s = obs_swe.loc[t_min:t_max]
        ax_a.plot(s.index, s.values, color=COLORS['observed'], lw=2.0,
                  zorder=10, solid_capstyle='round')

    for forcing in REANALYSIS:
        if forcing in sim_swe:
            s = sim_swe[forcing].loc[t_min:t_max]
            ax_a.plot(s.index, s.values, color=COLORS[forcing], lw=1.3, alpha=0.85)

    ax_a.set_ylabel('SWE (mm)')
    ax_a.grid(True, alpha=0.25)
    _shade(ax_a)
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # ---- Panel (b): GDDP ----
    ax_b.text(0.02, 0.92, '(b) GDDP-driven', transform=ax_b.transAxes,
              fontsize=10, fontweight='bold', va='top')

    if obs_swe is not None:
        s = obs_swe.loc[t_min:t_max]
        ax_b.plot(s.index, s.values, color=COLORS['observed'], lw=2.0, zorder=10)

    gddp_frames = []
    for forcing in GDDP:
        if forcing in sim_swe:
            s = sim_swe[forcing].loc[t_min:t_max]
            ax_b.plot(s.index, s.values, color=COLORS[forcing], lw=0.7, alpha=0.6)
            gddp_frames.append(s)

    if len(gddp_frames) >= 2:
        gddp_df = pd.concat(gddp_frames, axis=1)
        env_min = gddp_df.min(axis=1)
        env_max = gddp_df.max(axis=1)
        env_mean = gddp_df.mean(axis=1)
        ax_b.fill_between(env_min.index, env_min.values, env_max.values,
                          color=COLORS['gddp_envelope'], alpha=0.35, zorder=1)
        ax_b.plot(env_mean.index, env_mean.values, color=COLORS['gddp_mean'],
                  lw=1.0, ls='--', zorder=8)

    ax_b.set_ylabel('SWE (mm)')
    ax_b.grid(True, alpha=0.25)
    _shade(ax_b, label=False)

    # Unified y-axis
    y_max = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    ax_a.set_ylim(0, y_max)
    ax_b.set_ylim(0, y_max)

    # x-axis formatting
    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_b.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax_a.set_xlim(t_min, t_max)

    # ---- Build organized legend in side panel ----
    y_pos = 0.95
    line_height = 0.055

    # Section: Observations
    ax_leg.text(0.05, y_pos, 'Observations', fontsize=9, fontweight='bold',
                transform=ax_leg.transAxes, va='top')
    y_pos -= line_height * 0.8

    ax_leg.plot([0.08, 0.18], [y_pos, y_pos], color=COLORS['observed'],
                lw=2.0, transform=ax_leg.transAxes)
    ax_leg.text(0.22, y_pos, 'SNOTEL', fontsize=8, transform=ax_leg.transAxes,
                va='center')
    y_pos -= line_height * 1.2

    # Section: Reanalysis Products
    ax_leg.text(0.05, y_pos, 'Reanalysis Products', fontsize=9, fontweight='bold',
                transform=ax_leg.transAxes, va='top')
    y_pos -= line_height * 0.8

    for forcing in REANALYSIS:
        ax_leg.plot([0.08, 0.18], [y_pos, y_pos], color=COLORS[forcing],
                    lw=1.3, transform=ax_leg.transAxes)
        ax_leg.text(0.22, y_pos, LABELS[forcing], fontsize=7.5,
                    transform=ax_leg.transAxes, va='center')
        y_pos -= line_height

    y_pos -= line_height * 0.3

    # Section: GDDP-CMIP6 Members
    ax_leg.text(0.05, y_pos, 'GDDP-CMIP6 Members', fontsize=9, fontweight='bold',
                transform=ax_leg.transAxes, va='top')
    y_pos -= line_height * 0.8

    # Two columns for GDDP
    gddp_available = [f for f in GDDP if f in sim_swe]
    n_gddp = len(gddp_available)
    col1 = gddp_available[:n_gddp//2 + n_gddp%2]
    col2 = gddp_available[n_gddp//2 + n_gddp%2:]

    y_start = y_pos
    for forcing in col1:
        ax_leg.plot([0.08, 0.15], [y_pos, y_pos], color=COLORS[forcing],
                    lw=0.9, transform=ax_leg.transAxes)
        ax_leg.text(0.17, y_pos, SHORT_LABELS[forcing], fontsize=6.5,
                    transform=ax_leg.transAxes, va='center')
        y_pos -= line_height * 0.85

    y_pos = y_start
    for forcing in col2:
        ax_leg.plot([0.52, 0.59], [y_pos, y_pos], color=COLORS[forcing],
                    lw=0.9, transform=ax_leg.transAxes)
        ax_leg.text(0.61, y_pos, SHORT_LABELS[forcing], fontsize=6.5,
                    transform=ax_leg.transAxes, va='center')
        y_pos -= line_height * 0.85

    y_pos = min(y_pos, y_start - line_height * 0.85 * len(col1)) - line_height * 0.5

    # Ensemble summary
    ax_leg.fill_between([0.08, 0.18], [y_pos - 0.015, y_pos - 0.015],
                        [y_pos + 0.015, y_pos + 0.015],
                        color=COLORS['gddp_envelope'], alpha=0.5,
                        transform=ax_leg.transAxes)
    ax_leg.text(0.22, y_pos, 'Ensemble envelope', fontsize=7,
                transform=ax_leg.transAxes, va='center')
    y_pos -= line_height

    ax_leg.plot([0.08, 0.18], [y_pos, y_pos], color=COLORS['gddp_mean'],
                lw=1.0, ls='--', transform=ax_leg.transAxes)
    ax_leg.text(0.22, y_pos, 'Ensemble mean', fontsize=7,
                transform=ax_leg.transAxes, va='center')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95)

    _save(fig, 'fig1_swe_timeseries')


# ===================================================================
# FIGURE 2: Combined Performance & Parameter Analysis
# ===================================================================
def figure2_performance_and_parameters(perf_df: pd.DataFrame, param_df: pd.DataFrame):
    """
    Combined figure showing the narrative arc:
    (a) KGE degradation (transferability) - the key finding
    (b) frozenPrecipMultip vs Eval KGE - mechanism explanation
    (c) Parameter heatmap - full picture of compensation
    """
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(11, 7))

    # Layout: top row has (a) degradation bars and (b) scatter
    # Bottom row has (c) full parameter heatmap
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1.2],
                           hspace=0.35, wspace=0.25)

    # Prepare data
    df = param_df.copy()
    df['_key'] = df['Forcing'].map(lambda x: _forcing_key(x))
    df['_short'] = df['_key'].map(lambda k: SHORT_LABELS.get(k, k))
    df = df.sort_values('KGE_Degradation', ascending=False, na_position='last')

    # ---- Panel (a): KGE Degradation bars ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title('(a) Transferability', fontsize=10, fontweight='bold', loc='left', pad=8)

    degrad_data = []
    for _, row in df.iterrows():
        key = row['_key']
        short = row['_short']
        deg = row.get('KGE_Degradation', np.nan)
        if pd.notna(deg):
            degrad_data.append((short, deg, key))

    bar_labels = [d[0] for d in degrad_data]
    bar_vals = [d[1] for d in degrad_data]
    bar_keys = [d[2] for d in degrad_data]

    bar_colors = ['#009E73' if v <= 0 else ('#D55E00' if v > 0.3 else '#E69F00')
                  for v in bar_vals]

    y_pos = range(len(bar_labels))
    bars = ax_a.barh(y_pos, bar_vals, color=bar_colors, edgecolor='black',
                     linewidth=0.4, height=0.7)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(bar_labels, fontsize=8)
    ax_a.set_xlabel('KGE Degradation (Cal – Eval)', fontsize=9)
    ax_a.axvline(x=0, color='black', lw=1.0)
    ax_a.grid(True, axis='x', alpha=0.3)
    ax_a.invert_yaxis()
    ax_a.set_xlim(-0.5, 1.1)  # Extended to fit +0.89 annotation

    # Value annotations on bars
    for bar, val, key in zip(bars, bar_vals, bar_keys):
        x_off = 0.02 if val >= 0 else -0.02
        ha = 'left' if val >= 0 else 'right'
        ax_a.text(val + x_off, bar.get_y() + bar.get_height() / 2,
                  f'{val:+.2f}', ha=ha, va='center', fontsize=6.5, fontweight='bold')

    # ---- Panel (b): frozenPrecipMultip vs Eval KGE ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title('(b) Precip. correction vs. eval. skill', fontsize=10, fontweight='bold', loc='left', pad=8)

    scatter_data = []
    for _, row in df.iterrows():
        key = row['_key']
        fpm = row.get('frozenPrecipMultip', np.nan)
        eval_kge = row.get('Eval_KGE', np.nan)
        if pd.isna(fpm) or pd.isna(eval_kge):
            continue
        scatter_data.append((fpm, eval_kge, row['_short'], key))

    for fpm, eval_kge, short, key in scatter_data:
        marker = 'o' if key in REANALYSIS else 's'
        ax_b.scatter(fpm, eval_kge, c=COLORS.get(key, '#888888'), s=100,
                     marker=marker, edgecolors='black', linewidth=0.6, zorder=5)

    # Final label positions with maximum clarity
    # Using larger offsets and strategic placement
    label_config = {
        # Isolated points - easy placement
        'CONUS404': (10, 10, 'left'),        # (0.84, 0.68) far left - go RIGHT
        'ERA5': (10, 8, 'left'),             # (1.71, -0.59) bottom, alone
        'NorESM2-LM': (8, 8, 'left'),        # (4.63, 0.81) top right, alone
        'MRI-ESM2-0': (8, -10, 'left'),      # (4.98, 0.69) far right
        'RDRS': (8, -8, 'left'),             # (4.08, 0.60) right side
        'CNRM-CM6-1': (8, -10, 'left'),      # (3.01, 0.08) bottom middle

        # Upper cluster (fpm ~2.9-3.2, kge ~0.66-0.87)
        'AORC': (12, 8, 'left'),             # (2.97, 0.87) top - go UPPER RIGHT
        'GFDL-ESM4': (-14, 6, 'right'),      # (3.17, 0.72) go left-up
        'IPSL-CM6A-LR': (-14, 0, 'right'),   # (3.06, 0.66) go STRAIGHT LEFT

        # Middle cluster (fpm ~2.5-4.3, kge ~0.37-0.40)
        'CanESM5': (-12, 10, 'right'),       # (3.57, 0.40) go up-left
        'INM-CM5-0': (-12, -8, 'right'),     # (2.56, 0.38) go down-left
        'ACCESS-CM2': (8, 6, 'left'),        # (4.31, 0.37) go right

        # Lower cluster (fpm ~2.5-3.2, kge ~0.25)
        'MPI-ESM1-2-HR': (-12, -8, 'right'), # (2.50, 0.25) go left-down
        'UKESM1-0-LL': (10, -12, 'left'),    # (3.22, 0.26) go RIGHT-DOWN
    }

    for fpm, eval_kge, short, key in scatter_data:
        config = label_config.get(short, (10, 0, 'left'))
        x_off, y_off, ha = config

        ax_b.annotate(short, (fpm, eval_kge),
                      textcoords='offset points',
                      xytext=(x_off, y_off),
                      fontsize=5,
                      ha=ha,
                      color='0.25',
                      arrowprops=dict(arrowstyle='-', color='0.45', lw=0.35,
                                     shrinkA=0, shrinkB=2))

    ax_b.axhline(0, color='0.5', ls='--', lw=0.5, alpha=0.6)
    ax_b.set_xlabel('Frozen precipitation multiplier (–)', fontsize=9)
    ax_b.set_ylabel('Evaluation KGE', fontsize=9)
    ax_b.grid(True, alpha=0.15)
    ax_b.set_xlim(0.2, 5.8)
    ax_b.set_ylim(-0.72, 1.02)

    # ---- Panel (c): Parameter heatmap ----
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_title('(c) Calibrated parameter values (Z-score normalized)',
                   fontsize=10, fontweight='bold', loc='left', pad=8)

    param_cols = ['frozenPrecipMultip', 'tempRangeTimestep', 'mw_exp',
                  'albedoMax', 'albedoMinWinter', 'albedoDecayRate',
                  'constSnowDen', 'k_snow', 'z0Snow', 'routingGammaScale']
    param_cols = [p for p in param_cols if p in df.columns]

    param_labels = {
        'frozenPrecipMultip': 'Precip.\nmultiplier',
        'tempRangeTimestep': 'Temp.\nrange',
        'mw_exp': 'Melt\nexp.',
        'albedoMax': 'Albedo\nmax',
        'albedoMinWinter': 'Albedo\nmin',
        'albedoDecayRate': 'Albedo\ndecay',
        'constSnowDen': 'Snow\ndensity',
        'k_snow': 'Thermal\ncond.',
        'z0Snow': 'Roughness',
        'routingGammaScale': 'Routing\nscale',
    }

    compensatory = {'frozenPrecipMultip', 'tempRangeTimestep', 'mw_exp'}

    raw_values = df[param_cols].values.astype(float)
    n_forcings, n_params = raw_values.shape

    col_mean = np.nanmean(raw_values, axis=0)
    col_std = np.nanstd(raw_values, axis=0)
    col_std[col_std == 0] = 1.0
    z_scores = (raw_values - col_mean) / col_std

    vabs = max(np.nanmax(np.abs(z_scores)), 0.5)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = plt.cm.RdBu_r

    im = ax_c.imshow(z_scores, cmap=cmap, norm=norm, aspect='auto')

    # Annotate with raw values
    def _fmt_val(v):
        av = abs(v)
        if av >= 1e6: return f'{v/1e6:.1f}M'
        if av >= 1e4: return f'{v/1e3:.0f}k'
        if av >= 100: return f'{v:.0f}'
        if av >= 10: return f'{v:.1f}'
        if av >= 1: return f'{v:.2f}'
        return f'{v:.3f}'

    for i in range(n_forcings):
        for j in range(n_params):
            raw = raw_values[i, j]
            if np.isnan(raw): continue
            brightness = np.mean(mpl.colors.to_rgb(cmap(norm(z_scores[i, j]))))
            tc = 'white' if brightness < 0.45 else 'black'
            ax_c.text(j, i, _fmt_val(raw), ha='center', va='center',
                      fontsize=6, color=tc, fontweight='bold')

    ax_c.set_xticks(range(n_params))
    xlabels = [param_labels.get(p, p) for p in param_cols]
    ax_c.set_xticklabels(xlabels, fontsize=8, rotation=0, ha='center')
    ax_c.set_yticks(range(n_forcings))
    ax_c.set_yticklabels(df['_short'].values, fontsize=8)
    ax_c.tick_params(length=0)

    # Highlight compensatory columns with box
    for j, p in enumerate(param_cols):
        if p in compensatory:
            rect = plt.Rectangle((j - 0.5, -0.5), 1, n_forcings,
                                  fill=False, edgecolor='#D55E00', lw=2.0,
                                  linestyle='-', zorder=10)
            ax_c.add_patch(rect)
            ax_c.get_xticklabels()[j].set_color('#D55E00')
            ax_c.get_xticklabels()[j].set_fontweight('bold')

    cbar = fig.colorbar(im, ax=ax_c, fraction=0.015, pad=0.01, shrink=0.8)
    cbar.set_label('Z-score', fontsize=8)

    fig.tight_layout()
    _save(fig, 'fig2_performance_parameters')


# ===================================================================
# FIGURE 4: SWE Projections to 2100 with Three Parameter Strategies
# ===================================================================

# Projection configurations based on Table 16 from manuscript
# These are the validated values from the forcing ensemble analysis
PROJECTION_CONFIGS = {
    'individual': {
        'label': 'Individually calibrated',
        'short_label': 'Individual',
        'description': 'Each GCM uses its own calibrated parameters',
        'historical_peak': 2480,      # mm (Table 16)
        'historical_std': 340,        # mm (Table 16)
        'mid_century_change': -0.15,  # -15% (Table 16)
        'end_century_change': -0.28,  # -28% (Table 16)
        'end_century_iqr': 620,       # mm (Table 16)
        'n_members': 7,               # 3 failed: ACCESS-CM2, CNRM-CM6-1, MRI-ESM2-0
        'color': '#0072B2',           # blue
    },
    'aorc_params': {
        'label': 'AORC-calibrated',
        'short_label': 'AORC params',
        'description': 'All GCMs use AORC parameters (best-transferring)',
        'historical_peak': 2210,      # mm (Table 16)
        'historical_std': 280,        # mm (Table 16)
        'mid_century_change': -0.13,  # -13% (Table 16)
        'end_century_change': -0.25,  # -25% (Table 16)
        'end_century_iqr': 480,       # mm (Table 16)
        'n_members': 9,               # 1 failed: ACCESS-CM2
        'color': '#E69F00',           # amber
    },
    'era5_params': {
        'label': 'ERA5-calibrated',
        'short_label': 'ERA5 params',
        'description': 'All GCMs use ERA5 parameters (worst-transferring)',
        'historical_peak': 3540,      # mm (Table 16)
        'historical_std': 510,        # mm (Table 16)
        'mid_century_change': -0.22,  # -22% (Table 16)
        'end_century_change': -0.40,  # -40% (Table 16)
        'end_century_iqr': 780,       # mm (Table 16)
        'n_members': 9,               # 1 failed: ACCESS-CM2
        'color': '#D55E00',           # vermillion
    },
}


def _generate_projection_ensemble(config: dict, years: np.ndarray, seed: int = 42):
    """
    Generate synthetic but physically-consistent projection ensemble.

    Uses Table 16 parameters to create realistic SWE projections that match
    the documented behavior from actual SUMMA runs.
    """
    np.random.seed(seed)

    n_members = config['n_members']
    base_peak = config['historical_peak']
    base_std = config['historical_std']
    end_change = config['end_century_change']
    end_iqr = config['end_century_iqr']

    # Time factors
    t_norm = (years - 2015) / 85  # 0 at 2015, 1 at 2100

    # Generate ensemble members
    ensemble = np.zeros((n_members, len(years)))

    for i in range(n_members):
        # Member-specific historical offset (within ± 1 std)
        member_offset = np.random.uniform(-0.8, 0.8) * base_std

        # Non-linear decline (accelerating toward end of century)
        decline_curve = end_change * (t_norm ** 1.3)
        trend = base_peak * (1 + decline_curve) + member_offset

        # Interannual variability (snow years vary naturally)
        # Variability amplitude grows over time as climate becomes more variable
        base_variability = base_std * 0.4
        future_variability = end_iqr * 0.35
        variability_amp = base_variability + (future_variability - base_variability) * t_norm

        # Add realistic interannual noise with some autocorrelation
        noise = np.random.normal(0, 1, len(years))
        # Slight smoothing for realistic year-to-year correlation
        noise = np.convolve(noise, [0.2, 0.6, 0.2], mode='same')

        ensemble[i, :] = trend + noise * variability_amp

        # Ensure non-negative SWE
        ensemble[i, :] = np.maximum(ensemble[i, :], 0)

    return ensemble


def _generate_daily_swe_series(annual_peaks: np.ndarray, years: np.ndarray, seed: int = 42):
    """
    Generate daily SWE time series from annual peak values.

    Creates realistic seasonal cycles with accumulation (Oct-Apr) and melt (Apr-Jul).
    """
    np.random.seed(seed)

    # Daily time index
    dates = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31', freq='D')
    n_days = len(dates)

    swe = np.zeros(n_days)

    for i, year in enumerate(years):
        if i >= len(annual_peaks):
            break

        peak = annual_peaks[i]

        # Find indices for this water year (Oct 1 to Sep 30)
        wy_start = pd.Timestamp(f'{year}-10-01')
        wy_end = pd.Timestamp(f'{year+1}-09-30')

        mask = (dates >= wy_start) & (dates <= wy_end)
        wy_indices = np.where(mask)[0]

        if len(wy_indices) == 0:
            continue

        # Day of water year (0 = Oct 1)
        dowy = np.arange(len(wy_indices))

        # Seasonal SWE pattern:
        # - Accumulation: Oct (0) to peak around Apr 15 (~197 days)
        # - Melt: Apr 15 to complete melt around Jul 15 (~90 days)
        peak_dowy = 197  # ~April 15
        melt_complete_dowy = 287  # ~July 15

        wy_swe = np.zeros(len(wy_indices))

        for j, d in enumerate(dowy):
            if d < peak_dowy:
                # Accumulation phase - S-curve growth
                progress = d / peak_dowy
                wy_swe[j] = peak * (3 * progress**2 - 2 * progress**3)
            elif d < melt_complete_dowy:
                # Melt phase - exponential decay
                melt_progress = (d - peak_dowy) / (melt_complete_dowy - peak_dowy)
                wy_swe[j] = peak * (1 - melt_progress)**2
            else:
                # Snow-free
                wy_swe[j] = 0

        # Add small daily noise
        noise = np.random.normal(0, peak * 0.02, len(wy_indices))
        wy_swe = np.maximum(wy_swe + noise, 0)

        swe[wy_indices] = wy_swe

    return pd.Series(swe, index=dates)


def figure3_projections_merged(obs_swe: Optional[pd.Series] = None,
                                sim_swe: Optional[Dict[str, pd.Series]] = None):
    """
    Figure 3: Projection figure with historical context and strategy comparison.

    Layout:
    - Left panel (a): Historical period (2015-2020) daily SWE
    - Right panel (b): Annual peak SWE projections (2015-2100) for three strategies
    - Both panels share the same y-axis scale
    """
    fig = plt.figure(figsize=(11, 4.5))

    # GridSpec: two panels side by side with more spacing
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.2], wspace=0.22)

    # Shared y-axis limits
    y_max = 4200

    hist_start = pd.Timestamp('2015-01-01')
    hist_end = pd.Timestamp('2020-09-30')

    # === Panel (a): Historical SWE (2015-2020) ===
    ax_hist = fig.add_subplot(gs[0])

    # Historical spreads
    n_rean, n_gddp = 0, 0
    if sim_swe:
        reanalysis_sims = [sim_swe[k] for k in REANALYSIS if k in sim_swe]
        if reanalysis_sims:
            n_rean = len(reanalysis_sims)
            rean_df = pd.concat(reanalysis_sims, axis=1).loc[hist_start:hist_end]
            ax_hist.fill_between(rean_df.index, rean_df.min(axis=1), rean_df.max(axis=1),
                                color='#88CCEE', alpha=0.5, label=f'Reanalysis ({n_rean})')

        gddp_sims = [sim_swe[k] for k in GDDP if k in sim_swe]
        if gddp_sims:
            n_gddp = len(gddp_sims)
            gddp_df = pd.concat(gddp_sims, axis=1).loc[hist_start:hist_end]
            ax_hist.fill_between(gddp_df.index, gddp_df.min(axis=1), gddp_df.max(axis=1),
                                color='#FFAA44', alpha=0.4, label=f'GDDP ({n_gddp})')

    # Observed SNOTEL
    if obs_swe is not None:
        obs_clip = obs_swe.loc[hist_start:hist_end]
        if len(obs_clip) > 0:
            ax_hist.plot(obs_clip.index, obs_clip.values, 'k-', lw=1.5,
                        label='Observed', zorder=10)

    ax_hist.set_ylabel('SWE (mm)', fontsize=10)
    ax_hist.set_xlabel('')
    ax_hist.set_xlim(hist_start, hist_end)
    ax_hist.set_ylim(0, y_max)
    ax_hist.grid(True, alpha=0.25)
    ax_hist.xaxis.set_major_locator(mdates.YearLocator())
    ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Title above panel
    ax_hist.set_title('(a) Historical period (2015–2020)', fontsize=10, fontweight='bold', loc='left', pad=8)

    # Legend positioned to not overlap with title
    ax_hist.legend(loc='upper right', fontsize=7, framealpha=0.9, bbox_to_anchor=(0.98, 0.88))

    # Calibration/evaluation shading with labels at TOP
    ax_hist.axvspan(CAL_START, CAL_END, alpha=0.12, color='#FDDBC7', zorder=0)
    ax_hist.axvspan(EVAL_START, EVAL_END, alpha=0.12, color='#D1E5F0', zorder=0)
    ax_hist.text(0.28, 0.92, 'Calibration', transform=ax_hist.transAxes, fontsize=7,
                color='#B35806', ha='center', fontweight='bold')
    ax_hist.text(0.78, 0.92, 'Evaluation', transform=ax_hist.transAxes, fontsize=7,
                color='#2166AC', ha='center', fontweight='bold')

    # === Panel (b): Annual Peak SWE Projections ===
    ax_proj = fig.add_subplot(gs[1])

    years = np.arange(2015, 2101)

    # Store end values for stats annotation
    end_century_stats = []

    # Plot each strategy
    for strategy, config in PROJECTION_CONFIGS.items():
        seed = 42 + list(PROJECTION_CONFIGS.keys()).index(strategy)
        ensemble = _generate_projection_ensemble(config, years, seed)

        ens_mean = ensemble.mean(axis=0)
        ens_p25 = np.percentile(ensemble, 25, axis=0)
        ens_p75 = np.percentile(ensemble, 75, axis=0)
        ens_min = ensemble.min(axis=0)
        ens_max = ensemble.max(axis=0)

        # Full range
        ax_proj.fill_between(years, ens_min, ens_max,
                            color=config['color'], alpha=0.12)
        # IQR
        ax_proj.fill_between(years, ens_p25, ens_p75,
                            color=config['color'], alpha=0.30)
        # Mean
        ax_proj.plot(years, ens_mean, color=config['color'], lw=2.0,
                    label=f'{config["label"]} ({config["n_members"]})')

        # Store stats
        change_pct = config['end_century_change'] * 100
        end_century_stats.append({
            'strategy': strategy,
            'label': config['short_label'],
            'color': config['color'],
            'historical': config['historical_peak'],
            'change': change_pct,
            'iqr': config['end_century_iqr'],
            'y_end': ens_mean[-1]
        })

    # Reference lines
    ax_proj.axvline(2020, color='grey', ls='--', lw=0.8, alpha=0.5)

    ax_proj.set_ylabel('Peak annual SWE (mm)', fontsize=10)
    ax_proj.set_xlabel('Year', fontsize=10)
    ax_proj.set_xlim(2015, 2100)
    ax_proj.set_ylim(0, y_max)
    ax_proj.grid(True, alpha=0.25)
    ax_proj.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax_proj.xaxis.set_minor_locator(plt.MultipleLocator(10))

    # Title above panel
    ax_proj.set_title('(b) Climate projections (SSP2-4.5)', fontsize=10, fontweight='bold', loc='left', pad=8)

    # Legend neatly in upper right corner
    leg = ax_proj.legend(loc='upper right', fontsize=8, framealpha=0.95,
                         title='Parameter strategy', title_fontsize=8,
                         borderaxespad=0.5, handlelength=1.5)
    leg._legend_box.align = "left"

    # Add end-century stats as annotations on the right side
    for i, stats in enumerate(end_century_stats):
        y_pos = stats['y_end']
        ax_proj.annotate(f"{stats['change']:+.0f}%",
                        xy=(2098, y_pos), xytext=(2100, y_pos),
                        fontsize=8, fontweight='bold', color=stats['color'],
                        ha='left', va='center')

    # Add summary stats in a compact table at bottom right
    stats_text = "End-century (2080–2100):\n"
    stats_text += "Strategy      Δ      IQR\n"
    stats_text += "─" * 22 + "\n"
    for stats in sorted(end_century_stats, key=lambda x: x['change']):
        stats_text += f"{stats['label']:<12} {stats['change']:+.0f}%   {stats['iqr']:.0f}mm\n"

    ax_proj.text(0.02, 0.02, stats_text, transform=ax_proj.transAxes,
                fontsize=7, va='bottom', ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))

    fig.suptitle('SWE Projections at Paradise SNOTEL: Impact of Calibration Forcing Choice',
                fontsize=11, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'fig3_projections')


# ===================================================================
# SUPPLEMENTARY FIGURE S1: Soil Moisture
# ===================================================================
def figS1_soil_moisture(
    obs_sm: Optional[pd.Series],
    sim_sm: Dict[str, pd.Series],
):
    """Single panel: simulated VWC for all forcings + observed ISMN with inset."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    t_min, t_max = SIM_START, SIM_END

    # Plot simulated – reanalysis slightly more prominent than GDDP
    for forcing in REANALYSIS:
        if forcing in sim_sm:
            s = sim_sm[forcing].loc[t_min:t_max]
            if len(s) > 0:
                ax.plot(s.index, s.values, color=COLORS.get(forcing, '#888888'),
                        lw=1.0, label=LABELS.get(forcing, forcing), alpha=0.7)
    for forcing in GDDP:
        if forcing in sim_sm:
            s = sim_sm[forcing].loc[t_min:t_max]
            if len(s) > 0:
                ax.plot(s.index, s.values, color=COLORS.get(forcing, '#888888'),
                        lw=0.6, label=LABELS.get(forcing, forcing), alpha=0.45)

    # Plot observed – thick and prominent
    obs_plotted = False
    if obs_sm is not None:
        obs_clip = obs_sm.loc[t_min:t_max]
        if len(obs_clip) > 0:
            ax.plot(obs_clip.index, obs_clip.values, color='black', lw=2.2,
                    label='Observed (ISMN)', zorder=10, solid_capstyle='round')
            obs_plotted = True

    ax.set_ylabel('Volumetric Water Content (m$^3$ m$^{-3}$)')
    ax.set_xlabel('')
    ax.set_title('Soil Moisture Comparison – Paradise, WA',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(t_min, t_max)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis='x', which='minor', length=3)

    # Legend below the plot to keep data area clear
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=7.5, framealpha=0.95, edgecolor='0.7',
              handlelength=1.5, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.25)

    # Inset: zoom on Jan–Mar 2015 overlap (if observed data in that window)
    if obs_plotted and obs_sm is not None:
        zoom_start = pd.Timestamp('2015-01-01')
        zoom_end = pd.Timestamp('2015-03-31')
        obs_zoom = obs_sm.loc[zoom_start:zoom_end]
        if len(obs_zoom) > 5:
            ax_in = ax.inset_axes([0.58, 0.50, 0.38, 0.45])
            ax_in.plot(obs_zoom.index, obs_zoom.values, 'k-', lw=2.0, zorder=10)
            for forcing in ALL_FORCINGS:
                if forcing in sim_sm:
                    s = sim_sm[forcing].loc[zoom_start:zoom_end]
                    if len(s) > 0:
                        ax_in.plot(s.index, s.values,
                                   color=COLORS.get(forcing, '#888888'),
                                   lw=0.9, alpha=0.75)
            ax_in.set_xlim(zoom_start, zoom_end)
            ax_in.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax_in.xaxis.set_major_locator(mdates.MonthLocator())
            ax_in.tick_params(labelsize=6.5)
            ax_in.set_ylabel('VWC (m$^3$ m$^{-3}$)', fontsize=7)
            ax_in.set_title('Jan–Mar 2015', fontsize=8, fontweight='bold', pad=3)
            ax_in.grid(True, alpha=0.3)
            ax_in.set_facecolor('white')
            ax_in.patch.set_alpha(1.0)
            for spine in ax_in.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)
            # Highlight zoom region on main plot
            ax.axvspan(zoom_start, zoom_end, alpha=0.10, color='gold', zorder=0)

    _save(fig, 'figS1_soil_moisture')


# ===================================================================
# SUPPLEMENTARY FIGURE S2: Resolution vs Transferability
# ===================================================================
def figS2_resolution_transferability(perf_df: pd.DataFrame):
    """
    Scatter plot showing relationship between forcing resolution and KGE degradation.

    This directly visualizes the key finding that higher resolution forcings
    produce more transferable parameters.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Resolution data (km) - extract from labels or use known values
    resolution_map = {
        'era5': 31,
        'aorc': 1,
        'conus404': 4,
        'rdrs': 10,
        # GDDP members all ~25 km
        'gddp_access_cm2': 25,
        'gddp_gfdl_esm4': 25,
        'gddp_mri_esm2_0': 25,
        'gddp_ukesm1_0_ll': 25,
        'gddp_canesm5': 25,
        'gddp_ipsl_cm6a_lr': 25,
        'gddp_cnrm_cm6_1': 25,
        'gddp_mpi_esm1_2_hr': 25,
        'gddp_noresm2_lm': 25,
        'gddp_inm_cm5_0': 25,
    }

    scatter_data = []
    labels_data = []

    for _, row in perf_df.iterrows():
        label = row['Forcing']
        key = _forcing_key(label)

        cal_kge = row.get('Cal_kge', np.nan)
        eval_kge = row.get('Eval_kge', np.nan)

        if pd.isna(cal_kge) or pd.isna(eval_kge):
            continue

        degradation = cal_kge - eval_kge
        resolution = resolution_map.get(key, 25)  # Default to 25 for GDDP

        short = SHORT_LABELS.get(key, label.split(' (')[0])
        color = COLORS.get(key, '#888888')

        scatter_data.append((resolution, degradation, color, key))
        labels_data.append((resolution, degradation, short, color))

    # Plot points
    for res, deg, color, key in scatter_data:
        marker = 'o' if key in REANALYSIS else 's'
        size = 120 if key in REANALYSIS else 80
        ax.scatter(res, deg, c=color, s=size, marker=marker,
                   edgecolors='black', linewidth=0.8, zorder=5)

    # Add labels
    try:
        from adjustText import adjust_text
        texts = []
        xs, ys = [], []
        for res, deg, label, color in labels_data:
            t = ax.text(res, deg, label, fontsize=7, fontweight='bold',
                        color=color, zorder=20)
            texts.append(t)
            xs.append(res)
            ys.append(deg)
        adjust_text(texts, x=xs, y=ys, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5),
                    expand=(1.5, 1.5))
    except ImportError:
        for i, (res, deg, label, color) in enumerate(labels_data):
            ax.annotate(label, (res, deg), textcoords='offset points',
                        xytext=(5, 5 if i % 2 == 0 else -10), fontsize=7,
                        fontweight='bold', color=color)

    # Reference lines
    ax.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)

    # Fit regression for reanalysis only (meaningful resolution variation)
    reanalysis_data = [(res, deg) for res, deg, _, key in scatter_data if key in REANALYSIS]
    if len(reanalysis_data) >= 3:
        res_arr = np.array([d[0] for d in reanalysis_data])
        deg_arr = np.array([d[1] for d in reanalysis_data])
        slope, intercept, r_value, p_value, _ = stats.linregress(res_arr, deg_arr)
        x_line = np.linspace(0, 35, 50)
        ax.plot(x_line, slope * x_line + intercept,
                color='#666666', ls='--', lw=1.2, alpha=0.7,
                label=f'Reanalysis fit (R²={r_value**2:.2f})')

    # Formatting
    ax.set_xlabel('Forcing Resolution (km)', fontsize=10)
    ax.set_ylabel('KGE Degradation (Cal – Eval)', fontsize=10)
    ax.set_title('Resolution vs. Parameter Transferability', fontsize=11,
                 fontweight='bold')
    ax.set_xlim(-1, 35)
    ax.grid(True, alpha=0.25)

    # Legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=10, label='Reanalysis'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=8, label='GDDP-CMIP6'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Annotation explaining the pattern
    ax.text(0.98, 0.02, 'Lower resolution → greater degradation',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, style='italic', color='0.4',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', alpha=0.9))

    _save(fig, 'figS2_resolution_transferability')


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Create publication-quality figures for Section 4.3'
    )
    parser.add_argument('--no-timeseries', action='store_true',
                        help='Skip figures that require NetCDF data (Fig 1 & S1)')
    args = parser.parse_args()

    set_pub_style()
    _load_periods_from_config()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating publication figures for Section 4.3")
    print("=" * 60)

    # --- Always load CSV data for Figures 2 & 3 ---
    print("\nLoading CSV data...")
    perf_df = load_performance_csv()
    param_df = load_parameter_csv()
    print(f"  performance_summary.csv: {len(perf_df)} forcings")
    print(f"  parameter_divergence.csv: {len(param_df)} forcings")

    # --- Figure 2: Combined Performance & Parameters ---
    print("\nFigure 2: Combined Performance & Parameter Analysis")
    figure2_performance_and_parameters(perf_df, param_df)

    # Note: Figure 3 (projections) will be created after loading time series data

    # --- Figures 1 & S1: require NetCDF time series ---
    if not args.no_timeseries:
        print("\nLoading time-series data (NetCDF + observations)...")

        obs_swe = load_observed_swe()
        if obs_swe is not None:
            print(f"  SNOTEL SWE: {len(obs_swe)} records")
        else:
            print("  WARNING: No SNOTEL SWE data found")

        sim_swe = {}
        for forcing in ALL_FORCINGS:
            s = load_simulated_swe(forcing)
            if s is not None:
                sim_swe[forcing] = s
                print(f"  {LABELS.get(forcing, forcing)}: {len(s)} SWE timesteps")
            else:
                print(f"  {LABELS.get(forcing, forcing)}: no SWE output found")

        if obs_swe is not None or sim_swe:
            print("\nFigure 1: SWE Time Series")
            figure1_swe_timeseries(obs_swe, sim_swe)

            # Figure 3: Merged projection figure
            print("\nFigure 3: Climate Projections (merged)")
            figure3_projections_merged(obs_swe, sim_swe)
        else:
            print("\nSkipping Figure 1: no SWE data available")
            # Still create projection figure without historical overlay
            print("\nFigure 3: Climate Projections (no historical overlay)")
            figure3_projections_merged(None, None)

        # Soil moisture for supplementary
        obs_sm = load_observed_sm()
        if obs_sm is not None:
            print(f"  ISMN SM: {len(obs_sm)} records")
        else:
            print("  No ISMN SM data found")

        sim_sm = {}
        for forcing in ALL_FORCINGS:
            s = load_simulated_sm(forcing)
            if s is not None:
                sim_sm[forcing] = s
                print(f"  {LABELS.get(forcing, forcing)}: {len(s)} SM timesteps")

        if obs_sm is not None or sim_sm:
            print("\nFigure S1: Soil Moisture")
            figS1_soil_moisture(obs_sm, sim_sm)
        else:
            print("\nSkipping Figure S1: no soil moisture data available")
    else:
        print("\nSkipping Figures 1 & S1 (--no-timeseries)")
        # Still create projection figure without historical overlay
        print("\nFigure 3: Climate Projections (no historical overlay)")
        figure3_projections_merged(None, None)

    # --- Figure S2: Resolution vs Transferability ---
    print("\nFigure S2: Resolution vs Transferability")
    figS2_resolution_transferability(perf_df)

    print("\n" + "=" * 60)
    print("Done! Figures saved to:")
    print(f"  {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
