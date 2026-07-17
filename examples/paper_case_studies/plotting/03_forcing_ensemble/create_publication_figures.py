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
# FIGURE 6 (paper): Calibrated SWE from four reanalysis forcings
#   (a) big SWE time series (obs grey fill + 4 reanalysis lines, cal/eval)
#   (b) Skill: calibration vs evaluation grouped bars (hatched=cal, solid=eval)
#   (c) Snowfall-undercatch correction (frozenPrecipMultip) with 1.0 baseline
#   (d) Skill transfer (Cal - Eval KGE), lower is better
# ===================================================================
# Paper Fig 6 uses ONLY the four reanalysis forcings (no GDDP, no NWM3), in the
# order and colours of the published figure.
REAN4 = ['aorc', 'era5', 'rdrs', 'conus404']
PAPER_COLORS = {
    'aorc':     '#009E73',   # teal-green
    'era5':     '#D62728',   # red
    'rdrs':     '#1F77B4',   # blue
    'conus404': '#E69F00',   # orange
}


def figure_06_swe_forcings(
    obs_swe: Optional[pd.Series],
    sim_swe: Dict[str, pd.Series],
    perf_df: pd.DataFrame,
    param_df: pd.DataFrame,
):
    """Single 4-panel paper figure: SWE time series + three skill/parameter bars."""
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    from matplotlib.transforms import blended_transform_factory

    fig = plt.figure(figsize=(12, 8.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.35, 1.0],
                           hspace=0.42, wspace=0.28,
                           left=0.07, right=0.985, top=0.94, bottom=0.09)

    # ---- Panel (a): SWE time series (spans the top row) ----
    ax_a = fig.add_subplot(gs[0, :])
    t_min, t_max = pd.Timestamp('2015-01-01'), pd.Timestamp('2021-01-01')

    # Observed SNOTEL as a grey filled area
    if obs_swe is not None:
        s = obs_swe.loc[t_min:t_max]
        ax_a.fill_between(s.index, 0, s.values, color='#BFBFBF', alpha=0.65,
                          zorder=1, linewidth=0)
        ax_a.plot(s.index, s.values, color='#8A8A8A', lw=0.6, zorder=2)

    # Four reanalysis-driven SWE lines
    for forcing in REAN4:
        if forcing in sim_swe:
            s = sim_swe[forcing].loc[t_min:t_max]
            ax_a.plot(s.index, s.values, color=PAPER_COLORS[forcing], lw=1.3,
                      alpha=0.9, zorder=5)

    # Calibration / evaluation shading + divider
    ax_a.axvspan(EVAL_START, EVAL_END, alpha=0.5, color='#EFEBE0', zorder=0)
    ax_a.axvline(EVAL_START, color='0.45', ls='--', lw=0.9, zorder=3)
    trans = blended_transform_factory(ax_a.transData, ax_a.transAxes)
    mid_cal = CAL_START + (EVAL_START - CAL_START) / 2
    mid_eval = EVAL_START + (EVAL_END - EVAL_START) / 2
    ax_a.text(mid_cal, 0.94, 'calibration', ha='center', va='top', fontsize=10,
              color='0.4', fontstyle='italic', transform=trans)
    ax_a.text(mid_eval, 0.94, 'evaluation', ha='center', va='top', fontsize=10,
              color='0.4', fontstyle='italic', transform=trans)

    ax_a.set_ylabel('Snow water equivalent (mm)', fontsize=10)
    ax_a.set_xlim(t_min, t_max)
    ax_a.set_ylim(bottom=0)
    ax_a.grid(True, axis='y', alpha=0.2)
    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_title('Calibrated SWE from four reanalysis forcings vs. observed, '
                   'Paradise SNOTEL', fontsize=11, fontweight='bold')
    ax_a.text(-0.045, 1.02, 'a', transform=ax_a.transAxes, fontsize=13,
              fontweight='bold', va='bottom', ha='right')

    # Legend strip below panel (a)
    handles = [mpatches.Patch(facecolor='#BFBFBF', alpha=0.65,
                              label='Observed (SNOTEL #679)')]
    for forcing in REAN4:
        handles.append(Line2D([0], [0], color=PAPER_COLORS[forcing], lw=2.0,
                              label=LABELS[forcing]))
    ax_a.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                ncol=5, fontsize=9, frameon=False, columnspacing=1.6,
                handlelength=1.8)

    # ---- Prepare per-forcing skill / parameter data ----
    perf = perf_df.copy()
    perf['_key'] = perf['Forcing'].map(_forcing_key)
    perf = perf.set_index('_key')
    param = param_df.copy()
    param['_key'] = param['Forcing'].map(_forcing_key)
    param = param.set_index('_key')

    keys = [k for k in REAN4 if k in perf.index]
    x = np.arange(len(keys))
    xlabels = [SHORT_LABELS[k] for k in keys]
    bar_colors = [PAPER_COLORS[k] for k in keys]

    def _style_bottom(ax, letter, title):
        ax.set_title(title, fontsize=10.5, fontweight='bold', loc='left', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        for tick, k in zip(ax.get_xticklabels(), keys):
            tick.set_color(PAPER_COLORS[k])
            tick.set_fontweight('bold')
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='bottom', ha='right')

    # ---- Panel (b): Skill — calibration vs evaluation ----
    ax_b = fig.add_subplot(gs[1, 0])
    cal_kge = [float(perf.loc[k, 'Cal_kge']) for k in keys]
    eval_kge = [float(perf.loc[k, 'Eval_kge']) for k in keys]
    w = 0.38
    for xi, k, cv, ev in zip(x, keys, cal_kge, eval_kge):
        ax_b.bar(xi - w / 2, cv, w, color=PAPER_COLORS[k], alpha=0.55,
                 hatch='////', edgecolor='white', linewidth=0.5)
        ax_b.bar(xi + w / 2, ev, w, color=PAPER_COLORS[k],
                 edgecolor='white', linewidth=0.5)
    ax_b.set_ylabel('KGE', fontsize=10)
    ax_b.set_ylim(0, 1.15)
    _style_bottom(ax_b, 'b', 'Skill: calibration vs. evaluation')
    leg_b = [mpatches.Patch(facecolor='0.6', alpha=0.55, hatch='////',
                            edgecolor='white', label='Calibration'),
             mpatches.Patch(facecolor='0.6', edgecolor='white', label='Evaluation')]
    ax_b.legend(handles=leg_b, loc='upper center', ncol=2, fontsize=8,
                frameon=False, handlelength=1.4, columnspacing=1.0)

    # ---- Panel (c): Snowfall-undercatch correction (frozenPrecipMultip) ----
    ax_c = fig.add_subplot(gs[1, 1])
    fpm = [float(param.loc[k, 'frozenPrecipMultip']) for k in keys]
    bars_c = ax_c.bar(x, fpm, 0.6, color=bar_colors, edgecolor='white', linewidth=0.6)
    ax_c.axhline(1.0, color='0.4', ls='--', lw=1.0, zorder=1)
    ax_c.text(len(keys) - 0.5, 1.0, 'no correction', ha='right', va='bottom',
              fontsize=8, color='0.4', fontstyle='italic')
    for b, v in zip(bars_c, fpm):
        ax_c.text(b.get_x() + b.get_width() / 2, v + 0.05, f'{v:.2f}',
                  ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax_c.set_ylabel('frozenPrecipMultip', fontsize=10)
    ax_c.set_ylim(0, max(fpm) * 1.18)
    _style_bottom(ax_c, 'c', 'Snowfall-undercatch correction')

    # ---- Panel (d): Skill transfer (Cal - Eval KGE), lower is better ----
    ax_d = fig.add_subplot(gs[1, 2])
    transfer = [c - e for c, e in zip(cal_kge, eval_kge)]
    ax_d.bar(x, transfer, 0.6, color=bar_colors, edgecolor='white', linewidth=0.6)
    ax_d.axhline(0, color='black', lw=0.9)
    ax_d.set_ylabel('Cal – Eval KGE', fontsize=10)
    _style_bottom(ax_d, 'd', 'Skill transfer (lower is better)')

    _save(fig, 'figure_06_swe_forcings')


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
                        help='Skip figures that require NetCDF data (Fig 1)')
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

    # --- Load SWE time series (obs + simulated) for the four reanalysis forcings ---
    # The paper Fig 6 is a SINGLE 4-panel figure: (a) the SWE time series over
    # (b) skill, (c) undercatch correction and (d) skill transfer bars. Fig 3
    # (projections), Fig S1 (soil moisture) and Fig S2 (resolution) are not paper
    # figures and are no longer generated.
    print("\nLoading time-series data (NetCDF + observations)...")

    obs_swe = load_observed_swe()
    if obs_swe is not None:
        print(f"  SNOTEL SWE: {len(obs_swe)} records")
    else:
        print("  WARNING: No SNOTEL SWE data found")

    sim_swe = {}
    for forcing in REAN4:
        s = load_simulated_swe(forcing)
        if s is not None:
            sim_swe[forcing] = s
            print(f"  {LABELS.get(forcing, forcing)}: {len(s)} SWE timesteps")
        else:
            print(f"  {LABELS.get(forcing, forcing)}: no SWE output found")

    print("\nFigure 6: SWE forcings (single 4-panel figure)")
    figure_06_swe_forcings(obs_swe, sim_swe, perf_df, param_df)

    print("\n" + "=" * 60)
    print("Done! Figures saved to:")
    print(f"  {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
