#!/usr/bin/env python3
"""
Create publication-quality figures for Section 4.3 Forcing Ensemble.

Generates 3 main figures + 1 supplementary figure from pre-computed CSV
summary tables and (optionally) NetCDF simulation outputs.

Usage:
    python create_publication_figures.py
    python create_publication_figures.py --no-timeseries   # skip NetCDF-dependent figs
"""

import argparse
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
STUDY_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = STUDY_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFIGS_DIR = STUDY_DIR / "configs"
SYMFLUENCE_DATA_DIR = Path(
    "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data"
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

REANALYSIS = ['era5', 'aorc', 'conus404', 'rdrs']
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
        cfg_file = CONFIGS_DIR / "config_paradise_aorc.yaml"
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
    return SYMFLUENCE_DATA_DIR / f"domain_paradise_snotel_wa_{forcing}"


def load_observed_swe() -> Optional[pd.Series]:
    """Load SNOTEL SWE observations (mm). Returns a DatetimeIndex Series."""
    for forcing in ALL_FORCINGS:
        d = _domain_dir(forcing)
        for base in [d / "observations" / "snow" / "swe" / "preprocessed",
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
    opt_path = d / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation"
    nc_files = list(opt_path.glob("*_day.nc")) if opt_path.exists() else []
    if not nc_files:
        sim_path = d / "simulations" / "SUMMA"
        if sim_path.exists():
            nc_files = list(sim_path.glob("*_day.nc")) + list(sim_path.glob("*output*.nc"))
    if not nc_files:
        return None
    try:
        ds = xr.open_dataset(nc_files[0])
        for var in ['scalarSWE', 'SWE', 'swe', 'snow_water_equivalent']:
            if var in ds.data_vars:
                swe = ds[var].values.flatten()
                time = pd.to_datetime(ds['time'].values)
                return pd.Series(swe, index=time, name=forcing)
    except Exception:
        pass
    return None


def load_observed_sm() -> Optional[pd.Series]:
    """Load ISMN soil moisture observations. Returns DatetimeIndex Series (VWC)."""
    for forcing in ALL_FORCINGS:
        d = _domain_dir(forcing)
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
    opt_path = d / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation"
    nc_files = list(opt_path.glob("*_day.nc")) if opt_path.exists() else []
    if not nc_files:
        sim_path = d / "simulations" / "SUMMA"
        if sim_path.exists():
            nc_files = list(sim_path.glob("*_day.nc"))
    if not nc_files:
        return None
    try:
        ds = xr.open_dataset(nc_files[0])
        if 'mLayerVolFracLiq' not in ds or 'mLayerDepth' not in ds:
            return None
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
# FIGURE 1: SWE Time Series (2-row panel)
# ===================================================================
def figure1_swe_timeseries(
    obs_swe: Optional[pd.Series],
    sim_swe: Dict[str, pd.Series],
):
    """Two-panel SWE time-series: (a) reanalysis, (b) GDDP."""
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.5, 5.5), sharex=True,
        gridspec_kw={'hspace': 0.12}
    )

    t_min, t_max = SIM_START, SIM_END

    # --- helper: shade cal/eval with high-contrast styling ---
    def _shade(ax, label=True):
        from matplotlib.transforms import blended_transform_factory

        # Distinct colours: warm orange-tan for calibration, cool steel-blue for eval
        cal_color = '#FDDBC7'   # warm peach
        eval_color = '#D1E5F0'  # cool sky-blue

        ax.axvspan(CAL_START, CAL_END, alpha=0.35, color=cal_color,
                   zorder=0, linewidth=0)
        ax.axvspan(EVAL_START, EVAL_END, alpha=0.35, color=eval_color,
                   zorder=0, linewidth=0)

        # Vertical boundary lines at period transitions
        for ts in [CAL_START, CAL_END, EVAL_START, EVAL_END]:
            ax.axvline(ts, color='0.45', ls=':', lw=0.6, zorder=1)

        # Text annotations at top
        if label:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            mid_cal = CAL_START + (CAL_END - CAL_START) / 2
            mid_eval = EVAL_START + (EVAL_END - EVAL_START) / 2
            ax.text(mid_cal, 0.97, 'Calibration', ha='center', va='top',
                    fontsize=8.5, color='#B35806', fontweight='bold',
                    fontstyle='italic', transform=trans,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.7))
            ax.text(mid_eval, 0.97, 'Evaluation', ha='center', va='top',
                    fontsize=8.5, color='#2166AC', fontweight='bold',
                    fontstyle='italic', transform=trans,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.7))

    # ---- Panel (a): Reanalysis ----
    ax_a.text(0.015, 0.95, '(a)', transform=ax_a.transAxes,
              fontsize=11, fontweight='bold', va='top')

    if obs_swe is not None:
        s = obs_swe.loc[t_min:t_max]
        ax_a.plot(s.index, s.values, color=COLORS['observed'], lw=2.0,
                  label='Observed (SNOTEL)', zorder=10, solid_capstyle='round')

    for forcing in REANALYSIS:
        if forcing in sim_swe:
            s = sim_swe[forcing].loc[t_min:t_max]
            ax_a.plot(s.index, s.values, color=COLORS[forcing],
                      lw=1.3, label=LABELS[forcing], alpha=0.85)

    ax_a.set_ylabel('SWE (mm)')
    ax_a.grid(True, alpha=0.25)
    _shade(ax_a)

    # ---- Panel (b): GDDP ----
    ax_b.text(0.015, 0.95, '(b)', transform=ax_b.transAxes,
              fontsize=11, fontweight='bold', va='top')

    if obs_swe is not None:
        s = obs_swe.loc[t_min:t_max]
        ax_b.plot(s.index, s.values, color=COLORS['observed'], lw=2.0,
                  label='Observed (SNOTEL)', zorder=10, solid_capstyle='round')

    # Collect GDDP series for envelope
    gddp_frames = []
    for forcing in GDDP:
        if forcing in sim_swe:
            s = sim_swe[forcing].loc[t_min:t_max]
            ax_b.plot(s.index, s.values, color=COLORS[forcing],
                      lw=0.7, alpha=0.6, label=LABELS[forcing])
            gddp_frames.append(s)

    # Ensemble envelope + mean
    if len(gddp_frames) >= 2:
        gddp_df = pd.concat(gddp_frames, axis=1)
        env_min = gddp_df.min(axis=1)
        env_max = gddp_df.max(axis=1)
        env_mean = gddp_df.mean(axis=1)
        ax_b.fill_between(env_min.index, env_min.values, env_max.values,
                          color=COLORS['gddp_envelope'], alpha=0.35,
                          label='GDDP envelope', zorder=1)
        ax_b.plot(env_mean.index, env_mean.values, color=COLORS['gddp_mean'],
                  lw=1.0, ls='--', label='GDDP mean', zorder=8)

    ax_b.set_ylabel('SWE (mm)')
    ax_b.set_xlabel('')
    ax_b.grid(True, alpha=0.25)
    _shade(ax_b)

    # Unified y-axis limits across both panels
    y_max = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    ax_a.set_ylim(0, y_max)
    ax_b.set_ylim(0, y_max)

    # x-axis formatting – clean year ticks with minor month ticks
    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_b.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax_b.tick_params(axis='x', which='minor', length=3)
    ax_a.set_xlim(t_min, t_max)
    plt.setp(ax_b.get_xticklabels(), rotation=0, ha='center')

    # Shared legend below panels – two rows: reanalysis then GDDP
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    # Combine, de-duplicate (SNOTEL appears in both)
    seen = set()
    handles, labels = [], []
    for h, l in list(zip(handles_a, labels_a)) + list(zip(handles_b, labels_b)):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    # Determine ncol to keep it compact (aim for 2 rows)
    ncol = max(4, (len(handles) + 1) // 2)
    fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=7,
               frameon=True, framealpha=0.95, edgecolor='0.8',
               handlelength=1.6, handletextpad=0.5, columnspacing=0.8,
               bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(bottom=0.20)

    _save(fig, 'fig1_swe_timeseries')


# ===================================================================
# FIGURE 2: Performance Metrics  (heatmap + degradation bars)
# ===================================================================
def figure2_performance_metrics(perf_df: pd.DataFrame):
    """Heatmap of metrics + horizontal bar chart of KGE degradation."""
    fig = plt.figure(figsize=(7.5, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.40)

    # ---- Build data matrices ----
    # Rows: forcings with Cal & Eval grouped
    skill_cols = ['kge', 'corr', 'nse']
    error_col = 'rmse'
    all_metric_cols = skill_cols + [error_col]
    display_names = {'kge': 'KGE', 'corr': 'Corr', 'nse': 'NSE', 'rmse': 'RMSE\n(mm)'}

    row_labels = []
    data_rows = []
    forcings_order = []

    for _, row in perf_df.iterrows():
        label = row['Forcing']
        key = _forcing_key(label)
        short = SHORT_LABELS.get(key, label.split(' (')[0])

        cal_vals = [row.get(f'Cal_{m}', np.nan) for m in all_metric_cols]
        eval_vals = [row.get(f'Eval_{m}', np.nan) for m in all_metric_cols]

        row_labels.append(f"{short} – Cal")
        data_rows.append(cal_vals)
        row_labels.append(f"{short} – Eval")
        data_rows.append(eval_vals)
        forcings_order.append(key)

    data = np.array(data_rows, dtype=float)

    # ---- Panel (a): Heatmap ----
    ax_hm = fig.add_subplot(gs[0])
    ax_hm.text(-0.02, 1.05, '(a)', transform=ax_hm.transAxes,
               fontsize=11, fontweight='bold', va='top')

    n_rows, n_cols = data.shape

    # Separate colormaps: RdYlGn for skill metrics, YlOrRd_r for RMSE
    # We'll draw cell-by-cell for mixed colormaps
    from matplotlib.colors import Normalize
    cmap_skill = plt.cm.RdYlGn
    cmap_rmse = plt.cm.YlOrRd_r

    # Normalizations
    skill_data = data[:, :3]  # KGE, Corr, NSE
    rmse_data = data[:, 3]    # RMSE

    # Skill: vmin=-1, vmax=1 (appropriate for KGE/NSE/Corr)
    norm_skill = Normalize(vmin=-1, vmax=1)
    # RMSE: lower is better
    rmse_valid = rmse_data[~np.isnan(rmse_data)]
    if len(rmse_valid) > 0:
        norm_rmse = Normalize(vmin=rmse_valid.min() * 0.8, vmax=rmse_valid.max() * 1.05)
    else:
        norm_rmse = Normalize(vmin=0, vmax=1500)

    # Draw cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                color = '#F0F0F0'
            elif j < 3:  # skill metric
                color = cmap_skill(norm_skill(val))
            else:  # RMSE
                color = cmap_rmse(norm_rmse(val))
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                 facecolor=color, edgecolor='white', lw=0.8)
            ax_hm.add_patch(rect)
            # Annotate
            if not np.isnan(val):
                txt = f'{val:.2f}' if abs(val) < 10 else f'{val:.0f}'
                # Dark text on light bg, light on dark
                brightness = np.mean(mpl.colors.to_rgb(color))
                tc = 'white' if brightness < 0.45 else 'black'
                ax_hm.text(j + 0.5, n_rows - 1 - i + 0.5, txt,
                           ha='center', va='center', fontsize=7.5, color=tc,
                           fontweight='bold')
            else:
                ax_hm.text(j + 0.5, n_rows - 1 - i + 0.5, '–',
                           ha='center', va='center', fontsize=8, color='grey')

    ax_hm.set_xlim(0, n_cols)
    ax_hm.set_ylim(0, n_rows)
    ax_hm.set_xticks([c + 0.5 for c in range(n_cols)])
    ax_hm.set_xticklabels([display_names[m] for m in all_metric_cols], fontsize=9)
    ax_hm.set_yticks([r + 0.5 for r in range(n_rows)])
    ax_hm.set_yticklabels(row_labels[::-1], fontsize=8)
    ax_hm.tick_params(length=0)
    ax_hm.set_title('Performance Metrics', fontsize=11, fontweight='bold', pad=10)

    # Add thin horizontal lines separating forcing groups
    for i in range(2, n_rows, 2):
        ax_hm.axhline(y=i, color='grey', lw=0.4, alpha=0.6)

    # ---- Panel (b): KGE Degradation bars ----
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.text(-0.08, 1.05, '(b)', transform=ax_bar.transAxes,
                fontsize=11, fontweight='bold', va='top')

    # Compute degradation for each forcing
    degrad_data = []
    for _, row in perf_df.iterrows():
        label = row['Forcing']
        key = _forcing_key(label)
        short = SHORT_LABELS.get(key, label.split(' (')[0])
        cal_kge = row.get('Cal_kge', np.nan)
        eval_kge = row.get('Eval_kge', np.nan)
        if pd.notna(cal_kge) and pd.notna(eval_kge):
            degrad_data.append((short, cal_kge - eval_kge, key))

    # Sort worst-to-best (highest degradation at top)
    degrad_data.sort(key=lambda x: x[1], reverse=True)

    bar_labels = [d[0] for d in degrad_data]
    bar_vals = [d[1] for d in degrad_data]
    bar_keys = [d[2] for d in degrad_data]

    # Color: green for improvement (negative), vermillion for degradation (positive)
    bar_colors = []
    for v in bar_vals:
        if v <= 0:
            bar_colors.append('#009E73')  # green = improvement
        else:
            # Interpolate from amber to vermillion based on degradation magnitude
            t = min(v / 1.0, 1.0)
            bar_colors.append('#D55E00' if t > 0.3 else '#E69F00')

    y_pos = range(len(bar_labels))
    bars = ax_bar.barh(y_pos, bar_vals, color=bar_colors, edgecolor='black',
                       linewidth=0.5, height=0.6)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=8.5)
    ax_bar.set_xlabel('KGE Degradation\n(Cal – Eval)', fontsize=9)
    ax_bar.axvline(x=0, color='black', lw=0.8)
    ax_bar.set_title('Transferability', fontsize=11, fontweight='bold', pad=10)
    ax_bar.grid(True, axis='x')
    ax_bar.invert_yaxis()

    # Value annotations
    for bar, val in zip(bars, bar_vals):
        if val >= 0:
            x_off = 0.02
            ha = 'left'
        else:
            x_off = -0.02
            ha = 'right'
        ax_bar.text(val + x_off, bar.get_y() + bar.get_height() / 2,
                    f'{val:+.2f}', ha=ha, va='center', fontsize=7.5,
                    fontweight='bold')

    # Pad x-limits so annotations aren't clipped
    xmin, xmax = ax_bar.get_xlim()
    ax_bar.set_xlim(xmin - 0.10, xmax + 0.15)

    _save(fig, 'fig2_performance_metrics')


# ===================================================================
# FIGURE 3: Parameter Divergence  (heatmap + 2 scatters)
# ===================================================================
def figure3_parameter_divergence(param_df: pd.DataFrame):
    """Z-score heatmap + frozenPrecipMultip scatter + distortion scatter."""

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.38, wspace=0.35)

    # Parameters to display (11 calibrated parameters)
    param_cols = ['tempCritRain', 'tempRangeTimestep', 'frozenPrecipMultip',
                  'albedoMax', 'albedoMinWinter', 'albedoDecayRate',
                  'constSnowDen', 'mw_exp', 'k_snow', 'z0Snow',
                  'routingGammaScale']
    param_cols = [p for p in param_cols if p in param_df.columns]

    param_labels = {
        'tempCritRain': 'T$_{crit}$',
        'tempRangeTimestep': 'T$_{range}$',
        'frozenPrecipMultip': 'fPM',
        'albedoMax': r'$\alpha_{max}$',
        'albedoMinWinter': r'$\alpha_{min}$',
        'albedoDecayRate': r'$\alpha_{decay}$',
        'constSnowDen': r'$\rho_{snow}$',
        'mw_exp': 'MW$_{exp}$',
        'k_snow': 'k$_{snow}$',
        'z0Snow': 'z$_0$',
        'routingGammaScale': r'$\gamma_{route}$',
    }

    # Compensatory parameter flags
    compensatory = {'frozenPrecipMultip', 'tempRangeTimestep', 'mw_exp'}

    # Sort by KGE degradation (worst at top)
    df = param_df.copy()
    df['_key'] = df['Forcing'].map(lambda x: _forcing_key(x))
    df['_short'] = df['_key'].map(lambda k: SHORT_LABELS.get(k, k))

    # Sort by KGE_Degradation descending; NaN at bottom
    df = df.sort_values('KGE_Degradation', ascending=False, na_position='last')

    raw_values = df[param_cols].values.astype(float)
    n_forcings, n_params = raw_values.shape

    # Z-score normalize each parameter column
    col_mean = np.nanmean(raw_values, axis=0)
    col_std = np.nanstd(raw_values, axis=0)
    col_std[col_std == 0] = 1.0
    z_scores = (raw_values - col_mean) / col_std

    # ---- Panel (a): Z-score heatmap spanning full width ----
    ax_hm = fig.add_subplot(gs[0, :])
    ax_hm.text(-0.02, 1.05, '(a)', transform=ax_hm.transAxes,
               fontsize=11, fontweight='bold', va='top')

    # Diverging colormap centred at 0
    vabs = np.nanmax(np.abs(z_scores))
    vabs = max(vabs, 0.5)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = plt.cm.RdBu_r

    im = ax_hm.imshow(z_scores, cmap=cmap, norm=norm, aspect='auto')

    # Annotate with raw values (compact formatting for readability)
    def _fmt_val(v):
        """Format a parameter value for heatmap annotation."""
        av = abs(v)
        if av >= 1e6:
            return f'{v/1e6:.1f}M'
        if av >= 1e4:
            return f'{v/1e3:.0f}k'
        if av >= 1000:
            return f'{v:.0f}'
        if av >= 10:
            return f'{v:.1f}'
        if av >= 1:
            return f'{v:.2f}'
        return f'{v:.3f}'

    for i in range(n_forcings):
        for j in range(n_params):
            raw = raw_values[i, j]
            if np.isnan(raw):
                continue
            txt = _fmt_val(raw)
            brightness = np.mean(mpl.colors.to_rgb(cmap(norm(z_scores[i, j]))))
            tc = 'white' if brightness < 0.45 else 'black'
            ax_hm.text(j, i, txt, ha='center', va='center',
                       fontsize=5.5, color=tc, fontweight='bold')

    ax_hm.set_xticks(range(n_params))
    xlabels = [param_labels.get(p, p) for p in param_cols]
    ax_hm.set_xticklabels(xlabels, fontsize=8)
    ax_hm.set_yticks(range(n_forcings))
    ax_hm.set_yticklabels(df['_short'].values, fontsize=9)

    # Highlight compensatory columns
    for j, p in enumerate(param_cols):
        if p in compensatory:
            ax_hm.get_xticklabels()[j].set_color('#D55E00')
            ax_hm.get_xticklabels()[j].set_fontweight('bold')

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.02)
    cbar.set_label('Z-score', fontsize=9)
    ax_hm.set_title('Calibrated Parameters (Z-score normalized)',
                     fontsize=10, fontweight='bold', pad=8)

    # ---- Build metrics lookup for scatter panels ----
    degrad_map = {}  # key -> KGE_Degradation
    fpm_map = {}     # key -> frozenPrecipMultip
    for _, row in df.iterrows():
        key = row['_key']
        degrad_map[key] = row.get('KGE_Degradation', np.nan)
        fpm_map[key] = row.get('frozenPrecipMultip', np.nan)

    # ---- Helper: non-overlapping scatter labels ----
    def _add_scatter_labels(ax, texts_data):
        """Add non-overlapping labels using adjustText (with fallback)."""
        try:
            from adjustText import adjust_text
            texts = []
            xs, ys = [], []
            for x, y, label, color in texts_data:
                t = ax.text(x, y, label, fontsize=7, fontweight='bold',
                            color=color, zorder=20)
                texts.append(t)
                xs.append(x)
                ys.append(y)
            adjust_text(texts, x=xs, y=ys, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5),
                        expand=(2.0, 2.0),
                        force_text=(1.5, 1.5),
                        force_points=(1.0, 1.0),
                        ensure_inside_axes=True,
                        max_move=None)
        except ImportError:
            for i, (x, y, label, color) in enumerate(texts_data):
                dy = 8 if (i % 2 == 0) else -10
                dx = 6 + (i % 3) * 4
                ax.annotate(label, (x, y), textcoords='offset points',
                            xytext=(dx, dy), fontsize=7,
                            fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='-', color='0.6',
                                            lw=0.4, shrinkB=3))

    # ---- Panel (b): frozenPrecipMultip vs KGE degradation ----
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.text(-0.08, 1.05, '(b)', transform=ax_b.transAxes,
              fontsize=11, fontweight='bold', va='top')

    b_labels = []
    for _, row in df.iterrows():
        key = row['_key']
        fpm = row.get('frozenPrecipMultip', np.nan)
        deg = row.get('KGE_Degradation', np.nan)
        if pd.isna(fpm) or pd.isna(deg):
            continue
        ax_b.scatter(fpm, deg, c=COLORS.get(key, '#888888'), s=90,
                     edgecolors='black', linewidth=0.6, zorder=5)
        b_labels.append((fpm, deg, row['_short'], COLORS.get(key, '#444444')))

    _add_scatter_labels(ax_b, b_labels)

    ax_b.axhline(0, color='grey', ls='--', lw=0.6, alpha=0.5)
    ax_b.axvline(1.0, color='grey', ls='--', lw=0.6, alpha=0.5)
    ax_b.set_xlabel('frozenPrecipMultip (–)', fontsize=9)
    ax_b.set_ylabel('KGE Degradation (Cal – Eval)', fontsize=9)
    ax_b.set_title('Precip. Correction vs.\nTransferability', fontsize=10,
                    fontweight='bold')
    ax_b.grid(True, alpha=0.25)
    ax_b.margins(0.15)

    # ---- Panel (c): Composite distortion vs KGE degradation ----
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.text(-0.08, 1.05, '(c)', transform=ax_c.transAxes,
              fontsize=11, fontweight='bold', va='top')

    # Compute composite distortion: mean |z-score| of compensatory parameters
    comp_indices = [j for j, p in enumerate(param_cols) if p in compensatory]

    scatter_x, scatter_y, scatter_k = [], [], []
    c_labels = []
    for i_row in range(n_forcings):
        key = df.iloc[i_row]['_key']
        deg = df.iloc[i_row].get('KGE_Degradation', np.nan)
        if pd.isna(deg):
            continue
        comp_z = np.abs(z_scores[i_row, comp_indices])
        if np.all(np.isnan(comp_z)):
            continue
        distortion = np.nanmean(comp_z)
        scatter_x.append(distortion)
        scatter_y.append(deg)
        scatter_k.append(key)
        ax_c.scatter(distortion, deg, c=COLORS.get(key, '#888888'), s=90,
                     edgecolors='black', linewidth=0.6, zorder=5)
        c_labels.append((distortion, deg, df.iloc[i_row]['_short'],
                         COLORS.get(key, '#444444')))

    _add_scatter_labels(ax_c, c_labels)

    # Regression line + R²
    if len(scatter_x) >= 3:
        sx = np.array(scatter_x)
        sy = np.array(scatter_y)
        slope, intercept, r_value, p_value, _ = stats.linregress(sx, sy)
        x_line = np.linspace(sx.min(), sx.max(), 50)
        ax_c.plot(x_line, slope * x_line + intercept,
                  color='#666666', ls='--', lw=0.9, alpha=0.6)
        sig_str = '' if p_value < 0.05 else ' (n.s.)'
        ax_c.text(0.05, 0.95,
                  f'$R^2$ = {r_value**2:.2f}{sig_str}\n$p$ = {p_value:.2f}',
                  transform=ax_c.transAxes, ha='left', va='top',
                  fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', edgecolor='0.7',
                                        alpha=0.9))

    ax_c.axhline(0, color='grey', ls='--', lw=0.6, alpha=0.5)
    ax_c.set_xlabel('Composite Distortion\n(mean |z| of compensatory params)', fontsize=9)
    ax_c.set_ylabel('KGE Degradation (Cal – Eval)', fontsize=9)
    ax_c.set_title('Parameter Distortion vs.\nGeneralization Loss', fontsize=10,
                    fontweight='bold')
    ax_c.grid(True, alpha=0.25)
    ax_c.margins(0.15)

    _save(fig, 'fig3_parameter_divergence')


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

    # --- Figure 2: Performance Metrics ---
    print("\nFigure 2: Performance Metrics")
    figure2_performance_metrics(perf_df)

    # --- Figure 3: Parameter Divergence ---
    print("\nFigure 3: Parameter Divergence")
    figure3_parameter_divergence(param_df)

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
        else:
            print("\nSkipping Figure 1: no SWE data available")

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

    print("\n" + "=" * 60)
    print("Done! Figures saved to:")
    print(f"  {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
