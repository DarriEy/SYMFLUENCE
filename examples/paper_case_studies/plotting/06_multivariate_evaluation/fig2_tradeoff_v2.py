#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Figure 2: Calibration Trade-off - Redesigned
Multivariate Evaluation Section

Clear narrative showing:
- Panel (a): TWS time series - TWS-only excels here
- Panel (b): Streamflow time series - TWS-only fails catastrophically
- Panel (c): Trade-off space showing the fundamental tension

Design: Side-by-side comparison revealing why multi-objective is needed
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyBboxPatch
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

# Experiment directories (all re-run with consistent 17-parameter set, 2000 evals)
EXP1_DIR = DATA_DIR / "optimization/SUMMA/dds_bow_exp1_streamflow_only"   # Streamflow-only DDS
EXP2_DIR = DATA_DIR / "optimization/SUMMA/dds_bow_exp2_tws_only"          # TWS-only DDS
EXP3_DIR = DATA_DIR / "optimization/SUMMA/nsga-ii_bow_exp3_joint"  # Joint NSGA-II (shipped eid bow_exp3_joint)
EXP4_DIR = DATA_DIR / "optimization/SUMMA/moead_bow_exp4_moead_joint"     # Joint MOEA/D

# Observation data
GRACE_DATA = OBS_DIR / "grace/preprocessed/Bow_at_Banff_multivar_grace_tws_processed.csv"
STREAMFLOW_DATA = OBS_DIR / "streamflow/preprocessed/Bow_at_Banff_multivar_streamflow_processed.csv"

CALIBRATION_START = '2004-01-01'
CALIBRATION_END = '2010-12-31'
EVALUATION_START = '2011-01-01'
EVALUATION_END = '2017-12-31'
GRACE_BASELINE_START = '2004-01-01'
GRACE_BASELINE_END = '2009-12-31'

CATCHMENT_AREA_KM2 = 2210

# Experiment colors - colorblind friendly
EXP_COLORS = {
    'exp1': '#0072B2',   # Blue - Streamflow-only
    'exp2': '#D55E00',   # Vermillion - TWS-only
    'exp3': '#009E73',   # Green - Joint NSGA-II
    'exp4': '#CC79A7',   # Pink - Joint MOEA/D
    'obs': '#333333',    # Dark gray - Observations
}

EXP_LABELS = {
    'exp1': 'Q-only',
    'exp2': 'TWS-only',
    'exp3': 'NSGA-II',
    'exp4': 'MOEA/D',
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_experiment_results(exp_dir):
    """Load evaluation results for an experiment."""
    # Try final_evaluation.json first
    eval_files = list(exp_dir.glob("*final_evaluation.json"))
    if eval_files:
        with open(eval_files[0]) as f:
            return json.load(f)
    # Fall back to best_params.json (for MOEA/D)
    best_files = list(exp_dir.glob("*best_params.json"))
    if best_files:
        with open(best_files[0]) as f:
            return json.load(f)
    return None


def load_summa_output(exp_dir):
    """Load SUMMA simulation output from final_evaluation directory."""
    final_dir = exp_dir / "final_evaluation"
    daily_files = list(final_dir.glob("*_day.nc"))
    timestep_files = list(final_dir.glob("*_timestep.nc"))
    daily_path = daily_files[0] if daily_files else None
    timestep_path = timestep_files[0] if timestep_files else None

    # The TWS-component storage variables (scalarSWE and the soil/canopy/aquifer
    # stores) are written every timestep for the TWS objective (outputControl
    # `| 1`), so they live in the timestep file, not the daily file. Read them
    # from whichever file carries scalarSWE and aggregate to daily.
    store_path = None
    if daily_path and daily_path.exists() and 'scalarSWE' in xr.open_dataset(daily_path).variables:
        store_path = daily_path
    elif timestep_path and timestep_path.exists():
        store_path = timestep_path
    if store_path is None:
        return None

    ds_store = xr.open_dataset(store_path)
    store_times = pd.to_datetime(ds_store.time.values)
    store = pd.DataFrame({
        'SWE': ds_store['scalarSWE'].values.flatten(),
        'soil_water': ds_store['scalarTotalSoilWat'].values.flatten(),
        'canopy_water': ds_store['scalarCanopyWat'].values.flatten(),
        'aquifer': ds_store['scalarAquiferStorage'].values.flatten() * 1000,
    }, index=store_times)
    ds_store.close()

    df = store.resample('D').mean()  # no-op if already daily
    df['TWS'] = df['SWE'] + df['soil_water'] + df['canopy_water'] + df['aquifer']

    # Load streamflow from timestep
    if timestep_path and timestep_path.exists():
        ds_ts = xr.open_dataset(timestep_path)
        ts_times = pd.to_datetime(ds_ts.time.values)
        if 'averageRoutedRunoff' in ds_ts:
            runoff_ms = ds_ts['averageRoutedRunoff'].values.flatten()
            runoff_m3s = runoff_ms * CATCHMENT_AREA_KM2 * 1e6
            df_runoff = pd.DataFrame({'sim_Q': runoff_m3s}, index=ts_times)
            df = df.join(df_runoff.resample('D').mean(), how='left')
        ds_ts.close()

    return df


def load_streamflow_obs():
    """Load observed streamflow."""
    df = pd.read_csv(STREAMFLOW_DATA, parse_dates=['datetime'], index_col='datetime')
    df.columns = ['obs_Q']
    return df


def load_grace_data():
    """Load GRACE TWS observations."""
    df = pd.read_csv(GRACE_DATA, index_col=0, parse_dates=True)
    df['grace_mean'] = df[['grace_csr_anomaly', 'grace_gsfc_anomaly']].mean(axis=1) * 10
    return df


def calculate_tws_anomaly(df):
    """Calculate TWS anomaly relative to baseline."""
    baseline_mask = (df.index >= GRACE_BASELINE_START) & (df.index <= GRACE_BASELINE_END)
    baseline_mean = df.loc[baseline_mask, 'TWS'].mean()
    df['TWS_anomaly'] = df['TWS'] - baseline_mean
    return df


def calculate_kge(sim, obs, start=None, end=None):
    """Calculate Kling-Gupta Efficiency for calibration period."""
    if start is None:
        start = CALIBRATION_START
    if end is None:
        end = CALIBRATION_END

    # Filter to period
    sim_period = sim[(sim.index >= start) & (sim.index <= end)]
    obs_period = obs[(obs.index >= start) & (obs.index <= end)]

    common = sim_period.dropna().index.intersection(obs_period.dropna().index)
    if len(common) < 10:
        return np.nan

    s, o = sim_period.loc[common].values, obs_period.loc[common].values
    r = np.corrcoef(s, o)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def calculate_correlation(sim, obs):
    """Calculate Pearson correlation."""
    common = sim.dropna().index.intersection(obs.dropna().index)
    if len(common) < 10:
        return np.nan
    s, o = sim.loc[common].values, obs.loc[common].values
    return np.corrcoef(s, o)[0, 1]


def load_pareto_front(exp_dir, algorithm_name):
    """Load Pareto front CSV produced by SYMFLUENCE optimization.

    Returns DataFrame with columns obj_0 (KGE), obj_1 (TWS correlation),
    plus parameter columns, or None if not found.
    """
    import glob as globmod
    pattern = str(exp_dir / f"*{algorithm_name}*pareto_front.csv")
    files = globmod.glob(pattern)
    if not files:
        # Also try without algorithm name
        pattern = str(exp_dir / "*pareto_front.csv")
        files = globmod.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if 'obj_0' not in df.columns or 'obj_1' not in df.columns:
        return None
    return df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_tws_comparison(ax, grace_df, sim_dfs):
    """Panel (a): TWS time series - shows TWS-only excels here."""
    start = pd.Timestamp(CALIBRATION_START)
    end = pd.Timestamp(EVALUATION_END)

    # Observed GRACE (monthly - GRACE is monthly resolution)
    grace_monthly = grace_df['grace_mean'].resample('ME').mean()
    grace_monthly = grace_monthly[(grace_monthly.index >= start) & (grace_monthly.index <= end)]

    ax.plot(grace_monthly.index, grace_monthly.values, color=EXP_COLORS['obs'],
           linewidth=1.2, label='GRACE', zorder=5)

    # Simulations - weekly for more detail, thin distinct line styles
    line_styles = {'exp1': (0, (4, 1.5)), 'exp2': '-', 'exp3': (0, (2, 0.8, 0.8, 0.8)), 'exp4': (0, (1, 1))}  # dashed, solid, dash-dot, dotted
    metrics = {}

    for exp, df in sim_dfs.items():
        if df is not None and 'TWS_anomaly' in df.columns:
            sim_weekly = df['TWS_anomaly'].resample('W').mean()
            sim_weekly = sim_weekly[(sim_weekly.index >= start) & (sim_weekly.index <= end)]

            ax.plot(sim_weekly.index, sim_weekly.values, color=EXP_COLORS[exp],
                   linewidth=0.8, linestyle=line_styles.get(exp, '-'), alpha=1.0,
                   label=EXP_LABELS[exp], zorder=4 if exp != 'exp2' else 6)

            # Calculate correlation
            r = calculate_correlation(df['TWS_anomaly'].resample('ME').mean(), grace_df['grace_mean'].resample('ME').mean())
            metrics[exp] = r

    # Period shading
    ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
              alpha=0.08, color='#888888', zorder=0)

    ax.axhline(0, color='#CCCCCC', linewidth=0.5, zorder=0)

    ax.set_ylabel('TWS Anomaly (mm)', fontsize=FONT['size']['axis_label'])
    ax.set_xlim(start, end)
    ax.set_ylim(-250, 300)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.legend(loc='lower left', fontsize=FONT['size']['legend']-0.5, ncol=2,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

    despine(ax)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
           fontsize=FONT['size']['panel_label'], fontweight='bold', va='top', ha='left')

    return metrics


def plot_streamflow_comparison(ax, obs_df, sim_dfs):
    """Panel (b): Streamflow comparison."""
    start = pd.Timestamp(CALIBRATION_START)
    end = pd.Timestamp(EVALUATION_END)

    # Observed (weekly mean for more detail)
    obs_weekly = obs_df['obs_Q'].resample('W').mean()
    obs_weekly = obs_weekly[(obs_weekly.index >= start) & (obs_weekly.index <= end)]

    ax.plot(obs_weekly.index, obs_weekly.values, color=EXP_COLORS['obs'],
           linewidth=1.0, label='Observed', zorder=5)

    # Simulations - weekly for more detail, thin distinct line styles
    line_styles = {'exp1': (0, (4, 1.5)), 'exp2': '-', 'exp3': (0, (2, 0.8, 0.8, 0.8)), 'exp4': (0, (1, 1))}  # dashed, solid, dash-dot, dotted
    metrics = {}

    for exp, df in sim_dfs.items():
        if df is not None and 'sim_Q' in df.columns:
            sim_weekly = df['sim_Q'].resample('W').mean()
            sim_weekly = sim_weekly[(sim_weekly.index >= start) & (sim_weekly.index <= end)]

            ax.plot(sim_weekly.index, sim_weekly.values, color=EXP_COLORS[exp],
                   linewidth=0.8, linestyle=line_styles.get(exp, '-'), alpha=1.0,
                   label=EXP_LABELS[exp], zorder=4)

            # Calculate KGE from calibration period
            kge = calculate_kge(df['sim_Q'], obs_df['obs_Q'])
            metrics[exp] = kge

    # Period shading
    ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
              alpha=0.08, color='#888888', zorder=0)

    ax.set_ylabel('Discharge (m³/s)', fontsize=FONT['size']['axis_label'])
    ax.set_xlabel('Year', fontsize=FONT['size']['axis_label'])
    ax.set_xlim(start, end)
    ax.set_ylim(0, None)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.legend(loc='upper left', fontsize=FONT['size']['legend']-0.5, ncol=2,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

    despine(ax)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
           fontsize=FONT['size']['panel_label'], fontweight='bold', va='top', ha='left')

    return metrics


def plot_tradeoff_space(ax, tws_metrics, q_metrics, exp_results, pareto_fronts=None):
    """Panel (c): Objective space showing calibration trade-offs and Pareto fronts."""

    if pareto_fronts is None:
        pareto_fronts = {}

    marker_styles = {'exp1': 'o', 'exp2': 's', 'exp3': 'D', 'exp4': '^'}
    marker_sizes = {'exp1': 120, 'exp2': 120, 'exp3': 140, 'exp4': 140}

    # Plot Pareto fronts for multi-objective algorithms (NSGA-II, MOEA/D)
    for exp, pf_df in pareto_fronts.items():
        if pf_df is not None and len(pf_df) > 1:
            kge_vals = pf_df['obj_0'].values
            tws_vals = pf_df['obj_1'].values

            # Sort by KGE for clean front line
            sort_idx = np.argsort(kge_vals)
            kge_sorted = kge_vals[sort_idx]
            tws_sorted = tws_vals[sort_idx]

            # Plot all Pareto solutions as small faded markers
            ax.scatter(kge_vals, tws_vals, c=EXP_COLORS[exp], s=25,
                      marker=marker_styles.get(exp, 'o'), alpha=0.35,
                      edgecolors='none', zorder=7)

            # Plot Pareto front line connecting sorted solutions
            ax.plot(kge_sorted, tws_sorted, color=EXP_COLORS[exp],
                   linewidth=1.2, alpha=0.6, zorder=8, linestyle='-')

            n_solutions = len(pf_df)
            print(f"  {EXP_LABELS[exp]} Pareto front: {n_solutions} solutions")
            print(f"    KGE range: [{kge_vals.min():.3f}, {kge_vals.max():.3f}]")
            print(f"    TWS r range: [{tws_vals.min():.3f}, {tws_vals.max():.3f}]")

    # Plot best/representative points for each experiment (larger markers)
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in tws_metrics and exp in q_metrics:
            tws_r = tws_metrics[exp]
            q_kge = q_metrics[exp]

            # For multi-obj experiments with Pareto fronts, label indicates "best"
            if exp in pareto_fronts and pareto_fronts[exp] is not None:
                label = f'{EXP_LABELS[exp]} best ({q_kge:.2f}, {tws_r:.2f})'
            else:
                label = f'{EXP_LABELS[exp]} ({q_kge:.2f}, {tws_r:.2f})'

            ax.scatter(q_kge, tws_r, c=EXP_COLORS[exp], s=marker_sizes[exp],
                      marker=marker_styles[exp], edgecolors='white', linewidths=1.5,
                      zorder=10, label=label)

    # Light grid
    ax.grid(True, linestyle='-', alpha=0.15, zorder=0)

    ax.set_xlabel('Streamflow KGE', fontsize=FONT['size']['axis_label'])
    ax.set_ylabel('TWS Correlation (r)', fontsize=FONT['size']['axis_label'])

    # Dynamic axis limits based on data
    all_kge = [q_metrics.get(e, np.nan) for e in ['exp1', 'exp2', 'exp3', 'exp4']]
    all_tws = [tws_metrics.get(e, np.nan) for e in ['exp1', 'exp2', 'exp3', 'exp4']]
    for pf_df in pareto_fronts.values():
        if pf_df is not None and len(pf_df) > 0:
            all_kge.extend(pf_df['obj_0'].values)
            all_tws.extend(pf_df['obj_1'].values)
    all_kge = [v for v in all_kge if np.isfinite(v)]
    all_tws = [v for v in all_tws if np.isfinite(v)]

    if all_kge and all_tws:
        kge_pad = max(0.05, (max(all_kge) - min(all_kge)) * 0.1)
        tws_pad = max(0.03, (max(all_tws) - min(all_tws)) * 0.1)
        ax.set_xlim(min(all_kge) - kge_pad, max(all_kge) + kge_pad)
        ax.set_ylim(min(all_tws) - tws_pad, max(all_tws) + tws_pad)
    else:
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.60, 0.92)

    # Legend
    ax.legend(loc='lower left', fontsize=FONT['size']['legend']-0.5,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.95,
             title='Experiment (KGE, r)', title_fontsize=FONT['size']['legend']-0.5)

    despine(ax)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes,
           fontsize=FONT['size']['panel_label'], fontweight='bold', va='top', ha='left')


def plot_parameter_comparison(ax, exp_results):
    """Panel (d): Key parameter comparison - normalized to bounds (0-1 scale)."""

    # Key parameters with their bounds for normalization
    params_info = {
        'theta_sat': {'label': 'θ_sat\n(porosity)', 'bounds': (0.3, 0.7)},
        'k_soil': {'label': 'k_soil\n(conductivity)', 'bounds': (1e-7, 1e-2), 'log': True},
        'routingGammaScale': {'label': 'γ_scale\n(routing)', 'bounds': (100, 100000), 'log': True},
    }

    params = list(params_info.keys())
    param_labels = [params_info[p]['label'] for p in params]

    x = np.arange(len(params))
    width = 0.25

    for i, (exp, color) in enumerate([('exp1', EXP_COLORS['exp1']),
                                       ('exp2', EXP_COLORS['exp2']),
                                       ('exp3', EXP_COLORS['exp3']),
                                       ('exp4', EXP_COLORS['exp4'])]):
        if exp_results[exp] is None:
            continue

        norm_values = []
        raw_values = []

        for param in params:
            val = exp_results[exp]['best_params'].get(param, [None])[0]
            if val is None:
                val = params_info[param]['bounds'][0]
            raw_values.append(val)

            # Normalize to 0-1 based on bounds
            lo, hi = params_info[param]['bounds']
            if params_info[param].get('log', False):
                # Log-scale normalization
                norm_val = (np.log10(val) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            else:
                norm_val = (val - lo) / (hi - lo)
            norm_values.append(np.clip(norm_val, 0, 1))

        bars = ax.bar(x + (i-1)*width, norm_values, width, color=color, alpha=0.85,
              label=EXP_LABELS[exp], edgecolor='white', linewidth=0.8)

        # Add value annotations above bars
        for j, (nv, v) in enumerate(zip(norm_values, raw_values)):
            if params[j] == 'theta_sat':
                label = f'{v:.2f}'
            elif params[j] == 'k_soil':
                label = f'{v:.1e}'
            else:
                label = f'{int(v):,}'

            ax.text(x[j] + (i-1)*width, nv + 0.03, label,
                   ha='center', va='bottom', fontsize=FONT['size']['annotation']-1,
                   color=color, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, fontsize=FONT['size']['tick'])
    ax.set_ylabel('Normalized value', fontsize=FONT['size']['axis_label'])
    ax.set_ylim(0, 1.3)

    ax.axhline(0.5, color='#CCCCCC', linewidth=0.8, linestyle='--', zorder=0)

    ax.legend(loc='upper right', fontsize=FONT['size']['legend']-1, ncol=1,
             frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

    despine(ax)
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes,
           fontsize=FONT['size']['panel_label'], fontweight='bold', va='top', ha='left')


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print("=" * 70)
    print("Creating Figure 2: Calibration Trade-off (Redesigned)")
    print("=" * 70)

    # Load experiment results
    print("\nLoading experiment results...")
    exp_results = {
        'exp1': load_experiment_results(EXP1_DIR),
        'exp2': load_experiment_results(EXP2_DIR),
        'exp3': load_experiment_results(EXP3_DIR),
        'exp4': load_experiment_results(EXP4_DIR),
    }

    # Load simulation outputs
    print("\nLoading simulation outputs...")
    sim_dfs = {
        'exp1': load_summa_output(EXP1_DIR),
        'exp2': load_summa_output(EXP2_DIR),
        'exp3': load_summa_output(EXP3_DIR),
        'exp4': load_summa_output(EXP4_DIR),
    }

    # Calculate TWS anomalies
    grace_df = load_grace_data()
    for exp, df in sim_dfs.items():
        if df is not None:
            sim_dfs[exp] = calculate_tws_anomaly(df)
            print(f"  {exp}: loaded {len(df)} days")

    # Load observations
    obs_df = load_streamflow_obs()

    # Load Pareto front data (from re-run with save enabled)
    print("\nLoading Pareto front data...")
    pareto_fronts = {
        'exp3': load_pareto_front(EXP3_DIR, 'nsga'),
        'exp4': load_pareto_front(EXP4_DIR, 'moea'),
    }
    for exp, pf in pareto_fronts.items():
        if pf is not None:
            print(f"  {EXP_LABELS[exp]}: {len(pf)} Pareto solutions loaded")
        else:
            print(f"  {EXP_LABELS[exp]}: No Pareto front CSV found (using single point)")

    # Create figure - 2x2 layout
    print("\nCreating figure...")
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1],
                  wspace=0.25, hspace=0.3)

    # Panel (a): TWS comparison (top left)
    ax_tws = fig.add_subplot(gs[0, 0])
    tws_metrics = plot_tws_comparison(ax_tws, grace_df, sim_dfs)

    # Panel (b): Streamflow comparison (top right)
    ax_q = fig.add_subplot(gs[0, 1])
    q_metrics = plot_streamflow_comparison(ax_q, obs_df, sim_dfs)

    # Panel (c): Trade-off space with Pareto fronts (bottom left)
    ax_trade = fig.add_subplot(gs[1, 0])
    plot_tradeoff_space(ax_trade, tws_metrics, q_metrics, exp_results, pareto_fronts)

    # Panel (d): Parameter comparison (bottom right)
    ax_params = fig.add_subplot(gs[1, 1])
    plot_parameter_comparison(ax_params, exp_results)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "figure_10_multiobjective_tradeoff.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    # Print metrics
    print("\n" + "=" * 50)
    print("METRICS SUMMARY")
    print("=" * 50)
    print("\nTWS Correlation (r) - calculated from simulations:")
    for exp, r in tws_metrics.items():
        print(f"  {EXP_LABELS[exp]}: {r:.3f}")
    print("\nStreamflow KGE (from calibration JSON):")
    for exp, kge in q_metrics.items():
        print(f"  {EXP_LABELS[exp]}: {kge:.3f}")

    print("\nPareto Front Summary:")
    for exp, pf in pareto_fronts.items():
        if pf is not None:
            print(f"  {EXP_LABELS[exp]}: {len(pf)} non-dominated solutions")
            print(f"    KGE:   [{pf['obj_0'].min():.3f}, {pf['obj_0'].max():.3f}]")
            print(f"    TWS r: [{pf['obj_1'].min():.3f}, {pf['obj_1'].max():.3f}]")
        else:
            print(f"  {EXP_LABELS[exp]}: No Pareto front data")

    plt.close()
    print("\nFigure 2 (v2) complete!")


if __name__ == "__main__":
    main()
