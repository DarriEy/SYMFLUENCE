#!/usr/bin/env python3
"""
Analyze results from the 4.3 Forcing Ensemble Study.

This script generates comparison plots and summary statistics for the
forcing ensemble experiment, comparing SWE simulations across ERA5,
AORC, CONUS404, and RDRS forcing datasets, plus GDDP climate
projection ensemble members.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/pandas not available. Install with: pip install matplotlib pandas")

# Study configuration
STUDY_DIR = Path(__file__).parent.parent
CONFIGS_DIR = STUDY_DIR / "configs"
RESULTS_DIR = STUDY_DIR / "results"

# Data directory (from config)
SYMFLUENCE_DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

# Forcing datasets - reanalysis/gridded products
FORCINGS = ['era5', 'aorc', 'conus404', 'rdrs']
FORCING_LABELS = {
    'era5': 'ERA5 (~31 km)',
    'aorc': 'AORC (~1 km)',
    'conus404': 'CONUS404 (~4 km)',
    'rdrs': 'RDRS (~10 km)',
}
FORCING_COLORS = {
    'era5': '#1f77b4',
    'aorc': '#ff7f0e',
    'conus404': '#d62728',
    'rdrs': '#9467bd',
}

# GDDP climate projection ensemble members (10-member ensemble)
GDDP_FORCINGS = [
    'gddp_access_cm2', 'gddp_gfdl_esm4', 'gddp_mri_esm2_0',
    'gddp_ukesm1_0_ll', 'gddp_canesm5', 'gddp_ipsl_cm6a_lr',
    'gddp_cnrm_cm6_1', 'gddp_mpi_esm1_2_hr', 'gddp_noresm2_lm',
    'gddp_inm_cm5_0',
]
GDDP_LABELS = {
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
GDDP_COLORS = {
    'gddp_access_cm2':    '#88CCEE',
    'gddp_gfdl_esm4':     '#CC6677',
    'gddp_mri_esm2_0':    '#AA4499',
    'gddp_ukesm1_0_ll':   '#999933',
    'gddp_canesm5':        '#882255',
    'gddp_ipsl_cm6a_lr':  '#44AA99',
    'gddp_cnrm_cm6_1':    '#DDCC77',
    'gddp_mpi_esm1_2_hr': '#332288',
    'gddp_noresm2_lm':    '#117733',
    'gddp_inm_cm5_0':     '#CC3311',
}

# Combined lookups
ALL_FORCINGS = FORCINGS + GDDP_FORCINGS
ALL_LABELS = {**FORCING_LABELS, **GDDP_LABELS}
ALL_COLORS = {**FORCING_COLORS, **GDDP_COLORS}

# Unit conversion: SNOTEL reports SWE in inches, SUMMA outputs in mm
INCHES_TO_MM = 25.4


def load_observed_swe(domain_dir: Path) -> Optional[pd.DataFrame]:
    """Load observed SWE data from SNOTEL."""
    # Primary path: preprocessed SWE observations
    obs_path = domain_dir / "observations" / "snow" / "swe" / "preprocessed"

    if obs_path.exists():
        patterns = ["*swe*.csv", "*SWE*.csv", "*.csv"]
        for pattern in patterns:
            files = list(obs_path.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0], parse_dates=['Date'])
                    df = df.set_index('Date')
                    return df
                except Exception:
                    continue

    # Fallback: legacy path
    legacy_path = domain_dir / "observations" / "snotel"
    if legacy_path.exists():
        for pattern in ["*swe*.csv", "*SWE*.csv", "*.csv"]:
            files = list(legacy_path.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0], parse_dates=[0])
                    df.index.name = 'Date'
                    return df
                except Exception:
                    continue

    return None


def load_simulated_swe(domain_dir: Path, forcing: str) -> Optional[xr.Dataset]:
    """Load simulated SWE from SUMMA calibration final evaluation output."""
    # Primary path: DDS optimization final evaluation
    experiment_id = f"forcing_ensemble_{forcing}"
    opt_path = domain_dir / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation"

    if opt_path.exists():
        # Prefer daily output
        nc_files = list(opt_path.glob("*_day.nc"))
        if not nc_files:
            nc_files = list(opt_path.glob("*.nc"))
        if nc_files:
            try:
                ds = xr.open_dataset(nc_files[0])
                return ds
            except Exception:
                pass

    # Fallback: simulations directory
    sim_path = domain_dir / "simulations" / "SUMMA"
    if sim_path.exists():
        nc_files = list(sim_path.glob("*output*.nc")) + list(sim_path.glob("*_day.nc"))
        if nc_files:
            try:
                ds = xr.open_dataset(nc_files[0])
                return ds
            except Exception:
                pass

    return None


def load_calibration_metrics(domain_dir: Path, forcing: str) -> Optional[Dict]:
    """Load pre-computed calibration metrics from the final evaluation JSON."""
    experiment_id = f"forcing_ensemble_{forcing}"
    eval_json = (domain_dir / "optimization" / "SUMMA" / f"dds_{experiment_id}" /
                 f"{experiment_id}_dds_final_evaluation.json")

    if eval_json.exists():
        try:
            with open(eval_json) as f:
                data = json.load(f)
            return data
        except Exception:
            pass
    return None


def calculate_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics."""
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[mask]
    sim_clean = sim[mask]

    if len(obs_clean) == 0:
        return {'rmse': np.nan, 'bias': np.nan, 'kge': np.nan, 'corr': np.nan}

    # RMSE
    rmse = np.sqrt(np.mean((sim_clean - obs_clean) ** 2))

    # Bias
    bias = np.mean(sim_clean - obs_clean)

    # Correlation
    corr = np.corrcoef(obs_clean, sim_clean)[0, 1] if len(obs_clean) > 1 else np.nan

    # KGE
    if len(obs_clean) > 1 and np.std(obs_clean) > 0 and np.std(sim_clean) > 0:
        r = corr
        alpha = np.std(sim_clean) / np.std(obs_clean)
        beta = np.mean(sim_clean) / np.mean(obs_clean)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    else:
        kge = np.nan

    return {
        'rmse': rmse,
        'bias': bias,
        'kge': kge,
        'corr': corr,
    }


def plot_swe_timeseries(
    results: Dict[str, Dict],
    output_path: Path,
    forcings_to_plot: List[str] = None,
    date_range: tuple = None,
    title_suffix: str = ''
):
    """Create SWE time series comparison plot.

    Args:
        date_range: Optional (start, end) datetime tuple to restrict plot window.
    """
    if not HAS_PLOTTING:
        print("Skipping plot: matplotlib not available")
        return

    if forcings_to_plot is None:
        forcings_to_plot = ALL_FORCINGS

    fig, ax = plt.subplots(figsize=(14, 6))

    # Use explicit date range, or fall back to observed data range
    if date_range is not None:
        t_min, t_max = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    elif 'observed' in results:
        # Use observation period as the comparison window
        obs_idx = pd.DatetimeIndex(results['observed']['time'])
        t_min, t_max = obs_idx.min(), obs_idx.max()
    else:
        t_min, t_max = None, None

    # Plot observed
    if 'observed' in results:
        obs = results['observed']
        obs_series = pd.Series(obs['swe'], index=pd.DatetimeIndex(obs['time']))
        if t_min is not None:
            obs_series = obs_series.loc[t_min:t_max]
        ax.plot(obs_series.index, obs_series.values, 'k-', linewidth=2,
                label='Observed (SNOTEL)', zorder=10)

    # Plot each forcing - clipped to date range
    for forcing in forcings_to_plot:
        if forcing in results and 'swe' in results[forcing]:
            data = results[forcing]
            sim_series = pd.Series(data['swe'], index=pd.DatetimeIndex(data['time']))
            if t_min is not None:
                sim_series = sim_series.loc[t_min:t_max]
            if len(sim_series) > 0:
                ax.plot(
                    sim_series.index, sim_series.values,
                    color=ALL_COLORS[forcing],
                    linewidth=1.5,
                    label=ALL_LABELS[forcing],
                    alpha=0.8
                )

    ax.set_xlabel('Date')
    ax.set_ylabel('Snow Water Equivalent (mm)')
    title = 'Forcing Ensemble Study: SWE Comparison\nParadise SNOTEL Station'
    if title_suffix:
        title += f' - {title_suffix}'
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_metrics_comparison(
    metrics: Dict[str, Dict],
    output_path: Path,
    forcings_to_plot: List[str] = None,
    period: str = 'calibration'
):
    """Create bar chart comparing metrics across forcings."""
    if not HAS_PLOTTING:
        print("Skipping plot: matplotlib not available")
        return

    if forcings_to_plot is None:
        forcings_to_plot = ALL_FORCINGS

    metric_names = ['RMSE (mm)', 'Bias (mm)', 'KGE', 'Correlation']
    metric_keys = ['rmse', 'bias', 'kge', 'corr']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, name, key in zip(axes, metric_names, metric_keys):
        values = []
        colors = []
        labels = []

        for forcing in forcings_to_plot:
            if forcing in metrics:
                values.append(metrics[forcing].get(key, np.nan))
                colors.append(ALL_COLORS[forcing])
                labels.append(ALL_LABELS[forcing].split(' (')[0])  # Short label

        if values:
            bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_ylabel(name)
            ax.set_title(name)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    va = 'bottom' if val >= 0 else 'top'
                    ax.text(
                        bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va=va, fontsize=8
                    )

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Forcing Ensemble Study: {period.title()} Period Metrics', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_summary_table(
    cal_metrics: Dict[str, Dict],
    eval_metrics: Dict[str, Dict],
    output_path: Path,
    forcings_to_report: List[str] = None
):
    """Generate summary statistics table with calibration and evaluation periods."""
    if not cal_metrics and not eval_metrics:
        print("  No metrics to summarize")
        return

    if forcings_to_report is None:
        forcings_to_report = ALL_FORCINGS

    rows = []
    for forcing in forcings_to_report:
        if forcing in cal_metrics or forcing in eval_metrics:
            row = {'Forcing': ALL_LABELS.get(forcing, forcing)}
            if forcing in cal_metrics:
                for k, v in cal_metrics[forcing].items():
                    row[f'Cal_{k}'] = v
            if forcing in eval_metrics:
                for k, v in eval_metrics[forcing].items():
                    row[f'Eval_{k}'] = v
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df = df.round(3)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name}")

        # Print to console
        print("\n  Performance Summary:")
        print(df.to_string(index=False))


def load_observed_sm(domain_dir: Path) -> Optional[pd.DataFrame]:
    """Load observed soil moisture from the closest ISMN station.

    Returns a DataFrame indexed by Date with columns for each depth (e.g. 'sm_0.05').
    """
    ismn_dir = domain_dir / "observations" / "soil_moisture" / "ismn"
    if not ismn_dir.exists():
        return None

    # Find closest station from selection file
    sel_file = ismn_dir / "ismn_station_selection.csv"
    if not sel_file.exists():
        return None
    sel = pd.read_csv(sel_file)
    if sel.empty:
        return None
    closest = sel.sort_values('distance_km').iloc[0]
    station_id = str(int(closest['station_id']))

    # Load all depths for this station
    depth_data = {}
    for csv_file in sorted(ismn_dir.glob(f"{station_id}_depth_*.csv")):
        try:
            df = pd.read_csv(csv_file, parse_dates=['DateTime'])
            depth_m = df['depth_m'].iloc[0]
            daily = df.set_index('DateTime').resample('D')['soil_moisture'].mean()
            depth_data[f'sm_{depth_m:.2f}'] = daily
        except Exception:
            continue

    if not depth_data:
        return None

    result = pd.DataFrame(depth_data)
    result.index.name = 'Date'
    return result


def load_simulated_sm(domain_dir: Path, forcing: str) -> Optional[Dict]:
    """Load simulated soil moisture (top soil layer VFL) from SUMMA output.

    Extracts mLayerVolFracLiq for the top soil layer (0.2m depth) at each timestep.
    Returns dict with 'time' and 'sm' arrays.
    """
    experiment_id = f"forcing_ensemble_{forcing}"
    opt_path = domain_dir / "optimization" / "SUMMA" / f"dds_{experiment_id}" / "final_evaluation"

    nc_files = list(opt_path.glob("*_day.nc")) if opt_path.exists() else []
    if not nc_files:
        sim_path = domain_dir / "simulations" / "SUMMA"
        if sim_path.exists():
            nc_files = list(sim_path.glob("*_day.nc"))
    if not nc_files:
        return None

    try:
        ds = xr.open_dataset(nc_files[0])
    except Exception:
        return None

    if 'mLayerVolFracLiq' not in ds or 'mLayerDepth' not in ds:
        return None

    depths = ds['mLayerDepth'].values[:, :, 0]
    vfl = ds['mLayerVolFracLiq'].values[:, :, 0]
    n_time = len(ds.time)

    # Extract top soil layer (depth == 0.2m) at each timestep
    top_soil_vfl = np.full(n_time, np.nan)
    for t in range(n_time):
        for layer in range(depths.shape[1]):
            if abs(depths[t, layer] - 0.2) < 0.01 and vfl[t, layer] > -999:
                top_soil_vfl[t] = vfl[t, layer]
                break

    return {
        'time': pd.to_datetime(ds['time'].values),
        'sm': top_soil_vfl,
    }


def plot_swe_sm_combined(
    swe_results: Dict[str, Dict],
    sm_sim_results: Dict[str, Dict],
    sm_obs: Optional[pd.DataFrame],
    output_path: Path,
    forcings_to_plot: List[str] = None,
    date_range: tuple = None,
    cal_eval_periods: tuple = None,
):
    """Create a 2-panel figure: (a) SWE time series, (b) soil moisture time series.

    Args:
        cal_eval_periods: Optional ((cal_start, cal_end), (eval_start, eval_end)) for shading.
    """
    if not HAS_PLOTTING:
        print("Skipping plot: matplotlib not available")
        return

    if forcings_to_plot is None:
        forcings_to_plot = ALL_FORCINGS

    if date_range is not None:
        t_min, t_max = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        t_min, t_max = pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-31')

    fig, (ax_swe, ax_sm) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                         gridspec_kw={'height_ratios': [1.2, 1],
                                                      'hspace': 0.08})

    # --- Panel (a): SWE ---
    ax_swe.text(-0.02, 1.05, '(a)', transform=ax_swe.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Shade calibration/evaluation periods
    if cal_eval_periods:
        (cs, ce), (es, ee) = cal_eval_periods
        ax_swe.axvspan(cs, ce, alpha=0.06, color='blue', label='Calibration period')
        ax_swe.axvspan(es, ee, alpha=0.06, color='green', label='Evaluation period')
        ax_sm.axvspan(cs, ce, alpha=0.06, color='blue')
        ax_sm.axvspan(es, ee, alpha=0.06, color='green')

    # Plot observed SWE
    if 'observed' in swe_results:
        obs = swe_results['observed']
        obs_s = pd.Series(obs['swe'], index=pd.DatetimeIndex(obs['time'])).loc[t_min:t_max]
        ax_swe.plot(obs_s.index, obs_s.values, 'k-', linewidth=2.2,
                    label='Observed (SNOTEL)', zorder=10)

    # Plot simulated SWE
    for forcing in forcings_to_plot:
        if forcing in swe_results and 'swe' in swe_results[forcing]:
            data = swe_results[forcing]
            sim_s = pd.Series(data['swe'], index=pd.DatetimeIndex(data['time'])).loc[t_min:t_max]
            if len(sim_s) > 0:
                ax_swe.plot(sim_s.index, sim_s.values,
                           color=ALL_COLORS[forcing], linewidth=1.4,
                           label=ALL_LABELS[forcing], alpha=0.85)

    ax_swe.set_ylabel('Snow Water Equivalent (mm)', fontsize=11)
    ax_swe.set_title('Forcing Ensemble Study: SWE and Soil Moisture Comparison\n'
                      'Paradise SNOTEL Station, WA', fontsize=13, fontweight='bold')
    ax_swe.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax_swe.grid(True, alpha=0.2)
    ax_swe.set_xlim(t_min, t_max)

    # --- Panel (b): Soil Moisture ---
    ax_sm.text(-0.02, 1.05, '(b)', transform=ax_sm.transAxes,
               fontsize=14, fontweight='bold', va='top')

    # Plot observed SM (closest station, shallowest depth)
    sm_obs_plotted = False
    if sm_obs is not None:
        # Use the 0.20m depth to match SUMMA's top soil layer
        sm_col = None
        for col in ['sm_0.20', 'sm_0.10', 'sm_0.05']:
            if col in sm_obs.columns:
                sm_col = col
                break
        if sm_col is not None:
            obs_sm = sm_obs[sm_col].dropna()
            obs_sm = obs_sm.loc[t_min:t_max]
            if len(obs_sm) > 0:
                depth_label = sm_col.replace('sm_', '')
                ax_sm.plot(obs_sm.index, obs_sm.values, 'k-', linewidth=2.2,
                          label=f'Observed (ISMN, {depth_label} m)', zorder=10)
                sm_obs_plotted = True

    # Plot simulated SM
    for forcing in forcings_to_plot:
        if forcing in sm_sim_results:
            data = sm_sim_results[forcing]
            sim_s = pd.Series(data['sm'], index=pd.DatetimeIndex(data['time'])).loc[t_min:t_max]
            if len(sim_s) > 0:
                ax_sm.plot(sim_s.index, sim_s.values,
                          color=ALL_COLORS[forcing], linewidth=1.4,
                          label=ALL_LABELS[forcing], alpha=0.85)

    ax_sm.set_ylabel('Volumetric Water Content (m\u00b3/m\u00b3)', fontsize=11)
    ax_sm.set_xlabel('Date', fontsize=11)
    ax_sm.legend(loc='upper left', fontsize=8.5, ncol=2, framealpha=0.9)
    ax_sm.grid(True, alpha=0.2)

    # If observed SM exists, add an inset zooming into the observation period
    if sm_obs_plotted and sm_obs is not None:
        obs_start = sm_obs.index.min()
        obs_end = sm_obs.index.max()

        # Inset axes positioned in upper right, away from legend
        ax_inset = ax_sm.inset_axes([0.62, 0.42, 0.36, 0.55])

        # Plot observed SM in inset
        if sm_col in sm_obs.columns:
            obs_zoom = sm_obs[sm_col].dropna().loc[obs_start:obs_end]
            ax_inset.plot(obs_zoom.index, obs_zoom.values, 'k-', linewidth=2.2,
                         label='Observed', zorder=10)

        # Plot simulated SM in inset
        for forcing in forcings_to_plot:
            if forcing in sm_sim_results:
                data = sm_sim_results[forcing]
                sim_s = pd.Series(data['sm'], index=pd.DatetimeIndex(data['time']))
                sim_zoom = sim_s.loc[obs_start:obs_end].dropna()
                if len(sim_zoom) > 0:
                    ax_inset.plot(sim_zoom.index, sim_zoom.values,
                                color=ALL_COLORS[forcing], linewidth=1.3, alpha=0.85)

        ax_inset.set_xlim(obs_start, obs_end)
        ax_inset.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_inset.xaxis.set_major_locator(mdates.MonthLocator())
        ax_inset.tick_params(labelsize=7)
        ax_inset.set_ylabel('VWC (m\u00b3/m\u00b3)', fontsize=7)
        ax_inset.set_title('Observation overlap\n(Jan\u2013Mar 2015)',
                          fontsize=8, fontweight='bold', pad=3)
        ax_inset.grid(True, alpha=0.3)
        ax_inset.patch.set_alpha(0.95)
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(1.2)

        # Indicate inset region on main plot
        ax_sm.axvspan(obs_start, obs_end, alpha=0.10, color='gold',
                     zorder=0, label='_nolegend_')

    # Format x-axis
    ax_sm.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_sm.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def load_all_best_params(data_dir: Path) -> Dict[str, Dict]:
    """Load calibrated best parameters for all forcing domains."""
    params = {}
    for forcing in ALL_FORCINGS:
        experiment_id = f"forcing_ensemble_{forcing}"
        param_file = (data_dir / f"domain_paradise_snotel_wa_{forcing}" /
                      "optimization" / "SUMMA" / f"dds_{experiment_id}" /
                      f"{experiment_id}_dds_best_params.json")
        if param_file.exists():
            try:
                with open(param_file) as f:
                    data = json.load(f)
                bp = data.get('best_params', data)
                clean = {}
                for k, v in bp.items():
                    if isinstance(v, list):
                        clean[k] = v[0]
                    elif isinstance(v, (int, float)):
                        clean[k] = v
                if clean:
                    params[forcing] = clean
            except Exception:
                pass
    return params


# Parameter metadata for the divergence analysis
PARAM_INFO = {
    'frozenPrecipMultip': {
        'label': 'Frozen Precip.\nMultiplier',
        'units': '[-]',
        'description': 'Scales incoming frozen precipitation',
        'compensatory': True,
        'reference_value': 1.0,
    },
    'tempCritRain': {
        'label': 'Critical Rain\nTemperature',
        'units': '[K]',
        'description': 'Temperature threshold for rain vs snow',
        'compensatory': False,
    },
    'tempRangeTimestep': {
        'label': 'Rain-Snow\nTransition Range',
        'units': '[K]',
        'description': 'Temperature range for rain-snow transition',
        'compensatory': True,
    },
    'albedoMax': {
        'label': 'Maximum\nAlbedo',
        'units': '[-]',
        'description': 'Maximum fresh snow albedo',
        'compensatory': False,
    },
    'albedoMinWinter': {
        'label': 'Min. Winter\nAlbedo',
        'units': '[-]',
        'description': 'Minimum winter albedo',
        'compensatory': False,
    },
    'mw_exp': {
        'label': 'Meltwater\nExponent',
        'units': '[-]',
        'description': 'Controls melt rate nonlinearity',
        'compensatory': True,
    },
    'k_snow': {
        'label': 'Snow Thermal\nConductivity',
        'units': '[W m⁻¹ K⁻¹]',
        'description': 'Snow thermal conductivity parameter',
        'compensatory': False,
    },
    'constSnowDen': {
        'label': 'Constant Snow\nDensity',
        'units': '[kg m⁻³]',
        'description': 'Fixed snow density parameter',
        'compensatory': False,
    },
    'routingGammaScale': {
        'label': 'Routing Gamma\nScale',
        'units': '[s]',
        'description': 'Time delay parameter for routing',
        'compensatory': False,
    },
}


def plot_parameter_divergence(
    all_params: Dict[str, Dict],
    cal_metrics: Dict[str, Dict],
    eval_metrics: Dict[str, Dict],
    output_path: Path,
):
    """Create parameter divergence analysis figure.

    Multi-panel figure showing:
    (a) Key calibrated parameter values across forcings (dot plot)
    (b) frozenPrecipMultip vs evaluation KGE scatter (linking compensation to transferability)
    (c) Parameter distortion index vs evaluation degradation
    """
    if not HAS_PLOTTING:
        print("Skipping plot: matplotlib not available")
        return

    forcings_with_params = [f for f in ALL_FORCINGS if f in all_params]
    if len(forcings_with_params) < 2:
        print("  Skipping parameter divergence: need at least 2 forcings")
        return

    # Build parameter DataFrame
    df_params = pd.DataFrame(all_params).T.loc[forcings_with_params]

    # Select key parameters for display
    display_params = ['frozenPrecipMultip', 'tempRangeTimestep', 'mw_exp',
                      'albedoMax', 'albedoMinWinter', 'k_snow', 'constSnowDen']
    display_params = [p for p in display_params if p in df_params.columns]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

    # --- Panel (a): Parameter dot plot ---
    ax_dot = fig.add_subplot(gs[0, :])
    ax_dot.text(-0.02, 1.05, '(a)', transform=ax_dot.transAxes,
                fontsize=14, fontweight='bold', va='top')

    n_params = len(display_params)
    n_forcings = len(forcings_with_params)

    for i, param in enumerate(display_params):
        vals = df_params[param].values.astype(float)
        # Normalize to [0,1] range for display
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            norm_vals = (vals - vmin) / (vmax - vmin)
        else:
            norm_vals = np.full_like(vals, 0.5)

        for j, forcing in enumerate(forcings_with_params):
            marker_size = 120
            ax_dot.scatter(i, j, c=[ALL_COLORS[forcing]], s=marker_size,
                          edgecolors='black', linewidth=0.5, zorder=5)
            # Add actual value as text
            val = df_params.loc[forcing, param]
            if abs(val) >= 1000:
                label = f'{val:.0f}'
            elif abs(val) >= 10:
                label = f'{val:.1f}'
            else:
                label = f'{val:.2f}'
            ax_dot.annotate(label, (i, j), textcoords="offset points",
                           xytext=(15, 0), fontsize=7.5, va='center',
                           color='0.3')

    # Highlight compensatory parameters with background shading
    for i, param in enumerate(display_params):
        info = PARAM_INFO.get(param, {})
        if info.get('compensatory', False):
            ax_dot.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='red', zorder=0)

    ax_dot.set_xticks(range(n_params))
    ax_dot.set_xticklabels([PARAM_INFO.get(p, {}).get('label', p) for p in display_params],
                           fontsize=9, ha='center')
    ax_dot.set_yticks(range(n_forcings))
    ax_dot.set_yticklabels([ALL_LABELS.get(f, f) for f in forcings_with_params], fontsize=10)
    ax_dot.set_title('Calibrated Parameter Values by Forcing Dataset\n'
                     '(red shading = compensatory parameters)', fontsize=12, fontweight='bold')
    ax_dot.grid(True, alpha=0.15, axis='x')
    ax_dot.set_xlim(-0.5, n_params - 0.5)
    ax_dot.set_ylim(-0.5, n_forcings - 0.5)

    # --- Panel (b): frozenPrecipMultip vs Evaluation KGE ---
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_scatter.text(-0.08, 1.08, '(b)', transform=ax_scatter.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    for forcing in forcings_with_params:
        fpm = all_params[forcing].get('frozenPrecipMultip', np.nan)
        eval_kge = eval_metrics.get(forcing, {}).get('kge', np.nan)
        cal_kge = cal_metrics.get(forcing, {}).get('kge', np.nan)
        if not np.isnan(fpm) and not np.isnan(eval_kge):
            ax_scatter.scatter(fpm, eval_kge, c=ALL_COLORS[forcing], s=150,
                             edgecolors='black', linewidth=1, zorder=5)
            ax_scatter.annotate(ALL_LABELS.get(forcing, forcing).split(' (')[0],
                              (fpm, eval_kge), textcoords="offset points",
                              xytext=(8, 8), fontsize=8.5, fontweight='bold',
                              color=ALL_COLORS[forcing])

    ax_scatter.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_scatter.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8,
                      label='No correction (1.0)')
    ax_scatter.set_xlabel('Calibrated frozenPrecipMultip [-]', fontsize=11)
    ax_scatter.set_ylabel('Evaluation Period KGE', fontsize=11)
    ax_scatter.set_title('Precipitation Correction vs.\nEvaluation Transferability', fontsize=11,
                        fontweight='bold')
    ax_scatter.legend(fontsize=8, loc='lower left')
    ax_scatter.grid(True, alpha=0.2)

    # --- Panel (c): Parameter distortion index vs KGE degradation ---
    ax_distort = fig.add_subplot(gs[1, 1])
    ax_distort.text(-0.08, 1.08, '(c)', transform=ax_distort.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # Compute a "distortion index" based on compensatory parameters
    # Normalized distance from physically expected values
    compensatory_params = ['frozenPrecipMultip', 'tempRangeTimestep', 'mw_exp']
    compensatory_params = [p for p in compensatory_params if p in df_params.columns]

    for forcing in forcings_with_params:
        cal_kge = cal_metrics.get(forcing, {}).get('kge', np.nan)
        eval_kge = eval_metrics.get(forcing, {}).get('kge', np.nan)
        if np.isnan(cal_kge) or np.isnan(eval_kge):
            continue

        kge_degradation = cal_kge - eval_kge

        # Distortion: how far frozenPrecipMultip is from 1.0
        fpm = all_params[forcing].get('frozenPrecipMultip', 1.0)
        distortion = abs(fpm - 1.0)

        ax_distort.scatter(distortion, kge_degradation, c=ALL_COLORS[forcing],
                         s=150, edgecolors='black', linewidth=1, zorder=5)
        ax_distort.annotate(ALL_LABELS.get(forcing, forcing).split(' (')[0],
                          (distortion, kge_degradation),
                          textcoords="offset points", xytext=(8, 8),
                          fontsize=8.5, fontweight='bold',
                          color=ALL_COLORS[forcing])

    ax_distort.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8,
                      label='No degradation')
    ax_distort.set_xlabel('Precipitation Correction Distortion |fPM - 1.0|', fontsize=11)
    ax_distort.set_ylabel('KGE Degradation (Cal - Eval)', fontsize=11)
    ax_distort.set_title('Parameter Distortion vs.\nGeneralization Loss', fontsize=11,
                        fontweight='bold')
    ax_distort.legend(fontsize=8, loc='upper left')
    ax_distort.grid(True, alpha=0.2)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_parameter_table(
    all_params: Dict[str, Dict],
    cal_metrics: Dict[str, Dict],
    eval_metrics: Dict[str, Dict],
    output_path: Path,
):
    """Generate a CSV table of calibrated parameters with metrics context."""
    forcings = [f for f in ALL_FORCINGS if f in all_params]
    rows = []
    for forcing in forcings:
        row = {'Forcing': ALL_LABELS.get(forcing, forcing)}
        for k, v in all_params[forcing].items():
            row[k] = round(v, 4) if abs(v) < 10000 else round(v, 0)
        row['Cal_KGE'] = round(cal_metrics.get(forcing, {}).get('kge', float('nan')), 3)
        row['Eval_KGE'] = round(eval_metrics.get(forcing, {}).get('kge', float('nan')), 3)
        row['KGE_Degradation'] = round(row['Cal_KGE'] - row['Eval_KGE'], 3) if not (
            np.isnan(row['Cal_KGE']) or np.isnan(row['Eval_KGE'])) else float('nan')
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")

    # Console summary
    print("\n  Parameter Divergence Summary:")
    key_cols = ['Forcing', 'frozenPrecipMultip', 'mw_exp', 'tempRangeTimestep',
                'Cal_KGE', 'Eval_KGE', 'KGE_Degradation']
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].to_string(index=False))


def main():
    """Main analysis routine."""
    parser = argparse.ArgumentParser(
        description='Analyze results from the Forcing Ensemble Study'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=RESULTS_DIR,
        help='Output directory for plots and tables'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=SYMFLUENCE_DATA_DIR,
        help='SYMFLUENCE data directory'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Forcing Ensemble Study - Results Analysis")
    print("=" * 60)

    # Create output directories
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    results = {}
    cal_metrics = {}
    eval_metrics = {}
    available_forcings = []

    print("\nLoading data...")
    for forcing in ALL_FORCINGS:
        domain_name = f"domain_paradise_snotel_wa_{forcing}"
        domain_dir = args.data_dir / domain_name

        if not domain_dir.exists():
            print(f"  {ALL_LABELS.get(forcing, forcing)}: Domain directory not found")
            continue

        print(f"  {ALL_LABELS.get(forcing, forcing)}: Found domain directory")

        # Load pre-computed calibration metrics from JSON
        eval_data = load_calibration_metrics(domain_dir, forcing)
        if eval_data is not None:
            cm = eval_data.get('calibration_metrics', {})
            cal_metrics[forcing] = {
                'rmse': cm.get('RMSE', np.nan),
                'bias': cm.get('PBIAS', np.nan),
                'kge': cm.get('KGE', np.nan),
                'corr': cm.get('r', np.nan),
                'nse': cm.get('NSE', np.nan),
            }
            em = eval_data.get('evaluation_metrics', {})
            eval_metrics[forcing] = {
                'rmse': em.get('RMSE', np.nan),
                'bias': em.get('PBIAS', np.nan),
                'kge': em.get('KGE', np.nan),
                'corr': em.get('r', np.nan),
                'nse': em.get('NSE', np.nan),
            }
            print(f"    Calibration KGE: {cal_metrics[forcing]['kge']:.3f}, "
                  f"Evaluation KGE: {eval_metrics[forcing]['kge']:.3f}")

        # Load simulated SWE time series
        sim_ds = load_simulated_swe(domain_dir, forcing)
        if sim_ds is not None:
            swe_vars = ['scalarSWE', 'SWE', 'swe', 'snow_water_equivalent']
            for var in swe_vars:
                if var in sim_ds.data_vars:
                    results[forcing] = {
                        'time': pd.to_datetime(sim_ds['time'].values),
                        'swe': sim_ds[var].values.flatten()
                    }
                    available_forcings.append(forcing)
                    print(f"    Loaded SWE time series ({len(sim_ds['time'])} timesteps)")
                    break
        else:
            print("    No simulation output found (calibration may not be complete)")

    # Load observed data (from first available domain with observations)
    obs_loaded = False
    for forcing in ALL_FORCINGS:
        domain_dir = args.data_dir / f"domain_paradise_snotel_wa_{forcing}"
        if domain_dir.exists():
            obs_df = load_observed_swe(domain_dir)
            if obs_df is not None and 'swe' in obs_df.columns:
                # Convert SNOTEL inches to mm to match SUMMA output units
                swe_mm = obs_df['swe'].values * INCHES_TO_MM
                results['observed'] = {
                    'time': obs_df.index,
                    'swe': swe_mm
                }
                print(f"  Observed: Loaded SNOTEL data ({len(obs_df)} records, converted in -> mm)")
                obs_loaded = True
                break

    if not obs_loaded:
        print("  WARNING: Could not load observed SNOTEL data")

    # Load calibration/evaluation period definitions from a config
    cal_start, cal_end = pd.Timestamp('2015-10-01'), pd.Timestamp('2018-09-30')
    eval_start, eval_end = pd.Timestamp('2018-10-01'), pd.Timestamp('2020-09-30')
    sim_start, sim_end = pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-31')
    try:
        import yaml
        cfg_file = CONFIGS_DIR / "config_paradise_aorc.yaml"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            if 'CALIBRATION_PERIOD' in cfg:
                parts = [s.strip() for s in cfg['CALIBRATION_PERIOD'].split(',')]
                cal_start, cal_end = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
            if 'EVALUATION_PERIOD' in cfg:
                parts = [s.strip() for s in cfg['EVALUATION_PERIOD'].split(',')]
                eval_start, eval_end = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
            if 'EXPERIMENT_TIME_START' in cfg:
                sim_start = pd.Timestamp(cfg['EXPERIMENT_TIME_START'])
            if 'EXPERIMENT_TIME_END' in cfg:
                sim_end = pd.Timestamp(cfg['EXPERIMENT_TIME_END'])
            print(f"\n  Calibration period: {cal_start.date()} to {cal_end.date()}")
            print(f"  Evaluation period:  {eval_start.date()} to {eval_end.date()}")
    except ImportError:
        print("  (yaml not available, using default period definitions)")

    # Calculate metrics from time series for forcings without JSON evaluation
    print("\nComputing metrics from time series where needed...")
    if 'observed' in results:
        obs_time = pd.DatetimeIndex(results['observed']['time'])
        obs_series = pd.Series(results['observed']['swe'], index=obs_time)
        for forcing in available_forcings:
            if forcing not in cal_metrics:
                sim_time = pd.DatetimeIndex(results[forcing]['time'])
                sim_series = pd.Series(results[forcing]['swe'], index=sim_time)

                # Calibration period metrics
                cal_obs = obs_series.loc[cal_start:cal_end]
                cal_sim = sim_series.loc[cal_start:cal_end]
                common_cal = cal_obs.index.intersection(cal_sim.index)
                if len(common_cal) > 0:
                    cal_metrics[forcing] = calculate_metrics(
                        cal_obs.loc[common_cal].values,
                        cal_sim.loc[common_cal].values
                    )
                    print(f"  {ALL_LABELS.get(forcing, forcing)} (cal): "
                          f"KGE={cal_metrics[forcing]['kge']:.3f}, "
                          f"RMSE={cal_metrics[forcing]['rmse']:.1f} mm")

                # Evaluation period metrics
                eval_obs = obs_series.loc[eval_start:eval_end]
                eval_sim = sim_series.loc[eval_start:eval_end]
                common_eval = eval_obs.index.intersection(eval_sim.index)
                if len(common_eval) > 0:
                    eval_metrics[forcing] = calculate_metrics(
                        eval_obs.loc[common_eval].values,
                        eval_sim.loc[common_eval].values
                    )
                    print(f"  {ALL_LABELS.get(forcing, forcing)} (eval): "
                          f"KGE={eval_metrics[forcing]['kge']:.3f}, "
                          f"RMSE={eval_metrics[forcing]['rmse']:.1f} mm")

    # Summary
    print(f"\n  Forcings with results: {len(available_forcings)}/{len(ALL_FORCINGS)}")
    for f in available_forcings:
        print(f"    - {ALL_LABELS.get(f, f)}")
    missing = [f for f in ALL_FORCINGS if f not in available_forcings]
    if missing:
        print(f"  Awaiting calibration: {len(missing)}")
        for f in missing:
            print(f"    - {ALL_LABELS.get(f, f)}")

    # --- Load soil moisture data ---
    print("\nLoading soil moisture data...")
    sm_obs = None
    sm_sim_results = {}

    # Load observed SM from closest ISMN station (use first domain that has it)
    for forcing in ALL_FORCINGS:
        domain_dir = args.data_dir / f"domain_paradise_snotel_wa_{forcing}"
        if domain_dir.exists():
            sm_obs = load_observed_sm(domain_dir)
            if sm_obs is not None:
                print(f"  Observed SM: {len(sm_obs)} daily records "
                      f"({sm_obs.index.min().date()} to {sm_obs.index.max().date()}), "
                      f"depths: {[c.replace('sm_','') + 'm' for c in sm_obs.columns]}")
                break

    if sm_obs is None:
        print("  No observed soil moisture data found")

    # Load simulated SM for each available forcing
    for forcing in available_forcings:
        domain_dir = args.data_dir / f"domain_paradise_snotel_wa_{forcing}"
        sm_data = load_simulated_sm(domain_dir, forcing)
        if sm_data is not None:
            n_valid = (~np.isnan(sm_data['sm'])).sum()
            print(f"  {ALL_LABELS.get(forcing, forcing)}: "
                  f"Loaded simulated SM ({n_valid} valid timesteps)")
            sm_sim_results[forcing] = sm_data

    # Separate reanalysis and GDDP forcings that have results
    avail_reanalysis = [f for f in FORCINGS if f in available_forcings]
    avail_gddp = [f for f in GDDP_FORCINGS if f in available_forcings]

    # Forcings with metrics (from JSON) even if time series unavailable
    forcings_with_metrics = [f for f in ALL_FORCINGS
                             if f in cal_metrics or f in eval_metrics]

    # Generate outputs
    print("\nGenerating outputs...")

    # Main comparison plot: all forcings, clipped to simulation period
    if results:
        plot_swe_timeseries(results, plots_dir / "swe_timeseries_comparison.png",
                           forcings_to_plot=available_forcings,
                           date_range=(sim_start, sim_end))

    # Separate reanalysis-only plot if multiple reanalysis available
    if len(avail_reanalysis) > 1:
        plot_swe_timeseries(results, plots_dir / "swe_timeseries_reanalysis.png",
                           forcings_to_plot=avail_reanalysis,
                           date_range=(sim_start, sim_end),
                           title_suffix='Reanalysis Products')

    # GDDP projection plot if available
    if avail_gddp:
        plot_swe_timeseries(results, plots_dir / "swe_timeseries_gddp.png",
                           forcings_to_plot=avail_gddp,
                           date_range=(sim_start, sim_end),
                           title_suffix='GDDP Projections (Historical Period)')

    if cal_metrics:
        plot_metrics_comparison(cal_metrics, plots_dir / "metrics_comparison_calibration.png",
                               forcings_to_plot=forcings_with_metrics, period='calibration')
    if eval_metrics:
        plot_metrics_comparison(eval_metrics, plots_dir / "metrics_comparison_evaluation.png",
                               forcings_to_plot=forcings_with_metrics, period='evaluation')

    if cal_metrics or eval_metrics:
        generate_summary_table(cal_metrics, eval_metrics,
                              args.output_dir / "performance_summary.csv",
                              forcings_to_report=forcings_with_metrics)

    # Combined SWE + Soil Moisture panel figure
    if results and sm_sim_results:
        cal_eval = ((cal_start, cal_end), (eval_start, eval_end))

        # All forcings combined
        plot_swe_sm_combined(results, sm_sim_results, sm_obs,
                            plots_dir / "swe_sm_combined.png",
                            forcings_to_plot=available_forcings,
                            date_range=(sim_start, sim_end),
                            cal_eval_periods=cal_eval)

        # Reanalysis-only combined
        if len(avail_reanalysis) > 1:
            plot_swe_sm_combined(results, sm_sim_results, sm_obs,
                                plots_dir / "swe_sm_reanalysis.png",
                                forcings_to_plot=avail_reanalysis,
                                date_range=(sim_start, sim_end),
                                cal_eval_periods=cal_eval)

    # --- Parameter divergence analysis ---
    print("\nLoading calibrated parameters for divergence analysis...")
    all_params = load_all_best_params(args.data_dir)
    print(f"  Loaded parameters for {len(all_params)} forcings: "
          f"{[ALL_LABELS.get(f, f).split(' (')[0] for f in all_params]}")

    if len(all_params) >= 2:
        plot_parameter_divergence(all_params, cal_metrics, eval_metrics,
                                 plots_dir / "parameter_divergence.png")
        generate_parameter_table(all_params, cal_metrics, eval_metrics,
                                args.output_dir / "parameter_divergence.csv")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
