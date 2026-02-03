#!/usr/bin/env python3
"""
Bow at Banff Multivariate Evaluation - Publication Figure (v2)
Section 4.10a: GRACE TWS + Streamflow + SWE Comparison

Compact 4-panel layout for journal publication:
- (a) Domain map with inset
- (b) TWS components with GRACE overlay
- (c) Streamflow comparison with metrics
- (d) SWE comparison with CanSWE
- (e) Summary scatter plots (2x2)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path
try:
    SCRIPTS_DIR = Path(__file__).parent
except NameError:
    SCRIPTS_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/scripts")
sys.path.insert(0, str(SCRIPTS_DIR))

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time periods
SPINUP_END = '2003-12-31'
CALIBRATION_START = '2004-01-01'
CALIBRATION_END = '2010-12-31'
EVALUATION_START = '2011-01-01'
EVALUATION_END = '2017-12-31'
GRACE_BASELINE_START = '2004-01-01'
GRACE_BASELINE_END = '2009-12-31'

# Catchment info
BOW_LAT, BOW_LON = 51.17, -115.57
CATCHMENT_AREA_KM2 = 2210

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

COLORS = {
    'sim': '#2171B5',      # Blue
    'obs': '#D94801',      # Orange-red
    'grace': '#7A0177',    # Purple
    'swe_sim': '#6BAED6',  # Light blue
    'swe_obs': '#FD8D3C',  # Orange
    'soil': '#8B4513',     # Brown
    'aquifer': '#238B45',  # Green
    'cal_period': '#E5F5E0',   # Light green
    'eval_period': '#DEEBF7',  # Light blue
}


def load_summa_output(experiment_id='bow_tws_uncalibrated'):
    """Load SUMMA daily and timestep output."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"
    timestep_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_timestep.nc"

    print(f"Loading SUMMA output: {experiment_id}")
    ds_day = xr.open_dataset(daily_path)

    # Parse time
    time_values = ds_day.time.values
    if np.issubdtype(time_values.dtype, np.datetime64):
        times = pd.to_datetime(time_values)
    else:
        times = pd.to_datetime(time_values, unit='s', origin=pd.Timestamp('1990-01-01'))

    # Extract TWS components (all in kg/m2 = mm)
    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,  # m -> mm
    }
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    df = pd.DataFrame(data).set_index('time')
    ds_day.close()

    # Load runoff from timestep file
    if timestep_path.exists():
        ds_ts = xr.open_dataset(timestep_path)
        ts_time = ds_ts.time.values
        if np.issubdtype(ts_time.dtype, np.datetime64):
            ts_times = pd.to_datetime(ts_time)
        else:
            ts_times = pd.to_datetime(ts_time, unit='s', origin=pd.Timestamp('1990-01-01'))

        if 'averageRoutedRunoff' in ds_ts:
            runoff_ms = ds_ts['averageRoutedRunoff'].values.flatten()
            runoff_m3s = runoff_ms * CATCHMENT_AREA_KM2 * 1e6  # Convert to m³/s
            df_runoff = pd.DataFrame({'sim_Q': runoff_m3s}, index=ts_times)
            df = df.join(df_runoff.resample('D').mean(), how='left')
        ds_ts.close()

    print(f"  Loaded {len(df)} days: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df


def load_streamflow_obs():
    """Load observed streamflow."""
    obs_path = DATA_DIR / "observations/streamflow/preprocessed/Bow_at_Banff_multivar_streamflow_processed.csv"
    df = pd.read_csv(obs_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = ['obs_Q']
    return df


def load_grace_tws():
    """Load GRACE TWS anomalies.

    Note: GRACE mascon data is provided in cm of equivalent water height.
    Convert to mm for comparison with SUMMA output.
    """
    grace_path = DATA_DIR / "observations/grace/preprocessed/Bow_at_Banff_multivar_grace_tws_processed.csv"
    df = pd.read_csv(grace_path, index_col=0, parse_dates=True)
    df['GRACE_TWS'] = df['grace_csr_anomaly'] * 10  # cm -> mm
    return df


def load_canswe_swe():
    """Load CanSWE SWE observations."""
    swe_path = DATA_DIR / "observations/snow/preprocessed/Bow_at_Banff_multivar_swe_processed.csv"
    df = pd.read_csv(swe_path, parse_dates=['datetime'], index_col='datetime')
    return df


def load_catchment_shapefile():
    """Load catchment boundary."""
    shp_path = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"
    if shp_path.exists():
        return gpd.read_file(shp_path)
    # Fallback
    shp_path = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp")
    if shp_path.exists():
        return gpd.read_file(shp_path)
    return None


def filter_period(df, start=CALIBRATION_START, end=EVALUATION_END):
    """Filter to analysis period."""
    return df[(df.index >= start) & (df.index <= end)].copy()


def calculate_tws_anomaly(df):
    """Calculate TWS anomaly relative to baseline."""
    baseline_mask = (df.index >= GRACE_BASELINE_START) & (df.index <= GRACE_BASELINE_END)
    baseline_mean = df.loc[baseline_mask, 'TWS'].mean()
    df['TWS_anomaly'] = df['TWS'] - baseline_mean
    return df, baseline_mean


def calculate_metrics(sim, obs):
    """Calculate r, NSE, KGE, PBIAS, RMSE, bias."""
    # Handle both Series and DataFrame inputs
    if hasattr(sim, 'dropna'):
        sim_clean = sim.dropna()
    else:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    if hasattr(obs, 'dropna'):
        obs_clean = obs.dropna()
    else:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    # For monthly GRACE comparison, align by year-month
    sim_period = sim_clean.copy()
    obs_period = obs_clean.copy()

    # Convert to period for alignment
    sim_period.index = sim_period.index.to_period('M')
    obs_period.index = obs_period.index.to_period('M')

    common = sim_period.index.intersection(obs_period.index)

    if len(common) < 3:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    s = sim_period.loc[common].values
    o = obs_period.loc[common].values

    # Remove any remaining NaN pairs
    valid = ~(np.isnan(s) | np.isnan(o))
    s, o = s[valid], o[valid]

    if len(s) < 3:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    r = np.corrcoef(s, o)[0, 1]
    nse = 1 - np.sum((s - o)**2) / np.sum((o - np.mean(o))**2)
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(alpha) else np.nan
    pbias = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan
    rmse = np.sqrt(np.mean((s - o)**2))
    bias = np.mean(s - o)

    return {'r': r, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias, 'RMSE': rmse, 'bias': bias, 'n': len(s)}


def calculate_daily_metrics(sim, obs):
    """Calculate metrics for daily data (SWE, streamflow)."""
    if sim is None or obs is None:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    # Get common dates
    common = sim.dropna().index.intersection(obs.dropna().index)

    if len(common) < 10:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    s = sim.loc[common].values
    o = obs.loc[common].values

    # Remove NaN
    valid = ~(np.isnan(s) | np.isnan(o))
    s, o = s[valid], o[valid]

    if len(s) < 10:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    r = np.corrcoef(s, o)[0, 1]
    nse = 1 - np.sum((s - o)**2) / np.sum((o - np.mean(o))**2)
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(alpha) else np.nan
    pbias = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan
    rmse = np.sqrt(np.mean((s - o)**2))
    bias = np.mean(s - o)

    return {'r': r, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias, 'RMSE': rmse, 'bias': bias, 'n': len(s)}


def add_period_shading(ax, show_legend=False):
    """Add calibration/evaluation period shading."""
    cal_patch = ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
                           alpha=0.15, color='#2CA02C', label='Calibration (2004–2010)')
    eval_patch = ax.axvspan(pd.Timestamp(EVALUATION_START), pd.Timestamp(EVALUATION_END),
                            alpha=0.15, color='#1F77B4', label='Evaluation (2011–2017)')
    return cal_patch, eval_patch


def plot_domain_map(gdf, ax):
    """Plot domain map with regional inset."""
    if gdf is None:
        ax.text(0.5, 0.5, 'Shapefile not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(a) Bow River at Banff', fontweight='bold')
        return

    # Main map
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf_wgs.plot(ax=ax, facecolor='#9ECAE1', edgecolor='#08519C', linewidth=1.5, alpha=0.8)

    # Pour point
    ax.plot(BOW_LON, BOW_LAT, 'r^', markersize=8, markeredgecolor='white', markeredgewidth=1, zorder=10)

    # Bounds with margin
    bounds = gdf_wgs.total_bounds
    margin_x = (bounds[2] - bounds[0]) * 0.15
    margin_y = (bounds[3] - bounds[1]) * 0.15
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    # Info box
    area_km2 = gdf_wgs.to_crs(epsg=32611).area.sum() / 1e6
    info_text = f"Bow River at Banff\nArea: {area_km2:,.0f} km²\nElev: 1,400–3,400 m"
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # North arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.82), xycoords='axes fraction',
                fontsize=10, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

    # Regional inset
    inset_ax = ax.inset_axes([0.65, 0.02, 0.33, 0.35])
    inset_ax.set_xlim(-140, -100)
    inset_ax.set_ylim(45, 65)

    # Simple coastline approximation for inset
    inset_ax.fill_between([-140, -100], [45, 45], [65, 65], color='#f0f0f0', alpha=0.5)

    # Catchment location
    centroid = gdf_wgs.centroid.iloc[0]
    inset_ax.plot(centroid.x, centroid.y, 'ro', markersize=6, markeredgecolor='white')
    inset_ax.plot([centroid.x-2, centroid.x+2, centroid.x+2, centroid.x-2, centroid.x-2],
                  [centroid.y-1.5, centroid.y-1.5, centroid.y+1.5, centroid.y+1.5, centroid.y-1.5],
                  'r-', linewidth=1)

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title('Western Canada', fontsize=7)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.5)

    ax.set_xlabel('Longitude (°W)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('(a) Bow River at Banff', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)


def plot_tws_comparison(df_sim, df_grace, grace_metrics, ax):
    """Plot TWS anomaly comparison with GRACE."""
    df_sim = filter_period(df_sim)
    df_grace = filter_period(df_grace)

    # Monthly simulated TWS anomaly
    sim_monthly = df_sim['TWS_anomaly'].resample('M').mean()

    add_period_shading(ax)

    # Plot simulated
    ax.plot(sim_monthly.index, sim_monthly.values, color=COLORS['sim'],
            linewidth=1.2, alpha=0.9, label='Simulated TWS')

    # Plot GRACE
    ax.plot(df_grace.index, df_grace['GRACE_TWS'].values, color=COLORS['grace'],
            linewidth=1.8, marker='o', markersize=3, alpha=0.9, label='GRACE CSR')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Metrics box
    metrics_text = f"r = {grace_metrics['r']:.2f}\nRMSE = {grace_metrics['RMSE']:.0f} mm\nBias = {grace_metrics['bias']:.0f} mm"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('TWS Anomaly (mm)')
    ax.set_title('(b) Total Water Storage vs GRACE', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_streamflow_comparison(df_sim, df_obs, q_metrics, ax):
    """Plot streamflow comparison."""
    df_sim = filter_period(df_sim)
    df_obs = filter_period(df_obs)

    add_period_shading(ax)

    # Weekly means for clarity
    sim_weekly = df_sim['sim_Q'].resample('W').mean()
    obs_weekly = df_obs['obs_Q'].resample('W').mean()

    ax.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'],
            linewidth=0.8, alpha=0.8, label='Observed (WSC 05BB001)')
    ax.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'],
            linewidth=0.8, alpha=0.8, label='Simulated')

    # Metrics box (evaluation period)
    metrics_text = f"r = {q_metrics['r']:.2f}\nNSE = {q_metrics['NSE']:.2f}\nKGE = {q_metrics['KGE']:.2f}\nPBIAS = {q_metrics['PBIAS']:.0f}%"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('(c) Streamflow', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_swe_comparison(df_sim, df_swe, swe_metrics, ax):
    """Plot SWE comparison with CanSWE."""
    df_sim = filter_period(df_sim)
    df_swe = filter_period(df_swe)

    add_period_shading(ax, show_legend=True)

    # Weekly means
    sim_weekly = df_sim['SWE'].resample('W').mean()
    obs_weekly = df_swe['swe_mm'].resample('W').mean()

    ax.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'],
            linewidth=0.8, alpha=0.8, label='CanSWE (3 stations)')
    ax.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'],
            linewidth=0.8, alpha=0.8, label='Simulated')

    # Metrics box
    metrics_text = f"r = {swe_metrics['r']:.2f}\nRMSE = {swe_metrics['RMSE']:.0f} mm\nBias = {swe_metrics['bias']:.0f} mm"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_xlabel('Date')
    ax.set_ylabel('SWE (mm)')
    ax.set_title('(d) Snow Water Equivalent vs CanSWE', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_scatter_panels(df_sim, df_grace, df_obs, df_swe, axs):
    """Plot 2x2 scatter comparison panels."""
    df_sim = filter_period(df_sim)

    # (e) TWS scatter
    ax = axs[0]
    sim_monthly = df_sim['TWS_anomaly'].resample('M').mean()
    df_g = filter_period(df_grace)

    # Align by period
    sim_period = sim_monthly.copy()
    sim_period.index = sim_period.index.to_period('M')
    grace_period = df_g['GRACE_TWS'].copy()
    grace_period.index = grace_period.index.to_period('M')
    common = sim_period.dropna().index.intersection(grace_period.dropna().index)

    if len(common) > 0:
        ax.scatter(grace_period.loc[common], sim_period.loc[common],
                   c=COLORS['grace'], s=25, alpha=0.7, edgecolors='white', linewidth=0.5)
        lims = [min(grace_period.loc[common].min(), sim_period.loc[common].min()) - 20,
                max(grace_period.loc[common].max(), sim_period.loc[common].max()) + 20]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel('GRACE (mm)', fontsize=8)
    ax.set_ylabel('Simulated (mm)', fontsize=8)
    ax.set_title('(e) TWS Anomaly', fontsize=9, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)

    # (f) Streamflow scatter
    ax = axs[1]
    df_q = filter_period(df_obs).resample('M').mean()
    sim_q = df_sim['sim_Q'].resample('M').mean()
    common = sim_q.dropna().index.intersection(df_q['obs_Q'].dropna().index)
    # Filter to evaluation period only
    common = common[(common >= EVALUATION_START) & (common <= EVALUATION_END)]
    if len(common) > 0:
        ax.scatter(df_q.loc[common, 'obs_Q'], sim_q.loc[common],
                   c=COLORS['obs'], s=25, alpha=0.7, edgecolors='white', linewidth=0.5)
        max_val = max(df_q.loc[common, 'obs_Q'].max(), sim_q.loc[common].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel('Observed (m³/s)', fontsize=8)
    ax.set_ylabel('Simulated (m³/s)', fontsize=8)
    ax.set_title('(f) Streamflow', fontsize=9, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)

    # (g) SWE scatter
    ax = axs[2]
    df_s = filter_period(df_swe)
    common = df_sim['SWE'].dropna().index.intersection(df_s['swe_mm'].dropna().index)
    if len(common) > 0:
        # Subsample for clarity
        common_sub = common[::7]  # Weekly
        ax.scatter(df_s.loc[common_sub, 'swe_mm'], df_sim.loc[common_sub, 'SWE'],
                   c=COLORS['swe_sim'], s=15, alpha=0.5, edgecolors='white', linewidth=0.3)
        max_val = max(df_s.loc[common, 'swe_mm'].max(), df_sim.loc[common, 'SWE'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel('CanSWE (mm)', fontsize=8)
    ax.set_ylabel('Simulated (mm)', fontsize=8)
    ax.set_title('(g) SWE', fontsize=9, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)

    # (h) Mean seasonal TWS cycle
    ax = axs[3]
    sim_monthly = df_sim['TWS_anomaly'].copy()
    sim_monthly = sim_monthly[sim_monthly.index >= CALIBRATION_START]
    sim_seasonal = sim_monthly.groupby(sim_monthly.index.month).mean()

    df_g = filter_period(df_grace)
    grace_seasonal = df_g['GRACE_TWS'].groupby(df_g.index.month).mean()

    months = range(1, 13)
    ax.plot(months, [sim_seasonal.get(m, np.nan) for m in months], 'o-',
            color=COLORS['sim'], linewidth=1.5, markersize=4, label='Simulated')
    ax.plot(months, [grace_seasonal.get(m, np.nan) for m in months], 's-',
            color=COLORS['grace'], linewidth=1.5, markersize=4, label='GRACE')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Month', fontsize=8)
    ax.set_ylabel('TWS Anom. (mm)', fontsize=8)
    ax.set_title('(h) Seasonal Cycle', fontsize=9, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(loc='lower left', fontsize=6, framealpha=0.9)


def create_publication_figure():
    """Create the main publication figure."""
    print("=" * 60)
    print("Bow at Banff - Publication Figure Generation (v2)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_summa = load_summa_output('bow_tws_uncalibrated')
    df_obs = load_streamflow_obs()
    df_grace = load_grace_tws()
    df_swe = load_canswe_swe()
    gdf = load_catchment_shapefile()

    # Calculate TWS anomaly
    df_summa, baseline = calculate_tws_anomaly(df_summa)
    print(f"  TWS baseline mean: {baseline:.1f} mm")

    # Calculate metrics
    print("\nCalculating metrics...")

    # GRACE metrics (full period)
    sim_monthly = df_summa['TWS_anomaly'].resample('M').mean()
    grace_filt = filter_period(df_grace)
    grace_metrics = calculate_metrics(sim_monthly, grace_filt['GRACE_TWS'])
    print(f"  GRACE: r={grace_metrics['r']:.3f}, RMSE={grace_metrics['RMSE']:.1f} mm")

    # Streamflow metrics (evaluation period)
    df_sim_eval = df_summa[(df_summa.index >= EVALUATION_START) & (df_summa.index <= EVALUATION_END)]
    df_obs_eval = df_obs[(df_obs.index >= EVALUATION_START) & (df_obs.index <= EVALUATION_END)]
    q_metrics = calculate_daily_metrics(df_sim_eval['sim_Q'], df_obs_eval['obs_Q'])
    print(f"  Streamflow (eval): r={q_metrics['r']:.3f}, KGE={q_metrics['KGE']:.3f}, NSE={q_metrics['NSE']:.3f}")

    # SWE metrics (full analysis period)
    df_sim_filt = filter_period(df_summa)
    df_swe_filt = filter_period(df_swe)
    swe_metrics = calculate_daily_metrics(df_sim_filt['SWE'], df_swe_filt['swe_mm'])
    print(f"  SWE: r={swe_metrics['r']:.3f}, RMSE={swe_metrics['RMSE']:.1f} mm, Bias={swe_metrics['bias']:.1f} mm")

    # Create figure
    print("\nCreating figure...")
    fig = plt.figure(figsize=(7.5, 9.5))  # Single column width for journal

    # Layout: 2 columns, complex rows
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.75, 0.75, 1.0],
                  hspace=0.38, wspace=0.3)

    # (a) Domain map - left top
    ax_map = fig.add_subplot(gs[0, 0])
    plot_domain_map(gdf, ax_map)

    # (b) TWS comparison - right top, spanning
    ax_tws = fig.add_subplot(gs[0, 1])
    plot_tws_comparison(df_summa, df_grace, grace_metrics, ax_tws)

    # (c) Streamflow - full width
    ax_q = fig.add_subplot(gs[1, :])
    plot_streamflow_comparison(df_summa, df_obs, q_metrics, ax_q)

    # (d) SWE - full width
    ax_swe = fig.add_subplot(gs[2, :])
    plot_swe_comparison(df_summa, df_swe, swe_metrics, ax_swe)

    # (e-h) Scatter panels - 2x2
    ax_scatter = [
        fig.add_subplot(gs[3, 0]),  # TWS scatter + Streamflow scatter in one row
        fig.add_subplot(gs[3, 1]),
    ]

    # Actually, let's do 4 small panels in the bottom row
    # Recreate with nested gridspec
    gs_scatter = gs[3, :].subgridspec(1, 4, wspace=0.45)
    ax_scatter = [fig.add_subplot(gs_scatter[0, i]) for i in range(4)]
    plot_scatter_panels(df_summa, df_grace, df_obs, df_swe, ax_scatter)

    # Adjust bottom margin for scatter plot labels
    plt.subplots_adjust(bottom=0.06)

    # Title
    fig.suptitle('Bow River at Banff: Multivariate Evaluation (Uncalibrated)',
                 fontsize=11, fontweight='bold', y=0.98)

    # Save
    output_png = OUTPUT_DIR / "bow_banff_figure_v2.png"
    output_pdf = OUTPUT_DIR / "bow_banff_figure_v2.pdf"

    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_png}")
    print(f"Saved: {output_pdf}")

    plt.close(fig)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Variable': ['GRACE TWS', 'GRACE TWS', 'GRACE TWS', 'GRACE TWS',
                     'Streamflow', 'Streamflow', 'Streamflow', 'Streamflow', 'Streamflow',
                     'SWE', 'SWE', 'SWE', 'SWE'],
        'Metric': ['r', 'RMSE (mm)', 'Bias (mm)', 'n_months',
                   'r', 'NSE', 'KGE', 'PBIAS (%)', 'n_days',
                   'r', 'RMSE (mm)', 'Bias (mm)', 'n_days'],
        'Value': [grace_metrics['r'], grace_metrics['RMSE'], grace_metrics['bias'], grace_metrics['n'],
                  q_metrics['r'], q_metrics['NSE'], q_metrics['KGE'], q_metrics['PBIAS'], q_metrics['n'],
                  swe_metrics['r'], swe_metrics['RMSE'], swe_metrics['bias'], swe_metrics['n']],
        'Period': ['2004-2017', '2004-2017', '2004-2017', '2004-2017',
                   'Eval 2011-2017', 'Eval 2011-2017', 'Eval 2011-2017', 'Eval 2011-2017', 'Eval 2011-2017',
                   '2004-2017', '2004-2017', '2004-2017', '2004-2017']
    })
    metrics_path = OUTPUT_DIR / "bow_banff_metrics_v2.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

    return grace_metrics, q_metrics, swe_metrics


if __name__ == "__main__":
    create_publication_figure()
