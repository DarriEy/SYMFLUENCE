#!/usr/bin/env python3
"""
Bow at Banff Multivariate Evaluation - Publication Figure (v4)
Section 4.10a: GRACE TWS + Streamflow + SWE

Clean version without soil moisture (ISMN stations too far, satellite products unavailable).
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    'station_swe': '#2E86AB',   # Blue for SWE stations
    'station_q': '#E94F37',     # Red for streamflow gauge
}


def load_summa_output(experiment_id='bow_tws_uncalibrated'):
    """Load SUMMA daily output."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"
    timestep_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_timestep.nc"

    print(f"Loading SUMMA output: {experiment_id}")
    ds_day = xr.open_dataset(daily_path)

    times = pd.to_datetime(ds_day.time.values)

    # TWS components
    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,
    }
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    df = pd.DataFrame(data).set_index('time')
    ds_day.close()

    # Load runoff
    if timestep_path.exists():
        ds_ts = xr.open_dataset(timestep_path)
        ts_times = pd.to_datetime(ds_ts.time.values)
        if 'averageRoutedRunoff' in ds_ts:
            runoff_ms = ds_ts['averageRoutedRunoff'].values.flatten()
            runoff_m3s = runoff_ms * CATCHMENT_AREA_KM2 * 1e6
            df_runoff = pd.DataFrame({'sim_Q': runoff_m3s}, index=ts_times)
            df = df.join(df_runoff.resample('D').mean(), how='left')
        ds_ts.close()

    print(f"  Loaded {len(df)} days")
    return df


def load_streamflow_obs():
    """Load observed streamflow."""
    obs_path = DATA_DIR / "observations/streamflow/preprocessed/Bow_at_Banff_multivar_streamflow_processed.csv"
    df = pd.read_csv(obs_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = ['obs_Q']
    return df


def load_grace_tws():
    """Load GRACE TWS anomalies (convert cm to mm)."""
    grace_path = DATA_DIR / "observations/grace/preprocessed/Bow_at_Banff_multivar_grace_tws_processed.csv"
    df = pd.read_csv(grace_path, index_col=0, parse_dates=True)
    df['GRACE_TWS'] = df['grace_csr_anomaly'] * 10  # cm -> mm
    return df


def load_canswe_swe():
    """Load CanSWE SWE observations."""
    swe_path = DATA_DIR / "observations/snow/preprocessed/Bow_at_Banff_multivar_swe_processed.csv"
    df = pd.read_csv(swe_path, parse_dates=['datetime'], index_col='datetime')
    return df


def load_canswe_stations():
    """Load CanSWE station locations."""
    stations_path = DATA_DIR / "observations/snow/preprocessed/Bow_at_Banff_multivar_canswe_stations.csv"
    if stations_path.exists():
        return pd.read_csv(stations_path)
    return None


def load_catchment_shapefile():
    """Load catchment boundary."""
    shp_path = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"
    if shp_path.exists():
        return gpd.read_file(shp_path)
    shp_path = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp")
    if shp_path.exists():
        return gpd.read_file(shp_path)
    return None


def filter_period(df, start=CALIBRATION_START, end=EVALUATION_END):
    """Filter to analysis period."""
    if df is None:
        return None
    return df[(df.index >= start) & (df.index <= end)].copy()


def calculate_tws_anomaly(df):
    """Calculate TWS anomaly relative to baseline."""
    baseline_mask = (df.index >= GRACE_BASELINE_START) & (df.index <= GRACE_BASELINE_END)
    baseline_mean = df.loc[baseline_mask, 'TWS'].mean()
    df['TWS_anomaly'] = df['TWS'] - baseline_mean
    return df, baseline_mean


def calculate_metrics(sim, obs, use_period=True):
    """Calculate r, NSE, KGE, PBIAS, RMSE, bias."""
    if sim is None or obs is None:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    if use_period:
        sim_p = sim.copy()
        obs_p = obs.copy()
        sim_p.index = sim_p.index.to_period('M')
        obs_p.index = obs_p.index.to_period('M')
        common = sim_p.dropna().index.intersection(obs_p.dropna().index)
        if len(common) < 3:
            return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}
        s, o = sim_p.loc[common].values, obs_p.loc[common].values
    else:
        common = sim.dropna().index.intersection(obs.dropna().index)
        if len(common) < 10:
            return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}
        s, o = sim.loc[common].values, obs.loc[common].values

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


def add_period_shading(ax):
    """Add calibration/evaluation period shading."""
    ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
               alpha=0.12, color='#2CA02C', label='Calibration (2004-2010)')
    ax.axvspan(pd.Timestamp(EVALUATION_START), pd.Timestamp(EVALUATION_END),
               alpha=0.12, color='#1F77B4', label='Evaluation (2011-2017)')


def plot_domain_map_with_stations(gdf, df_swe_stations, ax):
    """Plot domain map with observation station locations."""
    from shapely.geometry import Point

    if gdf is None:
        ax.text(0.5, 0.5, 'Shapefile not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(a) Study Domain', fontweight='bold')
        return

    # Plot catchment
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf_wgs.plot(ax=ax, facecolor='#9ECAE1', edgecolor='#08519C', linewidth=1.5, alpha=0.7)

    # Get bounds
    bounds = gdf_wgs.total_bounds
    margin_x = (bounds[2] - bounds[0]) * 0.15
    margin_y = (bounds[3] - bounds[1]) * 0.15

    n_inside = 0
    n_outside = 0

    # Plot CanSWE stations - distinguish inside vs outside catchment
    if df_swe_stations is not None:
        # Create GeoDataFrame for spatial operations
        geometry = [Point(xy) for xy in zip(df_swe_stations['lon'], df_swe_stations['lat'])]
        gdf_stations = gpd.GeoDataFrame(df_swe_stations.copy(), geometry=geometry, crs='EPSG:4326')

        # Find stations inside catchment
        stations_inside = gpd.sjoin(gdf_stations, gdf_wgs, how='inner', predicate='within')
        inside_ids = set(stations_inside.index)

        # Filter to stations within map view
        mask = ((df_swe_stations['lon'] >= bounds[0] - margin_x) &
                (df_swe_stations['lon'] <= bounds[2] + margin_x) &
                (df_swe_stations['lat'] >= bounds[1] - margin_y) &
                (df_swe_stations['lat'] <= bounds[3] + margin_y))
        stations_in_view = df_swe_stations[mask]

        # Plot stations INSIDE catchment (filled markers)
        inside_mask = stations_in_view.index.isin(inside_ids)
        stations_in = stations_in_view[inside_mask]
        stations_out = stations_in_view[~inside_mask]

        n_inside = len(stations_in)
        n_outside = len(stations_out)

        if len(stations_in) > 0:
            ax.scatter(stations_in['lon'], stations_in['lat'],
                       c=COLORS['station_swe'], s=40, marker='s', alpha=0.9,
                       edgecolors='white', linewidth=1, zorder=9,
                       label=f'CanSWE (n={n_inside})')

        # Plot stations OUTSIDE catchment (hollow markers)
        if len(stations_out) > 0:
            ax.scatter(stations_out['lon'], stations_out['lat'],
                       facecolors='none', edgecolors=COLORS['station_swe'], s=35, marker='s',
                       alpha=0.6, linewidth=1.2, zorder=8,
                       label=f'CanSWE nearby ({n_outside})')

    # Pour point / WSC gauge
    ax.plot(BOW_LON, BOW_LAT, 'r^', markersize=11, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10, label='WSC 05BB001')

    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    # Info box
    area_km2 = gdf_wgs.to_crs(epsg=32611).area.sum() / 1e6
    info_text = f"Bow River at Banff\n{area_km2:,.0f} km²\n1400-3400 m elev."
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Legend
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9, markerscale=0.9)

    # North arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.82), xycoords='axes fraction',
                fontsize=10, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(a) Study Domain', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)


def plot_tws_comparison(df_sim, df_grace, metrics, ax):
    """Plot TWS anomaly comparison."""
    df_sim = filter_period(df_sim)
    df_grace = filter_period(df_grace)

    sim_monthly = df_sim['TWS_anomaly'].resample('M').mean()

    add_period_shading(ax)

    ax.plot(sim_monthly.index, sim_monthly.values, color=COLORS['sim'],
            linewidth=1.2, alpha=0.9, label='Simulated')
    ax.plot(df_grace.index, df_grace['GRACE_TWS'].values, color=COLORS['grace'],
            linewidth=1.8, marker='o', markersize=3, alpha=0.9, label='GRACE CSR')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    metrics_text = f"r = {metrics['r']:.2f}\nRMSE = {metrics['RMSE']:.0f} mm\nBias = {metrics['bias']:.0f} mm"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('TWS Anomaly (mm)')
    ax.set_title('(b) Total Water Storage vs GRACE', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_streamflow(df_sim, df_obs, metrics, ax):
    """Plot streamflow comparison."""
    df_sim = filter_period(df_sim)
    df_obs = filter_period(df_obs)

    add_period_shading(ax)

    sim_weekly = df_sim['sim_Q'].resample('W').mean()
    obs_weekly = df_obs['obs_Q'].resample('W').mean()

    ax.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'],
            linewidth=0.9, alpha=0.8, label='Observed')
    ax.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'],
            linewidth=0.9, alpha=0.8, label='Simulated')

    metrics_text = f"r = {metrics['r']:.2f}, NSE = {metrics['NSE']:.2f}\nKGE = {metrics['KGE']:.2f}, PBIAS = {metrics['PBIAS']:.0f}%"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('(c) Streamflow', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_swe(df_sim, df_swe, metrics, ax):
    """Plot SWE comparison."""
    df_sim = filter_period(df_sim)
    df_swe = filter_period(df_swe)

    add_period_shading(ax)

    sim_weekly = df_sim['SWE'].resample('W').mean()
    obs_weekly = df_swe['swe_mm'].resample('W').mean()

    ax.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'],
            linewidth=0.9, alpha=0.8, label='CanSWE')
    ax.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'],
            linewidth=0.9, alpha=0.8, label='Simulated')

    metrics_text = f"r = {metrics['r']:.2f}\nRMSE = {metrics['RMSE']:.0f} mm\nBias = {metrics['bias']:.0f} mm"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('SWE (mm)')
    ax.set_xlabel('Date')
    ax.set_title('(d) Snow Water Equivalent', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(CALIBRATION_START), pd.Timestamp(EVALUATION_END))


def plot_scatter_panels(df_sim, df_grace, df_obs, df_swe, axs):
    """Plot scatter comparison panels."""
    df_sim = filter_period(df_sim)

    # TWS scatter
    ax = axs[0]
    sim_monthly = df_sim['TWS_anomaly'].resample('M').mean()
    df_g = filter_period(df_grace)
    sim_p = sim_monthly.copy()
    sim_p.index = sim_p.index.to_period('M')
    grace_p = df_g['GRACE_TWS'].copy()
    grace_p.index = grace_p.index.to_period('M')
    common = sim_p.dropna().index.intersection(grace_p.dropna().index)
    if len(common) > 0:
        ax.scatter(grace_p.loc[common], sim_p.loc[common], c=COLORS['grace'], s=25, alpha=0.7)
        lims = [min(grace_p.loc[common].min(), sim_p.loc[common].min()) - 20,
                max(grace_p.loc[common].max(), sim_p.loc[common].max()) + 20]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel('GRACE (mm)', fontsize=8)
    ax.set_ylabel('Simulated (mm)', fontsize=8)
    ax.set_title('(e) TWS', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')

    # Streamflow scatter (evaluation period only)
    ax = axs[1]
    df_q = filter_period(df_obs).resample('M').mean()
    sim_q = df_sim['sim_Q'].resample('M').mean()
    common = sim_q.dropna().index.intersection(df_q['obs_Q'].dropna().index)
    common = common[(common >= EVALUATION_START) & (common <= EVALUATION_END)]
    if len(common) > 0:
        ax.scatter(df_q.loc[common, 'obs_Q'], sim_q.loc[common], c=COLORS['obs'], s=25, alpha=0.7)
        max_val = max(df_q.loc[common, 'obs_Q'].max(), sim_q.loc[common].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    ax.set_xlabel('Observed (m³/s)', fontsize=8)
    ax.set_ylabel('Simulated (m³/s)', fontsize=8)
    ax.set_title('(f) Streamflow (Eval)', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')

    # SWE scatter
    ax = axs[2]
    df_s = filter_period(df_swe)
    common = df_sim['SWE'].dropna().index.intersection(df_s['swe_mm'].dropna().index)
    if len(common) > 0:
        common_sub = common[::7]  # Weekly sampling
        ax.scatter(df_s.loc[common_sub, 'swe_mm'], df_sim.loc[common_sub, 'SWE'],
                   c=COLORS['swe_sim'], s=10, alpha=0.5)
        max_val = max(df_s.loc[common, 'swe_mm'].max(), df_sim.loc[common, 'SWE'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    ax.set_xlabel('CanSWE (mm)', fontsize=8)
    ax.set_ylabel('Simulated (mm)', fontsize=8)
    ax.set_title('(g) SWE', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')

    # Seasonal TWS cycle
    ax = axs[3]
    sim_monthly_full = df_sim['TWS_anomaly'].copy()
    sim_seasonal = sim_monthly_full.groupby(sim_monthly_full.index.month).mean()
    df_g = filter_period(df_grace)
    grace_seasonal = df_g['GRACE_TWS'].groupby(df_g.index.month).mean()

    months = range(1, 13)
    ax.plot(months, [sim_seasonal.get(m, np.nan) for m in months], 'o-',
            color=COLORS['sim'], linewidth=1.5, markersize=5, label='Simulated')
    ax.plot(months, [grace_seasonal.get(m, np.nan) for m in months], 's-',
            color=COLORS['grace'], linewidth=1.5, markersize=5, label='GRACE')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Month', fontsize=8)
    ax.set_ylabel('TWS Anom. (mm)', fontsize=8)
    ax.set_title('(h) Seasonal TWS Cycle', fontsize=9, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(loc='lower left', fontsize=7)


def main():
    """Create publication figure."""
    print("=" * 60)
    print("Bow at Banff - Publication Figure (v4)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_summa = load_summa_output()
    df_obs = load_streamflow_obs()
    df_grace = load_grace_tws()
    df_swe = load_canswe_swe()
    df_swe_stations = load_canswe_stations()
    gdf = load_catchment_shapefile()

    # Calculate TWS anomaly
    df_summa, baseline = calculate_tws_anomaly(df_summa)
    print(f"  TWS baseline: {baseline:.1f} mm")

    # Calculate metrics
    print("\nCalculating metrics...")
    sim_monthly = df_summa['TWS_anomaly'].resample('M').mean()
    grace_filt = filter_period(df_grace)
    grace_metrics = calculate_metrics(sim_monthly, grace_filt['GRACE_TWS'], use_period=True)
    print(f"  GRACE TWS: r={grace_metrics['r']:.3f}, RMSE={grace_metrics['RMSE']:.1f} mm, Bias={grace_metrics['bias']:.1f} mm")

    df_sim_eval = df_summa[(df_summa.index >= EVALUATION_START) & (df_summa.index <= EVALUATION_END)]
    df_obs_eval = df_obs[(df_obs.index >= EVALUATION_START) & (df_obs.index <= EVALUATION_END)]
    q_metrics = calculate_metrics(df_sim_eval['sim_Q'], df_obs_eval['obs_Q'], use_period=False)
    print(f"  Streamflow (Eval): r={q_metrics['r']:.3f}, NSE={q_metrics['NSE']:.3f}, KGE={q_metrics['KGE']:.3f}, PBIAS={q_metrics['PBIAS']:.1f}%")

    df_sim_filt = filter_period(df_summa)
    df_swe_filt = filter_period(df_swe)
    swe_metrics = calculate_metrics(df_sim_filt['SWE'], df_swe_filt['swe_mm'], use_period=False)
    print(f"  SWE: r={swe_metrics['r']:.3f}, RMSE={swe_metrics['RMSE']:.1f} mm, Bias={swe_metrics['bias']:.1f} mm")

    # Create figure
    print("\nCreating figure...")
    fig = plt.figure(figsize=(7.5, 9.5))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.1, 0.8, 0.8, 0.9],
                  hspace=0.35, wspace=0.3)

    # (a) Domain map with stations
    ax_map = fig.add_subplot(gs[0, 0])
    plot_domain_map_with_stations(gdf, df_swe_stations, ax_map)

    # (b) TWS vs GRACE
    ax_tws = fig.add_subplot(gs[0, 1])
    plot_tws_comparison(df_summa, df_grace, grace_metrics, ax_tws)

    # (c) Streamflow
    ax_q = fig.add_subplot(gs[1, :])
    plot_streamflow(df_summa, df_obs, q_metrics, ax_q)

    # (d) SWE
    ax_swe = fig.add_subplot(gs[2, :])
    plot_swe(df_summa, df_swe, swe_metrics, ax_swe)

    # Scatter panels
    gs_scatter = gs[3, :].subgridspec(1, 4, wspace=0.5)
    ax_scatter = [fig.add_subplot(gs_scatter[0, i]) for i in range(4)]
    plot_scatter_panels(df_summa, df_grace, df_obs, df_swe, ax_scatter)

    plt.subplots_adjust(bottom=0.06)
    fig.suptitle('Bow River at Banff: Multivariate Evaluation (Uncalibrated)',
                 fontsize=11, fontweight='bold', y=0.98)

    # Save
    output_png = OUTPUT_DIR / "bow_banff_figure_v4.png"
    output_pdf = OUTPUT_DIR / "bow_banff_figure_v4.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_png}")

    plt.close(fig)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([
        {'Variable': 'GRACE TWS', 'Metric': 'r', 'Value': grace_metrics['r'], 'Period': '2004-2017'},
        {'Variable': 'GRACE TWS', 'Metric': 'RMSE (mm)', 'Value': grace_metrics['RMSE'], 'Period': '2004-2017'},
        {'Variable': 'GRACE TWS', 'Metric': 'Bias (mm)', 'Value': grace_metrics['bias'], 'Period': '2004-2017'},
        {'Variable': 'Streamflow', 'Metric': 'r', 'Value': q_metrics['r'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'Streamflow', 'Metric': 'NSE', 'Value': q_metrics['NSE'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'Streamflow', 'Metric': 'KGE', 'Value': q_metrics['KGE'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'Streamflow', 'Metric': 'PBIAS (%)', 'Value': q_metrics['PBIAS'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'SWE', 'Metric': 'r', 'Value': swe_metrics['r'], 'Period': '2004-2017'},
        {'Variable': 'SWE', 'Metric': 'RMSE (mm)', 'Value': swe_metrics['RMSE'], 'Period': '2004-2017'},
        {'Variable': 'SWE', 'Metric': 'Bias (mm)', 'Value': swe_metrics['bias'], 'Period': '2004-2017'},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "bow_banff_metrics_v4.csv", index=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
