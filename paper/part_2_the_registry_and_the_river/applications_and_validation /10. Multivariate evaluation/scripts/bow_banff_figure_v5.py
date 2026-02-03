#!/usr/bin/env python3
"""
Bow at Banff Multivariate Evaluation - Publication Figure (v5)
Section 4.10a: GRACE TWS + Streamflow + SWE + ET (SSEBop)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPEFILE = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"
SSEBOP_DIR = DATA_DIR / "observations/et/ssebop"

# Time periods
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
    'sim': '#2171B5',
    'obs': '#D94801',
    'grace': '#7A0177',
    'et': '#228B22',
    'station_swe': '#2E86AB',
}


def load_summa_output(experiment_id='bow_tws_uncalibrated'):
    """Load SUMMA daily output."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"
    timestep_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_timestep.nc"

    ds_day = xr.open_dataset(daily_path)
    times = pd.to_datetime(ds_day.time.values)

    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,
        'ET': -ds_day['scalarTotalET'].values.flatten() * 86400,  # kg/m²/s to mm/day
    }
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    df = pd.DataFrame(data).set_index('time')
    ds_day.close()

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


def load_streamflow_obs():
    obs_path = DATA_DIR / "observations/streamflow/preprocessed/Bow_at_Banff_multivar_streamflow_processed.csv"
    df = pd.read_csv(obs_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = ['obs_Q']
    return df


def load_grace_tws():
    grace_path = DATA_DIR / "observations/grace/preprocessed/Bow_at_Banff_multivar_grace_tws_processed.csv"
    df = pd.read_csv(grace_path, index_col=0, parse_dates=True)
    df['GRACE_TWS'] = df['grace_csr_anomaly'] * 10  # cm -> mm
    return df


def load_canswe_swe():
    swe_path = DATA_DIR / "observations/snow/preprocessed/Bow_at_Banff_multivar_swe_processed.csv"
    df = pd.read_csv(swe_path, parse_dates=['datetime'], index_col='datetime')
    return df


def load_canswe_stations():
    stations_path = DATA_DIR / "observations/snow/preprocessed/Bow_at_Banff_multivar_canswe_stations.csv"
    if stations_path.exists():
        return pd.read_csv(stations_path)
    return None


def load_ssebop_et():
    """Load and process SSEBop ET data."""
    gdf = gpd.read_file(SHAPEFILE)
    tif_files = sorted(SSEBOP_DIR.glob("m*.tif"))

    all_data = []
    for tif_file in tif_files:
        try:
            fname = tif_file.stem
            year = int(fname[1:5])
            month = int(fname[5:7])
            date = pd.Timestamp(year=year, month=month, day=15)

            with rasterio.open(tif_file) as src:
                gdf_reproj = gdf.to_crs(src.crs)
                out_image, _ = mask(src, gdf_reproj.geometry, crop=True)
                data = out_image[0]
                nodata = src.nodata if src.nodata is not None else -9999
                data = np.ma.masked_equal(data, nodata)
                data = np.ma.masked_less_equal(data, 0)
                data = np.ma.masked_greater(data, 1000)

                if data.count() == 0:
                    continue

                days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
                et_mm_day = float(data.mean()) / days_in_month
                all_data.append({'date': date, 'ET_obs': et_mm_day})
        except Exception:
            continue

    if not all_data:
        return None

    df = pd.DataFrame(all_data).set_index('date').sort_index()
    return df


def load_catchment_shapefile():
    if SHAPEFILE.exists():
        return gpd.read_file(SHAPEFILE)
    return None


def filter_period(df, start=CALIBRATION_START, end=EVALUATION_END):
    if df is None:
        return None
    return df[(df.index >= start) & (df.index <= end)].copy()


def calculate_tws_anomaly(df):
    baseline_mask = (df.index >= GRACE_BASELINE_START) & (df.index <= GRACE_BASELINE_END)
    baseline_mean = df.loc[baseline_mask, 'TWS'].mean()
    df['TWS_anomaly'] = df['TWS'] - baseline_mean
    return df, baseline_mean


def calculate_metrics(sim, obs, use_period=True):
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
    ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
               alpha=0.12, color='#2CA02C', label='Calibration')
    ax.axvspan(pd.Timestamp(EVALUATION_START), pd.Timestamp(EVALUATION_END),
               alpha=0.12, color='#1F77B4', label='Evaluation')


def main():
    print("=" * 60)
    print("Bow at Banff - Publication Figure (v5 with ET)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_summa = load_summa_output()
    df_obs = load_streamflow_obs()
    df_grace = load_grace_tws()
    df_swe = load_canswe_swe()
    df_swe_stations = load_canswe_stations()
    df_et_obs = load_ssebop_et()
    gdf = load_catchment_shapefile()

    df_summa, baseline = calculate_tws_anomaly(df_summa)
    print(f"  TWS baseline: {baseline:.1f} mm")

    # Calculate metrics
    print("\nCalculating metrics...")
    sim_monthly = df_summa['TWS_anomaly'].resample('M').mean()
    grace_filt = filter_period(df_grace)
    grace_metrics = calculate_metrics(sim_monthly, grace_filt['GRACE_TWS'], use_period=True)
    print(f"  GRACE TWS: r={grace_metrics['r']:.3f}")

    df_sim_eval = df_summa[(df_summa.index >= EVALUATION_START) & (df_summa.index <= EVALUATION_END)]
    df_obs_eval = df_obs[(df_obs.index >= EVALUATION_START) & (df_obs.index <= EVALUATION_END)]
    q_metrics = calculate_metrics(df_sim_eval['sim_Q'], df_obs_eval['obs_Q'], use_period=False)
    print(f"  Streamflow: KGE={q_metrics['KGE']:.3f}")

    df_sim_filt = filter_period(df_summa)
    df_swe_filt = filter_period(df_swe)
    swe_metrics = calculate_metrics(df_sim_filt['SWE'], df_swe_filt['swe_mm'], use_period=False)
    print(f"  SWE: r={swe_metrics['r']:.3f}")

    # ET metrics
    if df_et_obs is not None:
        et_sim_monthly = df_summa['ET'].resample('ME').mean()
        et_metrics = calculate_metrics(et_sim_monthly, df_et_obs['ET_obs'], use_period=True)
        print(f"  ET: r={et_metrics['r']:.3f}")
    else:
        et_metrics = {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    # Create figure
    print("\nCreating figure...")
    fig = plt.figure(figsize=(7.5, 11))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.1, 0.75, 0.75, 0.75, 0.9],
                  hspace=0.35, wspace=0.3)

    # (a) Domain map
    ax_map = fig.add_subplot(gs[0, 0])
    if gdf is not None:
        from shapely.geometry import Point
        gdf_wgs = gdf.to_crs(epsg=4326)
        gdf_wgs.plot(ax=ax_map, facecolor='#9ECAE1', edgecolor='#08519C', linewidth=1.5, alpha=0.7)
        bounds = gdf_wgs.total_bounds
        margin_x = (bounds[2] - bounds[0]) * 0.15
        margin_y = (bounds[3] - bounds[1]) * 0.15

        if df_swe_stations is not None:
            geometry = [Point(xy) for xy in zip(df_swe_stations['lon'], df_swe_stations['lat'])]
            gdf_stations = gpd.GeoDataFrame(df_swe_stations.copy(), geometry=geometry, crs='EPSG:4326')
            stations_inside = gpd.sjoin(gdf_stations, gdf_wgs, how='inner', predicate='within')
            inside_ids = set(stations_inside.index)
            mask = ((df_swe_stations['lon'] >= bounds[0] - margin_x) &
                    (df_swe_stations['lon'] <= bounds[2] + margin_x) &
                    (df_swe_stations['lat'] >= bounds[1] - margin_y) &
                    (df_swe_stations['lat'] <= bounds[3] + margin_y))
            stations_in_view = df_swe_stations[mask]
            inside_mask = stations_in_view.index.isin(inside_ids)
            stations_in = stations_in_view[inside_mask]
            if len(stations_in) > 0:
                ax_map.scatter(stations_in['lon'], stations_in['lat'],
                               c=COLORS['station_swe'], s=40, marker='s', alpha=0.9,
                               edgecolors='white', linewidth=1, zorder=9,
                               label=f'CanSWE (n={len(stations_in)})')

        ax_map.plot(BOW_LON, BOW_LAT, 'r^', markersize=11, markeredgecolor='white',
                    markeredgewidth=1.5, zorder=10, label='WSC 05BB001')
        ax_map.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
        ax_map.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)
        area_km2 = gdf_wgs.to_crs(epsg=32611).area.sum() / 1e6
        ax_map.text(0.03, 0.97, f"Bow River at Banff\n{area_km2:,.0f} km²",
                    transform=ax_map.transAxes, fontsize=8, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax_map.legend(loc='lower right', fontsize=7)
        ax_map.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.82), xycoords='axes fraction',
                        fontsize=10, fontweight='bold', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.set_title('(a) Study Domain', fontweight='bold')
    ax_map.grid(True, linestyle='--', alpha=0.3)

    # (b) TWS vs GRACE
    ax_tws = fig.add_subplot(gs[0, 1])
    df_sim_f = filter_period(df_summa)
    df_grace_f = filter_period(df_grace)
    sim_monthly = df_sim_f['TWS_anomaly'].resample('M').mean()
    add_period_shading(ax_tws)
    ax_tws.plot(sim_monthly.index, sim_monthly.values, color=COLORS['sim'], linewidth=1.2, label='Simulated')
    ax_tws.plot(df_grace_f.index, df_grace_f['GRACE_TWS'].values, color=COLORS['grace'],
                linewidth=1.8, marker='o', markersize=3, label='GRACE')
    ax_tws.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_tws.text(0.02, 0.97, f"r = {grace_metrics['r']:.2f}\nRMSE = {grace_metrics['RMSE']:.0f} mm",
                transform=ax_tws.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax_tws.set_ylabel('TWS Anomaly (mm)')
    ax_tws.set_title('(b) Total Water Storage vs GRACE', fontweight='bold')
    ax_tws.legend(loc='upper right', fontsize=7)
    ax_tws.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_tws.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # (c) Streamflow
    ax_q = fig.add_subplot(gs[1, :])
    df_sim_f = filter_period(df_summa)
    df_obs_f = filter_period(df_obs)
    add_period_shading(ax_q)
    sim_weekly = df_sim_f['sim_Q'].resample('W').mean()
    obs_weekly = df_obs_f['obs_Q'].resample('W').mean()
    ax_q.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'], linewidth=0.9, label='Observed')
    ax_q.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'], linewidth=0.9, label='Simulated')
    ax_q.text(0.02, 0.97, f"r = {q_metrics['r']:.2f}, KGE = {q_metrics['KGE']:.2f}\nPBIAS = {q_metrics['PBIAS']:.0f}%",
              transform=ax_q.transAxes, fontsize=8, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax_q.set_ylabel('Discharge (m³/s)')
    ax_q.set_title('(c) Streamflow', fontweight='bold')
    ax_q.legend(loc='upper right', fontsize=7)
    ax_q.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # (d) SWE
    ax_swe = fig.add_subplot(gs[2, :])
    df_swe_f = filter_period(df_swe)
    add_period_shading(ax_swe)
    sim_weekly = df_sim_f['SWE'].resample('W').mean()
    obs_weekly = df_swe_f['swe_mm'].resample('W').mean()
    ax_swe.plot(obs_weekly.index, obs_weekly.values, color=COLORS['obs'], linewidth=0.9, label='CanSWE')
    ax_swe.plot(sim_weekly.index, sim_weekly.values, color=COLORS['sim'], linewidth=0.9, label='Simulated')
    ax_swe.text(0.02, 0.97, f"r = {swe_metrics['r']:.2f}, Bias = {swe_metrics['bias']:.0f} mm",
                transform=ax_swe.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax_swe.set_ylabel('SWE (mm)')
    ax_swe.set_title('(d) Snow Water Equivalent', fontweight='bold')
    ax_swe.legend(loc='upper right', fontsize=7)
    ax_swe.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_swe.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # (e) ET
    ax_et = fig.add_subplot(gs[3, :])
    if df_et_obs is not None and len(df_et_obs) > 0:
        et_sim_monthly = df_sim_f['ET'].resample('ME').mean()
        # Align for plotting
        et_sim_p = et_sim_monthly.copy()
        et_obs_p = df_et_obs['ET_obs'].copy()
        et_sim_p.index = et_sim_p.index.to_period('M')
        et_obs_p.index = et_obs_p.index.to_period('M')
        common = et_sim_p.index.intersection(et_obs_p.index)
        et_sim_plot = et_sim_p.loc[common].copy()
        et_obs_plot = et_obs_p.loc[common].copy()
        et_sim_plot.index = et_sim_plot.index.to_timestamp()
        et_obs_plot.index = et_obs_plot.index.to_timestamp()

        add_period_shading(ax_et)
        ax_et.plot(et_obs_plot.index, et_obs_plot.values, color=COLORS['obs'], linewidth=1.2, label='SSEBop')
        ax_et.plot(et_sim_plot.index, et_sim_plot.values, color=COLORS['sim'], linewidth=1.2, label='Simulated')
        ax_et.text(0.02, 0.97, f"r = {et_metrics['r']:.2f}, Bias = {et_metrics['bias']:.2f} mm/day",
                   transform=ax_et.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    else:
        ax_et.text(0.5, 0.5, 'SSEBop ET data not available', ha='center', va='center', transform=ax_et.transAxes)
    ax_et.set_ylabel('ET (mm/day)')
    ax_et.set_xlabel('Date')
    ax_et.set_title('(e) Evapotranspiration vs SSEBop', fontweight='bold')
    ax_et.legend(loc='upper right', fontsize=7)
    ax_et.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_et.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Scatter panels
    gs_scatter = gs[4, :].subgridspec(1, 4, wspace=0.5)

    # TWS scatter
    ax = fig.add_subplot(gs_scatter[0, 0])
    sim_p = sim_monthly.copy()
    sim_p.index = sim_p.index.to_period('M')
    grace_p = df_grace_f['GRACE_TWS'].copy()
    grace_p.index = grace_p.index.to_period('M')
    common = sim_p.dropna().index.intersection(grace_p.dropna().index)
    if len(common) > 0:
        ax.scatter(grace_p.loc[common], sim_p.loc[common], c=COLORS['grace'], s=25, alpha=0.7)
        lims = [min(grace_p.loc[common].min(), sim_p.loc[common].min()) - 20,
                max(grace_p.loc[common].max(), sim_p.loc[common].max()) + 20]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('GRACE (mm)')
    ax.set_ylabel('Simulated (mm)')
    ax.set_title('(f) TWS', fontweight='bold')
    ax.tick_params(labelsize=7)

    # Streamflow scatter
    ax = fig.add_subplot(gs_scatter[0, 1])
    df_q_eval = df_obs_f[(df_obs_f.index >= EVALUATION_START)].resample('M').mean()
    sim_q_eval = df_sim_f[(df_sim_f.index >= EVALUATION_START)]['sim_Q'].resample('M').mean()
    common = sim_q_eval.dropna().index.intersection(df_q_eval['obs_Q'].dropna().index)
    if len(common) > 0:
        ax.scatter(df_q_eval.loc[common, 'obs_Q'], sim_q_eval.loc[common], c=COLORS['obs'], s=25, alpha=0.7)
        max_val = max(df_q_eval.loc[common, 'obs_Q'].max(), sim_q_eval.loc[common].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Observed (m³/s)')
    ax.set_ylabel('Simulated (m³/s)')
    ax.set_title('(g) Q (Eval)', fontweight='bold')
    ax.tick_params(labelsize=7)

    # SWE scatter
    ax = fig.add_subplot(gs_scatter[0, 2])
    common = df_sim_f['SWE'].dropna().index.intersection(df_swe_f['swe_mm'].dropna().index)
    if len(common) > 0:
        common_sub = common[::7]
        ax.scatter(df_swe_f.loc[common_sub, 'swe_mm'], df_sim_f.loc[common_sub, 'SWE'],
                   c=COLORS['sim'], s=10, alpha=0.5)
        max_val = max(df_swe_f.loc[common, 'swe_mm'].max(), df_sim_f.loc[common, 'SWE'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('CanSWE (mm)')
    ax.set_ylabel('Simulated (mm)')
    ax.set_title('(h) SWE', fontweight='bold')
    ax.tick_params(labelsize=7)

    # ET scatter
    ax = fig.add_subplot(gs_scatter[0, 3])
    if df_et_obs is not None and len(common) > 0:
        ax.scatter(et_obs_plot.values, et_sim_plot.values, c=COLORS['et'], s=25, alpha=0.7)
        max_val = max(et_obs_plot.max(), et_sim_plot.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('SSEBop (mm/day)')
    ax.set_ylabel('Simulated (mm/day)')
    ax.set_title('(i) ET', fontweight='bold')
    ax.tick_params(labelsize=7)

    plt.subplots_adjust(bottom=0.05)
    fig.suptitle('Bow River at Banff: Multivariate Evaluation (Uncalibrated)',
                 fontsize=11, fontweight='bold', y=0.98)

    # Save
    output_png = OUTPUT_DIR / "bow_banff_figure_v5.png"
    output_pdf = OUTPUT_DIR / "bow_banff_figure_v5.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_png}")

    plt.close(fig)

    # Save metrics
    metrics_df = pd.DataFrame([
        {'Variable': 'GRACE TWS', 'Metric': 'r', 'Value': grace_metrics['r'], 'Period': '2004-2017'},
        {'Variable': 'GRACE TWS', 'Metric': 'RMSE (mm)', 'Value': grace_metrics['RMSE'], 'Period': '2004-2017'},
        {'Variable': 'Streamflow', 'Metric': 'r', 'Value': q_metrics['r'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'Streamflow', 'Metric': 'KGE', 'Value': q_metrics['KGE'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'Streamflow', 'Metric': 'PBIAS (%)', 'Value': q_metrics['PBIAS'], 'Period': 'Eval 2011-2017'},
        {'Variable': 'SWE', 'Metric': 'r', 'Value': swe_metrics['r'], 'Period': '2004-2017'},
        {'Variable': 'SWE', 'Metric': 'Bias (mm)', 'Value': swe_metrics['bias'], 'Period': '2004-2017'},
        {'Variable': 'ET vs SSEBop', 'Metric': 'r', 'Value': et_metrics['r'], 'Period': 'Available months'},
        {'Variable': 'ET vs SSEBop', 'Metric': 'Bias (mm/day)', 'Value': et_metrics['bias'], 'Period': 'Available months'},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "bow_banff_metrics_v5.csv", index=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
