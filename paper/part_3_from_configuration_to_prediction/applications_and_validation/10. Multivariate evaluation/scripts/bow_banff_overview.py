#!/usr/bin/env python3
"""
Bow at Banff Multivariate Evaluation - Overview Plots and Map
Section 4.10a: GRACE TWS Comparison

This script generates:
1. Domain map with catchment boundary
2. TWS components time series (SWE, Soil Water, Aquifer)
3. Simulated vs Observed streamflow comparison
4. Summary statistics and performance metrics
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

# Set up paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
GRACE_PATH = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Gulkana/observations/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc")
MODIS_SCA_PATH = DATA_DIR / "observations/modis_sca"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time periods (from config)
SPINUP_END = '2003-12-31'  # Exclude spinup period
CALIBRATION_START = '2004-01-01'
CALIBRATION_END = '2010-12-31'
EVALUATION_START = '2011-01-01'
EVALUATION_END = '2017-12-31'
GRACE_BASELINE_START = '2004-01-01'
GRACE_BASELINE_END = '2009-12-31'

# Bow at Banff catchment centroid
BOW_LAT = 51.17
BOW_LON = -115.57

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'sim': '#2E86AB',      # Blue for simulated
    'obs': '#E94F37',      # Red for observed
    'grace': '#9B59B6',    # Purple for GRACE
    'modis': '#FF6B35',    # Orange for MODIS
    'swe': '#5C8DFF',      # Light blue for SWE
    'soil': '#8B4513',     # Brown for soil water
    'aquifer': '#228B22',  # Green for aquifer
    'total': '#1a1a1a',    # Black for total TWS
}


def load_summa_output(experiment_id='bow_tws_uncalibrated'):
    """Load SUMMA daily and timestep output."""
    # Load daily file for TWS components
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"
    timestep_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_timestep.nc"

    print(f"Loading SUMMA daily output from: {daily_path}")
    ds_day = xr.open_dataset(daily_path)

    # Convert time to datetime - handle both numeric and datetime64 formats
    time_values = ds_day.time.values
    if np.issubdtype(time_values.dtype, np.datetime64):
        times = pd.to_datetime(time_values)
    else:
        time_units = ds_day.time.attrs.get('units', 'seconds since 1990-1-1 0:0:0.0 -0:00')
        times = pd.to_datetime(time_values, unit='s', origin=pd.Timestamp('1990-01-01'))

    # Extract TWS components
    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),  # kg/m2
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),  # kg/m2
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),  # kg/m2
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,  # m -> kg/m2 (mm)
    }

    # Calculate total TWS
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    ds_day.close()

    # Load timestep file for runoff
    if timestep_path.exists():
        print(f"Loading SUMMA timestep output from: {timestep_path}")
        ds_ts = xr.open_dataset(timestep_path)

        ts_time = ds_ts.time.values
        if np.issubdtype(ts_time.dtype, np.datetime64):
            ts_times = pd.to_datetime(ts_time)
        else:
            ts_times = pd.to_datetime(ts_time, unit='s', origin=pd.Timestamp('1990-01-01'))

        if 'averageRoutedRunoff' in ds_ts:
            # Get runoff (m/s) and convert to m³/s using catchment area
            runoff_ms = ds_ts['averageRoutedRunoff'].values.flatten()  # m/s

            # Get catchment area (approximate for Bow at Banff ~2210 km²)
            catchment_area_m2 = 2210 * 1e6  # m²

            # Convert m/s to m³/s
            runoff_m3s = runoff_ms * catchment_area_m2

            # Create runoff dataframe and resample to daily
            df_runoff = pd.DataFrame({'sim_runoff': runoff_m3s}, index=ts_times)
            df_runoff_daily = df_runoff.resample('D').mean()

            # Merge with main dataframe
            df = df.join(df_runoff_daily, how='left')
            print(f"  Loaded runoff: {df['sim_runoff'].notna().sum()} daily values")

        ds_ts.close()

    return df


def load_streamflow_obs():
    """Load observed streamflow data."""
    obs_path = DATA_DIR / "observations/streamflow/preprocessed/Bow_at_Banff_multivar_streamflow_processed.csv"
    print(f"Loading streamflow observations from: {obs_path}")

    df = pd.read_csv(obs_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = ['obs_discharge']  # Rename to clarify
    return df


def load_grace_tws(catchment_gdf=None):
    """Load GRACE TWS data and extract for catchment area.

    Note: GRACE CSR Mascon data is already provided as anomalies relative to
    the 2004.0-2009.999 baseline period, so no additional baseline subtraction needed.
    """
    print(f"Loading GRACE data from: {GRACE_PATH}")

    ds = xr.open_dataset(GRACE_PATH)

    # Get time - days since 2002-01-01
    grace_time = ds.time.values
    times = pd.to_datetime('2002-01-01') + pd.to_timedelta(grace_time, unit='D')

    # Get lwe_thickness (liquid water equivalent anomaly in cm)
    lwe = ds['lwe_thickness'].values  # (time, lat, lon)

    # Convert longitude to 0-360 for lookup
    lon_360 = BOW_LON + 360 if BOW_LON < 0 else BOW_LON

    # Find nearest grid cell to catchment centroid
    lat_idx = np.abs(ds.lat.values - BOW_LAT).argmin()
    lon_idx = np.abs(ds.lon.values - lon_360).argmin()

    print(f"  GRACE grid cell: lat={ds.lat.values[lat_idx]:.2f}, lon={ds.lon.values[lon_idx]:.2f}")

    # Extract time series for catchment (single cell for lumped catchment)
    # GRACE data is already anomaly relative to 2004-2009, convert cm to mm
    grace_anomaly = lwe[:, lat_idx, lon_idx] * 10  # Convert cm to mm

    ds.close()

    # Create dataframe - use GRACE anomaly directly (already baseline-subtracted)
    df_grace = pd.DataFrame({
        'GRACE_TWS': grace_anomaly,  # This is actually the anomaly
        'GRACE_anomaly': grace_anomaly  # Same value since data is already anomaly
    }, index=times)

    # Resample to month-end for alignment with simulated monthly data
    df_grace = df_grace.resample('M').mean()

    print(f"  GRACE time range: {times[0].strftime('%Y-%m')} to {times[-1].strftime('%Y-%m')}")
    print(f"  GRACE anomaly range: {df_grace['GRACE_anomaly'].min():.1f} to {df_grace['GRACE_anomaly'].max():.1f} mm")
    print(f"  Valid GRACE months: {df_grace['GRACE_anomaly'].notna().sum()}")

    return df_grace


def load_modis_sca():
    """Load MODIS snow cover area data.

    Supports multiple formats:
    - NetCDF files (merged or individual)
    - AppEEARS CSV output
    - GeoTIFF files
    """
    print(f"Looking for MODIS SCA data in: {MODIS_SCA_PATH}")

    if not MODIS_SCA_PATH.exists():
        print("  MODIS SCA directory not found")
        print("  Download data from AppEEARS and place in:", MODIS_SCA_PATH)
        return None

    # Try NetCDF files first
    nc_files = list(MODIS_SCA_PATH.glob("*.nc"))
    if not nc_files:
        merged_files = list(MODIS_SCA_PATH.glob("*merged*.nc")) + \
                       list(MODIS_SCA_PATH.glob("*MODIS*.nc")) + \
                       list(DATA_DIR.glob("observations/**/modis*.nc"))
        nc_files = merged_files

    if nc_files:
        return _load_modis_netcdf(nc_files[0])

    # Try AppEEARS CSV files
    csv_files = list(MODIS_SCA_PATH.glob("*MOD10A1*.csv")) + \
                list(MODIS_SCA_PATH.glob("*MYD10A1*.csv")) + \
                list(MODIS_SCA_PATH.glob("*statistics*.csv"))
    if csv_files:
        return _load_modis_appeears_csv(csv_files)

    # Try GeoTIFF files
    tif_files = list(MODIS_SCA_PATH.glob("*.tif")) + list(MODIS_SCA_PATH.glob("*.tiff"))
    if tif_files:
        return _load_modis_geotiff(tif_files)

    print("  No MODIS SCA files found (NetCDF, CSV, or GeoTIFF)")
    print("  Download data from AppEEARS: https://appeears.earthdatacloud.nasa.gov/")
    return None


def _load_modis_netcdf(nc_file):
    """Load MODIS data from NetCDF file."""
    print(f"  Loading NetCDF: {nc_file.name}")
    ds = xr.open_dataset(nc_file)

    # Try to extract snow cover fraction (variable names vary)
    sca_var = None
    for var_name in ['NDSI_Snow_Cover', 'snow_cover', 'SCA', 'snow_fraction', 'SCF',
                     'MOD10A1_061_NDSI_Snow_Cover', 'MYD10A1_061_NDSI_Snow_Cover']:
        if var_name in ds:
            sca_var = var_name
            break

    if sca_var is None:
        print(f"  Could not find snow cover variable. Available: {list(ds.data_vars)}")
        ds.close()
        return None

    sca = ds[sca_var].values
    if sca.ndim > 1:
        sca = np.nanmean(sca, axis=tuple(range(1, sca.ndim)))

    times = pd.to_datetime(ds.time.values)
    ds.close()

    # MODIS NDSI is 0-100, convert to fraction 0-1
    if np.nanmax(sca) > 1:
        sca = sca / 100.0

    df_modis = pd.DataFrame({'MODIS_SCF': sca}, index=times)
    df_modis['MODIS_snow'] = (df_modis['MODIS_SCF'] > 0.5).astype(float)

    print(f"  MODIS time range: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
    print(f"  Valid MODIS days: {df_modis['MODIS_SCF'].notna().sum()}")

    return df_modis


def _load_modis_appeears_csv(csv_files):
    """Load MODIS data from AppEEARS CSV output."""
    print(f"  Loading {len(csv_files)} AppEEARS CSV file(s)")

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # AppEEARS format has Date and various columns
            date_col = None
            for col in ['Date', 'date', 'time', 'Time']:
                if col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                continue

            # Look for NDSI Snow Cover column
            sca_col = None
            for col in df.columns:
                if 'NDSI_Snow_Cover' in col or 'snow' in col.lower():
                    sca_col = col
                    break

            if sca_col is None:
                continue

            df['time'] = pd.to_datetime(df[date_col])
            df['sca'] = pd.to_numeric(df[sca_col], errors='coerce')

            all_data.append(df[['time', 'sca']])
        except Exception as e:
            print(f"    Warning: Could not read {csv_file.name}: {e}")
            continue

    if not all_data:
        return None

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna().groupby('time').mean()

    # Convert to fraction if needed
    if combined['sca'].max() > 1:
        combined['sca'] = combined['sca'] / 100.0

    df_modis = pd.DataFrame({'MODIS_SCF': combined['sca']})
    df_modis['MODIS_snow'] = (df_modis['MODIS_SCF'] > 0.5).astype(float)

    print(f"  MODIS time range: {df_modis.index[0].strftime('%Y-%m-%d')} to {df_modis.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Valid MODIS days: {df_modis['MODIS_SCF'].notna().sum()}")

    return df_modis


def _load_modis_geotiff(tif_files):
    """Load MODIS data from GeoTIFF files (extract basin mean)."""
    print(f"  Found {len(tif_files)} GeoTIFF files - this may take a while...")

    try:
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        print("  rasterio not available for GeoTIFF processing")
        return None

    # Load catchment for masking
    gdf = load_catchment_shapefile()
    if gdf is None:
        print("  No catchment shapefile for masking GeoTIFFs")
        return None

    gdf_wgs = gdf.to_crs(epsg=4326)

    data = []
    for tif_file in sorted(tif_files):
        try:
            # Extract date from filename (typical AppEEARS format)
            date_str = None
            parts = tif_file.stem.split('_')
            for part in parts:
                if len(part) == 7 and part[0] == 'A':  # Julian date format AYYYYDDD
                    year = int(part[1:5])
                    doy = int(part[5:])
                    date_str = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
                    break
                elif len(part) == 8 and part.isdigit():  # YYYYMMDD
                    date_str = pd.Timestamp(part)
                    break

            if date_str is None:
                continue

            with rasterio.open(tif_file) as src:
                out_image, _ = mask(src, gdf_wgs.geometry, crop=True, nodata=np.nan)
                # Average over basin
                valid = out_image[out_image <= 100]  # NDSI valid range
                if len(valid) > 0:
                    mean_val = np.nanmean(valid)
                    data.append({'time': date_str, 'sca': mean_val})

        except Exception:
            continue

    if not data:
        return None

    df = pd.DataFrame(data)
    df = df.set_index('time').sort_index()

    # Convert to fraction
    if df['sca'].max() > 1:
        df['sca'] = df['sca'] / 100.0

    df_modis = pd.DataFrame({'MODIS_SCF': df['sca']})
    df_modis['MODIS_snow'] = (df_modis['MODIS_SCF'] > 0.5).astype(float)

    print(f"  MODIS time range: {df_modis.index[0].strftime('%Y-%m-%d')} to {df_modis.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Valid MODIS days: {df_modis['MODIS_SCF'].notna().sum()}")

    return df_modis


def load_simulated_scf(experiment_id='bow_tws_uncalibrated'):
    """Load simulated snow cover fraction from SUMMA output."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"

    ds = xr.open_dataset(daily_path)

    times = pd.to_datetime(ds.time.values)
    scf = ds['scalarGroundSnowFraction'].values.flatten()

    ds.close()

    df_scf = pd.DataFrame({
        'sim_SCF': scf
    }, index=times)

    # Binary snow cover
    df_scf['sim_snow'] = (df_scf['sim_SCF'] > 0.5).astype(float)

    return df_scf


def calculate_sca_metrics(df_sim, df_modis):
    """Calculate snow cover accuracy metrics."""
    if df_modis is None:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    # Get common dates (excluding spinup)
    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)

    common_dates = df_sim_filt.index.intersection(df_modis_filt.index)

    if len(common_dates) == 0:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    sim_snow = df_sim_filt.loc[common_dates, 'sim_snow'].values
    obs_snow = df_modis_filt.loc[common_dates, 'MODIS_snow'].values
    sim_scf = df_sim_filt.loc[common_dates, 'sim_SCF'].values
    obs_scf = df_modis_filt.loc[common_dates, 'MODIS_SCF'].values

    # Remove NaN
    valid = ~(np.isnan(sim_snow) | np.isnan(obs_snow))
    sim_snow = sim_snow[valid]
    obs_snow = obs_snow[valid]
    sim_scf = sim_scf[valid]
    obs_scf = obs_scf[valid]

    if len(sim_snow) == 0:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    # Contingency table
    hits = np.sum((sim_snow == 1) & (obs_snow == 1))  # Both snow
    misses = np.sum((sim_snow == 0) & (obs_snow == 1))  # Obs snow, sim no snow
    false_alarms = np.sum((sim_snow == 1) & (obs_snow == 0))  # Sim snow, obs no snow
    correct_negatives = np.sum((sim_snow == 0) & (obs_snow == 0))  # Both no snow

    n_total = len(sim_snow)

    # Calculate metrics
    accuracy = (hits + correct_negatives) / n_total if n_total > 0 else np.nan

    # Probability of Detection (POD) = hits / (hits + misses)
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan

    # False Alarm Ratio (FAR) = false_alarms / (hits + false_alarms)
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan

    # Critical Success Index (CSI) = hits / (hits + misses + false_alarms)
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan

    # Correlation of SCF values
    r = np.corrcoef(sim_scf, obs_scf)[0, 1] if len(sim_scf) > 2 else np.nan

    return {
        'accuracy': accuracy,
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'r': r,
        'n_days': len(sim_snow),
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'correct_negatives': correct_negatives
    }


def plot_sca_comparison(df_sim, df_modis, sca_metrics, ax=None):
    """Plot snow cover fraction comparison time series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if df_modis is None:
        ax.text(0.5, 0.5, 'MODIS SCA data not available\nRun: symfluence workflow steps process_observed_data',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Snow Cover Fraction: Simulated vs MODIS', fontsize=12, fontweight='bold')
        return ax

    # Filter spinup
    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)

    # Resample to weekly for cleaner plot
    sim_weekly = df_sim_filt['sim_SCF'].resample('W').mean()
    modis_weekly = df_modis_filt['MODIS_SCF'].resample('W').mean()

    ax.plot(modis_weekly.index, modis_weekly.values,
            color=COLORS['modis'], linewidth=1.5, alpha=0.8, label='MODIS')
    ax.plot(sim_weekly.index, sim_weekly.values,
            color=COLORS['sim'], linewidth=1.5, alpha=0.8, label='Simulated')

    # Add metrics text
    acc = sca_metrics.get('accuracy', np.nan)
    r = sca_metrics.get('r', np.nan)
    csi = sca_metrics.get('CSI', np.nan)
    metrics_text = f"Accuracy = {acc:.2%}\nr = {r:.3f}\nCSI = {csi:.3f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('Snow Cover Fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Snow Cover Fraction: Simulated vs MODIS', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_sca_scatter(df_sim, df_modis, sca_metrics, ax=None):
    """Plot scatter of simulated vs MODIS snow cover fraction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if df_modis is None:
        ax.text(0.5, 0.5, 'MODIS data\nnot available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Filter spinup and get common dates
    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)
    common_dates = df_sim_filt.index.intersection(df_modis_filt.index)

    if len(common_dates) == 0:
        ax.text(0.5, 0.5, 'No overlapping data', ha='center', va='center', transform=ax.transAxes)
        return ax

    sim_scf = df_sim_filt.loc[common_dates, 'sim_SCF'].values
    modis_scf = df_modis_filt.loc[common_dates, 'MODIS_SCF'].values

    # Remove NaN
    valid = ~(np.isnan(sim_scf) | np.isnan(modis_scf))
    sim_scf = sim_scf[valid]
    modis_scf = modis_scf[valid]

    # Scatter plot with density coloring
    ax.scatter(modis_scf, sim_scf, alpha=0.3, s=5, c=COLORS['modis'])

    # 1:1 line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='1:1 line')

    # Add metrics
    r = sca_metrics.get('r', np.nan)
    acc = sca_metrics.get('accuracy', np.nan)
    metrics_text = f"r = {r:.3f}\nAccuracy = {acc:.2%}\nn = {len(sim_scf)} days"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('MODIS Snow Cover Fraction')
    ax.set_ylabel('Simulated Snow Cover Fraction')
    ax.set_title('Snow Cover Comparison', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    return ax


def load_catchment_shapefile():
    """Load catchment boundary shapefile."""
    # Try multiple possible locations
    shp_paths = [
        DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp",
        Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp"),
    ]

    for shp_path in shp_paths:
        if shp_path.exists():
            print(f"Loading catchment from: {shp_path}")
            return gpd.read_file(shp_path)

    print("Warning: Could not find catchment shapefile")
    return None


def calculate_tws_anomaly(df, baseline_start=GRACE_BASELINE_START, baseline_end=GRACE_BASELINE_END):
    """Calculate TWS anomaly relative to baseline period mean."""
    baseline_mask = (df.index >= baseline_start) & (df.index <= baseline_end)
    baseline_mean = df.loc[baseline_mask, 'TWS'].mean()

    df['TWS_anomaly'] = df['TWS'] - baseline_mean

    # Also calculate monthly means for comparison with GRACE
    df_monthly = df.resample('M').mean()
    df_monthly['TWS_anomaly'] = df_monthly['TWS'] - baseline_mean

    return df, df_monthly, baseline_mean


def filter_spinup(df, spinup_end=SPINUP_END):
    """Remove spinup period from dataframe."""
    return df[df.index > spinup_end].copy()


def calculate_grace_metrics(sim_monthly, grace_df):
    """Calculate metrics comparing simulated TWS anomaly to GRACE."""
    # Resample simulated to monthly if not already
    if not isinstance(sim_monthly.index, pd.DatetimeIndex):
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    # Get common months
    common_months = sim_monthly.index.intersection(grace_df.index)

    # Filter to evaluation period
    eval_months = common_months[(common_months >= CALIBRATION_START) & (common_months <= EVALUATION_END)]

    if len(eval_months) == 0:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    sim_vals = sim_monthly.loc[eval_months, 'TWS_anomaly'].values
    grace_vals = grace_df.loc[eval_months, 'GRACE_anomaly'].values

    # Remove NaN pairs
    valid = ~(np.isnan(sim_vals) | np.isnan(grace_vals))
    sim_vals = sim_vals[valid]
    grace_vals = grace_vals[valid]

    if len(sim_vals) < 3:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    # Calculate metrics
    r = np.corrcoef(sim_vals, grace_vals)[0, 1]
    rmse = np.sqrt(np.mean((sim_vals - grace_vals)**2))
    bias = np.mean(sim_vals - grace_vals)

    return {'r': r, 'RMSE': rmse, 'bias': bias, 'n_months': len(sim_vals)}


def calculate_metrics(sim, obs):
    """Calculate performance metrics."""
    # Align data
    common_idx = sim.index.intersection(obs.index)
    sim_aligned = sim.loc[common_idx].dropna()
    obs_aligned = obs.loc[common_idx].dropna()

    common_idx = sim_aligned.index.intersection(obs_aligned.index)
    sim_vals = sim_aligned.loc[common_idx].values
    obs_vals = obs_aligned.loc[common_idx].values

    if len(sim_vals) == 0:
        return {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan}

    # Correlation
    r = np.corrcoef(sim_vals, obs_vals)[0, 1]

    # NSE
    nse = 1 - np.sum((sim_vals - obs_vals)**2) / np.sum((obs_vals - np.mean(obs_vals))**2)

    # KGE
    alpha = np.std(sim_vals) / np.std(obs_vals)
    beta = np.mean(sim_vals) / np.mean(obs_vals)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Percent Bias
    pbias = 100 * np.sum(sim_vals - obs_vals) / np.sum(obs_vals)

    return {'r': r, 'NSE': nse, 'KGE': kge, 'PBIAS': pbias}


def plot_domain_map(gdf, ax=None):
    """Plot the domain map."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if gdf is not None:
        gdf.plot(ax=ax, facecolor='lightblue', edgecolor='darkblue', linewidth=2, alpha=0.7)

        # Get bounds for context
        bounds = gdf.total_bounds
        ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
        ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)

        # Add pour point marker (Banff)
        ax.plot(-115.5717, 51.1722, 'r^', markersize=12, label='Pour Point (Banff)')

        # Add area annotation
        area_km2 = gdf.to_crs(epsg=32611).area.sum() / 1e6  # UTM zone 11N for Alberta
        ax.text(0.05, 0.95, f'Area: {area_km2:.0f} km²',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bow River at Banff\nCatchment Domain', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')

    return ax


def plot_tws_components(df, ax=None):
    """Plot TWS component time series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Filter spinup and resample to monthly
    df_filtered = filter_spinup(df)
    df_monthly = df_filtered.resample('M').mean()

    ax.fill_between(df_monthly.index, 0, df_monthly['SWE'],
                    alpha=0.7, label='SWE', color=COLORS['swe'])
    ax.fill_between(df_monthly.index, df_monthly['SWE'],
                    df_monthly['SWE'] + df_monthly['soil_water'],
                    alpha=0.7, label='Soil Water', color=COLORS['soil'])
    ax.fill_between(df_monthly.index, df_monthly['SWE'] + df_monthly['soil_water'],
                    df_monthly['SWE'] + df_monthly['soil_water'] + df_monthly['aquifer'],
                    alpha=0.7, label='Aquifer', color=COLORS['aquifer'])

    ax.plot(df_monthly.index, df_monthly['TWS'], 'k-', linewidth=1.5, label='Total TWS')

    ax.set_xlabel('Date')
    ax.set_ylabel('Water Storage (kg/m² = mm)')
    ax.set_title('Simulated Total Water Storage Components', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_tws_anomaly(df_monthly, ax=None):
    """Plot TWS anomaly time series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Filter out spinup
    df_plot = filter_spinup(df_monthly)

    # Plot anomaly as bar chart
    colors = ['blue' if x < 0 else 'red' for x in df_plot['TWS_anomaly']]
    ax.bar(df_plot.index, df_plot['TWS_anomaly'], width=25, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('TWS Anomaly (mm)')
    ax.set_title('Simulated TWS Anomaly (relative to 2004-2009 baseline)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_grace_comparison(df_monthly, df_grace, grace_metrics, ax=None):
    """Plot GRACE vs simulated TWS anomaly comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Filter out spinup period
    df_sim = filter_spinup(df_monthly)
    df_obs = filter_spinup(df_grace)

    # Plot simulated TWS anomaly
    ax.plot(df_sim.index, df_sim['TWS_anomaly'],
            color=COLORS['sim'], linewidth=1.5, alpha=0.8, label='Simulated TWS')

    # Plot GRACE TWS anomaly
    ax.plot(df_obs.index, df_obs['GRACE_anomaly'],
            color=COLORS['grace'], linewidth=2, marker='o', markersize=4,
            alpha=0.8, label='GRACE TWS')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add calibration/evaluation period shading
    ax.axvspan(pd.Timestamp(CALIBRATION_START), pd.Timestamp(CALIBRATION_END),
               alpha=0.1, color='blue', label='Calibration Period')
    ax.axvspan(pd.Timestamp(EVALUATION_START), pd.Timestamp(EVALUATION_END),
               alpha=0.1, color='green', label='Evaluation Period')

    # Add metrics text
    metrics_text = f"r = {grace_metrics['r']:.3f}\nRMSE = {grace_metrics['RMSE']:.1f} mm\nBias = {grace_metrics['bias']:.1f} mm"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('TWS Anomaly (mm)')
    ax.set_title('TWS Anomaly: Simulated vs GRACE', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_grace_scatter(df_monthly, df_grace, grace_metrics, ax=None):
    """Plot scatter of GRACE vs simulated TWS anomaly."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Get common months (excluding spinup)
    df_sim = filter_spinup(df_monthly)
    df_obs = filter_spinup(df_grace)
    common_months = df_sim.index.intersection(df_obs.index)

    if len(common_months) == 0:
        ax.text(0.5, 0.5, 'No overlapping data', ha='center', va='center', transform=ax.transAxes)
        return ax

    sim_vals = df_sim.loc[common_months, 'TWS_anomaly'].values
    grace_vals = df_obs.loc[common_months, 'GRACE_anomaly'].values

    # Remove NaN
    valid = ~(np.isnan(sim_vals) | np.isnan(grace_vals))
    sim_vals = sim_vals[valid]
    grace_vals = grace_vals[valid]

    if len(sim_vals) == 0:
        ax.text(0.5, 0.5, 'No valid data pairs', ha='center', va='center', transform=ax.transAxes)
        return ax

    # Scatter plot
    ax.scatter(grace_vals, sim_vals, alpha=0.6, s=30, c=COLORS['grace'], edgecolors='white', linewidth=0.5)

    # 1:1 line
    min_val = min(grace_vals.min(), sim_vals.min())
    max_val = max(grace_vals.max(), sim_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 line')

    # Add metrics text
    r_val = grace_metrics.get('r', np.nan)
    rmse_val = grace_metrics.get('RMSE', np.nan)
    metrics_text = f"r = {r_val:.3f}\nRMSE = {rmse_val:.1f} mm\nn = {len(sim_vals)} months"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('GRACE TWS Anomaly (mm)')
    ax.set_ylabel('Simulated TWS Anomaly (mm)')
    ax.set_title('TWS Anomaly Comparison', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_streamflow_comparison(df_sim, df_obs, ax=None, period='all'):
    """Plot simulated vs observed streamflow."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if 'sim_runoff' not in df_sim.columns:
        ax.text(0.5, 0.5, 'No simulated runoff available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Filter spinup from simulation
    df_sim_filt = filter_spinup(df_sim)

    # Resample observations to daily
    obs_daily = df_obs.resample('D').mean()

    # Get common dates
    common_dates = df_sim_filt.index.intersection(obs_daily.index)
    sim_aligned = df_sim_filt.loc[common_dates, 'sim_runoff'].dropna()
    obs_aligned = obs_daily.loc[common_dates, 'obs_discharge'].dropna()

    # Get final common dates
    final_dates = sim_aligned.index.intersection(obs_aligned.index)
    sim_plot = sim_aligned.loc[final_dates]
    obs_plot = obs_aligned.loc[final_dates]

    # Apply period filter if specified
    if period == 'eval':
        mask = (final_dates >= EVALUATION_START) & (final_dates <= EVALUATION_END)
        sim_plot = sim_plot.loc[mask]
        obs_plot = obs_plot.loc[mask]

    ax.plot(obs_plot.index, obs_plot.values,
            color=COLORS['obs'], linewidth=0.8, alpha=0.7, label='Observed')
    ax.plot(sim_plot.index, sim_plot.values,
            color=COLORS['sim'], linewidth=0.8, alpha=0.7, label='Simulated')

    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Streamflow: Simulated vs Observed', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_streamflow_scatter(df_sim, df_obs, ax=None, period='eval'):
    """Plot scatter plot of simulated vs observed streamflow."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if 'sim_runoff' not in df_sim.columns:
        ax.text(0.5, 0.5, 'No simulated runoff available',
                ha='center', va='center', transform=ax.transAxes)
        return ax, {}

    # Filter spinup
    df_sim_filt = filter_spinup(df_sim)

    # Resample observations to daily
    obs_daily = df_obs.resample('D').mean()

    # Get common dates
    common_dates = df_sim_filt.index.intersection(obs_daily.index)
    sim_aligned = df_sim_filt.loc[common_dates, 'sim_runoff'].dropna()
    obs_aligned = obs_daily.loc[common_dates, 'obs_discharge'].dropna()

    final_dates = sim_aligned.index.intersection(obs_aligned.index)

    # Apply period filter
    if period == 'eval':
        mask = (final_dates >= EVALUATION_START) & (final_dates <= EVALUATION_END)
        final_dates = final_dates[mask]

    sim_vals = sim_aligned.loc[final_dates].values
    obs_vals = obs_aligned.loc[final_dates].values

    # Calculate metrics
    metrics = calculate_metrics(
        pd.Series(sim_vals, index=final_dates),
        pd.Series(obs_vals, index=final_dates)
    )

    # Scatter plot
    ax.scatter(obs_vals, sim_vals, alpha=0.3, s=10, c=COLORS['sim'])

    # 1:1 line
    max_val = max(obs_vals.max(), sim_vals.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')

    # Add metrics text
    metrics_text = f"r = {metrics['r']:.3f}\nNSE = {metrics['NSE']:.3f}\nKGE = {metrics['KGE']:.3f}\nPBIAS = {metrics['PBIAS']:.1f}%"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Observed Discharge (m³/s)')
    ax.set_ylabel('Simulated Discharge (m³/s)')
    ax.set_title(f'Streamflow Scatter Plot\n(Evaluation: {EVALUATION_START[:4]}-{EVALUATION_END[:4]})', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    return ax, metrics


def create_overview_figure(df_summa, df_monthly, df_obs, df_grace, gdf, q_metrics, grace_metrics,
                           df_modis=None, df_sim_scf=None, sca_metrics=None):
    """Create comprehensive overview figure with GRACE and MODIS comparison."""
    fig = plt.figure(figsize=(16, 28))
    gs = GridSpec(7, 2, figure=fig, height_ratios=[1, 0.9, 0.8, 0.9, 0.8, 0.9, 0.9], hspace=0.35, wspace=0.25)

    # Filter spinup for stats
    df_summa_filt = filter_spinup(df_summa)
    df_monthly_filt = filter_spinup(df_monthly)

    # Top left: Domain map
    ax1 = fig.add_subplot(gs[0, 0])
    plot_domain_map(gdf, ax1)

    # Top right: Summary statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Get MODIS metrics if available
    if sca_metrics is None:
        sca_metrics = {}

    # Create summary text
    summary_text = f"""
    Bow River at Banff - Uncalibrated SUMMA Evaluation
    ══════════════════════════════════════════════════

    Analysis Period: {CALIBRATION_START} to {EVALUATION_END}
    (Spinup excluded: before {SPINUP_END})

    TWS Statistics (mm):
    ────────────────────
    Mean SWE:           {df_summa_filt['SWE'].mean():.1f}
    Mean Soil Water:    {df_summa_filt['soil_water'].mean():.1f}
    Mean Aquifer:       {df_summa_filt['aquifer'].mean():.1f}
    Mean Total TWS:     {df_summa_filt['TWS'].mean():.1f}

    Streamflow Metrics (Eval 2011-2017):
    ────────────────────────────────────
    Correlation (r):    {q_metrics.get('r', np.nan):.3f}
    NSE:                {q_metrics.get('NSE', np.nan):.3f}
    KGE:                {q_metrics.get('KGE', np.nan):.3f}
    PBIAS (%):          {q_metrics.get('PBIAS', np.nan):.1f}

    GRACE TWS Metrics (2004-2017):
    ──────────────────────────────
    Correlation (r):    {grace_metrics.get('r', np.nan):.3f}
    RMSE (mm):          {grace_metrics.get('RMSE', np.nan):.1f}
    Bias (mm):          {grace_metrics.get('bias', np.nan):.1f}

    MODIS SCA Metrics:
    ──────────────────
    Accuracy:           {sca_metrics.get('accuracy', np.nan):.2%}
    SCF Correlation:    {sca_metrics.get('r', np.nan):.3f}
    CSI:                {sca_metrics.get('CSI', np.nan):.3f}
    """

    ax2.text(0.05, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: TWS components
    ax3 = fig.add_subplot(gs[1, :])
    plot_tws_components(df_summa, ax3)

    # Row 3: Streamflow time series
    ax4 = fig.add_subplot(gs[2, :])
    plot_streamflow_comparison(df_summa, df_obs, ax4, period='all')

    # Row 4: GRACE comparison time series
    ax5 = fig.add_subplot(gs[3, :])
    plot_grace_comparison(df_monthly, df_grace, grace_metrics, ax5)

    # Row 5: MODIS SCA time series
    ax6 = fig.add_subplot(gs[4, :])
    if df_modis is not None and df_sim_scf is not None:
        plot_sca_comparison(df_sim_scf, df_modis, sca_metrics, ax6)
    else:
        ax6.text(0.5, 0.5, 'MODIS SCA data not available\nRun: symfluence workflow steps process_observed_data',
                ha='center', va='center', transform=ax6.transAxes, fontsize=11)
        ax6.set_title('Snow Cover Fraction: Simulated vs MODIS', fontsize=12, fontweight='bold')

    # Row 6 left: Streamflow scatter
    ax7 = fig.add_subplot(gs[5, 0])
    plot_streamflow_scatter(df_summa, df_obs, ax7, period='eval')

    # Row 6 right: GRACE scatter
    ax8 = fig.add_subplot(gs[5, 1])
    plot_grace_scatter(df_monthly, df_grace, grace_metrics, ax8)

    # Row 7 left: MODIS SCA scatter
    ax9 = fig.add_subplot(gs[6, 0])
    if df_modis is not None and df_sim_scf is not None:
        plot_sca_scatter(df_sim_scf, df_modis, sca_metrics, ax9)
    else:
        ax9.text(0.5, 0.5, 'MODIS data\nnot available',
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Snow Cover Comparison', fontsize=11, fontweight='bold')

    # Row 7 right: Contingency table visualization
    ax10 = fig.add_subplot(gs[6, 1])
    if sca_metrics and 'hits' in sca_metrics:
        # Create contingency table heatmap
        contingency = np.array([
            [sca_metrics.get('hits', 0), sca_metrics.get('misses', 0)],
            [sca_metrics.get('false_alarms', 0), sca_metrics.get('correct_negatives', 0)]
        ])
        im = ax10.imshow(contingency, cmap='Blues')
        ax10.set_xticks([0, 1])
        ax10.set_yticks([0, 1])
        ax10.set_xticklabels(['Obs: Snow', 'Obs: No Snow'])
        ax10.set_yticklabels(['Sim: Snow', 'Sim: No Snow'])
        ax10.set_title('Snow Cover Contingency Table', fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax10.text(j, i, f'{contingency[i, j]:.0f}',
                               ha='center', va='center', color='white' if contingency[i, j] > contingency.max()/2 else 'black',
                               fontsize=12)

        # Add colorbar
        plt.colorbar(im, ax=ax10, shrink=0.8, label='Count (days)')
    else:
        ax10.text(0.5, 0.5, 'MODIS metrics\nnot available',
                ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Snow Cover Contingency Table', fontsize=11, fontweight='bold')

    plt.suptitle('Bow at Banff: Multivariate Evaluation Overview (Uncalibrated)',
                 fontsize=14, fontweight='bold', y=0.99)

    return fig


def main():
    """Main execution function."""
    print("=" * 60)
    print("Bow at Banff Multivariate Evaluation - Overview Generation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_summa = load_summa_output()
    print(f"  SUMMA output: {len(df_summa)} days")

    df_obs = load_streamflow_obs()
    print(f"  Streamflow obs: {len(df_obs)} records")

    gdf = load_catchment_shapefile()
    if gdf is not None:
        print(f"  Catchment loaded: {len(gdf)} features")

    # Load GRACE data
    df_grace = load_grace_tws(gdf)

    # Calculate TWS anomaly
    print("\nCalculating TWS anomaly...")
    df_summa, df_monthly, baseline_mean = calculate_tws_anomaly(df_summa)
    print(f"  Baseline mean TWS: {baseline_mean:.1f} mm")

    # Filter spinup for stats
    df_summa_filt = filter_spinup(df_summa)
    df_monthly_filt = filter_spinup(df_monthly)

    # Calculate GRACE metrics
    print("\nCalculating GRACE comparison metrics...")
    grace_metrics = calculate_grace_metrics(df_monthly, df_grace)
    print(f"  GRACE correlation: {grace_metrics.get('r', np.nan):.3f}")
    print(f"  GRACE RMSE: {grace_metrics.get('RMSE', np.nan):.1f} mm")

    # Calculate streamflow metrics (evaluation period: 2011-2017)
    print("\nCalculating streamflow performance metrics...")
    if 'sim_runoff' in df_summa.columns:
        # Align and calculate metrics for evaluation period
        obs_daily = df_obs.resample('D').mean()
        common_dates = df_summa_filt.index.intersection(obs_daily.index)
        eval_dates = common_dates[(common_dates >= EVALUATION_START) & (common_dates <= EVALUATION_END)]

        sim_eval = df_summa_filt.loc[eval_dates, 'sim_runoff'].dropna()
        obs_eval = obs_daily.loc[eval_dates, 'obs_discharge'].dropna()
        final_dates = sim_eval.index.intersection(obs_eval.index)

        q_metrics = calculate_metrics(sim_eval.loc[final_dates], obs_eval.loc[final_dates])
        print(f"  Evaluation period: {len(final_dates)} days")
    else:
        q_metrics = {'r': np.nan, 'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan}
        print("  Warning: No simulated runoff found in output")
    print(f"  Streamflow KGE: {q_metrics.get('KGE', np.nan):.3f}")

    # Load MODIS SCA data
    print("\nLoading MODIS SCA data...")
    df_modis = load_modis_sca()
    df_sim_scf = load_simulated_scf()
    print(f"  Simulated SCF: {len(df_sim_scf)} days")

    # Calculate MODIS SCA metrics
    print("\nCalculating MODIS SCA metrics...")
    if df_modis is not None:
        sca_metrics = calculate_sca_metrics(df_sim_scf, df_modis)
        print(f"  SCA Accuracy: {sca_metrics.get('accuracy', np.nan):.2%}")
        print(f"  SCF Correlation: {sca_metrics.get('r', np.nan):.3f}")
        print(f"  CSI: {sca_metrics.get('CSI', np.nan):.3f}")
    else:
        sca_metrics = {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}
        print("  MODIS SCA data not available")

    # Create overview figure
    print("\nGenerating overview figure...")
    fig = create_overview_figure(df_summa, df_monthly, df_obs, df_grace, gdf, q_metrics, grace_metrics,
                                 df_modis=df_modis, df_sim_scf=df_sim_scf, sca_metrics=sca_metrics)

    # Save figure
    output_path = OUTPUT_DIR / "bow_banff_overview_uncalibrated.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")

    # Also save as PDF for publication
    pdf_path = OUTPUT_DIR / "bow_banff_overview_uncalibrated.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")

    plt.close(fig)

    # Save summary statistics to CSV (using filtered data)
    stats_path = OUTPUT_DIR / "bow_banff_summary_stats.csv"
    stats_df = pd.DataFrame({
        'Metric': ['Mean SWE (mm)', 'Mean Soil Water (mm)', 'Mean Aquifer (mm)',
                   'Mean Total TWS (mm)', 'TWS Anomaly Min (mm)', 'TWS Anomaly Max (mm)',
                   'Streamflow r', 'Streamflow NSE', 'Streamflow KGE', 'Streamflow PBIAS (%)',
                   'GRACE r', 'GRACE RMSE (mm)', 'GRACE Bias (mm)',
                   'MODIS SCA Accuracy', 'MODIS SCF r', 'MODIS SCA POD', 'MODIS SCA FAR', 'MODIS SCA CSI'],
        'Value': [df_summa_filt['SWE'].mean(), df_summa_filt['soil_water'].mean(),
                  df_summa_filt['aquifer'].mean(), df_summa_filt['TWS'].mean(),
                  df_monthly_filt['TWS_anomaly'].min(), df_monthly_filt['TWS_anomaly'].max(),
                  q_metrics.get('r', np.nan), q_metrics.get('NSE', np.nan),
                  q_metrics.get('KGE', np.nan), q_metrics.get('PBIAS', np.nan),
                  grace_metrics.get('r', np.nan), grace_metrics.get('RMSE', np.nan),
                  grace_metrics.get('bias', np.nan),
                  sca_metrics.get('accuracy', np.nan), sca_metrics.get('r', np.nan),
                  sca_metrics.get('POD', np.nan), sca_metrics.get('FAR', np.nan),
                  sca_metrics.get('CSI', np.nan)]
    })
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats to: {stats_path}")

    print("\n" + "=" * 60)
    print("Overview generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
