#!/usr/bin/env python3
"""
Paradise Multivariate Evaluation - Overview Plots
Section 4.10b: SCA & Soil Moisture Comparison

This script generates:
1. Domain map with catchment boundary
2. TWS components time series (SWE, Soil Water, Aquifer)
3. Simulated vs MODIS snow cover fraction comparison
4. Simulated vs SMAP soil moisture comparison
5. Summary statistics and performance metrics
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

# Add scripts directory to path for map_utils
try:
    SCRIPTS_DIR = Path(__file__).parent
except NameError:
    SCRIPTS_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/scripts")
sys.path.insert(0, str(SCRIPTS_DIR))
from map_utils import plot_domain_map as plot_domain_map_styled, get_region_inset_extent, DOMAIN_COLORS

# Set up paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_paradise_multivar")
MODIS_SCA_PATH = DATA_DIR / "observations/modis_sca"
SMAP_PATH = DATA_DIR / "observations/smap_sm"
SNOTEL_PATH = DATA_DIR / "observations/snotel"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/paradise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time periods (from config)
SPINUP_END = '2016-09-30'  # Exclude spinup period
CALIBRATION_START = '2016-10-01'
CALIBRATION_END = '2020-09-30'
EVALUATION_START = '2020-10-01'
EVALUATION_END = '2023-09-30'

# Paradise catchment centroid (Mt. Rainier area)
PARADISE_LAT = 46.79
PARADISE_LON = -121.75

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'sim': '#2E86AB',      # Blue for simulated
    'obs': '#E94F37',      # Red for observed
    'modis': '#FF6B35',    # Orange for MODIS
    'smap': '#9B59B6',     # Purple for SMAP
    'snotel': '#27AE60',   # Green for SNOTEL
    'swe': '#5C8DFF',      # Light blue for SWE
    'soil': '#8B4513',     # Brown for soil water
    'aquifer': '#228B22',  # Green for aquifer
    'total': '#1a1a1a',    # Black for total TWS
}


def load_summa_output(experiment_id='paradise_sca_sm'):
    """Load SUMMA daily output."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"

    print(f"Loading SUMMA daily output from: {daily_path}")

    if not daily_path.exists():
        print(f"  File not found: {daily_path}")
        return None

    ds_day = xr.open_dataset(daily_path)

    # Convert time to datetime
    time_values = ds_day.time.values
    if np.issubdtype(time_values.dtype, np.datetime64):
        times = pd.to_datetime(time_values)
    else:
        times = pd.to_datetime(time_values, unit='s', origin=pd.Timestamp('1990-01-01'))

    # Extract variables
    data = {
        'time': times,
        'SWE': ds_day['scalarSWE'].values.flatten(),  # kg/m2 = mm
        'soil_water': ds_day['scalarTotalSoilWat'].values.flatten(),  # kg/m2
        'canopy_water': ds_day['scalarCanopyWat'].values.flatten(),  # kg/m2
        'aquifer': ds_day['scalarAquiferStorage'].values.flatten() * 1000,  # m -> mm
    }

    # Snow cover fraction
    if 'scalarGroundSnowFraction' in ds_day:
        data['sim_SCF'] = ds_day['scalarGroundSnowFraction'].values.flatten()

    # Surface soil moisture (top layer)
    if 'mLayerVolFracWat' in ds_day:
        # Get first soil layer (surface)
        soil_moisture = ds_day['mLayerVolFracWat'].values
        if soil_moisture.ndim > 1:
            # First layer is typically index 0 after snow layers
            # Take mean of top layers as approximation
            data['sim_SM'] = np.nanmean(soil_moisture[:, :, -3:], axis=(1, 2))  # Bottom 3 layers (soil)
        else:
            data['sim_SM'] = soil_moisture.flatten()

    # Calculate total TWS
    data['TWS'] = data['SWE'] + data['soil_water'] + data['canopy_water'] + data['aquifer']

    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    ds_day.close()

    return df


def load_modis_sca():
    """Load MODIS snow cover area data."""
    print(f"Looking for MODIS SCA data in: {MODIS_SCA_PATH}")

    if not MODIS_SCA_PATH.exists():
        print("  MODIS SCA directory not found")
        return None

    # Look for NetCDF files
    nc_files = list(MODIS_SCA_PATH.glob("*.nc"))
    csv_files = list(MODIS_SCA_PATH.glob("*.csv"))

    if nc_files:
        return _load_modis_netcdf(nc_files[0])
    elif csv_files:
        return _load_modis_csv(csv_files)
    else:
        print("  No MODIS SCA files found")
        return None


def _load_modis_netcdf(nc_file):
    """Load MODIS data from NetCDF."""
    print(f"  Loading NetCDF: {nc_file.name}")
    ds = xr.open_dataset(nc_file)

    sca_var = None
    for var_name in ['NDSI_Snow_Cover', 'snow_cover', 'SCA', 'MOD10A1_061_NDSI_Snow_Cover']:
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

    # Convert to fraction if needed
    if np.nanmax(sca) > 1:
        sca = sca / 100.0

    df_modis = pd.DataFrame({'MODIS_SCF': sca}, index=times)
    df_modis['MODIS_snow'] = (df_modis['MODIS_SCF'] > 0.5).astype(float)

    print(f"  MODIS time range: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
    print(f"  Valid MODIS days: {df_modis['MODIS_SCF'].notna().sum()}")

    return df_modis


def _load_modis_csv(csv_files):
    """Load MODIS from CSV files."""
    print(f"  Loading {len(csv_files)} CSV file(s)")
    all_data = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            date_col = next((c for c in ['Date', 'date', 'time'] if c in df.columns), None)
            sca_col = next((c for c in df.columns if 'NDSI' in c or 'snow' in c.lower()), None)

            if date_col and sca_col:
                df['time'] = pd.to_datetime(df[date_col])
                df['sca'] = pd.to_numeric(df[sca_col], errors='coerce')
                all_data.append(df[['time', 'sca']])
        except Exception:
            continue

    if not all_data:
        return None

    combined = pd.concat(all_data).groupby('time').mean()
    if combined['sca'].max() > 1:
        combined['sca'] = combined['sca'] / 100.0

    df_modis = pd.DataFrame({'MODIS_SCF': combined['sca']})
    df_modis['MODIS_snow'] = (df_modis['MODIS_SCF'] > 0.5).astype(float)

    return df_modis


def load_smap_sm():
    """Load SMAP soil moisture data."""
    print(f"Looking for SMAP data in: {SMAP_PATH}")

    if not SMAP_PATH.exists():
        print("  SMAP directory not found")
        return None

    nc_files = list(SMAP_PATH.glob("*.nc"))
    csv_files = list(SMAP_PATH.glob("*.csv"))

    if nc_files:
        return _load_smap_netcdf(nc_files[0])
    elif csv_files:
        return _load_smap_csv(csv_files)
    else:
        print("  No SMAP files found")
        return None


def _load_smap_netcdf(nc_file):
    """Load SMAP from NetCDF."""
    print(f"  Loading NetCDF: {nc_file.name}")
    ds = xr.open_dataset(nc_file)

    sm_var = None
    for var_name in ['soil_moisture', 'sm_surface', 'Geophysical_Data_sm_surface',
                     'sm', 'volumetric_soil_moisture']:
        if var_name in ds:
            sm_var = var_name
            break

    if sm_var is None:
        print(f"  Could not find soil moisture variable. Available: {list(ds.data_vars)}")
        ds.close()
        return None

    sm = ds[sm_var].values
    if sm.ndim > 1:
        sm = np.nanmean(sm, axis=tuple(range(1, sm.ndim)))

    times = pd.to_datetime(ds.time.values)
    ds.close()

    df_smap = pd.DataFrame({'SMAP_SM': sm}, index=times)

    print(f"  SMAP time range: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
    print(f"  Valid SMAP days: {df_smap['SMAP_SM'].notna().sum()}")

    return df_smap


def _load_smap_csv(csv_files):
    """Load SMAP from CSV."""
    print(f"  Loading {len(csv_files)} CSV file(s)")
    all_data = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
            sm_col = next((c for c in df.columns if 'moisture' in c.lower() or 'sm' in c.lower()), None)
            if sm_col:
                all_data.append(df[[sm_col]].rename(columns={sm_col: 'sm'}))
        except:
            continue

    if not all_data:
        return None

    combined = pd.concat(all_data).groupby(level=0).mean()
    return pd.DataFrame({'SMAP_SM': combined['sm']})


def load_snotel_swe():
    """Load SNOTEL SWE observations."""
    print(f"Looking for SNOTEL data in: {SNOTEL_PATH}")

    if not SNOTEL_PATH.exists():
        print("  SNOTEL directory not found")
        return None

    csv_files = list(SNOTEL_PATH.glob("*.csv"))
    nc_files = list(SNOTEL_PATH.glob("*.nc"))

    if csv_files:
        return _load_snotel_csv(csv_files)
    elif nc_files:
        return _load_snotel_netcdf(nc_files[0])
    else:
        print("  No SNOTEL files found")
        return None


def _load_snotel_csv(csv_files):
    """Load SNOTEL from CSV."""
    all_data = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, parse_dates=True)
            date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), df.columns[0])
            swe_col = next((c for c in df.columns if 'swe' in c.lower()), None)

            if swe_col:
                df['time'] = pd.to_datetime(df[date_col])
                df['swe'] = pd.to_numeric(df[swe_col], errors='coerce')
                all_data.append(df[['time', 'swe']].set_index('time'))
        except:
            continue

    if not all_data:
        return None

    # Average across stations
    combined = pd.concat(all_data, axis=1).mean(axis=1)
    return pd.DataFrame({'SNOTEL_SWE': combined})


def _load_snotel_netcdf(nc_file):
    """Load SNOTEL from NetCDF."""
    ds = xr.open_dataset(nc_file)

    swe_var = next((v for v in ds.data_vars if 'swe' in v.lower()), None)
    if swe_var is None:
        ds.close()
        return None

    swe = ds[swe_var].values.flatten()
    times = pd.to_datetime(ds.time.values)
    ds.close()

    return pd.DataFrame({'SNOTEL_SWE': swe}, index=times)


def load_catchment_shapefile():
    """Load catchment boundary shapefile."""
    shp_paths = [
        DATA_DIR / "shapefiles/catchment/lumped/paradise_sca_sm/paradise_multivar_HRUs_GRUS.shp",
        DATA_DIR / "shapefiles/catchment",
    ]

    for shp_path in shp_paths:
        if shp_path.exists():
            if shp_path.is_dir():
                shp_files = list(shp_path.rglob("*.shp"))
                if shp_files:
                    print(f"Loading catchment from: {shp_files[0]}")
                    return gpd.read_file(shp_files[0])
            else:
                print(f"Loading catchment from: {shp_path}")
                return gpd.read_file(shp_path)

    print("Warning: Could not find catchment shapefile")
    return None


def filter_spinup(df, spinup_end=SPINUP_END):
    """Remove spinup period from dataframe."""
    if df is None:
        return None
    return df[df.index > spinup_end].copy()


def calculate_sca_metrics(df_sim, df_modis):
    """Calculate snow cover accuracy metrics."""
    if df_modis is None or df_sim is None:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    if 'sim_SCF' not in df_sim.columns:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    # Add binary snow column if not present
    if 'sim_snow' not in df_sim.columns:
        df_sim = df_sim.copy()
        df_sim['sim_snow'] = (df_sim['sim_SCF'] > 0.5).astype(float)

    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)

    common_dates = df_sim_filt.index.intersection(df_modis_filt.index)

    if len(common_dates) == 0:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    sim_snow = df_sim_filt.loc[common_dates, 'sim_snow'].values
    obs_snow = df_modis_filt.loc[common_dates, 'MODIS_snow'].values
    sim_scf = df_sim_filt.loc[common_dates, 'sim_SCF'].values
    obs_scf = df_modis_filt.loc[common_dates, 'MODIS_SCF'].values

    valid = ~(np.isnan(sim_snow) | np.isnan(obs_snow))
    sim_snow = sim_snow[valid]
    obs_snow = obs_snow[valid]
    sim_scf = sim_scf[valid]
    obs_scf = obs_scf[valid]

    if len(sim_snow) == 0:
        return {'accuracy': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'r': np.nan}

    # Contingency table
    hits = np.sum((sim_snow == 1) & (obs_snow == 1))
    misses = np.sum((sim_snow == 0) & (obs_snow == 1))
    false_alarms = np.sum((sim_snow == 1) & (obs_snow == 0))
    correct_negatives = np.sum((sim_snow == 0) & (obs_snow == 0))

    n_total = len(sim_snow)
    accuracy = (hits + correct_negatives) / n_total if n_total > 0 else np.nan
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    r = np.corrcoef(sim_scf, obs_scf)[0, 1] if len(sim_scf) > 2 else np.nan

    return {
        'accuracy': accuracy, 'POD': pod, 'FAR': far, 'CSI': csi, 'r': r,
        'n_days': len(sim_snow), 'hits': hits, 'misses': misses,
        'false_alarms': false_alarms, 'correct_negatives': correct_negatives
    }


def calculate_sm_metrics(df_sim, df_smap):
    """Calculate soil moisture comparison metrics."""
    if df_smap is None or df_sim is None:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    if 'sim_SM' not in df_sim.columns:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    df_sim_filt = filter_spinup(df_sim)
    df_smap_filt = filter_spinup(df_smap)

    common_dates = df_sim_filt.index.intersection(df_smap_filt.index)

    if len(common_dates) == 0:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    sim_vals = df_sim_filt.loc[common_dates, 'sim_SM'].values
    smap_vals = df_smap_filt.loc[common_dates, 'SMAP_SM'].values

    valid = ~(np.isnan(sim_vals) | np.isnan(smap_vals))
    sim_vals = sim_vals[valid]
    smap_vals = smap_vals[valid]

    if len(sim_vals) < 3:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan}

    r = np.corrcoef(sim_vals, smap_vals)[0, 1]
    rmse = np.sqrt(np.mean((sim_vals - smap_vals)**2))
    bias = np.mean(sim_vals - smap_vals)

    return {'r': r, 'RMSE': rmse, 'bias': bias, 'n_days': len(sim_vals)}


def plot_domain_map(gdf, ax=None):
    """Plot the domain map with improved styling."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    colors = DOMAIN_COLORS['paradise']
    plot_domain_map_styled(
        gdf, ax,
        pour_point_coords=(PARADISE_LAT, PARADISE_LON),
        title='Paradise, Mt. Rainier',
        catchment_color=colors['catchment'],
        edge_color=colors['edge'],
        show_scale=True,
        show_north=True,
        show_inset=True,
        inset_extent=get_region_inset_extent('pacific_northwest')
    )

    return ax


def plot_tws_components(df, ax=None):
    """Plot TWS component time series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if df is None:
        ax.text(0.5, 0.5, 'No SUMMA output available', ha='center', va='center', transform=ax.transAxes)
        return ax

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
    ax.set_ylabel('Water Storage (mm)')
    ax.set_title('Simulated Total Water Storage Components', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_sca_comparison(df_sim, df_modis, sca_metrics, ax=None):
    """Plot snow cover fraction comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if df_modis is None or df_sim is None or 'sim_SCF' not in df_sim.columns:
        ax.text(0.5, 0.5, 'MODIS SCA data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Snow Cover Fraction: Simulated vs MODIS', fontsize=12, fontweight='bold')
        return ax

    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)

    # Weekly means for cleaner plot
    sim_weekly = df_sim_filt['sim_SCF'].resample('W').mean()
    modis_weekly = df_modis_filt['MODIS_SCF'].resample('W').mean()

    ax.plot(modis_weekly.index, modis_weekly.values,
            color=COLORS['modis'], linewidth=1.5, alpha=0.8, label='MODIS')
    ax.plot(sim_weekly.index, sim_weekly.values,
            color=COLORS['sim'], linewidth=1.5, alpha=0.8, label='Simulated')

    acc = sca_metrics.get('accuracy', np.nan)
    r = sca_metrics.get('r', np.nan)
    metrics_text = f"Accuracy = {acc:.2%}\nr = {r:.3f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('Snow Cover Fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Snow Cover Fraction: Simulated vs MODIS', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_sm_comparison(df_sim, df_smap, sm_metrics, ax=None):
    """Plot soil moisture comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if df_smap is None or df_sim is None or 'sim_SM' not in df_sim.columns:
        ax.text(0.5, 0.5, 'SMAP soil moisture data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Soil Moisture: Simulated vs SMAP', fontsize=12, fontweight='bold')
        return ax

    df_sim_filt = filter_spinup(df_sim)
    df_smap_filt = filter_spinup(df_smap)

    # Weekly means
    sim_weekly = df_sim_filt['sim_SM'].resample('W').mean()
    smap_weekly = df_smap_filt['SMAP_SM'].resample('W').mean()

    ax.plot(smap_weekly.index, smap_weekly.values,
            color=COLORS['smap'], linewidth=1.5, alpha=0.8, label='SMAP')
    ax.plot(sim_weekly.index, sim_weekly.values,
            color=COLORS['sim'], linewidth=1.5, alpha=0.8, label='Simulated')

    r = sm_metrics.get('r', np.nan)
    rmse = sm_metrics.get('RMSE', np.nan)
    metrics_text = f"r = {r:.3f}\nRMSE = {rmse:.3f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('Soil Moisture (m³/m³)')
    ax.set_title('Soil Moisture: Simulated vs SMAP', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_sca_scatter(df_sim, df_modis, sca_metrics, ax=None):
    """Plot SCA scatter plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if df_modis is None or df_sim is None or 'sim_SCF' not in df_sim.columns:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return ax

    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)
    common_dates = df_sim_filt.index.intersection(df_modis_filt.index)

    if len(common_dates) == 0:
        ax.text(0.5, 0.5, 'No overlapping data', ha='center', va='center', transform=ax.transAxes)
        return ax

    sim_scf = df_sim_filt.loc[common_dates, 'sim_SCF'].values
    modis_scf = df_modis_filt.loc[common_dates, 'MODIS_SCF'].values

    valid = ~(np.isnan(sim_scf) | np.isnan(modis_scf))
    sim_scf = sim_scf[valid]
    modis_scf = modis_scf[valid]

    ax.scatter(modis_scf, sim_scf, alpha=0.3, s=5, c=COLORS['modis'])
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='1:1 line')

    r = sca_metrics.get('r', np.nan)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(sim_scf)}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('MODIS SCF')
    ax.set_ylabel('Simulated SCF')
    ax.set_title('Snow Cover Scatter', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    return ax


def plot_sm_scatter(df_sim, df_smap, sm_metrics, ax=None):
    """Plot soil moisture scatter plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if df_smap is None or df_sim is None or 'sim_SM' not in df_sim.columns:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return ax

    df_sim_filt = filter_spinup(df_sim)
    df_smap_filt = filter_spinup(df_smap)
    common_dates = df_sim_filt.index.intersection(df_smap_filt.index)

    if len(common_dates) == 0:
        ax.text(0.5, 0.5, 'No overlapping data', ha='center', va='center', transform=ax.transAxes)
        return ax

    sim_sm = df_sim_filt.loc[common_dates, 'sim_SM'].values
    smap_sm = df_smap_filt.loc[common_dates, 'SMAP_SM'].values

    valid = ~(np.isnan(sim_sm) | np.isnan(smap_sm))
    sim_sm = sim_sm[valid]
    smap_sm = smap_sm[valid]

    ax.scatter(smap_sm, sim_sm, alpha=0.5, s=20, c=COLORS['smap'])

    min_val = min(smap_sm.min(), sim_sm.min())
    max_val = max(smap_sm.max(), sim_sm.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 line')

    r = sm_metrics.get('r', np.nan)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(sim_sm)}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('SMAP Soil Moisture (m³/m³)')
    ax.set_ylabel('Simulated Soil Moisture (m³/m³)')
    ax.set_title('Soil Moisture Scatter', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')

    return ax


def create_overview_figure(df_summa, gdf, df_modis, df_smap, sca_metrics, sm_metrics):
    """Create comprehensive overview figure."""
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 0.9, 0.8, 0.8, 0.9], hspace=0.35, wspace=0.25)

    df_filt = filter_spinup(df_summa) if df_summa is not None else None

    # Row 1 left: Domain map
    ax1 = fig.add_subplot(gs[0, 0])
    plot_domain_map(gdf, ax1)

    # Row 1 right: Summary statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    if df_filt is not None:
        summary_text = f"""
    Paradise (Mt. Rainier) - SUMMA Evaluation
    ══════════════════════════════════════════

    Analysis Period: {CALIBRATION_START} to {EVALUATION_END}
    (Spinup excluded: before {SPINUP_END})

    TWS Statistics (mm):
    ────────────────────
    Mean SWE:           {df_filt['SWE'].mean():.1f}
    Mean Soil Water:    {df_filt['soil_water'].mean():.1f}
    Mean Aquifer:       {df_filt['aquifer'].mean():.1f}
    Mean Total TWS:     {df_filt['TWS'].mean():.1f}

    MODIS SCA Metrics:
    ──────────────────
    Accuracy:           {sca_metrics.get('accuracy', np.nan):.2%}
    SCF Correlation:    {sca_metrics.get('r', np.nan):.3f}
    CSI:                {sca_metrics.get('CSI', np.nan):.3f}

    SMAP Soil Moisture Metrics:
    ───────────────────────────
    Correlation (r):    {sm_metrics.get('r', np.nan):.3f}
    RMSE:               {sm_metrics.get('RMSE', np.nan):.4f}
    Bias:               {sm_metrics.get('bias', np.nan):.4f}
        """
    else:
        summary_text = "No SUMMA output available"

    ax2.text(0.05, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: TWS components
    ax3 = fig.add_subplot(gs[1, :])
    plot_tws_components(df_summa, ax3)

    # Row 3: MODIS SCA comparison
    ax4 = fig.add_subplot(gs[2, :])
    plot_sca_comparison(df_summa, df_modis, sca_metrics, ax4)

    # Row 4: SMAP soil moisture comparison
    ax5 = fig.add_subplot(gs[3, :])
    plot_sm_comparison(df_summa, df_smap, sm_metrics, ax5)

    # Row 5 left: SCA scatter
    ax6 = fig.add_subplot(gs[4, 0])
    plot_sca_scatter(df_summa, df_modis, sca_metrics, ax6)

    # Row 5 right: SM scatter
    ax7 = fig.add_subplot(gs[4, 1])
    plot_sm_scatter(df_summa, df_smap, sm_metrics, ax7)

    plt.suptitle('Paradise (Mt. Rainier): Multivariate Evaluation Overview',
                 fontsize=14, fontweight='bold', y=0.99)

    return fig


def main():
    """Main execution function."""
    print("=" * 60)
    print("Paradise Multivariate Evaluation - Overview Generation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_summa = load_summa_output()
    if df_summa is not None:
        print(f"  SUMMA output: {len(df_summa)} days")

    gdf = load_catchment_shapefile()
    if gdf is not None:
        print(f"  Catchment loaded: {len(gdf)} features")

    # Load observation data
    df_modis = load_modis_sca()
    df_smap = load_smap_sm()
    df_snotel = load_snotel_swe()

    # Calculate metrics
    print("\nCalculating metrics...")
    sca_metrics = calculate_sca_metrics(df_summa, df_modis)
    print(f"  SCA Accuracy: {sca_metrics.get('accuracy', np.nan):.2%}")
    print(f"  SCF Correlation: {sca_metrics.get('r', np.nan):.3f}")

    sm_metrics = calculate_sm_metrics(df_summa, df_smap)
    print(f"  SM Correlation: {sm_metrics.get('r', np.nan):.3f}")

    # Create overview figure
    print("\nGenerating overview figure...")
    fig = create_overview_figure(df_summa, gdf, df_modis, df_smap, sca_metrics, sm_metrics)

    # Save figure
    output_path = OUTPUT_DIR / "paradise_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")

    pdf_path = OUTPUT_DIR / "paradise_overview.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")

    plt.close(fig)

    # Save summary statistics
    if df_summa is not None:
        df_filt = filter_spinup(df_summa)
        stats_path = OUTPUT_DIR / "paradise_summary_stats.csv"
        stats_df = pd.DataFrame({
            'Metric': ['Mean SWE (mm)', 'Mean Soil Water (mm)', 'Mean Aquifer (mm)',
                       'Mean Total TWS (mm)', 'MODIS SCA Accuracy', 'MODIS SCF r',
                       'MODIS SCA POD', 'MODIS SCA FAR', 'MODIS SCA CSI',
                       'SMAP SM r', 'SMAP SM RMSE', 'SMAP SM Bias'],
            'Value': [df_filt['SWE'].mean(), df_filt['soil_water'].mean(),
                      df_filt['aquifer'].mean(), df_filt['TWS'].mean(),
                      sca_metrics.get('accuracy', np.nan), sca_metrics.get('r', np.nan),
                      sca_metrics.get('POD', np.nan), sca_metrics.get('FAR', np.nan),
                      sca_metrics.get('CSI', np.nan),
                      sm_metrics.get('r', np.nan), sm_metrics.get('RMSE', np.nan),
                      sm_metrics.get('bias', np.nan)]
        })
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved stats to: {stats_path}")

    print("\n" + "=" * 60)
    print("Overview generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
