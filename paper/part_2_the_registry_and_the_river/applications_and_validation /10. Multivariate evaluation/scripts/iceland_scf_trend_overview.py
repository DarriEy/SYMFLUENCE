#!/usr/bin/env python3
"""
Iceland Regional SCF Trend Study - Overview Plots
Section 4.10c: Snow Cover Fraction Trend Analysis (2000-2023)

This script generates:
1. Regional domain map showing all Iceland basins
2. SCF time series and trends (simulated vs MODIS)
3. Elevation-band SCF analysis
4. Seasonal decomposition
5. Mann-Kendall trend statistics
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
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

# Set up paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Iceland_multivar_scf_trend")
MODIS_PATH = DATA_DIR / "observations/modis_scf"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/iceland_scf_trend")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time periods
SPINUP_END = '2001-12-31'
CALIBRATION_START = '2002-01-01'
CALIBRATION_END = '2012-12-31'
EVALUATION_START = '2013-01-01'
EVALUATION_END = '2023-12-31'

# Iceland centroid
ICELAND_LAT = 64.96
ICELAND_LON = -19.02

# Elevation bands for analysis (m)
ELEVATION_BANDS = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000]

# Seasons
SEASONS = {
    'accumulation': [10, 11, 12, 1, 2, 3],
    'ablation': [4, 5, 6],
    'snow_free': [7, 8, 9]
}

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'sim': '#2E86AB',
    'modis': '#E74C3C',
    'trend_pos': '#27AE60',
    'trend_neg': '#C0392B',
    'accumulation': '#3498DB',
    'ablation': '#E67E22',
    'snow_free': '#27AE60',
}

# Custom colormap for SCF
SCF_CMAP = LinearSegmentedColormap.from_list('scf', ['#8B4513', '#F5DEB3', '#87CEEB', '#FFFFFF'])


def load_basins_shapefile():
    """Load the river basins shapefile (with coastal watersheds if available)."""
    # Prefer the coastal watersheds version
    shp_path = DATA_DIR / "shapefiles/river_basins/Iceland_multivar_scf_trend_riverBasins_with_coastal.shp"
    if not shp_path.exists():
        # Fall back to semidistributed version
        shp_path = DATA_DIR / "shapefiles/river_basins/Iceland_multivar_scf_trend_riverBasins_semidistributed.shp"

    if not shp_path.exists():
        print(f"Basins shapefile not found: {shp_path}")
        return None

    print(f"Loading basins from: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"  Loaded {len(gdf)} basins")
    return gdf


def load_catchment_shapefile():
    """Load the HRU/catchment shapefile."""
    shp_path = DATA_DIR / "shapefiles/catchment/semidistributed/iceland_scf_trend/Iceland_multivar_scf_trend_HRUs_GRUS.shp"

    if not shp_path.exists():
        # Try basins as fallback
        return load_basins_shapefile()

    print(f"Loading catchment from: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"  Loaded {len(gdf)} HRUs")
    return gdf


def load_summa_output(experiment_id='iceland_scf_trend'):
    """Load SUMMA output for SCF analysis."""
    daily_path = DATA_DIR / f"simulations/{experiment_id}/SUMMA/{experiment_id}_day.nc"

    if not daily_path.exists():
        print(f"SUMMA output not found: {daily_path}")
        print("  Run the model first: symfluence workflow run --config <iceland_config>")
        return None

    print(f"Loading SUMMA output from: {daily_path}")
    ds = xr.open_dataset(daily_path)

    # Extract SCF and SWE
    times = pd.to_datetime(ds.time.values)

    data = {'time': times}

    if 'scalarGroundSnowFraction' in ds:
        # Multi-basin: average or keep all
        scf = ds['scalarGroundSnowFraction'].values
        if scf.ndim > 1:
            data['sim_SCF'] = np.nanmean(scf, axis=1)  # Regional average
            data['sim_SCF_all'] = scf  # Keep all basins
        else:
            data['sim_SCF'] = scf.flatten()

    if 'scalarSWE' in ds:
        swe = ds['scalarSWE'].values
        if swe.ndim > 1:
            data['SWE'] = np.nanmean(swe, axis=1)
        else:
            data['SWE'] = swe.flatten()

    ds.close()

    df = pd.DataFrame({k: v for k, v in data.items() if k != 'sim_SCF_all'})
    df.set_index('time', inplace=True)

    return df


def load_modis_scf():
    """Load MODIS MOD10A2 snow cover fraction data."""
    print(f"Looking for MODIS SCF data in: {MODIS_PATH}")

    if not MODIS_PATH.exists():
        print("  MODIS SCF directory not found")
        return None

    nc_files = list(MODIS_PATH.glob("*.nc"))
    if not nc_files:
        print("  No MODIS SCF files found")
        return None

    print(f"  Found {len(nc_files)} MODIS file(s)")

    # Load and combine
    datasets = []
    for f in nc_files:
        ds = xr.open_dataset(f)
        datasets.append(ds)

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.concat(datasets, dim='time')

    # Extract SCF
    scf_var = None
    for var in ['Maximum_Snow_Extent', 'NDSI_Snow_Cover', 'snow_cover', 'SCF']:
        if var in ds:
            scf_var = var
            break

    if scf_var is None:
        print(f"  Could not find SCF variable. Available: {list(ds.data_vars)}")
        return None

    scf = ds[scf_var].values
    times = pd.to_datetime(ds.time.values)

    # Regional average if multi-dimensional
    if scf.ndim > 1:
        scf = np.nanmean(scf, axis=tuple(range(1, scf.ndim)))

    # Convert to fraction if needed
    if np.nanmax(scf) > 1:
        scf = scf / 100.0

    ds.close()

    df = pd.DataFrame({'MODIS_SCF': scf}, index=times)
    print(f"  MODIS time range: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")

    return df


def filter_spinup(df, spinup_end=SPINUP_END):
    """Remove spinup period."""
    if df is None:
        return None
    return df[df.index > spinup_end].copy()


def calculate_trend(series, alpha=0.05):
    """Calculate Mann-Kendall trend statistics."""
    try:
        from scipy import stats

        # Remove NaN
        valid = series.dropna()
        if len(valid) < 10:
            return {'slope': np.nan, 'p_value': np.nan, 'significant': False}

        # Simple linear regression for slope
        x = np.arange(len(valid))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, valid.values)

        # Convert slope to per-decade
        slope_per_decade = slope * 365.25 * 10  # Assuming daily data

        return {
            'slope': slope,
            'slope_per_decade': slope_per_decade,
            'r_value': r_value,
            'p_value': p_value,
            'significant': p_value < alpha
        }
    except:
        return {'slope': np.nan, 'p_value': np.nan, 'significant': False}


def calculate_scf_metrics(df_sim, df_modis):
    """Calculate SCF comparison metrics."""
    if df_sim is None or df_modis is None:
        return {'r': np.nan, 'rmse': np.nan, 'bias': np.nan}

    if 'sim_SCF' not in df_sim.columns:
        return {'r': np.nan, 'rmse': np.nan, 'bias': np.nan}

    df_sim_filt = filter_spinup(df_sim)
    df_modis_filt = filter_spinup(df_modis)

    common = df_sim_filt.index.intersection(df_modis_filt.index)

    if len(common) == 0:
        return {'r': np.nan, 'rmse': np.nan, 'bias': np.nan}

    sim = df_sim_filt.loc[common, 'sim_SCF'].values
    obs = df_modis_filt.loc[common, 'MODIS_SCF'].values

    valid = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[valid]
    obs = obs[valid]

    if len(sim) < 10:
        return {'r': np.nan, 'rmse': np.nan, 'bias': np.nan}

    r = np.corrcoef(sim, obs)[0, 1]
    rmse = np.sqrt(np.mean((sim - obs)**2))
    bias = np.mean(sim - obs)

    return {'r': r, 'rmse': rmse, 'bias': bias, 'n': len(sim)}


def plot_iceland_map(gdf, ax, title='Iceland Regional Domain'):
    """Plot Iceland regional map with basins."""
    if gdf is None:
        ax.text(0.5, 0.5, 'Shapefile not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax

    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf_wgs = gdf.to_crs(epsg=4326)

    # Plot basins
    gdf_wgs.plot(ax=ax, facecolor='#9B59B6', edgecolor='#6C3483',
                 linewidth=0.1, alpha=0.6)

    # Get bounds
    bounds = gdf_wgs.total_bounds
    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    # Info box
    n_basins = len(gdf)
    try:
        gdf_utm = gdf_wgs.to_crs(epsg=32627)  # UTM 27N for Iceland
        total_area = gdf_utm.area.sum() / 1e6
        area_text = f'{total_area:,.0f} km²'
    except:
        area_text = 'N/A'

    info_text = f'{title}\n{n_basins:,} basins\nArea: {area_text}'
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=10,
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#BDC3C7', alpha=0.95))

    # North arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=12, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2),
                color='#2C3E50')

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    return ax


def plot_scf_timeseries(df_sim, df_modis, ax):
    """Plot SCF time series comparison."""
    if df_sim is None and df_modis is None:
        ax.text(0.5, 0.5, 'No data available\nRun model and/or download MODIS data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Snow Cover Fraction Time Series', fontsize=12, fontweight='bold')
        return ax

    # Plot MODIS if available
    if df_modis is not None:
        df_m = filter_spinup(df_modis)
        monthly = df_m['MODIS_SCF'].resample('M').mean()
        ax.plot(monthly.index, monthly.values, color=COLORS['modis'],
                linewidth=1.5, alpha=0.8, label='MODIS MOD10A2')

    # Plot simulated if available
    if df_sim is not None and 'sim_SCF' in df_sim.columns:
        df_s = filter_spinup(df_sim)
        monthly = df_s['sim_SCF'].resample('M').mean()
        ax.plot(monthly.index, monthly.values, color=COLORS['sim'],
                linewidth=1.5, alpha=0.8, label='Simulated')

    ax.set_xlabel('Date')
    ax.set_ylabel('Snow Cover Fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Snow Cover Fraction: Regional Average (2002-2023)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return ax


def plot_seasonal_scf(df_sim, df_modis, ax):
    """Plot seasonal SCF patterns."""
    ax.set_title('Seasonal SCF Patterns', fontsize=12, fontweight='bold')

    has_data = False

    # MODIS seasonal
    if df_modis is not None:
        df_m = filter_spinup(df_modis)
        df_m['month'] = df_m.index.month
        monthly_clim = df_m.groupby('month')['MODIS_SCF'].mean()
        ax.plot(monthly_clim.index, monthly_clim.values, 'o-',
                color=COLORS['modis'], linewidth=2, markersize=8, label='MODIS')
        has_data = True

    # Simulated seasonal
    if df_sim is not None and 'sim_SCF' in df_sim.columns:
        df_s = filter_spinup(df_sim)
        df_s['month'] = df_s.index.month
        monthly_clim = df_s.groupby('month')['sim_SCF'].mean()
        ax.plot(monthly_clim.index, monthly_clim.values, 's-',
                color=COLORS['sim'], linewidth=2, markersize=8, label='Simulated')
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return ax

    # Add season shading
    ax.axvspan(0.5, 3.5, alpha=0.1, color=COLORS['accumulation'], label='Accumulation')
    ax.axvspan(3.5, 6.5, alpha=0.1, color=COLORS['ablation'], label='Ablation')
    ax.axvspan(6.5, 9.5, alpha=0.1, color=COLORS['snow_free'], label='Snow-free')
    ax.axvspan(9.5, 12.5, alpha=0.1, color=COLORS['accumulation'])

    ax.set_xlabel('Month')
    ax.set_ylabel('Mean SCF')
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.legend(loc='upper right', fontsize=8)

    return ax


def plot_trend_analysis(df_modis, ax):
    """Plot SCF trend analysis."""
    ax.set_title('SCF Trend Analysis (Mann-Kendall)', fontsize=12, fontweight='bold')

    if df_modis is None:
        ax.text(0.5, 0.5, 'MODIS data not available', ha='center', va='center', transform=ax.transAxes)
        return ax

    df_m = filter_spinup(df_modis)

    # Annual mean SCF
    annual = df_m['MODIS_SCF'].resample('Y').mean()

    # Calculate trend
    trend = calculate_trend(annual)

    # Plot
    ax.plot(annual.index, annual.values, 'o-', color=COLORS['modis'],
            linewidth=2, markersize=6, label='Annual Mean SCF')

    # Add trend line
    if not np.isnan(trend['slope']):
        x = np.arange(len(annual))
        y_trend = trend['slope'] * x + annual.values[0]
        color = COLORS['trend_neg'] if trend['slope'] < 0 else COLORS['trend_pos']
        linestyle = '-' if trend['significant'] else '--'
        ax.plot(annual.index, y_trend, color=color, linewidth=2,
                linestyle=linestyle, label=f"Trend: {trend['slope_per_decade']:.3f}/decade")

    # Add stats text
    sig_text = 'Significant' if trend['significant'] else 'Not significant'
    stats_text = f"Slope: {trend.get('slope_per_decade', np.nan):.4f}/decade\np-value: {trend['p_value']:.3f}\n{sig_text}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Mean SCF')
    ax.legend(loc='upper right')

    return ax


def create_overview_figure(gdf, df_sim, df_modis, scf_metrics):
    """Create comprehensive Iceland SCF trend overview figure."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)

    # Row 1 left: Iceland map
    ax1 = fig.add_subplot(gs[0, 0])
    plot_iceland_map(gdf, ax1, title='Iceland SCF Study Domain')

    # Row 1 right: Summary statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    summary_text = f"""
    Iceland Regional SCF Trend Study
    ═══════════════════════════════════

    Study Period: {CALIBRATION_START} to {EVALUATION_END}
    (Spinup: before {SPINUP_END})

    Domain Statistics:
    ──────────────────
    Number of basins:   {len(gdf) if gdf is not None else 'N/A':,}

    SCF Comparison Metrics:
    ───────────────────────
    Correlation (r):    {scf_metrics.get('r', np.nan):.3f}
    RMSE:               {scf_metrics.get('rmse', np.nan):.3f}
    Bias:               {scf_metrics.get('bias', np.nan):.3f}

    Data Status:
    ────────────
    Simulations:        {'Available' if df_sim is not None else 'Not run yet'}
    MODIS MOD10A2:      {'Available' if df_modis is not None else 'Not downloaded'}
    """

    ax2.text(0.05, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 2: SCF time series
    ax3 = fig.add_subplot(gs[1, :])
    plot_scf_timeseries(df_sim, df_modis, ax3)

    # Row 3 left: Seasonal patterns
    ax4 = fig.add_subplot(gs[2, 0])
    plot_seasonal_scf(df_sim, df_modis, ax4)

    # Row 3 right: Trend analysis
    ax5 = fig.add_subplot(gs[2, 1])
    plot_trend_analysis(df_modis, ax5)

    plt.suptitle('Iceland: Regional Snow Cover Fraction Trend Analysis (2000-2023)',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("Iceland Regional SCF Trend Study - Overview Generation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    gdf = load_basins_shapefile()

    df_sim = load_summa_output()
    df_modis = load_modis_scf()

    # Calculate metrics
    print("\nCalculating metrics...")
    scf_metrics = calculate_scf_metrics(df_sim, df_modis)
    print(f"  SCF correlation: {scf_metrics.get('r', np.nan):.3f}")

    # Create figure
    print("\nGenerating overview figure...")
    fig = create_overview_figure(gdf, df_sim, df_modis, scf_metrics)

    # Save
    output_path = OUTPUT_DIR / "iceland_scf_trend_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")

    pdf_path = OUTPUT_DIR / "iceland_scf_trend_overview.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")

    plt.close(fig)

    # Save summary stats
    stats_path = OUTPUT_DIR / "iceland_scf_trend_stats.csv"
    stats_df = pd.DataFrame({
        'Metric': ['Number of basins', 'SCF r', 'SCF RMSE', 'SCF Bias',
                   'Simulations available', 'MODIS available'],
        'Value': [len(gdf) if gdf is not None else np.nan,
                  scf_metrics.get('r', np.nan), scf_metrics.get('rmse', np.nan),
                  scf_metrics.get('bias', np.nan),
                  'Yes' if df_sim is not None else 'No',
                  'Yes' if df_modis is not None else 'No']
    })
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats to: {stats_path}")

    print("\n" + "=" * 70)
    print("Overview generation complete!")
    if df_sim is None:
        print("\nNOTE: Run SUMMA simulations to get model comparison:")
        print("  symfluence workflow run --config <iceland_scf_config>")
    if df_modis is None:
        print("\nNOTE: Download MODIS MOD10A2 data:")
        print("  symfluence workflow steps process_observed_data --config <iceland_scf_config>")
    print("=" * 70)


if __name__ == "__main__":
    main()
