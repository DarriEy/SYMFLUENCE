#!/usr/bin/env python3
"""
Compare SUMMA ET with SSEBop observations for Bow at Banff.

This script processes the downloaded SSEBop data and compares it with
SUMMA-simulated ET, creating a figure panel and calculating metrics.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
SSEBOP_DIR = DATA_DIR / "observations/et/ssebop"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPEFILE = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"
CATCHMENT_AREA_KM2 = 2210

# Analysis period
ANALYSIS_START = '2004-01-01'
ANALYSIS_END = '2017-12-31'


def load_summa_et():
    """Load SUMMA simulated ET."""
    daily_path = DATA_DIR / "simulations/bow_tws_uncalibrated/SUMMA/bow_tws_uncalibrated_day.nc"
    ds = xr.open_dataset(daily_path)

    times = pd.to_datetime(ds.time.values)
    # SUMMA: negative = upward flux (ET), convert kg/m²/s to mm/day
    et = -ds['scalarTotalET'].values.flatten() * 86400

    df = pd.DataFrame({'ET_sim': et}, index=times)
    ds.close()

    # Filter to analysis period
    df = df[(df.index >= ANALYSIS_START) & (df.index <= ANALYSIS_END)]

    # Monthly mean
    df_monthly = df.resample('ME').mean()

    print(f"SUMMA ET loaded: {len(df_monthly)} months")
    print(f"  Mean: {df_monthly['ET_sim'].mean():.2f} mm/day")

    return df_monthly


def process_ssebop_files():
    """Process all downloaded SSEBop files to extract catchment mean ET."""
    gdf = gpd.read_file(SHAPEFILE)

    tif_files = sorted(SSEBOP_DIR.glob("m*.tif"))
    print(f"Found {len(tif_files)} SSEBop files")

    all_data = []

    for tif_file in tif_files:
        try:
            # Extract date from filename (m200401.tif)
            fname = tif_file.stem  # m200401
            year = int(fname[1:5])
            month = int(fname[5:7])
            date = pd.Timestamp(year=year, month=month, day=15)  # Mid-month

            # Skip if outside analysis period
            if date < pd.Timestamp(ANALYSIS_START) or date > pd.Timestamp(ANALYSIS_END):
                continue

            with rasterio.open(tif_file) as src:
                # Ensure same CRS
                gdf_reproj = gdf.to_crs(src.crs)

                # Mask to catchment
                out_image, out_transform = mask(src, gdf_reproj.geometry, crop=True)
                data = out_image[0]

                # Mask no-data values (typically negative or very high)
                nodata = src.nodata if src.nodata is not None else -9999
                data = np.ma.masked_equal(data, nodata)
                data = np.ma.masked_less_equal(data, 0)
                data = np.ma.masked_greater(data, 1000)  # Sanity check

                if data.count() == 0:
                    continue

                # SSEBop is in mm/month, convert to mm/day
                days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
                et_mm_day = float(data.mean()) / days_in_month

                all_data.append({'date': date, 'ET_obs': et_mm_day})

        except Exception as e:
            print(f"  Error processing {tif_file.name}: {e}")
            continue

    if not all_data:
        print("No SSEBop data could be processed")
        return None

    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Convert to monthly period index for easier comparison
    df.index = df.index.to_period('M').to_timestamp()

    print(f"SSEBop processed: {len(df)} months")
    print(f"  Mean: {df['ET_obs'].mean():.2f} mm/day")

    return df


def calculate_metrics(sim, obs):
    """Calculate correlation metrics between simulated and observed ET."""
    # Convert both to period index for proper alignment
    sim_p = sim.copy()
    obs_p = obs.copy()
    sim_p.index = sim_p.index.to_period('M')
    obs_p.index = obs_p.index.to_period('M')

    # Align on common periods
    common = sim_p.index.intersection(obs_p.index)
    if len(common) < 12:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    s = sim_p.loc[common].values.flatten()
    o = obs_p.loc[common].values.flatten()

    valid = ~(np.isnan(s) | np.isnan(o))
    s, o = s[valid], o[valid]

    if len(s) < 12:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    r = np.corrcoef(s, o)[0, 1]
    rmse = np.sqrt(np.mean((s - o)**2))
    bias = np.mean(s - o)

    return {'r': r, 'RMSE': rmse, 'bias': bias, 'n': len(s)}


def create_et_figure(df_sim, df_obs, metrics):
    """Create ET comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Convert to period for alignment
    sim_p = df_sim.copy()
    obs_p = df_obs.copy()
    sim_p.index = sim_p.index.to_period('M')
    obs_p.index = obs_p.index.to_period('M')

    # Time series
    ax = axes[0]
    common = sim_p.index.intersection(obs_p.index)
    # Convert back to timestamp for plotting
    sim_plot = sim_p.loc[common].copy()
    obs_plot = obs_p.loc[common].copy()
    sim_plot.index = sim_plot.index.to_timestamp()
    obs_plot.index = obs_plot.index.to_timestamp()

    ax.plot(sim_plot.index, sim_plot['ET_sim'],
            color='#2171B5', linewidth=1.2, label='SUMMA')
    ax.plot(obs_plot.index, obs_plot['ET_obs'],
            color='#D94801', linewidth=1.2, label='SSEBop')

    metrics_text = f"r = {metrics['r']:.2f}\nRMSE = {metrics['RMSE']:.2f} mm/day\nBias = {metrics['bias']:.2f} mm/day"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('ET (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title('(a) Monthly ET Time Series', fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Scatter plot
    ax = axes[1]
    ax.scatter(obs_plot['ET_obs'], sim_plot['ET_sim'],
               c='#7A0177', s=30, alpha=0.7)
    max_val = max(obs_plot['ET_obs'].max(), sim_plot['ET_sim'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('SSEBop (mm/day)')
    ax.set_ylabel('SUMMA (mm/day)')
    ax.set_title('(b) ET Scatter', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # Seasonal cycle
    ax = axes[2]
    sim_seasonal = sim_plot['ET_sim'].groupby(sim_plot.index.month).mean()
    obs_seasonal = obs_plot['ET_obs'].groupby(obs_plot.index.month).mean()

    months = range(1, 13)
    ax.plot(months, [sim_seasonal.get(m, np.nan) for m in months], 'o-',
            color='#2171B5', linewidth=1.5, markersize=6, label='SUMMA')
    ax.plot(months, [obs_seasonal.get(m, np.nan) for m in months], 's-',
            color='#D94801', linewidth=1.5, markersize=6, label='SSEBop')

    ax.set_xlabel('Month')
    ax.set_ylabel('ET (mm/day)')
    ax.set_title('(c) Mean Seasonal Cycle', fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.legend(loc='upper left')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "bow_banff_et_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.close(fig)


def main():
    """Main comparison workflow."""
    print("=" * 60)
    print("ET Comparison: SUMMA vs SSEBop")
    print("=" * 60)

    # Check if SSEBop files exist
    tif_files = list(SSEBOP_DIR.glob("m*.tif"))
    if len(tif_files) < 12:
        print(f"Only {len(tif_files)} SSEBop files available. Need at least 12 for meaningful comparison.")
        print("SSEBop download still in progress. Run again when complete.")
        return

    # Load data
    print("\nLoading SUMMA ET...")
    df_sim = load_summa_et()

    print("\nProcessing SSEBop...")
    df_obs = process_ssebop_files()

    if df_obs is None or len(df_obs) < 12:
        print("Insufficient SSEBop data")
        return

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df_sim['ET_sim'], df_obs['ET_obs'])
    print(f"  r = {metrics['r']:.3f}")
    print(f"  RMSE = {metrics['RMSE']:.3f} mm/day")
    print(f"  Bias = {metrics['bias']:.3f} mm/day")

    # Create figure
    print("\nCreating figure...")
    create_et_figure(df_sim, df_obs, metrics)

    # Save metrics
    metrics_df = pd.DataFrame([{
        'Variable': 'ET vs SSEBop',
        'Metric': 'r',
        'Value': metrics['r'],
        'Period': f"{ANALYSIS_START} to {ANALYSIS_END}"
    }, {
        'Variable': 'ET vs SSEBop',
        'Metric': 'RMSE (mm/day)',
        'Value': metrics['RMSE'],
        'Period': f"{ANALYSIS_START} to {ANALYSIS_END}"
    }, {
        'Variable': 'ET vs SSEBop',
        'Metric': 'Bias (mm/day)',
        'Value': metrics['bias'],
        'Period': f"{ANALYSIS_START} to {ANALYSIS_END}"
    }])
    metrics_df.to_csv(OUTPUT_DIR / "bow_banff_et_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
