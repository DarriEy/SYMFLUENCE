#!/usr/bin/env python3
"""
Compare SUMMA soil moisture with SMAP observations for Bow at Banff.

This script compares SUMMA-simulated soil moisture against SMAP satellite
retrievals, creating a figure panel and calculating metrics.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
SMAP_DIR = DATA_DIR / "observations/soil_moisture/smap"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SUMMA soil depth - surface layer (approximately 0-10 cm)
# SMAP retrieves top ~5 cm soil moisture


def load_summa_sm():
    """Load SUMMA simulated soil moisture."""
    daily_path = DATA_DIR / "simulations/bow_tws_uncalibrated/SUMMA/bow_tws_uncalibrated_day.nc"
    ds = xr.open_dataset(daily_path)

    times = pd.to_datetime(ds.time.values)

    # SUMMA soil moisture variables
    # scalarVolFracLiq: volumetric fraction of liquid water
    # scalarTotalSoilLiq: total liquid water in soil column (kg/m²)
    # For comparison with SMAP (surface), we should use the top layer

    # Check available variables
    sm_vars = [v for v in ds.data_vars if 'soil' in v.lower() or 'volFrac' in v.lower()]
    print(f"Available SM-related variables: {sm_vars}")

    # Try scalarVolFracLiq (volumetric water content)
    if 'scalarVolFracLiq' in ds:
        # This is the volumetric liquid water fraction (m³/m³)
        sm = ds['scalarVolFracLiq'].values.flatten()
    elif 'mLayerVolFracLiq' in ds:
        # Multi-layer - take the surface layer
        sm = ds['mLayerVolFracLiq'].values[:, 0, 0]  # First layer
    else:
        print("No volumetric SM variable found")
        # Fallback: estimate from total soil water
        if 'scalarTotalSoilLiq' in ds:
            # Convert kg/m² to m³/m³ assuming 1m soil column
            sm = ds['scalarTotalSoilLiq'].values.flatten() / 1000.0  # Very rough
        else:
            return None

    df = pd.DataFrame({'SM_sim': sm}, index=times)
    ds.close()

    # Filter valid values
    df = df[(df['SM_sim'] > 0) & (df['SM_sim'] < 1)]

    print(f"SUMMA SM loaded: {len(df)} days")
    print(f"  Range: {df['SM_sim'].min():.3f} - {df['SM_sim'].max():.3f} m³/m³")

    return df


def load_smap_sm():
    """Load SMAP observed soil moisture."""
    smap_file = SMAP_DIR / "smap_processed.csv"

    if not smap_file.exists():
        print(f"SMAP file not found: {smap_file}")
        return None

    df = pd.read_csv(smap_file, parse_dates=['date'], index_col='date')

    # Rename for consistency
    df = df.rename(columns={'soil_moisture': 'SM_obs'})

    print(f"SMAP SM loaded: {len(df)} days")
    print(f"  Range: {df['SM_obs'].min():.3f} - {df['SM_obs'].max():.3f} m³/m³")
    print(f"  Period: {df.index.min()} to {df.index.max()}")

    return df


def calculate_metrics(sim, obs):
    """Calculate correlation metrics between simulated and observed SM."""
    # Align on common dates
    common = sim.index.intersection(obs.index)
    if len(common) < 5:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    s = sim.loc[common].values.flatten()
    o = obs.loc[common].values.flatten()

    valid = ~(np.isnan(s) | np.isnan(o))
    s, o = s[valid], o[valid]

    if len(s) < 5:
        return {'r': np.nan, 'RMSE': np.nan, 'bias': np.nan, 'n': 0}

    r = np.corrcoef(s, o)[0, 1]
    rmse = np.sqrt(np.mean((s - o)**2))
    bias = np.mean(s - o)

    return {'r': r, 'RMSE': rmse, 'bias': bias, 'n': len(s)}


def create_sm_figure(df_sim, df_obs, metrics):
    """Create soil moisture comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Align on common dates
    common = df_sim.index.intersection(df_obs.index)
    sim_plot = df_sim.loc[common].copy()
    obs_plot = df_obs.loc[common].copy()

    # Time series
    ax = axes[0]
    ax.plot(sim_plot.index, sim_plot['SM_sim'],
            'o-', color='#2171B5', linewidth=1, markersize=4, label='SUMMA')
    ax.plot(obs_plot.index, obs_plot['SM_obs'],
            's-', color='#D94801', linewidth=1, markersize=4, label='SMAP')

    metrics_text = f"r = {metrics['r']:.2f}\nRMSE = {metrics['RMSE']:.3f} m³/m³\nBias = {metrics['bias']:.3f} m³/m³\nn = {metrics['n']}"
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.set_ylabel('Soil Moisture (m³/m³)')
    ax.set_xlabel('Date')
    ax.set_title('(a) Daily Soil Moisture', fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Scatter plot
    ax = axes[1]
    ax.scatter(obs_plot['SM_obs'], sim_plot['SM_sim'],
               c='#7A0177', s=40, alpha=0.7)
    max_val = max(obs_plot['SM_obs'].max(), sim_plot['SM_sim'].max()) * 1.1
    min_val = min(obs_plot['SM_obs'].min(), sim_plot['SM_sim'].min()) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('SMAP (m³/m³)')
    ax.set_ylabel('SUMMA (m³/m³)')
    ax.set_title('(b) Scatter Plot', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "bow_banff_sm_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.close(fig)


def main():
    """Main comparison workflow."""
    print("=" * 60)
    print("Soil Moisture Comparison: SUMMA vs SMAP")
    print("=" * 60)

    # Load SUMMA SM
    print("\nLoading SUMMA soil moisture...")
    df_sim = load_summa_sm()

    if df_sim is None:
        print("Could not load SUMMA data")
        return

    # Load SMAP SM
    print("\nLoading SMAP soil moisture...")
    df_obs = load_smap_sm()

    if df_obs is None or len(df_obs) < 5:
        print("Insufficient SMAP data")
        return

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df_sim['SM_sim'], df_obs['SM_obs'])
    print(f"  r = {metrics['r']:.3f}")
    print(f"  RMSE = {metrics['RMSE']:.4f} m³/m³")
    print(f"  Bias = {metrics['bias']:.4f} m³/m³")
    print(f"  n = {metrics['n']} days")

    # Create figure
    print("\nCreating figure...")
    create_sm_figure(df_sim, df_obs, metrics)

    # Save metrics
    metrics_df = pd.DataFrame([{
        'Variable': 'SM vs SMAP',
        'Metric': 'r',
        'Value': metrics['r'],
        'Period': f"{df_obs.index.min().strftime('%Y-%m-%d')} to {df_obs.index.max().strftime('%Y-%m-%d')}"
    }, {
        'Variable': 'SM vs SMAP',
        'Metric': 'RMSE (m³/m³)',
        'Value': metrics['RMSE'],
        'Period': f"{df_obs.index.min().strftime('%Y-%m-%d')} to {df_obs.index.max().strftime('%Y-%m-%d')}"
    }, {
        'Variable': 'SM vs SMAP',
        'Metric': 'Bias (m³/m³)',
        'Value': metrics['bias'],
        'Period': f"{df_obs.index.min().strftime('%Y-%m-%d')} to {df_obs.index.max().strftime('%Y-%m-%d')}"
    }])
    metrics_df.to_csv(OUTPUT_DIR / "bow_banff_sm_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
