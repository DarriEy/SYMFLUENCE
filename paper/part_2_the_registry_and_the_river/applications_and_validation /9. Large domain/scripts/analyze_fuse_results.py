#!/usr/bin/env python3
"""Analyze FUSE baseline results for the Iceland large domain (Section 4.9).

Creates multi-panel figures showing:
  - Time series of domain-averaged runoff, precipitation, temperature
  - Spatial distribution of mean runoff
  - Water balance components
  - Comparison with LamaH-Ice large sample catchments (where available)

Follows the publication style of analyze_large_domain.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
ANALYSIS_DIR = BASE_DIR / "analysis"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = SYMFLUENCE_DATA / "domain_Iceland_Multivar"
FUSE_OUTPUT_DIR = DOMAIN_DIR / "simulations" / "large_domain" / "FUSE"
LAMAHICE_DIR = SYMFLUENCE_DATA / "lamahice"

# Large sample catchment stats (from Section 4.8)
LARGE_SAMPLE_DIR = BASE_DIR.parent / "8. Large sample"
CATCHMENT_STATS_FILE = LARGE_SAMPLE_DIR / "analysis" / "catchment_stats.csv"

# ── Publication style (matching Section 4.8) ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_fuse_output() -> "xr.Dataset | None":
    """Load FUSE model output for the large domain."""
    if not HAS_XARRAY:
        print("Warning: xarray not available")
        return None

    # Find the runs file
    runs_files = list(FUSE_OUTPUT_DIR.glob("*_runs_def.nc"))
    if not runs_files:
        runs_files = list(FUSE_OUTPUT_DIR.glob("*_runs*.nc"))

    if not runs_files:
        print(f"Warning: No FUSE output found in {FUSE_OUTPUT_DIR}")
        return None

    print(f"  Loading: {runs_files[0].name}")
    return xr.open_dataset(runs_files[0])


def load_catchment_stats() -> "pd.DataFrame | None":
    """Load large sample catchment statistics for comparison."""
    if not CATCHMENT_STATS_FILE.exists():
        print(f"  Large sample stats not found: {CATCHMENT_STATS_FILE}")
        return None

    print(f"  Loading: {CATCHMENT_STATS_FILE.name}")
    return pd.read_csv(CATCHMENT_STATS_FILE)


def load_large_sample_observations(catchment_stats: pd.DataFrame,
                                    start_date: str = '2008-01-01',
                                    end_date: str = '2010-12-31') -> pd.DataFrame:
    """Load and aggregate observations from large sample catchments for comparison.

    Returns DataFrame with daily aggregated observations across all valid catchments.
    """
    # Filter catchments with observations covering the period
    stats = catchment_stats.copy()
    stats['record_start'] = pd.to_datetime(stats['record_start'])
    stats['record_end'] = pd.to_datetime(stats['record_end'])

    mask = ((stats['record_start'] <= start_date) &
            (stats['record_end'] >= end_date) &
            (stats['pct_valid'] > 50))
    valid_catchments = stats[mask]

    print(f"  Found {len(valid_catchments)} catchments with valid observations")

    if len(valid_catchments) == 0:
        return None

    # Load observations from each catchment
    all_obs = []
    total_area = 0

    for _, row in valid_catchments.iterrows():
        domain_id = row['domain_id']
        area_km2 = row['area_km2']

        obs_file = LAMAHICE_DIR / f"domain_{domain_id}" / "observations" / "streamflow" / "preprocessed" / f"{domain_id}_streamflow_processed.csv"

        if not obs_file.exists():
            continue

        try:
            df = pd.read_csv(obs_file, parse_dates=['datetime'], index_col='datetime')
            # Filter to period
            df = df[start_date:end_date]

            if len(df) == 0:
                continue

            # Convert discharge (m³/s) to specific runoff (mm/day)
            # Q (mm/day) = Q (m³/s) * 86400 / (area_km² * 1e6) * 1000
            df['runoff_mm'] = df['discharge_cms'] * 86400 / (area_km2 * 1e6) * 1000
            df['area_km2'] = area_km2
            df['domain_id'] = domain_id

            # Resample to daily
            daily = df.resample('D').mean()
            all_obs.append(daily)
            total_area += area_km2

        except Exception as e:
            print(f"    Warning: Could not load domain {domain_id}: {e}")
            continue

    if not all_obs:
        return None

    # Combine all observations
    combined = pd.concat(all_obs)

    # Compute area-weighted mean daily runoff
    daily_stats = combined.groupby(combined.index).agg({
        'runoff_mm': lambda x: np.nanmean(x),  # Simple mean for now
        'area_km2': 'sum',
        'domain_id': 'count'
    }).rename(columns={'domain_id': 'n_catchments'})

    print(f"  Total observed area: {total_area:.0f} km²")
    print(f"  Daily records: {len(daily_stats)}")

    return daily_stats


def compute_annual_stats(ds: "xr.Dataset") -> pd.DataFrame:
    """Compute annual water balance statistics."""
    years = np.unique(ds.time.dt.year.values)
    records = []

    for year in years:
        year_mask = ds.time.dt.year == year
        year_data = ds.sel(time=year_mask)

        ppt_sum = float(year_data['ppt'].sum(dim='time').mean().values)
        q_sum = float(year_data['q_routed'].sum(dim='time').mean().values)
        pet_sum = float(year_data['pet'].sum(dim='time').mean().values) if 'pet' in ds else np.nan
        temp_mean = float(year_data['temp'].mean().values) if 'temp' in ds else np.nan

        records.append({
            'year': year,
            'precipitation_mm': ppt_sum,
            'runoff_mm': q_sum,
            'pet_mm': pet_sum,
            'temp_mean_C': temp_mean,
            'runoff_ratio': q_sum / ppt_sum if ppt_sum > 0 else np.nan,
        })

    return pd.DataFrame(records)


def plot_baseline_results(ds: "xr.Dataset", figures_dir: Path):
    """Generate the main FUSE baseline results figure."""

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1],
                           wspace=0.25, hspace=0.35)

    # ── Panel (a): Domain-averaged runoff time series ──
    ax_q = fig.add_subplot(gs[0, 0])
    q_mean = ds['q_routed'].mean(dim=['latitude', 'longitude'])
    ax_q.plot(ds.time, q_mean, 'b-', linewidth=0.5)
    ax_q.set_ylabel('Runoff (mm day$^{-1}$)')
    ax_q.set_xlabel('Date')
    ax_q.grid(True, linewidth=0.3, alpha=0.4)
    ax_q.text(0.02, 0.95, '(a)', transform=ax_q.transAxes,
              fontsize=10, fontweight='bold', va='top')
    ax_q.set_title('Domain-averaged routed runoff')

    # ── Panel (b): Precipitation time series ──
    ax_p = fig.add_subplot(gs[0, 1])
    ppt_mean = ds['ppt'].mean(dim=['latitude', 'longitude'])
    ax_p.plot(ds.time, ppt_mean, 'g-', linewidth=0.5)
    ax_p.set_ylabel('Precipitation (mm day$^{-1}$)')
    ax_p.set_xlabel('Date')
    ax_p.grid(True, linewidth=0.3, alpha=0.4)
    ax_p.text(0.02, 0.95, '(b)', transform=ax_p.transAxes,
              fontsize=10, fontweight='bold', va='top')
    ax_p.set_title('Domain-averaged precipitation')

    # ── Panel (c): Spatial distribution of mean runoff ──
    ax_hist = fig.add_subplot(gs[1, 0])
    q_spatial = ds['q_routed'].mean(dim='time').values.flatten()
    q_spatial = q_spatial[~np.isnan(q_spatial)]
    ax_hist.hist(q_spatial, bins=50, color='#4a90d9', edgecolor='#333333',
                 linewidth=0.4, alpha=0.85)
    median_q = np.median(q_spatial)
    ax_hist.axvline(median_q, color='#b03a2e', linewidth=1.0, linestyle='--')
    ax_hist.text(0.95, 0.85, f'median\n{median_q:.2f} mm d$^{{-1}}$',
                 transform=ax_hist.transAxes, fontsize=8, color='#b03a2e',
                 ha='right', va='top')
    ax_hist.set_xlabel('Mean runoff (mm day$^{-1}$)')
    ax_hist.set_ylabel('Number of subcatchments')
    ax_hist.grid(True, linewidth=0.3, alpha=0.4, axis='y')
    ax_hist.text(0.02, 0.95, '(c)', transform=ax_hist.transAxes,
                 fontsize=10, fontweight='bold', va='top')
    ax_hist.set_title('Spatial distribution of mean runoff')

    # ── Panel (d): Monthly P-Q relationship ──
    ax_pq = fig.add_subplot(gs[1, 1])
    q_monthly = ds['q_routed'].resample(time='ME').mean().mean(dim=['latitude', 'longitude'])
    ppt_monthly = ds['ppt'].resample(time='ME').mean().mean(dim=['latitude', 'longitude'])
    ax_pq.scatter(ppt_monthly, q_monthly, alpha=0.7, s=40, edgecolors='black',
                  linewidths=0.5, c='steelblue')
    ax_pq.set_xlabel('Monthly mean precipitation (mm day$^{-1}$)')
    ax_pq.set_ylabel('Monthly mean runoff (mm day$^{-1}$)')
    ax_pq.grid(True, linewidth=0.3, alpha=0.4)
    ax_pq.text(0.02, 0.95, '(d)', transform=ax_pq.transAxes,
               fontsize=10, fontweight='bold', va='top')
    ax_pq.set_title('Monthly P-Q relationship')

    # ── Panel (e): Temperature time series ──
    ax_t = fig.add_subplot(gs[2, 0])
    if 'temp' in ds:
        temp_mean = ds['temp'].mean(dim=['latitude', 'longitude'])
        ax_t.plot(ds.time, temp_mean, 'r-', linewidth=0.5)
        ax_t.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_t.set_ylabel('Temperature (\u00b0C)')
    ax_t.set_xlabel('Date')
    ax_t.grid(True, linewidth=0.3, alpha=0.4)
    ax_t.text(0.02, 0.95, '(e)', transform=ax_t.transAxes,
              fontsize=10, fontweight='bold', va='top')
    ax_t.set_title('Domain-averaged temperature')

    # ── Panel (f): Annual water balance ──
    ax_wb = fig.add_subplot(gs[2, 1])
    annual_stats = compute_annual_stats(ds)
    years = annual_stats['year'].values
    x = np.arange(len(years))
    width = 0.25

    ax_wb.bar(x - width, annual_stats['precipitation_mm'], width,
              label='Precipitation', color='#4a90d9', edgecolor='#333', linewidth=0.4)
    ax_wb.bar(x, annual_stats['runoff_mm'], width,
              label='Runoff', color='#6ab04c', edgecolor='#333', linewidth=0.4)
    if not annual_stats['pet_mm'].isna().all():
        ax_wb.bar(x + width, annual_stats['pet_mm'], width,
                  label='PET', color='#f39c12', edgecolor='#333', linewidth=0.4)

    ax_wb.set_xticks(x)
    ax_wb.set_xticklabels([str(int(y)) for y in years])
    ax_wb.set_ylabel('Annual total (mm)')
    ax_wb.set_xlabel('Year')
    ax_wb.legend(loc='upper right', framealpha=0.9)
    ax_wb.grid(True, linewidth=0.3, alpha=0.4, axis='y')
    ax_wb.text(0.02, 0.95, '(f)', transform=ax_wb.transAxes,
               fontsize=10, fontweight='bold', va='top')
    ax_wb.set_title('Annual water balance components')

    # Add summary text box
    n_subcatch = ds.dims['longitude']
    total_ppt = annual_stats['precipitation_mm'].sum()
    total_q = annual_stats['runoff_mm'].sum()
    mean_rr = total_q / total_ppt if total_ppt > 0 else 0

    summary = (f"n = {n_subcatch} subcatchments\n"
               f"Period: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}\n"
               f"Mean runoff ratio: {mean_rr:.2f}")

    fig.text(0.98, 0.02, summary, fontsize=8, ha='right', va='bottom',
             family='monospace', bbox=dict(boxstyle='round,pad=0.4',
             facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # Save
    for ext in ('png', 'pdf'):
        out = figures_dir / f'fig_fuse_baseline_results.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)


def compute_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Compute Kling-Gupta Efficiency."""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    if mask.sum() < 10:
        return np.nan
    sim_v, obs_v = sim[mask], obs[mask]
    r = np.corrcoef(sim_v, obs_v)[0, 1]
    alpha = np.std(sim_v) / np.std(obs_v) if np.std(obs_v) > 0 else np.nan
    beta = np.mean(sim_v) / np.mean(obs_v) if np.mean(obs_v) > 0 else np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def plot_hru_level_comparison(analysis_dir: Path, figures_dir: Path):
    """Create improved HRU-level comparison figure using matched HRU-catchment pairs.

    This approach is more appropriate than domain-average comparison because it
    compares simulated runoff at specific HRU locations with observed discharge
    from the corresponding gauged catchments.
    """

    # Load HRU-observation comparison data
    hru_obs_file = analysis_dir / "hru_obs_comparison_data.csv"
    if not hru_obs_file.exists():
        print(f"  Skipping HRU-level comparison (no data: {hru_obs_file})")
        return None

    print(f"  Loading: {hru_obs_file.name}")
    df = pd.read_csv(hru_obs_file, parse_dates=['date'])

    # Load matches for area information
    matches_file = analysis_dir / "lamahice_hru_matches.csv"
    if matches_file.exists():
        matches = pd.read_csv(matches_file)
        df = df.merge(
            matches[['lamahice_id', 'lamahice_area_km2', 'hru_area_km2']],
            left_on='domain_id', right_on='lamahice_id', how='left'
        )

    # Compute per-catchment metrics
    per_catchment = []
    for domain_id, group in df.groupby('domain_id'):
        sim = group['sim_mm'].values
        obs = group['obs_mm'].values
        mask = ~(np.isnan(sim) | np.isnan(obs))

        if mask.sum() < 30:
            continue

        kge = compute_kge(sim, obs)
        r = np.corrcoef(sim[mask], obs[mask])[0, 1] if mask.sum() > 1 else np.nan
        pbias = 100 * np.sum(sim[mask] - obs[mask]) / np.sum(obs[mask]) if np.sum(obs[mask]) > 0 else np.nan

        per_catchment.append({
            'domain_id': domain_id,
            'kge': kge,
            'r': r,
            'pbias': pbias,
            'n': mask.sum(),
            'mean_sim': np.mean(sim[mask]),
            'mean_obs': np.mean(obs[mask]),
        })

    metrics_df = pd.DataFrame(per_catchment)

    # Save per-catchment KGE to CSV
    kge_out = analysis_dir / "per_catchment_kge.csv"
    metrics_df[['domain_id', 'kge', 'n']].to_csv(kge_out, index=False)
    print(f"  Saved: {kge_out}")

    # Create 4-panel figure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # ── Panel (a): KGE distribution ──
    ax = fig.add_subplot(gs[0, 0])
    kge_valid = metrics_df['kge'].dropna()
    ax.hist(kge_valid, bins=25, color='#4a90d9', edgecolor='#333333',
            linewidth=0.4, alpha=0.85)
    ax.axvline(kge_valid.median(), color='#b03a2e', linewidth=1.5, linestyle='--',
               label=f'Median: {kge_valid.median():.2f}')
    ax.axvline(0, color='k', linewidth=1.0, linestyle='-', alpha=0.5,
               label='KGE = 0')
    ax.set_xlabel('KGE')
    ax.set_ylabel('Number of catchments')
    ax.set_title('(a) Distribution of per-catchment KGE')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis='y')
    ax.text(0.98, 0.98, f'n = {len(kge_valid)} catchments\n'
                        f'KGE > 0: {(kge_valid > 0).sum()} ({100*(kge_valid > 0).mean():.0f}%)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ── Panel (b): Percent bias distribution ──
    ax = fig.add_subplot(gs[0, 1])
    pbias_valid = metrics_df['pbias'].dropna()
    pbias_clipped = pbias_valid.clip(-200, 50)  # Clip for visualization
    ax.hist(pbias_clipped, bins=25, color='#6ab04c', edgecolor='#333333',
            linewidth=0.4, alpha=0.85)
    ax.axvline(pbias_valid.median(), color='#b03a2e', linewidth=1.5, linestyle='--',
               label=f'Median: {pbias_valid.median():.0f}%')
    ax.axvline(0, color='k', linewidth=1.0, linestyle='-', alpha=0.5)
    ax.set_xlabel('Percent Bias (%)')
    ax.set_ylabel('Number of catchments')
    ax.set_title('(b) Distribution of percent bias')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis='y')
    ax.text(0.98, 0.98, f'Underestimation\n(PBIAS < 0): {(pbias_valid < 0).sum()} catchments',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ── Panel (c): Aggregated time series (calibration + evaluation) ──
    ax = fig.add_subplot(gs[1, :])

    # Compute daily mean across all catchments
    daily_agg = df.groupby('date').agg({
        'sim_mm': 'mean',
        'obs_mm': 'mean'
    }).dropna()

    ax.plot(daily_agg.index, daily_agg['sim_mm'], 'b-', linewidth=0.6,
            label='Simulated (HRU mean)', alpha=0.8)
    ax.plot(daily_agg.index, daily_agg['obs_mm'], 'r-', linewidth=0.6,
            label='Observed (catchment mean)', alpha=0.8)

    # Add period annotations
    ax.axvline(pd.Timestamp('2009-01-01'), color='gray', linestyle=':', alpha=0.7)
    ax.axvline(pd.Timestamp('2010-01-01'), color='gray', linestyle=':', alpha=0.7)
    ax.text(pd.Timestamp('2008-07-01'), ax.get_ylim()[1]*0.95, 'Spinup',
            ha='center', fontsize=8, color='gray')
    ax.text(pd.Timestamp('2009-07-01'), ax.get_ylim()[1]*0.95, 'Calibration',
            ha='center', fontsize=8, color='gray')
    ax.text(pd.Timestamp('2010-07-01'), ax.get_ylim()[1]*0.95, 'Evaluation',
            ha='center', fontsize=8, color='gray')

    ax.set_ylabel('Runoff (mm day$^{-1}$)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title('(c) Aggregated daily runoff across 72 validation catchments')

    # Compute aggregate metrics
    sim_all = daily_agg['sim_mm'].values
    obs_all = daily_agg['obs_mm'].values
    r_agg = np.corrcoef(sim_all, obs_all)[0, 1]
    kge_agg = compute_kge(sim_all, obs_all)
    pbias_agg = 100 * np.sum(sim_all - obs_all) / np.sum(obs_all)

    ax.text(0.02, 0.02,
            f'Aggregate metrics: r = {r_agg:.2f}, KGE = {kge_agg:.2f}, PBIAS = {pbias_agg:.0f}%',
            transform=ax.transAxes, fontsize=8, va='bottom',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = figures_dir / f'fig_hru_obs_comparison.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)

    return metrics_df


def plot_obs_comparison(ds: "xr.Dataset", obs_daily: "pd.DataFrame | None",
                        figures_dir: Path):
    """Compare large domain simulated runoff with aggregated observations.

    Note: This comparison has methodological limitations due to scale mismatch
    between domain-averaged simulation and mean catchment observations. The
    HRU-level comparison (plot_hru_level_comparison) provides a more appropriate
    validation approach.
    """

    if obs_daily is None:
        print("  Skipping observation comparison (no data)")
        return

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # Get simulated daily runoff (domain average)
    sim_daily = ds['q_routed'].mean(dim=['latitude', 'longitude']).to_pandas()
    sim_daily.index = pd.to_datetime(sim_daily.index)

    # Align time periods
    common_idx = sim_daily.index.intersection(obs_daily.index)
    sim_aligned = sim_daily.loc[common_idx]
    obs_aligned = obs_daily.loc[common_idx, 'runoff_mm']

    # ── Panel (a): Time series comparison ──
    ax = fig.add_subplot(gs[0, :])
    ax.plot(sim_aligned.index, sim_aligned.values, 'b-', linewidth=0.6,
            label='Simulated (domain average)', alpha=0.8)
    ax.plot(obs_aligned.index, obs_aligned.values, 'r-', linewidth=0.6,
            label='Observed (72-catchment mean)', alpha=0.8)
    ax.set_ylabel('Runoff (mm day$^{-1}$)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_title('Daily runoff: Domain average vs Catchment mean (Note: scale mismatch)')

    # ── Panel (b): Scatter plot ──
    ax = fig.add_subplot(gs[1, 0])
    mask = ~(np.isnan(sim_aligned.values) | np.isnan(obs_aligned.values))
    sim_valid = sim_aligned.values[mask]
    obs_valid = obs_aligned.values[mask]

    ax.scatter(obs_valid, sim_valid, alpha=0.3, s=10, c='steelblue', edgecolors='none')

    # Add 1:1 line
    max_val = max(np.nanmax(sim_valid), np.nanmax(obs_valid))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')

    # Compute metrics
    r, kge, pbias, rmse = np.nan, np.nan, np.nan, np.nan
    if len(sim_valid) > 10:
        r = np.corrcoef(sim_valid, obs_valid)[0, 1]
        bias = np.mean(sim_valid - obs_valid)
        pbias = 100 * np.sum(sim_valid - obs_valid) / np.sum(obs_valid)
        rmse = np.sqrt(np.mean((sim_valid - obs_valid)**2))

        # KGE
        alpha = np.std(sim_valid) / np.std(obs_valid)
        beta = np.mean(sim_valid) / np.mean(obs_valid)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        metrics_text = f'r = {r:.2f}\nKGE = {kge:.2f}\nPBIAS = {pbias:.1f}%\nRMSE = {rmse:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Observed runoff (mm day$^{-1}$)')
    ax.set_ylabel('Simulated runoff (mm day$^{-1}$)')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.set_title('Daily scatter plot')

    # ── Panel (c): Monthly comparison ──
    ax = fig.add_subplot(gs[1, 1])

    # Resample to monthly
    sim_monthly = sim_aligned.resample('ME').mean()
    obs_monthly = obs_aligned.resample('ME').mean()

    x = np.arange(len(sim_monthly))
    width = 0.35

    ax.bar(x - width/2, obs_monthly.values, width, label='Observed',
           color='#e74c3c', edgecolor='#333', linewidth=0.4, alpha=0.8)
    ax.bar(x + width/2, sim_monthly.values, width, label='Simulated',
           color='#3498db', edgecolor='#333', linewidth=0.4, alpha=0.8)

    # X-axis labels (show every 6 months)
    tick_positions = x[::6]
    tick_labels = [sim_monthly.index[i].strftime('%Y-%m') for i in range(0, len(sim_monthly), 6)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_ylabel('Monthly mean runoff (mm day$^{-1}$)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis='y')
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_title('Monthly comparison')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = figures_dir / f'fig_sim_obs_comparison.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)

    return {'r': r, 'kge': kge, 'pbias': pbias, 'rmse': rmse} if len(sim_valid) > 10 else None


def plot_spatial_kge(figures_dir: Path, analysis_dir: Path):
    """Create a spatial map of per-catchment KGE values on the Iceland domain."""

    # Load per-catchment KGE data
    kge_file = analysis_dir / "per_catchment_kge.csv"
    if not kge_file.exists():
        print(f"  Skipping spatial KGE (no data: {kge_file})")
        return

    kge_df = pd.read_csv(kge_file)

    # Load HRU matches to get coordinates
    matches_file = analysis_dir / "lamahice_hru_matches.csv"
    if not matches_file.exists():
        print(f"  Skipping spatial KGE (no matches: {matches_file})")
        return

    matches_df = pd.read_csv(matches_file)

    # Merge KGE with coordinates
    merged = kge_df.merge(
        matches_df[['lamahice_id', 'lamahice_lat', 'lamahice_lon', 'lamahice_area_km2']],
        left_on='domain_id', right_on='lamahice_id', how='left'
    )
    merged = merged.dropna(subset=['lamahice_lat', 'lamahice_lon'])

    if len(merged) == 0:
        print("  No valid coordinates for spatial KGE plot")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): Spatial map of KGE ──
    ax = axes[0]

    # Load GRU shapefile for background if available
    gru_shp = DOMAIN_DIR / "shapefiles" / "catchment" / "catchment_Iceland_Multivar.shp"
    if HAS_GEOPANDAS and gru_shp.exists():
        try:
            grus = gpd.read_file(gru_shp)
            grus.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.1)
        except Exception as e:
            print(f"    Warning: Could not load GRU shapefile: {e}")

    # Plot KGE values
    scatter = ax.scatter(
        merged['lamahice_lon'], merged['lamahice_lat'],
        c=merged['kge'], cmap='RdYlGn', vmin=-1, vmax=0.5,
        s=60, edgecolors='black', linewidths=0.5, alpha=0.9, zorder=5
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('KGE', fontsize=9)

    # Mark positive KGE catchments
    positive_kge = merged[merged['kge'] > 0]
    if len(positive_kge) > 0:
        ax.scatter(
            positive_kge['lamahice_lon'], positive_kge['lamahice_lat'],
            facecolors='none', edgecolors='blue', s=120, linewidths=2,
            zorder=6, label=f'KGE > 0 (n={len(positive_kge)})'
        )
        ax.legend(loc='lower left', fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(a) Spatial distribution of KGE')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Add summary stats
    stats_text = (f"n = {len(merged)} catchments\n"
                  f"Median KGE = {merged['kge'].median():.2f}\n"
                  f"KGE > 0: {len(positive_kge)} ({100*len(positive_kge)/len(merged):.0f}%)")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ── Panel (b): KGE vs catchment area ──
    ax = axes[1]
    ax.scatter(merged['lamahice_area_km2'], merged['kge'],
               c=merged['kge'], cmap='RdYlGn', vmin=-1, vmax=0.5,
               s=50, edgecolors='black', linewidths=0.4, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Catchment area (km²)')
    ax.set_ylabel('KGE')
    ax.set_title('(b) KGE vs catchment area')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Add trend annotation
    small = merged[merged['lamahice_area_km2'] < 200]
    large = merged[merged['lamahice_area_km2'] >= 200]
    if len(small) > 5 and len(large) > 5:
        ax.text(0.98, 0.98,
                f"Small (<200 km²): median KGE = {small['kge'].median():.2f}\n"
                f"Large (≥200 km²): median KGE = {large['kge'].median():.2f}",
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = figures_dir / f'fig_spatial_kge.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_runoff_ratio_comparison(ds: "xr.Dataset", catchment_stats: "pd.DataFrame | None",
                                  figures_dir: Path):
    """Compare large domain runoff ratios with large sample catchments."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Large domain runoff ratio distribution
    ax = axes[0]
    q_annual = ds['q_routed'].resample(time='YE').sum().mean(dim='time')
    ppt_annual = ds['ppt'].resample(time='YE').sum().mean(dim='time')
    rr_spatial = (q_annual / ppt_annual).values.flatten()
    rr_spatial = rr_spatial[~np.isnan(rr_spatial) & (rr_spatial > 0) & (rr_spatial < 2)]

    ax.hist(rr_spatial, bins=40, color='#4a90d9', edgecolor='#333333',
            linewidth=0.4, alpha=0.85, density=True, label='Large domain GRUs')
    ax.axvline(np.median(rr_spatial), color='#b03a2e', linewidth=1.5, linestyle='--',
               label=f'Median: {np.median(rr_spatial):.2f}')
    ax.set_xlabel('Runoff ratio (Q/P)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Large domain runoff ratio distribution')
    ax.legend(loc='upper right')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # Comparison with large sample if available
    ax = axes[1]
    if catchment_stats is not None and len(catchment_stats) > 0:
        # Plot large sample catchment locations colored by some attribute
        ax.scatter(catchment_stats['centroid_lon'], catchment_stats['centroid_lat'],
                   c=catchment_stats['area_km2'], cmap='viridis', s=50,
                   edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('(b) Large sample catchment locations')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=catchment_stats['area_km2'].min(),
                                                      vmax=catchment_stats['area_km2'].max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Area (km\u00b2)', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Large sample data\nnot available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='#888888')
        ax.set_title('(b) Large sample comparison')

    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = figures_dir / f'fig_runoff_ratio_comparison.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FUSE baseline results for Section 4.9 (Iceland large domain)."
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory for figures")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR,
                        help="Output directory for CSV statistics")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Section 4.9 - FUSE Baseline Results Analysis")
    print("=" * 55)

    # Load FUSE output
    print("\nLoading FUSE output...")
    ds = load_fuse_output()

    if ds is None:
        print("\nError: No FUSE output available. Run the model first.")
        sys.exit(1)

    print(f"  Time range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
    print(f"  Subcatchments: {ds.dims['longitude']}")

    # Compute and save annual statistics
    print("\nComputing annual statistics...")
    annual_stats = compute_annual_stats(ds)
    stats_path = args.analysis_dir / "fuse_annual_stats.csv"
    annual_stats.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")
    print(annual_stats.to_string(index=False))

    # Load large sample catchment stats for comparison
    print("\nLoading large sample catchment stats...")
    catchment_stats = load_catchment_stats()

    # Load observations from large sample catchments
    obs_daily = None
    if catchment_stats is not None:
        print("\nLoading large sample observations...")
        obs_daily = load_large_sample_observations(catchment_stats)

    # Generate figures
    print("\nGenerating figures...")
    plot_baseline_results(ds, args.figures_dir)
    plot_runoff_ratio_comparison(ds, catchment_stats, args.figures_dir)
    plot_spatial_kge(args.figures_dir, args.analysis_dir)

    # Observation comparison (domain average - note methodological limitations)
    metrics = None
    if obs_daily is not None:
        print("\nGenerating domain-average observation comparison...")
        metrics = plot_obs_comparison(ds, obs_daily, args.figures_dir)

    # HRU-level comparison (more appropriate methodology)
    print("\nGenerating HRU-level observation comparison...")
    hru_metrics = plot_hru_level_comparison(args.analysis_dir, args.figures_dir)

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY - FUSE Baseline Results")
    print("=" * 55)
    total_ppt = annual_stats['precipitation_mm'].sum()
    total_q = annual_stats['runoff_mm'].sum()
    print(f"Total precipitation (3 years): {total_ppt:.0f} mm")
    print(f"Total runoff (3 years): {total_q:.0f} mm")
    print(f"Mean annual runoff ratio: {total_q/total_ppt:.3f}")

    if metrics is not None:
        print("\nDomain-Average Comparison (methodological caveats apply):")
        print(f"  Correlation (r): {metrics['r']:.3f}")
        print(f"  KGE: {metrics['kge']:.3f}")
        print(f"  Percent Bias: {metrics['pbias']:.1f}%")
        print(f"  RMSE: {metrics['rmse']:.2f} mm/day")

    if hru_metrics is not None and len(hru_metrics) > 0:
        print(f"\nHRU-Level Validation (n={len(hru_metrics)} catchments):")
        print(f"  Median KGE: {hru_metrics['kge'].median():.3f}")
        print(f"  KGE > 0: {(hru_metrics['kge'] > 0).sum()} catchments "
              f"({100*(hru_metrics['kge'] > 0).mean():.0f}%)")
        print(f"  Median Percent Bias: {hru_metrics['pbias'].median():.1f}%")

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
