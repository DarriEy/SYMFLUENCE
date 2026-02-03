#!/usr/bin/env python3
"""Generate validation figures from existing analysis CSV files.

This script creates:
  - fig_spatial_kge.{png,pdf}: Spatial map of per-catchment KGE values
  - fig_hru_obs_comparison.{png,pdf}: HRU-level validation summary

Can be run independently without FUSE NetCDF output files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ── Publication style ──
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


def plot_spatial_kge():
    """Create spatial map of per-catchment KGE values with Iceland background."""

    # Load per-catchment KGE data
    kge_file = ANALYSIS_DIR / "per_catchment_kge.csv"
    if not kge_file.exists():
        print(f"  Error: {kge_file} not found")
        return False

    kge_df = pd.read_csv(kge_file)
    print(f"  Loaded {len(kge_df)} catchment KGE values")

    # Load HRU matches to get coordinates
    matches_file = ANALYSIS_DIR / "lamahice_hru_matches.csv"
    if not matches_file.exists():
        print(f"  Error: {matches_file} not found")
        return False

    matches_df = pd.read_csv(matches_file)

    # Merge KGE with coordinates
    merged = kge_df.merge(
        matches_df[['lamahice_id', 'lamahice_lat', 'lamahice_lon', 'lamahice_area_km2']],
        left_on='domain_id', right_on='lamahice_id', how='left'
    )
    merged = merged.dropna(subset=['lamahice_lat', 'lamahice_lon'])

    if len(merged) == 0:
        print("  No valid coordinates for spatial KGE plot")
        return False

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): Spatial map of KGE ──
    ax = axes[0]

    # Load river basins shapefile for Iceland background
    basins_shp = DOMAIN_DIR / "shapefiles" / "river_basins" / "Iceland_Multivar_riverBasins_with_coastal.shp"
    map_loaded = False
    if HAS_GEOPANDAS and basins_shp.exists():
        try:
            basins = gpd.read_file(basins_shp)
            # Dissolve to get outline, or plot all basins with light fill
            basins.plot(ax=ax, color='#e8e8e8', edgecolor='#cccccc', linewidth=0.15)
            map_loaded = True
            print(f"  Loaded basins shapefile ({len(basins)} basins)")
        except Exception as e:
            print(f"  Warning: Could not load basins shapefile: {e}")

    if not map_loaded:
        # Fallback: try HRU shapefile
        hru_shp = DOMAIN_DIR / "shapefiles" / "catchment" / "semidistributed" / "large_domain" / "Iceland_Multivar_HRUs_GRUs.shp"
        if HAS_GEOPANDAS and hru_shp.exists():
            try:
                hrus = gpd.read_file(hru_shp)
                hrus.plot(ax=ax, color='#e8e8e8', edgecolor='#cccccc', linewidth=0.1)
                map_loaded = True
                print(f"  Loaded HRU shapefile ({len(hrus)} HRUs)")
            except Exception as e:
                print(f"  Warning: Could not load HRU shapefile: {e}")

    # Plot KGE values
    scatter = ax.scatter(
        merged['lamahice_lon'], merged['lamahice_lat'],
        c=merged['kge'], cmap='RdYlGn', vmin=-1, vmax=0.5,
        s=80, edgecolors='black', linewidths=0.6, alpha=0.95, zorder=5
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('KGE', fontsize=9)

    # Mark positive KGE catchments with circles
    positive_kge = merged[merged['kge'] > 0]
    if len(positive_kge) > 0:
        ax.scatter(
            positive_kge['lamahice_lon'], positive_kge['lamahice_lat'],
            facecolors='none', edgecolors='blue', s=150, linewidths=2.5,
            zorder=6, label=f'KGE > 0 (n={len(positive_kge)})'
        )
        ax.legend(loc='lower left', fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(a) Spatial distribution of per-catchment KGE')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # Set axis limits to Iceland bounds
    ax.set_xlim(-25.5, -12.5)
    ax.set_ylim(63.0, 66.6)
    ax.set_aspect('equal', adjustable='box')

    # Add summary stats box
    stats_text = (f"n = {len(merged)} catchments\n"
                  f"Median KGE = {merged['kge'].median():.2f}\n"
                  f"KGE > 0: {len(positive_kge)} ({100*len(positive_kge)/len(merged):.0f}%)")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999', alpha=0.95))

    # ── Panel (b): CDF of KGE values ──
    ax = axes[1]

    kge_sorted = np.sort(merged['kge'].dropna().values)
    cdf = np.arange(1, len(kge_sorted) + 1) / len(kge_sorted)

    ax.plot(kge_sorted, cdf, 'b-', linewidth=2, label='KGE CDF')
    ax.fill_between(kge_sorted, 0, cdf, alpha=0.2, color='steelblue')

    # Add reference lines
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='KGE = 0')
    ax.axvline(x=merged['kge'].median(), color='#b03a2e', linestyle='--', linewidth=1.5,
               label=f'Median = {merged["kge"].median():.2f}')

    # Mark key percentiles
    p25 = np.percentile(kge_sorted, 25)
    p75 = np.percentile(kge_sorted, 75)
    ax.axhline(y=0.25, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.75, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Shade the "acceptable" KGE region (> 0)
    ax.axvspan(0, 0.5, alpha=0.1, color='green', label='KGE > 0 region')

    ax.set_xlabel('KGE')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('(b) Cumulative distribution of KGE')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_xlim(-1.0, 0.6)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Add percentile annotations
    pct_positive = 100 * (kge_sorted > 0).sum() / len(kge_sorted)
    ax.text(0.98, 0.5, f'25th pctl: {p25:.2f}\n'
                       f'75th pctl: {p75:.2f}\n'
                       f'KGE > 0: {pct_positive:.0f}%',
            transform=ax.transAxes, fontsize=8, ha='right', va='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999', alpha=0.95))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = FIGURES_DIR / f'fig_spatial_kge.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)
    return True


def plot_hru_level_comparison():
    """Create HRU-level validation summary figure with diagnostic panels."""

    # Load HRU-observation comparison data
    hru_obs_file = ANALYSIS_DIR / "hru_obs_comparison_data.csv"
    if not hru_obs_file.exists():
        print(f"  Error: {hru_obs_file} not found")
        return False

    print(f"  Loading: {hru_obs_file.name}")
    df = pd.read_csv(hru_obs_file, parse_dates=['date'])
    print(f"  Loaded {len(df)} observation pairs")

    # Load area information
    matches_file = ANALYSIS_DIR / "lamahice_hru_matches.csv"
    if matches_file.exists():
        matches = pd.read_csv(matches_file)
        df = df.merge(matches[['lamahice_id', 'lamahice_area_km2', 'hru_area_km2']],
                      left_on='domain_id', right_on='lamahice_id', how='left')

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
    print(f"  Computed metrics for {len(metrics_df)} catchments")

    # Create 2x2 figure with diagnostic panels
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.35)

    # ── Panel (a): Time series comparison ──
    ax = fig.add_subplot(gs[0, :])

    # Compute daily mean across all catchments
    daily_agg = df.groupby('date').agg({
        'sim_mm': 'mean',
        'obs_mm': 'mean'
    }).dropna()

    ax.plot(daily_agg.index, daily_agg['sim_mm'], 'b-', linewidth=0.7,
            label='Simulated (HRU local runoff)', alpha=0.85)
    ax.plot(daily_agg.index, daily_agg['obs_mm'], 'r-', linewidth=0.7,
            label='Observed (catchment discharge)', alpha=0.85)

    # Add period annotations
    ymax = max(daily_agg['sim_mm'].max(), daily_agg['obs_mm'].max())
    ax.axvline(pd.Timestamp('2009-01-01'), color='gray', linestyle=':', alpha=0.7)
    ax.axvline(pd.Timestamp('2010-01-01'), color='gray', linestyle=':', alpha=0.7)
    ax.text(pd.Timestamp('2008-07-01'), ymax*0.92, 'Spinup', ha='center', fontsize=8, color='gray')
    ax.text(pd.Timestamp('2009-07-01'), ymax*0.92, 'Calibration', ha='center', fontsize=8, color='gray')
    ax.text(pd.Timestamp('2010-07-01'), ymax*0.92, 'Evaluation', ha='center', fontsize=8, color='gray')

    ax.set_ylabel('Runoff (mm day$^{-1}$)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title('(a) Daily runoff: HRU local generation vs catchment outlet discharge')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    # Compute aggregate metrics
    sim_all = daily_agg['sim_mm'].values
    obs_all = daily_agg['obs_mm'].values
    r_agg = np.corrcoef(sim_all, obs_all)[0, 1]
    kge_agg = compute_kge(sim_all, obs_all)
    pbias_agg = 100 * np.sum(sim_all - obs_all) / np.sum(obs_all)

    ax.text(0.02, 0.02,
            f'Aggregate: r = {r_agg:.2f}, KGE = {kge_agg:.2f}, PBIAS = {pbias_agg:.0f}%',
            transform=ax.transAxes, fontsize=8, va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ── Panel (b): Seasonal sim/obs ratio - shows glacier signal ──
    ax = fig.add_subplot(gs[1, 0])

    df['month'] = df['date'].dt.month
    seasonal = df.groupby('month').agg({'sim_mm': 'mean', 'obs_mm': 'mean'})
    seasonal['ratio'] = seasonal['sim_mm'] / seasonal['obs_mm']

    months = np.arange(1, 13)
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    # Bar chart of sim/obs ratio by month
    colors = ['#3498db' if m in [12, 1, 2, 3, 4, 5] else '#e74c3c' for m in months]
    bars = ax.bar(months, seasonal.loc[months, 'ratio'], color=colors,
                  edgecolor='#333', linewidth=0.5, alpha=0.85)

    ax.axhline(y=seasonal['ratio'].mean(), color='k', linestyle='--', linewidth=1,
               label=f'Annual mean: {seasonal["ratio"].mean():.2f}')

    # Add annotations
    summer_ratio = seasonal.loc[[6, 7, 8], 'ratio'].mean()
    winter_ratio = seasonal.loc[[12, 1, 2], 'ratio'].mean()

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month')
    ax.set_ylabel('Sim / Obs ratio')
    ax.set_title('(b) Seasonal pattern of underestimation')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    ax.set_ylim(0, 0.3)
    ax.grid(True, linewidth=0.3, alpha=0.4, axis='y')

    # Add legend/annotation explaining the pattern
    ax.text(0.98, 0.98,
            f'Summer (JJA): {summer_ratio:.2f}\n'
            f'Winter (DJF): {winter_ratio:.2f}\n\n'
            'Summer underestimation\n'
            '3× worse → missing\n'
            'glacier melt signal',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#856404', alpha=0.95))

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', edgecolor='#333', label='Cold season'),
                       Patch(facecolor='#e74c3c', edgecolor='#333', label='Warm season (glacier melt)')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    # ── Panel (c): Scale mismatch explanation ──
    ax = fig.add_subplot(gs[1, 1])

    if 'lamahice_area_km2' in df.columns:
        # Get unique catchment info
        catch_info = df.groupby('domain_id').agg({
            'lamahice_area_km2': 'first',
            'hru_area_km2': 'first',
            'sim_mm': 'mean',
            'obs_mm': 'mean'
        }).dropna()
        catch_info['ratio'] = catch_info['sim_mm'] / catch_info['obs_mm']

        # Scatter: catchment area vs sim/obs ratio
        scatter = ax.scatter(catch_info['lamahice_area_km2'], catch_info['ratio'],
                            c=catch_info['hru_area_km2'], cmap='viridis',
                            s=50, edgecolors='black', linewidths=0.4, alpha=0.8)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('HRU area (km²)', fontsize=8)

        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect match')
        ax.set_xscale('log')
        ax.set_xlabel('LamaH-Ice catchment area (km²)')
        ax.set_ylabel('Sim / Obs ratio')
        ax.set_title('(c) Scale mismatch: HRU vs catchment area')
        ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_ylim(0, 1.5)

        # Annotation explaining the issue
        mean_catch = catch_info['lamahice_area_km2'].mean()
        mean_hru = catch_info['hru_area_km2'].mean()
        ax.text(0.98, 0.02,
                f'Mean catchment: {mean_catch:.0f} km²\n'
                f'Mean HRU: {mean_hru:.0f} km²\n'
                f'Ratio: {mean_catch/mean_hru:.0f}×\n\n'
                'HRU runoff ≠\n'
                'Catchment discharge',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8d7da', edgecolor='#721c24', alpha=0.95))
    else:
        ax.text(0.5, 0.5, 'Area data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = FIGURES_DIR / f'fig_hru_obs_comparison.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    plt.close(fig)
    return True


def main():
    print("Generating Validation Figures from CSV Data")
    print("=" * 50)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Spatial KGE Figure...")
    if not plot_spatial_kge():
        print("   Failed to generate spatial KGE figure")

    print("\n2. HRU-Level Comparison Figure...")
    if not plot_hru_level_comparison():
        print("   Failed to generate HRU comparison figure")

    print("\nDone.")


if __name__ == "__main__":
    main()
