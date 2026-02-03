#!/usr/bin/env python3
"""
Comprehensive cross-catchment analysis for the Large Sample study.

Produces publication-ready figures:
- Figure A: Performance CDFs (KGE, NSE calibration and validation)
- Figure B: Spatial map of catchment performance
- Figure C: Parameter distributions across catchments
- Figure D: Performance vs catchment attributes

Handles partial results gracefully - run anytime during the campaign.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")

# Parameter names and display labels
PARAM_NAMES = [
    "MAXWATR_1", "MAXWATR_2", "BASERTE", "QB_POWR", "TIMEDELAY",
    "PERCRTE", "FRACTEN", "RTFRAC1", "MBASE", "MFMAX", "MFMIN", "PXTEMP", "LAPSE"
]

PARAM_LABELS = {
    "MAXWATR_1": "Max Water 1\n(mm)",
    "MAXWATR_2": "Max Water 2\n(mm)",
    "BASERTE": "Baseflow Rate\n(day⁻¹)",
    "QB_POWR": "Baseflow Power",
    "TIMEDELAY": "Time Delay\n(days)",
    "PERCRTE": "Percolation Rate",
    "FRACTEN": "Tension Fraction",
    "RTFRAC1": "Routing Frac 1",
    "MBASE": "Melt Base\n(°C)",
    "MFMAX": "Max Melt Factor\n(mm/°C/day)",
    "MFMIN": "Min Melt Factor\n(mm/°C/day)",
    "PXTEMP": "Rain/Snow Temp\n(°C)",
    "LAPSE": "Lapse Rate\n(°C/km)"
}

PARAM_BOUNDS = {
    "MAXWATR_1": (50, 500),
    "MAXWATR_2": (50, 1000),
    "BASERTE": (0.001, 0.5),
    "QB_POWR": (1.0, 5.0),
    "TIMEDELAY": (0.5, 5.0),
    "PERCRTE": (0.01, 10.0),
    "FRACTEN": (0.1, 0.7),
    "RTFRAC1": (0.1, 0.9),
    "MBASE": (-2.0, 2.0),
    "MFMAX": (1.0, 8.0),
    "MFMIN": (0.5, 4.0),
    "PXTEMP": (-2.0, 4.0),
    "LAPSE": (-8.0, -4.0)
}

# Plot styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_domain_ids() -> list:
    """Discover all configured domain IDs from config files."""
    ids = []
    for f in CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"):
        parts = f.stem.replace("config_lamahice_", "").replace("_FUSE", "")
        try:
            ids.append(int(parts))
        except ValueError:
            continue
    return sorted(ids)


def load_catchment_stats() -> pd.DataFrame:
    """Load catchment attributes from stats file."""
    stats_file = ANALYSIS_DIR / "catchment_stats.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        df = df.rename(columns={"domain_id": "domain_id"})
        return df
    return pd.DataFrame()


def load_benchmark_scores(domain_id: int) -> Optional[dict]:
    """Load evaluation benchmark scores for a domain."""
    eval_dir = SYMFLUENCE_DATA / f"domain_{domain_id}" / "evaluation"
    scores_file = eval_dir / "benchmark_scores.csv"

    if not scores_file.exists():
        return None

    try:
        df = pd.read_csv(scores_file)
        # Extract daily_mean_flow metrics (row index 6)
        daily_row = df[df['benchmarks'] == 'daily_mean_flow']
        if daily_row.empty:
            return None

        row = daily_row.iloc[0]
        return {
            'domain_id': domain_id,
            'nse_cal': row.get('nse_cal', np.nan),
            'nse_val': row.get('nse_val', np.nan),
            'kge_cal': row.get('kge_cal', np.nan),
            'kge_val': row.get('kge_val', np.nan),
        }
    except Exception as e:
        print(f"  Warning: Could not parse scores for domain {domain_id}: {e}")
        return None


def load_best_params(domain_id: int) -> Optional[dict]:
    """Load best parameters from optimization."""
    opt_dir = SYMFLUENCE_DATA / f"domain_{domain_id}" / "optimization" / "FUSE" / "dds_run_1"
    params_file = opt_dir / "run_1_dds_best_params.json"

    if not params_file.exists():
        return None

    try:
        with open(params_file) as f:
            data = json.load(f)

        result = {
            'domain_id': domain_id,
            'best_score': data.get('best_score', np.nan),
            'best_iteration': data.get('best_iteration', np.nan),
        }

        # Add individual parameters
        best_params = data.get('best_params', {})
        for param in PARAM_NAMES:
            result[param] = best_params.get(param, np.nan)

        return result
    except Exception as e:
        print(f"  Warning: Could not parse params for domain {domain_id}: {e}")
        return None


def load_iteration_history(domain_id: int) -> Optional[pd.DataFrame]:
    """Load optimization iteration history."""
    opt_dir = SYMFLUENCE_DATA / f"domain_{domain_id}" / "optimization" / "FUSE" / "dds_run_1"
    iter_file = opt_dir / "run_1_parallel_iteration_results.csv"

    if not iter_file.exists():
        return None

    try:
        df = pd.read_csv(iter_file)
        df['domain_id'] = domain_id
        return df
    except:
        return None


def load_all_results(domain_ids: list) -> tuple:
    """Load all available results."""
    print("Loading results...")

    scores_list = []
    params_list = []

    completed = 0
    failed = 0
    pending = 0

    for did in domain_ids:
        scores = load_benchmark_scores(did)
        params = load_best_params(did)

        if scores is not None and params is not None:
            # Check if optimization actually succeeded (not all crashes)
            if params['best_score'] > -9000:  # Not penalty score
                scores_list.append(scores)
                params_list.append(params)
                completed += 1
            else:
                failed += 1
        elif scores is not None or params is not None:
            failed += 1
        else:
            pending += 1

    print(f"  Completed: {completed}, Failed: {failed}, Pending: {pending}")

    scores_df = pd.DataFrame(scores_list) if scores_list else pd.DataFrame()
    params_df = pd.DataFrame(params_list) if params_list else pd.DataFrame()

    return scores_df, params_df, {'completed': completed, 'failed': failed, 'pending': pending}


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_performance_cdfs(scores_df: pd.DataFrame, stats: dict, output_dir: Path):
    """
    Figure A: CDFs of KGE and NSE for calibration and validation periods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    metrics = [
        ('kge_cal', 'kge_val', 'KGE', axes[0]),
        ('nse_cal', 'nse_val', 'NSE', axes[1]),
    ]

    colors = {'cal': '#2166ac', 'val': '#b2182b'}

    for cal_col, val_col, label, ax in metrics:
        # Calibration CDF
        if cal_col in scores_df.columns:
            cal_data = scores_df[cal_col].dropna().sort_values()
            if len(cal_data) > 0:
                cal_cdf = np.arange(1, len(cal_data) + 1) / len(cal_data)
                ax.plot(cal_data, cal_cdf, '-', color=colors['cal'],
                       linewidth=2, label=f'Calibration (n={len(cal_data)})')
                ax.axvline(cal_data.median(), color=colors['cal'], linestyle='--',
                          alpha=0.7, linewidth=1)

        # Validation CDF
        if val_col in scores_df.columns:
            val_data = scores_df[val_col].dropna().sort_values()
            if len(val_data) > 0:
                val_cdf = np.arange(1, len(val_data) + 1) / len(val_data)
                ax.plot(val_data, val_cdf, '-', color=colors['val'],
                       linewidth=2, label=f'Validation (n={len(val_data)})')
                ax.axvline(val_data.median(), color=colors['val'], linestyle='--',
                          alpha=0.7, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel('Cumulative Probability')
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0, 1)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Add status annotation
    status_text = f"Completed: {stats['completed']}/111 | Pending: {stats['pending']} | Failed: {stats['failed']}"
    fig.suptitle(f'Large Sample Performance Distribution\n({status_text})', fontsize=12)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_performance_cdfs.{ext}')
    plt.close(fig)
    print("  Saved: fig_performance_cdfs.png/pdf")


def plot_performance_map(scores_df: pd.DataFrame, catchment_stats: pd.DataFrame,
                         stats: dict, output_dir: Path):
    """
    Figure B: Spatial map of catchment KGE performance.
    """
    if catchment_stats.empty or scores_df.empty:
        print("  Skipping map: insufficient data")
        return

    # Merge scores with catchment locations
    merged = catchment_stats.merge(scores_df, on='domain_id', how='left')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration map
    ax = axes[0]
    has_cal = merged['kge_cal'].notna()

    # Plot pending/failed as gray
    if (~has_cal).any():
        ax.scatter(merged.loc[~has_cal, 'centroid_lon'],
                  merged.loc[~has_cal, 'centroid_lat'],
                  c='lightgray', s=40, alpha=0.5, edgecolors='gray',
                  linewidths=0.5, label='Pending/Failed')

    # Plot completed with color
    if has_cal.any():
        sc = ax.scatter(merged.loc[has_cal, 'centroid_lon'],
                       merged.loc[has_cal, 'centroid_lat'],
                       c=merged.loc[has_cal, 'kge_cal'],
                       cmap='RdYlBu', vmin=0, vmax=1,
                       s=60, edgecolors='black', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='KGE', shrink=0.8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Calibration KGE')
    ax.set_aspect('equal')

    # Validation map
    ax = axes[1]
    has_val = merged['kge_val'].notna()

    if (~has_val).any():
        ax.scatter(merged.loc[~has_val, 'centroid_lon'],
                  merged.loc[~has_val, 'centroid_lat'],
                  c='lightgray', s=40, alpha=0.5, edgecolors='gray',
                  linewidths=0.5, label='Pending/Failed')

    if has_val.any():
        sc = ax.scatter(merged.loc[has_val, 'centroid_lon'],
                       merged.loc[has_val, 'centroid_lat'],
                       c=merged.loc[has_val, 'kge_val'],
                       cmap='RdYlBu', vmin=0, vmax=1,
                       s=60, edgecolors='black', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='KGE', shrink=0.8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Validation KGE')
    ax.set_aspect('equal')

    status_text = f"Completed: {stats['completed']}/111"
    fig.suptitle(f'Spatial Distribution of Model Performance\n({status_text})', fontsize=12)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_performance_map.{ext}')
    plt.close(fig)
    print("  Saved: fig_performance_map.png/pdf")


def plot_parameter_distributions(params_df: pd.DataFrame, stats: dict, output_dir: Path):
    """
    Figure C: Distribution of calibrated parameters across catchments.
    """
    if params_df.empty:
        print("  Skipping parameter plot: no data")
        return

    # Select subset of most interpretable parameters
    key_params = ['MAXWATR_1', 'MAXWATR_2', 'BASERTE', 'MFMAX', 'MBASE', 'LAPSE']

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()

    for i, param in enumerate(key_params):
        ax = axes[i]

        if param in params_df.columns:
            data = params_df[param].dropna()

            if len(data) > 0:
                # Histogram
                ax.hist(data, bins=15, color='steelblue', edgecolor='white',
                       alpha=0.7, density=True)

                # Add parameter bounds as vertical lines
                bounds = PARAM_BOUNDS.get(param, (None, None))
                if bounds[0] is not None:
                    ax.axvline(bounds[0], color='red', linestyle='--',
                              alpha=0.7, label='Bounds')
                    ax.axvline(bounds[1], color='red', linestyle='--', alpha=0.7)

                # Add median
                ax.axvline(data.median(), color='black', linestyle='-',
                          linewidth=2, label=f'Median: {data.median():.2f}')

                ax.set_xlabel(PARAM_LABELS.get(param, param))
                ax.set_ylabel('Density')

                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel(PARAM_LABELS.get(param, param))

    status_text = f"n = {stats['completed']} catchments"
    fig.suptitle(f'Calibrated Parameter Distributions\n({status_text})', fontsize=12)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_parameter_distributions.{ext}')
    plt.close(fig)
    print("  Saved: fig_parameter_distributions.png/pdf")


def plot_performance_vs_attributes(scores_df: pd.DataFrame, params_df: pd.DataFrame,
                                   catchment_stats: pd.DataFrame, stats: dict,
                                   output_dir: Path):
    """
    Figure D: KGE vs catchment attributes (area, elevation, glacier fraction).
    """
    if scores_df.empty or catchment_stats.empty:
        print("  Skipping attribute plot: insufficient data")
        return

    # Merge all data
    merged = catchment_stats.merge(scores_df, on='domain_id', how='inner')

    if len(merged) < 3:
        print("  Skipping attribute plot: too few data points")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    attributes = [
        ('area_km2', 'Catchment Area (km²)', True),   # log scale
        ('elev_mean_m', 'Mean Elevation (m)', False),
        ('glac_fra', 'Glacier Fraction', False),
    ]

    for ax, (attr, label, log_scale) in zip(axes, attributes):
        if attr not in merged.columns:
            ax.text(0.5, 0.5, f'{attr} not available', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        x = merged[attr]
        y_cal = merged['kge_cal'] if 'kge_cal' in merged.columns else None
        y_val = merged['kge_val'] if 'kge_val' in merged.columns else None

        if y_cal is not None:
            mask = y_cal.notna() & x.notna()
            ax.scatter(x[mask], y_cal[mask], c='#2166ac', s=50, alpha=0.7,
                      edgecolors='white', linewidths=0.5, label='Calibration')

        if y_val is not None:
            mask = y_val.notna() & x.notna()
            ax.scatter(x[mask], y_val[mask], c='#b2182b', s=50, alpha=0.7,
                      edgecolors='white', linewidths=0.5, label='Validation')

        ax.set_xlabel(label)
        ax.set_ylabel('KGE')
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        if log_scale and x.min() > 0:
            ax.set_xscale('log')

        if ax == axes[0]:
            ax.legend(loc='lower left')

    status_text = f"n = {len(merged)} catchments"
    fig.suptitle(f'Model Performance vs Catchment Attributes\n({status_text})', fontsize=12)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_performance_vs_attributes.{ext}')
    plt.close(fig)
    print("  Saved: fig_performance_vs_attributes.png/pdf")


def plot_summary_table(scores_df: pd.DataFrame, params_df: pd.DataFrame,
                       stats: dict, output_dir: Path):
    """
    Generate a summary statistics table as a figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    # Compute statistics
    rows = []

    metrics = [
        ('KGE (cal)', 'kge_cal', scores_df),
        ('KGE (val)', 'kge_val', scores_df),
        ('NSE (cal)', 'nse_cal', scores_df),
        ('NSE (val)', 'nse_val', scores_df),
    ]

    for name, col, df in metrics:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                rows.append([
                    name,
                    f"{len(data)}",
                    f"{data.mean():.3f}",
                    f"{data.median():.3f}",
                    f"{data.std():.3f}",
                    f"{data.min():.3f}",
                    f"{data.max():.3f}",
                ])

    if rows:
        col_labels = ['Metric', 'n', 'Mean', 'Median', 'Std', 'Min', 'Max']
        table = ax.table(cellText=rows, colLabels=col_labels,
                        loc='center', cellLoc='center',
                        colColours=['lightgray']*7)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

    status_text = f"Completed: {stats['completed']}/111 | Pending: {stats['pending']} | Failed: {stats['failed']}"
    ax.set_title(f'Large Sample Study - Summary Statistics\n({status_text})',
                fontsize=12, pad=20)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_summary_table.{ext}')
    plt.close(fig)
    print("  Saved: fig_summary_table.png/pdf")


def save_results_csv(scores_df: pd.DataFrame, params_df: pd.DataFrame,
                     catchment_stats: pd.DataFrame, output_dir: Path):
    """Save comprehensive results to CSV."""

    # Merge all data
    if not scores_df.empty and not params_df.empty:
        merged = scores_df.merge(params_df, on='domain_id', how='outer')
        if not catchment_stats.empty:
            merged = merged.merge(catchment_stats, on='domain_id', how='outer')

        output_file = output_dir / 'large_sample_results.csv'
        merged.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    # Save scores separately
    if not scores_df.empty:
        scores_df.to_csv(output_dir / 'large_sample_scores.csv', index=False)

    # Save params separately
    if not params_df.empty:
        params_df.to_csv(output_dir / 'large_sample_params.csv', index=False)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Large Sample analysis with partial results support'
    )
    parser.add_argument('--output-dir', type=Path, default=ANALYSIS_DIR)
    parser.add_argument('--figures-dir', type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Large Sample Study - Comprehensive Analysis")
    print("=" * 70)

    # Get domain IDs
    domain_ids = get_domain_ids()
    print(f"\nConfigured domains: {len(domain_ids)}")

    # Load data
    scores_df, params_df, stats = load_all_results(domain_ids)
    catchment_stats = load_catchment_stats()

    print(f"\nCatchment stats loaded: {len(catchment_stats)} rows")

    # Print current status
    print("\n" + "-" * 50)
    print("CURRENT STATUS")
    print("-" * 50)
    print(f"  Completed calibrations: {stats['completed']}")
    print(f"  Failed calibrations:    {stats['failed']}")
    print(f"  Pending:                {stats['pending']}")
    print(f"  Total configured:       {len(domain_ids)}")

    if not scores_df.empty:
        print("\n" + "-" * 50)
        print("PERFORMANCE SUMMARY (completed catchments)")
        print("-" * 50)
        for col in ['kge_cal', 'kge_val', 'nse_cal', 'nse_val']:
            if col in scores_df.columns:
                data = scores_df[col].dropna()
                if len(data) > 0:
                    print(f"  {col:10s}: mean={data.mean():.3f}, median={data.median():.3f}, "
                          f"min={data.min():.3f}, max={data.max():.3f}")

    # Generate figures
    print("\n" + "-" * 50)
    print("GENERATING FIGURES")
    print("-" * 50)

    plot_performance_cdfs(scores_df, stats, args.figures_dir)
    plot_performance_map(scores_df, catchment_stats, stats, args.figures_dir)
    plot_parameter_distributions(params_df, stats, args.figures_dir)
    plot_performance_vs_attributes(scores_df, params_df, catchment_stats, stats, args.figures_dir)
    plot_summary_table(scores_df, params_df, stats, args.figures_dir)

    # Save CSV outputs
    print("\n" + "-" * 50)
    print("SAVING DATA")
    print("-" * 50)
    save_results_csv(scores_df, params_df, catchment_stats, args.output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
