#!/usr/bin/env python3
"""
Benchmark analysis for the Large Sample study.

Compares FUSE model performance against naive benchmarks:
- Mean flow (climatological mean)
- Monthly mean flow (monthly climatology)
- Rainfall-runoff ratio benchmarks

Computes skill scores showing improvement over benchmarks.

Output:
- benchmark_comparison.csv: All benchmark scores for all domains
- fig_4_8_benchmarks.pdf: Benchmark comparison figure
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
CONFIGS_DIR = BASE_DIR / "configs"

# Benchmarks to analyze
BENCHMARKS = {
    'mean_flow': 'Mean Flow',
    'monthly_mean_flow': 'Monthly Climatology',
    'daily_mean_flow': 'FUSE Model',
    'rainfall_runoff_ratio_to_monthly': 'P-scaled (monthly)',
    'adjusted_smoothed_precipitation_benchmark': 'Smoothed P',
}

# Key comparison: Model vs these benchmarks
BENCHMARK_COMPARISONS = [
    ('daily_mean_flow', 'mean_flow', 'vs Mean'),
    ('daily_mean_flow', 'monthly_mean_flow', 'vs Monthly Clim.'),
]

# Plot styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'model': '#0072B2',
    'benchmark1': '#D55E00',
    'benchmark2': '#009E73',
    'benchmark3': '#CC79A7',
    'neutral': '#999999',
}


# ============================================================================
# Data Loading
# ============================================================================

def get_domain_ids():
    ids = []
    for f in CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"):
        parts = f.stem.replace("config_lamahice_", "").replace("_FUSE", "")
        try:
            ids.append(int(parts))
        except ValueError:
            continue
    return sorted(ids)


def load_benchmark_scores(domain_id):
    """Load all benchmark scores for a domain."""
    scores_file = SYMFLUENCE_DATA / f"domain_{domain_id}" / "evaluation" / "benchmark_scores.csv"
    if not scores_file.exists():
        return None

    try:
        df = pd.read_csv(scores_file)

        result = {'domain_id': domain_id}

        for _, row in df.iterrows():
            bench_name = row['benchmarks']
            for metric in ['nse_cal', 'nse_val', 'kge_cal', 'kge_val']:
                key = f"{bench_name}_{metric}"
                result[key] = row.get(metric, np.nan)

        return result
    except Exception as e:
        print(f"  Warning: Could not parse benchmarks for domain {domain_id}: {e}")
        return None


def load_best_params(domain_id):
    """Check if optimization succeeded (not all crashes)."""
    params_file = SYMFLUENCE_DATA / f"domain_{domain_id}" / "optimization" / "FUSE" / "dds_run_1" / "run_1_dds_best_params.json"
    if not params_file.exists():
        return None
    try:
        with open(params_file) as f:
            data = json.load(f)
        if data.get('best_score', -9999) < -9000:
            return None
        return data
    except:
        return None


def load_all_benchmarks():
    """Load benchmark data for all domains with successful calibration."""
    domain_ids = get_domain_ids()

    all_scores = []

    for did in domain_ids:
        # Only include domains with successful optimization
        params = load_best_params(did)
        if params is None:
            continue

        scores = load_benchmark_scores(did)
        if scores is not None:
            all_scores.append(scores)

    if all_scores:
        return pd.DataFrame(all_scores)
    return pd.DataFrame()


def load_catchment_stats():
    stats_file = ANALYSIS_DIR / "catchment_stats.csv"
    if stats_file.exists():
        return pd.read_csv(stats_file)
    return pd.DataFrame()


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_skill_score(model_score, benchmark_score):
    """
    Compute skill score: improvement of model over benchmark.

    SS = (model - benchmark) / (1 - benchmark)

    SS > 0: model better than benchmark
    SS = 1: perfect model
    SS < 0: model worse than benchmark
    """
    if pd.isna(model_score) or pd.isna(benchmark_score):
        return np.nan
    if benchmark_score >= 1:
        return np.nan
    return (model_score - benchmark_score) / (1 - benchmark_score)


def analyze_benchmark_skill(df):
    """Compute skill scores for model vs benchmarks."""

    results = df[['domain_id']].copy()

    # Model performance
    results['model_kge_cal'] = df['daily_mean_flow_kge_cal']
    results['model_kge_val'] = df['daily_mean_flow_kge_val']
    results['model_nse_cal'] = df['daily_mean_flow_nse_cal']
    results['model_nse_val'] = df['daily_mean_flow_nse_val']

    # Benchmark performance
    results['mean_flow_kge_val'] = df['mean_flow_kge_val']
    results['monthly_clim_kge_val'] = df['monthly_mean_flow_kge_val']

    # Skill scores (validation period)
    results['skill_vs_mean'] = df.apply(
        lambda r: compute_skill_score(r['daily_mean_flow_kge_val'], r['mean_flow_kge_val']),
        axis=1
    )
    results['skill_vs_monthly'] = df.apply(
        lambda r: compute_skill_score(r['daily_mean_flow_kge_val'], r['monthly_mean_flow_kge_val']),
        axis=1
    )

    # Does model beat benchmark?
    results['beats_mean'] = results['model_kge_val'] > df['mean_flow_kge_val']
    results['beats_monthly'] = results['model_kge_val'] > df['monthly_mean_flow_kge_val']

    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_benchmark_comparison(df, skill_df, output_dir):
    """
    Create benchmark comparison figure.

    (a) KGE comparison: Model vs Mean Flow vs Monthly Climatology (CDFs)
    (b) Skill score distribution
    (c) Model vs Monthly benchmark scatter (1:1 line)
    """

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

    # -------------------------------------------------------------------------
    # (a) KGE CDFs for Model and Benchmarks
    # -------------------------------------------------------------------------
    ax = axes[0]

    # Model (daily_mean_flow)
    model_kge = df['daily_mean_flow_kge_val'].dropna().sort_values()
    if len(model_kge) > 0:
        cdf = np.arange(1, len(model_kge) + 1) / len(model_kge)
        ax.plot(model_kge, cdf, '-', color=COLORS['model'], linewidth=2.5,
               label=f'FUSE Model (n={len(model_kge)})')

    # Mean flow benchmark
    mean_kge = df['mean_flow_kge_val'].dropna().sort_values()
    if len(mean_kge) > 0:
        cdf = np.arange(1, len(mean_kge) + 1) / len(mean_kge)
        ax.plot(mean_kge, cdf, '--', color=COLORS['benchmark1'], linewidth=2,
               label='Mean Flow')

    # Monthly climatology benchmark
    monthly_kge = df['monthly_mean_flow_kge_val'].dropna().sort_values()
    if len(monthly_kge) > 0:
        cdf = np.arange(1, len(monthly_kge) + 1) / len(monthly_kge)
        ax.plot(monthly_kge, cdf, '--', color=COLORS['benchmark2'], linewidth=2,
               label='Monthly Climatology')

    ax.set_xlabel('KGE (validation)')
    ax.set_ylabel('Cumulative probability')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 1)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('(a) Model vs benchmarks', loc='left', fontweight='bold')

    # -------------------------------------------------------------------------
    # (b) Skill Score Distribution
    # -------------------------------------------------------------------------
    ax = axes[1]

    skill_mean = skill_df['skill_vs_mean'].dropna()
    skill_monthly = skill_df['skill_vs_monthly'].dropna()

    positions = [1, 2]
    box_data = [skill_mean, skill_monthly]
    labels = ['vs Mean\nFlow', 'vs Monthly\nClimatology']

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                   showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.5))

    colors_box = [COLORS['benchmark1'], COLORS['benchmark2']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual points
    for i, (data, color) in enumerate(zip(box_data, colors_box)):
        x = np.random.normal(positions[i], 0.08, size=len(data))
        ax.scatter(x, data, c=color, s=25, alpha=0.6, edgecolors='white', linewidths=0.5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Skill Score')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.5, 1.5)

    # Annotation
    n_beat_mean = skill_df['beats_mean'].sum()
    n_beat_monthly = skill_df['beats_monthly'].sum()
    n_total = len(skill_df)
    ax.text(0.98, 0.98, f'Beats mean: {n_beat_mean}/{n_total}\nBeats monthly: {n_beat_monthly}/{n_total}',
           transform=ax.transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_title('(b) Skill scores', loc='left', fontweight='bold')

    # -------------------------------------------------------------------------
    # (c) Scatter: Model vs Monthly Benchmark
    # -------------------------------------------------------------------------
    ax = axes[2]

    model_vals = df['daily_mean_flow_kge_val']
    monthly_vals = df['monthly_mean_flow_kge_val']

    mask = model_vals.notna() & monthly_vals.notna()

    ax.scatter(monthly_vals[mask], model_vals[mask],
              c=COLORS['model'], s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

    # 1:1 line
    lims = [-0.6, 1.0]
    ax.plot(lims, lims, '--', color='gray', linewidth=1.5, alpha=0.7, label='1:1 line')

    # Points above line = model better
    n_above = (model_vals[mask] > monthly_vals[mask]).sum()
    n_total = mask.sum()

    ax.set_xlabel('Monthly Climatology KGE')
    ax.set_ylabel('FUSE Model KGE')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray')

    # Shade regions
    ax.fill_between(lims, lims, [1, 1], alpha=0.1, color=COLORS['model'], label='Model better')
    ax.fill_between(lims, [-1, -1], lims, alpha=0.1, color=COLORS['benchmark2'], label='Benchmark better')

    ax.text(0.05, 0.95, f'Model better:\n{n_above}/{n_total} ({100*n_above/n_total:.0f}%)',
           transform=ax.transAxes, ha='left', va='top', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_title('(c) Model vs benchmark', loc='left', fontweight='bold')

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'fig_S_4_8_benchmarks.{ext}')
    plt.close(fig)
    print("Saved: fig_S_4_8_benchmarks.png/pdf (supplementary)")


def create_summary_table(df, skill_df, output_dir):
    """Create summary statistics table."""

    summary = []

    # Model performance
    model_kge_val = df['daily_mean_flow_kge_val'].dropna()
    summary.append({
        'Benchmark': 'FUSE Model',
        'n': len(model_kge_val),
        'KGE_val_mean': model_kge_val.mean(),
        'KGE_val_median': model_kge_val.median(),
        'KGE_val_min': model_kge_val.min(),
        'KGE_val_max': model_kge_val.max(),
    })

    # Mean flow benchmark
    mean_kge_val = df['mean_flow_kge_val'].dropna()
    summary.append({
        'Benchmark': 'Mean Flow',
        'n': len(mean_kge_val),
        'KGE_val_mean': mean_kge_val.mean(),
        'KGE_val_median': mean_kge_val.median(),
        'KGE_val_min': mean_kge_val.min(),
        'KGE_val_max': mean_kge_val.max(),
    })

    # Monthly climatology
    monthly_kge_val = df['monthly_mean_flow_kge_val'].dropna()
    summary.append({
        'Benchmark': 'Monthly Climatology',
        'n': len(monthly_kge_val),
        'KGE_val_mean': monthly_kge_val.mean(),
        'KGE_val_median': monthly_kge_val.median(),
        'KGE_val_min': monthly_kge_val.min(),
        'KGE_val_max': monthly_kge_val.max(),
    })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'benchmark_summary.csv', index=False)

    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Benchmark Analysis for Large Sample Study")
    print("=" * 60)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading benchmark data...")
    df = load_all_benchmarks()

    if df.empty:
        print("No benchmark data found!")
        return

    print(f"  Loaded benchmarks for {len(df)} domains")

    # Analyze skill
    print("\nComputing skill scores...")
    skill_df = analyze_benchmark_skill(df)

    # Save full results
    df.to_csv(ANALYSIS_DIR / 'benchmark_comparison.csv', index=False)
    skill_df.to_csv(ANALYSIS_DIR / 'benchmark_skill_scores.csv', index=False)
    print("  Saved: benchmark_comparison.csv, benchmark_skill_scores.csv")

    # Summary statistics
    print("\n" + "-" * 50)
    print("BENCHMARK COMPARISON SUMMARY")
    print("-" * 50)

    summary_df = create_summary_table(df, skill_df, ANALYSIS_DIR)
    print(summary_df.to_string(index=False))

    # Skill score summary
    print("\n" + "-" * 50)
    print("SKILL SCORES (validation period)")
    print("-" * 50)

    skill_mean = skill_df['skill_vs_mean'].dropna()
    skill_monthly = skill_df['skill_vs_monthly'].dropna()

    print(f"  vs Mean Flow:        median={skill_mean.median():.3f}, "
          f"mean={skill_mean.mean():.3f}, "
          f">{0}: {(skill_mean > 0).sum()}/{len(skill_mean)}")
    print(f"  vs Monthly Clim:     median={skill_monthly.median():.3f}, "
          f"mean={skill_monthly.mean():.3f}, "
          f">{0}: {(skill_monthly > 0).sum()}/{len(skill_monthly)}")

    # Generate figure
    print("\n" + "-" * 50)
    print("GENERATING FIGURE")
    print("-" * 50)
    plot_benchmark_comparison(df, skill_df, FIGURES_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
