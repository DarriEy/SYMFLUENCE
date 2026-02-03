#!/usr/bin/env python3
"""
Publication-ready figures for Section 4.8: Large Sample Study

Figure 1: Cross-catchment model performance (includes benchmark comparison)
Figure 2: Calibration diagnostics

Designed for two-column journal format (~180mm width for full-width figures).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# Cartopy for proper map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Optional: geopandas for catchment boundaries
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

warnings.filterwarnings('ignore')


def compute_skill_score(model_score, benchmark_score):
    """Compute skill score: improvement of model over benchmark."""
    if pd.isna(model_score) or pd.isna(benchmark_score):
        return np.nan
    if benchmark_score >= 1:
        return np.nan
    return (model_score - benchmark_score) / (1 - benchmark_score)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
CONFIGS_DIR = BASE_DIR / "configs"

# Publication styling
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
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'cal': '#0072B2',      # Blue
    'val': '#D55E00',      # Vermillion/Orange
    'neutral': '#CCCCCC',  # Light gray
    'pending': '#E0E0E0',  # Very light gray
    'accent': '#009E73',   # Teal
    'bounds': '#CC79A7',   # Pink
    'land': '#F5F5F5',     # Off-white for land
    'ocean': '#E6F2FF',    # Light blue for ocean
}

PARAM_BOUNDS = {
    "MAXWATR_1": (50, 500),
    "MAXWATR_2": (50, 1000),
    "MBASE": (-2.0, 2.0),
    "MFMAX": (1.0, 8.0),
}

# Iceland extent
ICELAND_EXTENT = [-25, -13, 63, 67]  # [lon_min, lon_max, lat_min, lat_max]


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


def load_catchment_stats():
    stats_file = ANALYSIS_DIR / "catchment_stats.csv"
    if stats_file.exists():
        return pd.read_csv(stats_file)
    return pd.DataFrame()


def load_benchmark_scores(domain_id):
    scores_file = SYMFLUENCE_DATA / f"domain_{domain_id}" / "evaluation" / "benchmark_scores.csv"
    if not scores_file.exists():
        return None
    try:
        df = pd.read_csv(scores_file)
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
    except:
        return None


def load_all_benchmark_scores(domain_id):
    """Load all benchmark scores (model + naive benchmarks) for a domain."""
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
    except:
        return None


def load_best_params(domain_id):
    params_file = SYMFLUENCE_DATA / f"domain_{domain_id}" / "optimization" / "FUSE" / "dds_run_1" / "run_1_dds_best_params.json"
    if not params_file.exists():
        return None
    try:
        with open(params_file) as f:
            data = json.load(f)
        if data.get('best_score', -9999) < -9000:
            return None
        result = {'domain_id': domain_id, 'best_score': data.get('best_score', np.nan)}
        for param, val in data.get('best_params', {}).items():
            result[param] = val
        return result
    except:
        return None


def load_catchment_boundaries(domain_ids):
    """Load catchment boundary polygons for map display."""
    if not HAS_GEOPANDAS:
        return None

    gdfs = []
    for did in domain_ids:
        shp_file = SYMFLUENCE_DATA / f"domain_{did}" / "shapefiles" / "catchment" / f"{did}_HRUs_GRUs.shp"
        if shp_file.exists():
            try:
                gdf = gpd.read_file(shp_file)
                gdf['domain_id'] = did
                # Dissolve to single polygon per domain
                gdf = gdf.dissolve().reset_index(drop=True)
                gdf['domain_id'] = did
                gdfs.append(gdf)
            except:
                pass

    if gdfs:
        return pd.concat(gdfs, ignore_index=True)
    return None


def load_all_data():
    domain_ids = get_domain_ids()

    scores_list = []
    params_list = []
    benchmarks_list = []

    for did in domain_ids:
        scores = load_benchmark_scores(did)
        params = load_best_params(did)
        benchmarks = load_all_benchmark_scores(did)
        if scores and params:
            scores_list.append(scores)
            params_list.append(params)
            if benchmarks:
                benchmarks_list.append(benchmarks)

    scores_df = pd.DataFrame(scores_list) if scores_list else pd.DataFrame()
    params_df = pd.DataFrame(params_list) if params_list else pd.DataFrame()
    benchmarks_df = pd.DataFrame(benchmarks_list) if benchmarks_list else pd.DataFrame()
    catchment_stats = load_catchment_stats()

    n_total = len(domain_ids)
    n_complete = len(scores_df)

    return scores_df, params_df, catchment_stats, benchmarks_df, n_total, n_complete, domain_ids


# ============================================================================
# Figure 1: Cross-Catchment Performance
# ============================================================================

def create_figure_1(scores_df, params_df, catchment_stats, benchmarks_df, n_total, n_complete, domain_ids):
    """
    Figure 1: Cross-catchment model performance

    (a) KGE CDFs - calibration vs validation
    (b) Spatial map of validation KGE with Iceland coastline
    (c) KGE vs glacier fraction
    (d) Model vs benchmark comparison
    """

    # Create figure with constrained layout for equal panels
    fig = plt.figure(figsize=(7.5, 6.5), constrained_layout=True)

    # Use GridSpec for precise control - 2x2 grid with equal sizes
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1],
                          wspace=0.15, hspace=0.25)

    # Merge data for scatter plots
    if not catchment_stats.empty and not scores_df.empty:
        merged = catchment_stats.merge(scores_df, on='domain_id', how='left')
    else:
        merged = pd.DataFrame()

    # -------------------------------------------------------------------------
    # (a) KGE CDFs - Model vs Benchmarks
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    # Benchmark definitions: (column_name, label, color, linestyle, linewidth)
    benchmark_configs = [
        ('daily_mean_flow_kge_val', 'FUSE Model', COLORS['cal'], '-', 2.5),
        ('monthly_mean_flow_kge_val', 'Monthly Clim.', '#009E73', '--', 1.8),
        ('mean_flow_kge_val', 'Mean Flow', '#D55E00', '--', 1.8),
        ('rainfall_runoff_ratio_to_monthly_kge_val', 'P-scaled Monthly', '#CC79A7', ':', 1.5),
        ('adjusted_smoothed_precipitation_benchmark_kge_val', 'Smoothed P', '#E69F00', ':', 1.5),
    ]

    if not benchmarks_df.empty:
        for col, label, color, ls, lw in benchmark_configs:
            if col in benchmarks_df.columns:
                data = benchmarks_df[col].dropna().sort_values()
                if len(data) > 0:
                    cdf = np.arange(1, len(data) + 1) / len(data)
                    ax_a.plot(data, cdf, ls, color=color, linewidth=lw, label=label)

                    # Add median marker for main model only
                    if col == 'daily_mean_flow_kge_val':
                        med_idx = np.argmin(np.abs(cdf - 0.5))
                        ax_a.plot(data.iloc[med_idx], 0.5, 'o', color=color,
                                 markersize=7, markeredgecolor='white', markeredgewidth=1.5)

    ax_a.set_xlabel('KGE (validation)')
    ax_a.set_ylabel('Cumulative probability')
    ax_a.set_xlim(-0.7, 1.0)
    ax_a.set_ylim(0, 1)
    ax_a.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax_a.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.95, fontsize=7)
    ax_a.set_title('(a) Model vs benchmarks', loc='left', fontweight='bold', fontsize=10)

    # -------------------------------------------------------------------------
    # (b) Spatial map with cartopy
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    # Set extent
    ax_b.set_extent(ICELAND_EXTENT, crs=ccrs.PlateCarree())

    # Add map features
    ax_b.add_feature(cfeature.OCEAN, facecolor=COLORS['ocean'], zorder=0)
    ax_b.add_feature(cfeature.LAND, facecolor=COLORS['land'], zorder=1)
    ax_b.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', zorder=3)
    ax_b.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='gray', zorder=2)

    # Add gridlines
    gl = ax_b.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                        alpha=0.5, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator([-24, -20, -16])
    gl.ylocator = plt.FixedLocator([63.5, 64.5, 65.5, 66.5])
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    if not merged.empty and 'centroid_lon' in merged.columns:
        # Pending catchments (gray circles)
        pending = merged[merged['kge_val'].isna()]
        if len(pending) > 0:
            ax_b.scatter(pending['centroid_lon'], pending['centroid_lat'],
                        c=COLORS['pending'], s=35, alpha=0.6,
                        edgecolors='#999999', linewidths=0.3,
                        transform=ccrs.PlateCarree(), zorder=4)

        # Completed catchments (colored by KGE)
        complete = merged[merged['kge_val'].notna()]
        if len(complete) > 0:
            sc = ax_b.scatter(complete['centroid_lon'], complete['centroid_lat'],
                             c=complete['kge_val'], cmap='RdYlBu',
                             vmin=-0.2, vmax=0.8,
                             s=55, edgecolors='black', linewidths=0.6,
                             transform=ccrs.PlateCarree(), zorder=5)

            # Colorbar
            cbar = plt.colorbar(sc, ax=ax_b, shrink=0.7, pad=0.02,
                               orientation='vertical', aspect=20)
            cbar.set_label('Validation KGE', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    # Legend for pending
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pending'],
               markersize=7, markeredgecolor='#999999', markeredgewidth=0.3,
               label=f'Pending (n={n_total - n_complete})')
    ]
    ax_b.legend(handles=legend_elements, loc='lower left', frameon=True,
               fancybox=False, edgecolor='gray', framealpha=0.95, fontsize=8)

    ax_b.set_title('(b) Spatial distribution', loc='left', fontweight='bold', fontsize=10)

    # -------------------------------------------------------------------------
    # (c) KGE vs glacier fraction
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    if not merged.empty and 'glac_fra' in merged.columns:
        complete = merged[merged['kge_cal'].notna()]

        if len(complete) > 0:
            # Calibration points
            ax_c.scatter(complete['glac_fra'], complete['kge_cal'],
                        c=COLORS['cal'], s=45, alpha=0.7, edgecolors='white',
                        linewidths=0.5, label='Calibration', zorder=3)
            # Validation points
            ax_c.scatter(complete['glac_fra'], complete['kge_val'],
                        c=COLORS['val'], s=45, alpha=0.7, edgecolors='white',
                        linewidths=0.5, label='Validation', zorder=3)

            # Add trend line for validation
            mask = complete['kge_val'].notna() & complete['glac_fra'].notna()
            if mask.sum() > 5:
                x = complete.loc[mask, 'glac_fra']
                y = complete.loc[mask, 'kge_val']
                slope, intercept, r, p, se = stats.linregress(x, y)
                x_line = np.array([0, 0.8])
                ax_c.plot(x_line, intercept + slope * x_line, '--',
                         color=COLORS['val'], alpha=0.6, linewidth=1.5, zorder=2)
                # Annotate r value
                ax_c.text(0.72, intercept + slope * 0.72 + 0.08, f'r={r:.2f}',
                         fontsize=8, color=COLORS['val'], alpha=0.8)

    ax_c.set_xlabel('Glacier fraction')
    ax_c.set_ylabel('KGE')
    ax_c.set_xlim(-0.03, 0.85)
    ax_c.set_ylim(-0.5, 1.05)
    ax_c.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax_c.legend(loc='lower left', frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.95)
    ax_c.set_title('(c) Effect of glacier coverage', loc='left', fontweight='bold', fontsize=10)

    # -------------------------------------------------------------------------
    # (d) Model vs Monthly Climatology Benchmark
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    if not benchmarks_df.empty:
        model_kge = benchmarks_df['daily_mean_flow_kge_val']
        monthly_kge = benchmarks_df['monthly_mean_flow_kge_val']
        mask = model_kge.notna() & monthly_kge.notna()

        if mask.sum() > 0:
            # Scatter plot
            ax_d.scatter(monthly_kge[mask], model_kge[mask],
                        c=COLORS['cal'], s=50, alpha=0.7, edgecolors='white',
                        linewidths=0.5, zorder=3)

            # 1:1 line
            lims = [-0.5, 1.0]
            ax_d.plot(lims, lims, '--', color='gray', linewidth=1.5, alpha=0.7)

            # Shade region where model is better (above line)
            ax_d.fill_between(lims, lims, [1.1, 1.1], alpha=0.08,
                             color=COLORS['cal'], zorder=1)

            # Count how many beat benchmark
            n_better = (model_kge[mask] > monthly_kge[mask]).sum()
            n_total_bench = mask.sum()

            ax_d.text(0.05, 0.95, f'Model better:\n{n_better}/{n_total_bench} ({100*n_better/n_total_bench:.0f}%)',
                     transform=ax_d.transAxes, ha='left', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.9))

            ax_d.set_xlim(lims)
            ax_d.set_ylim(lims)

    ax_d.set_xlabel('Monthly climatology KGE')
    ax_d.set_ylabel('FUSE model KGE')
    ax_d.set_aspect('equal')
    ax_d.set_title('(d) Model vs benchmark', loc='left', fontweight='bold', fontsize=10)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES_DIR / f'fig_4_8_performance.{ext}')
    plt.close(fig)
    print("Saved: fig_4_8_performance.png/pdf")


# ============================================================================
# Figure 2: Calibration Diagnostics
# ============================================================================

def create_figure_2(scores_df, params_df, catchment_stats, n_total, n_complete):
    """
    Figure 2: Calibration diagnostics - parameter distributions

    (a) MAXWATR_1 - upper zone storage
    (b) MAXWATR_2 - lower zone storage
    (c) MFMAX - maximum melt factor
    (d) MBASE - temperature threshold for melt
    """

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), constrained_layout=True)
    axes = axes.flatten()

    params_to_plot = [
        ('MAXWATR_1', 'Upper zone capacity (mm)', '(a)'),
        ('MAXWATR_2', 'Lower zone capacity (mm)', '(b)'),
        ('MFMAX', 'Max melt factor (mm °C⁻¹ d⁻¹)', '(c)'),
        ('MBASE', 'Melt temperature threshold (°C)', '(d)'),
    ]

    for idx, (param, label, panel) in enumerate(params_to_plot):
        ax = axes[idx]

        if param in params_df.columns:
            data = params_df[param].dropna()

            if len(data) > 0:
                # Get bounds
                bounds = PARAM_BOUNDS.get(param, (data.min(), data.max()))
                bins = np.linspace(bounds[0], bounds[1], 16)

                # Histogram with better styling
                n, bins_out, patches = ax.hist(data, bins=bins, color=COLORS['cal'],
                                               edgecolor='white', alpha=0.85, linewidth=0.8)

                # Parameter bounds (dashed pink lines)
                ax.axvline(bounds[0], color=COLORS['bounds'], linestyle='--',
                          linewidth=1.8, alpha=0.9, zorder=4)
                ax.axvline(bounds[1], color=COLORS['bounds'], linestyle='--',
                          linewidth=1.8, alpha=0.9, zorder=4)

                # Median line (solid black)
                median_val = data.median()
                ax.axvline(median_val, color='black', linestyle='-',
                          linewidth=2.2, zorder=5)

                # Stats box
                stats_text = f'n = {len(data)}\nmedian = {median_val:.1f}'
                ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                       ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='gray', alpha=0.9))

        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{panel} {param}', loc='left', fontweight='bold', fontsize=10)

    # Add legend at bottom
    legend_elements = [
        Line2D([0], [0], color=COLORS['bounds'], linestyle='--', linewidth=2,
               label='Parameter bounds'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2.2,
               label='Median'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=False,
              edgecolor='gray')

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES_DIR / f'fig_4_8_calibration.{ext}')
    plt.close(fig)
    print("Saved: fig_4_8_calibration.png/pdf")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Generating publication figures for Section 4.8")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    scores_df, params_df, catchment_stats, benchmarks_df, n_total, n_complete, domain_ids = load_all_data()
    print(f"  Completed: {n_complete}/{n_total} catchments")
    print(f"  Benchmark data: {len(benchmarks_df)} domains")

    if n_complete == 0:
        print("\nNo completed results yet. Figures will show placeholders.")

    # Generate figures
    print("\nGenerating figures...")
    create_figure_1(scores_df, params_df, catchment_stats, benchmarks_df, n_total, n_complete, domain_ids)
    create_figure_2(scores_df, params_df, catchment_stats, n_total, n_complete)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - {FIGURES_DIR / 'fig_4_8_performance.pdf'}")
    print(f"  - {FIGURES_DIR / 'fig_4_8_calibration.pdf'}")


if __name__ == "__main__":
    main()
