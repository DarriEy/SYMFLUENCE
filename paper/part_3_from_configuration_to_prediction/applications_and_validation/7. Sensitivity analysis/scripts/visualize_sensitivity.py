#!/usr/bin/env python3
"""
Multi-Model Sensitivity Analysis Visualization for SYMFLUENCE Paper Section 4.7

Creates two publication-quality figures for the sensitivity analysis section:

    Figure 1 (fig_process_sensitivity):
        (a) Process sensitivity heatmap — which processes matter in which models
        (b) Radar chart — model sensitivity profiles at a glance

    Figure 2 (fig_parameter_sensitivity):
        Per-model horizontal bar charts showing individual parameter sensitivity,
        colored by hydrological process to link back to Figure 1.

Usage:
    python visualize_sensitivity.py [--output-dir DIR] [--format pdf|png|svg]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Add SYMFLUENCE to path for process mapping import
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"

# Colorblind-friendly palette (Wong 2011)
MODEL_COLORS = {
    "FUSE": "#009E73",
    "GR4J": "#E69F00",
    "HBV": "#0072B2",
    "HYPE": "#CC79A7",
    "SUMMA": "#56B4E9",
}

MODEL_MARKERS = {
    "FUSE": "o",
    "GR4J": "s",
    "HBV": "D",
    "HYPE": "^",
    "SUMMA": "v",
}

PROCESS_COLORS = {
    "Snow": "#56B4E9",
    "Evapotranspiration": "#009E73",
    "Soil Storage": "#A67C52",
    "Surface Runoff": "#E69F00",
    "Percolation": "#999999",
    "Baseflow": "#0072B2",
    "Groundwater Exchange": "#44AA99",
    "Routing": "#CC79A7",
}

PROCESS_ORDER = [
    "Snow",
    "Evapotranspiration",
    "Soil Storage",
    "Surface Runoff",
    "Percolation",
    "Baseflow",
    "Groundwater Exchange",
    "Routing",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sensitivity_visualization")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_analysis_data(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {}
    for key, pat in [
        ("process", "process_sensitivity_*.csv"),
        ("per_model", "per_model_sensitivity_*.csv"),
        ("rankings", "cross_model_ranking_*.csv"),
    ]:
        f = find_latest_file(analysis_dir, pat)
        if f:
            data[key] = pd.read_csv(f, index_col=0 if key == "process" else None)
            logger.info(f"Loaded {key}: {f.name}")
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enable_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def _add_nan_hatching(ax, data):
    """Hatch NaN cells so 'not represented' is distinct from low values."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1, 1, boxstyle="square,pad=0",
                    facecolor='#F5F5F5', edgecolor='#BDBDBD',
                    linewidth=0.5, hatch='///', zorder=2,
                ))
                ax.text(j, i, '\u2014', ha='center', va='center',
                        color='#9E9E9E', fontsize=9)


def _order_process_df(process_df: pd.DataFrame) -> pd.DataFrame:
    ordered = [p for p in PROCESS_ORDER if p in process_df.index]
    remaining = [p for p in process_df.index if p not in ordered]
    process_df = process_df.reindex(ordered + remaining)
    model_order = [m for m in MODEL_COLORS if m in process_df.columns]
    return process_df[model_order]


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if path.suffix == '.pdf':
        fig.savefig(path.with_name(path.stem + '_preview.png'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Process-level sensitivity (heatmap + radar)
# ---------------------------------------------------------------------------

def create_figure_1(process_df: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 1 — Process Sensitivity Across Models

    (a) Heatmap: mean sensitivity index per process per model, with hatched
        NaN cells for processes absent from a model's parameter set.
    (b) Radar: same data as a spider chart, emphasising model-specific
        sensitivity profiles and structural gaps.
    """
    process_df = _order_process_df(process_df.copy())
    data = process_df.values

    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.38, width_ratios=[1.15, 1])

    # --- (a) Heatmap ---
    ax_h = fig.add_subplot(gs[0, 0])
    _enable_spines(ax_h)

    im = ax_h.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    _add_nan_hatching(ax_h, data)

    ax_h.set_xticks(range(len(process_df.columns)))
    ax_h.set_xticklabels(process_df.columns, rotation=45, ha='right',
                         fontweight='bold')
    ax_h.set_yticks(range(len(process_df.index)))
    ax_h.set_yticklabels(process_df.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax_h.text(j, i, f'{v:.2f}', ha='center', va='center',
                          color='white' if v > 0.6 else 'black',
                          fontsize=8.5, fontweight='medium')

    fig.colorbar(im, ax=ax_h, shrink=0.82, label='Mean Sensitivity Index')
    ax_h.legend(
        handles=[mpatches.Patch(facecolor='#F5F5F5', edgecolor='#BDBDBD',
                                hatch='///', label='Not represented')],
        loc='lower right', fontsize=8, framealpha=0.9,
    )
    ax_h.set_xlabel('Model')
    ax_h.set_ylabel('Hydrological Process')
    ax_h.set_title('(a) Process Sensitivity', fontweight='bold', pad=8)

    # --- (b) Radar ---
    ax_r = fig.add_subplot(gs[0, 1], polar=True)

    categories = list(process_df.index)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for model in process_df.columns:
        vals = process_df[model].values
        color = MODEL_COLORS[model]
        marker = MODEL_MARKERS[model]

        # Markers
        for i in range(N):
            if not np.isnan(vals[i]):
                ax_r.plot(angles[i], vals[i], marker, color=color,
                          markersize=6, zorder=5, markeredgecolor='white',
                          markeredgewidth=0.5)
        # Lines between consecutive valid points
        for i in range(N):
            j = (i + 1) % N
            if not np.isnan(vals[i]) and not np.isnan(vals[j]):
                ax_r.plot([angles[i], angles[j]], [vals[i], vals[j]],
                          '-', linewidth=1.8, color=color)
        # Legend entry
        ax_r.plot([], [], marker + '-', linewidth=1.8, color=color,
                  markersize=6, label=model, markeredgecolor='white',
                  markeredgewidth=0.5)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(categories, fontsize=9)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax_r.set_rlabel_position(30)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.28, 1.12), fontsize=9,
                framealpha=0.9)
    ax_r.set_title('(b) Sensitivity Profiles', fontweight='bold', pad=18)

    # Footnote
    fig.text(0.5, -0.04,
             'Sensitivity indices derived from optimization trajectories '
             '(screening-level analysis). Hatched cells = process absent '
             'from model parameter set.',
             ha='center', fontsize=8, fontstyle='italic', color='#616161')

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: Parameter-level sensitivity (bar charts)
# ---------------------------------------------------------------------------

def create_figure_2(per_model_df: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 2 — Parameter Sensitivity by Model

    Horizontal bars showing normalised ensemble-mean sensitivity per
    parameter, colored by hydrological process. Each subplot annotates the
    number of methods that contributed to the ensemble mean.
    """
    from analyze_sensitivity import PROCESS_MAPPING

    models = [m for m in MODEL_COLORS if m in per_model_df['Model'].values]
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        md = per_model_df[per_model_df['Model'] == model].copy()
        method_cols = [c for c in md.columns if c not in ['Model', 'Parameter']]

        md[method_cols] = md[method_cols].replace(-999, np.nan).abs()
        n_methods = int(md[method_cols].notna().sum(axis=1).iloc[0])
        md['ens_mean'] = md[method_cols].mean(axis=1)

        mx = md['ens_mean'].max()
        md['norm'] = md['ens_mean'] / mx if mx > 0 else 0
        md = md.sort_values('norm', ascending=True)

        colors = [PROCESS_COLORS.get(PROCESS_MAPPING.get(p, 'Other'), '#BDBDBD')
                  for p in md['Parameter']]

        ax.barh(range(len(md)), md['norm'].values, color=colors,
                edgecolor='gray', linewidth=0.5)
        ax.set_yticks(range(len(md)))
        ax.set_yticklabels(md['Parameter'].values, fontsize=8)
        ax.set_xlabel('Normalised Sensitivity')
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.35, lw=0.5)
        ax.grid(axis='x', alpha=0.2)

        method_label = f'{n_methods} method{"s" if n_methods != 1 else ""}'
        ax.set_title(f'{model}  ({method_label})', fontweight='bold',
                     color=MODEL_COLORS.get(model, 'black'))

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    # Process colour legend
    legend_handles = [Patch(facecolor=c, edgecolor='gray', label=p)
                      for p, c in PROCESS_COLORS.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=8,
               title='Hydrological Process', title_fontsize=9,
               framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_visualization(analysis_dir: Path, output_dir: Path,
                      fig_format: str = "pdf") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(analysis_dir)

    if not data:
        logger.error("No analysis data found. Run analyze_sensitivity.py first.")
        return

    if "process" in data:
        create_figure_1(data["process"],
                        output_dir / f"fig_process_sensitivity.{fig_format}")
    if "per_model" in data:
        create_figure_2(data["per_model"],
                        output_dir / f"fig_parameter_sensitivity.{fig_format}")

    logger.info(f"All figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create publication figures for Section 4.7 sensitivity analysis"
    )
    parser.add_argument("--analysis-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"],
                        default="pdf")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else ANALYSIS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR
    run_visualization(analysis_dir, output_dir, args.format)


if __name__ == "__main__":
    main()
