#!/usr/bin/env python3
"""
Publication Figure Generator for SYMFLUENCE Paper Section 4.5

Creates two figures comparing benchmark performance against the multi-model
ensemble from Section 4.2. Matches visual style of Section 4.2 figures
(serif fonts, (a)/(b) panel labels, consistent sizing).

Figures:
    1. fig_benchmarking.{png,pdf}:
       (a) Horizontal bar chart — models + ensemble vs deduplicated benchmarks
       (b) Grouped benchmark KGE by category (dot plot with ranges)

    2. fig_benchmark_flows.{png,pdf}:
       (a) Evaluation-period time series — best benchmark, ensemble mean, observed
       (b) Selected benchmark flows showing increasing temporal resolution

Usage:
    python create_publication_figures.py [--data-dir DIR] [--output-dir DIR]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"

# ---------------------------------------------------------------------------
# Publication style — matches Section 4.2 (visualize_ensemble.py)
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
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
    })

# ---------------------------------------------------------------------------
# Section 4.2 model evaluation KGE (Table 14 of the paper)
# ---------------------------------------------------------------------------
SECTION_4_2_MODELS = {
    "SUMMA":  {"Eval_KGE": 0.88, "type": "Process-based"},
    "FUSE":   {"Eval_KGE": 0.88, "type": "Conceptual"},
    "GR4J":   {"Eval_KGE": 0.79, "type": "Conceptual"},
    "HBV":    {"Eval_KGE": 0.70, "type": "Conceptual"},
    "HYPE":   {"Eval_KGE": 0.81, "type": "Conceptual"},
    "LSTM":   {"Eval_KGE": 0.88, "type": "Data-driven"},
}
ENSEMBLE_MEAN_KGE = 0.94
ENSEMBLE_MEDIAN_KGE = 0.92

# ---------------------------------------------------------------------------
# Deduplicated benchmark groups
#
# Several HydroBM benchmarks produce identical flow series on this domain
# (confirmed numerically):
#   rainfall_runoff_ratio_to_daily == rainfall_runoff_ratio_to_timestep
#       == scaled_precipitation_benchmark == adjusted_precipitation_benchmark
#   monthly_rainfall_runoff_ratio_to_daily == monthly_rainfall_runoff_ratio_to_timestep
#
# We keep one representative from each duplicate set.
# ---------------------------------------------------------------------------
BENCHMARK_GROUPS = {
    "Time-invariant": [
        "mean_flow",
        "median_flow",
    ],
    "Seasonal": [
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
    ],
    "Rainfall-runoff": [
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
        "rainfall_runoff_ratio_to_daily",        # representative of 4 duplicates
    ],
    "Schaefli & Gupta\n(2007)": [
        "scaled_precipitation_benchmark",         # representative of 2 duplicates
        "adjusted_smoothed_precipitation_benchmark",
    ],
}

# Display names for benchmarks (short, publication-friendly)
BENCHMARK_DISPLAY = {
    "mean_flow":               "Mean flow",
    "median_flow":             "Median flow",
    "monthly_mean_flow":       "Monthly mean",
    "monthly_median_flow":     "Monthly median",
    "daily_mean_flow":         "Daily mean",
    "daily_median_flow":       "Daily median",
    "rainfall_runoff_ratio_to_all":     "R-R ratio (all)",
    "rainfall_runoff_ratio_to_annual":  "R-R ratio (annual)",
    "rainfall_runoff_ratio_to_monthly": "R-R ratio (monthly)",
    "rainfall_runoff_ratio_to_daily":   "R-R ratio (daily)",
    "scaled_precipitation_benchmark":   "Scaled precip.",
    "adjusted_smoothed_precipitation_benchmark": "Adj. smoothed precip.",
}

# Color scheme — consistent with Section 4.2
MODEL_TYPE_COLORS = {
    "Process-based": "#2166ac",
    "Conceptual":    "#b2182b",
    "Data-driven":   "#4dac26",
    "Ensemble":      "#000000",
}
BENCHMARK_COLOR = "#969696"

# Group colors for the dot plot
GROUP_COLORS = {
    "Time-invariant":           "#bdbdbd",
    "Seasonal":                 "#74a9cf",
    "Rainfall-runoff":          "#fd8d3c",
    "Schaefli & Gupta\n(2007)": "#78c679",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("figure_generator")


# ===================================================================
# Helpers
# ===================================================================
def get_kge_val_column(scores: pd.DataFrame) -> str:
    """Return the validation-KGE column name."""
    for col in scores.columns:
        if "kge" in col.lower() and "val" in col.lower():
            return col
    for col in scores.columns:
        if "kge" in col.lower():
            return col
    raise ValueError(f"No KGE column found. Available: {list(scores.columns)}")


def get_kge_cal_column(scores: pd.DataFrame) -> str:
    """Return the calibration-KGE column name."""
    for col in scores.columns:
        if "kge" in col.lower() and "cal" in col.lower():
            return col
    return get_kge_val_column(scores)


def valid_kge(scores: pd.DataFrame, col: str) -> pd.Series:
    """Return a Series of finite KGE values."""
    return scores[col].replace([np.inf, -np.inf], np.nan).dropna()


def deduplicated_benchmarks(scores: pd.DataFrame, kge_col: str) -> pd.Series:
    """Return KGE series for only the deduplicated benchmark set."""
    keep = []
    for group_name, bms in BENCHMARK_GROUPS.items():
        for bm in bms:
            if bm in scores.index:
                keep.append(bm)
    return valid_kge(scores.loc[scores.index.isin(keep)], kge_col)


# ===================================================================
# Figure 1 — Benchmarking comparison (two panels)
# ===================================================================
def fig_benchmarking(scores: pd.DataFrame, output_dir: Path):
    """
    Two-panel figure:
      (a) Horizontal bar chart: models + ensemble vs benchmarks (KGE, val)
      (b) Grouped dot plot: benchmark KGE by category with group ranges
    """
    if not HAS_MATPLOTLIB:
        return

    kge_col = get_kge_val_column(scores)
    kge_cal_col = get_kge_cal_column(scores)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.30,
                          left=0.08, right=0.88, bottom=0.15, top=0.92)
    ax_bar = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # Panel (a): Horizontal bar chart
    # ------------------------------------------------------------------
    bm_kge = deduplicated_benchmarks(scores, kge_col).sort_values(ascending=True)

    # Model entries sorted by KGE
    model_entries = sorted(SECTION_4_2_MODELS.items(), key=lambda x: x[1]["Eval_KGE"])
    model_names = [m[0] for m in model_entries]
    model_kge = [m[1]["Eval_KGE"] for m in model_entries]
    model_types = [m[1]["type"] for m in model_entries]

    # Add ensemble
    model_names += ["Ens. median", "Ens. mean"]
    model_kge += [ENSEMBLE_MEDIAN_KGE, ENSEMBLE_MEAN_KGE]
    model_types += ["Ensemble", "Ensemble"]

    # Positions: benchmarks then gap then models
    n_bm = len(bm_kge)
    n_mod = len(model_names)
    gap = 1.0
    bm_pos = np.arange(n_bm)
    mod_pos = np.arange(n_mod) + n_bm + gap

    # Draw benchmark bars
    ax_bar.barh(bm_pos, bm_kge.values, height=0.65, color=BENCHMARK_COLOR,
                edgecolor="white", alpha=0.75)

    # Draw model bars
    mod_colors = [MODEL_TYPE_COLORS.get(t, "#333") for t in model_types]
    ax_bar.barh(mod_pos, model_kge, height=0.65, color=mod_colors, edgecolor="white")

    # Best-benchmark reference line (annotation added after sep_y is defined)
    best_bm_kge = bm_kge.max()
    ax_bar.axvline(best_bm_kge, color="#555555", ls="--", lw=0.9, alpha=0.8)

    # Y labels
    bm_labels = [BENCHMARK_DISPLAY.get(b, b.replace("_", " ")) for b in bm_kge.index]
    all_labels = bm_labels + model_names
    all_pos = list(bm_pos) + list(mod_pos)
    ax_bar.set_yticks(all_pos)
    ax_bar.set_yticklabels(all_labels, fontsize=8.5)

    # Separator line between benchmarks and models
    sep_y = n_bm + gap / 2 - 0.5
    ax_bar.axhline(sep_y, color="black", lw=0.6, ls="-")

    # Best-benchmark annotation — sits at the separator, clear of all bars
    ax_bar.annotate(
        f"Best benchmark\nKGE = {best_bm_kge:.2f}",
        xy=(best_bm_kge, sep_y), xytext=(best_bm_kge + 0.06, sep_y),
        fontsize=7.5, color="#555555", va="center",
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
    )

    ax_bar.set_xlabel("KGE (validation period)")
    ax_bar.set_xlim(-0.55, 1.02)
    ax_bar.set_title("(a) Model ensemble vs simple benchmarks", loc="left")
    ax_bar.grid(axis="x", alpha=0.25)

    # Legend — place below the x-axis so it doesn't obscure any bars
    legend_elements = [
        Patch(fc=BENCHMARK_COLOR, alpha=0.75, label="Benchmark"),
        Patch(fc=MODEL_TYPE_COLORS["Process-based"], label="Process-based"),
        Patch(fc=MODEL_TYPE_COLORS["Conceptual"], label="Conceptual"),
        Patch(fc=MODEL_TYPE_COLORS["Data-driven"], label="Data-driven"),
        Patch(fc=MODEL_TYPE_COLORS["Ensemble"], label="Ensemble"),
    ]
    ax_bar.legend(handles=legend_elements, loc="upper center",
                  bbox_to_anchor=(0.55, -0.12), ncol=5, fontsize=8,
                  frameon=False)

    # ------------------------------------------------------------------
    # Panel (b): Grouped dot plot with cal/val pairs
    # ------------------------------------------------------------------
    group_names = list(BENCHMARK_GROUPS.keys())
    y_positions = []
    y_labels = []
    y_group_spans = []
    current_y = 0

    for gi, gname in enumerate(group_names):
        bms = [b for b in BENCHMARK_GROUPS[gname] if b in scores.index]
        if not bms:
            continue
        group_start = current_y
        for bm in bms:
            kge_v = scores.loc[bm, kge_col]
            kge_c = scores.loc[bm, kge_cal_col]
            kge_v = kge_v if np.isfinite(kge_v) else np.nan
            kge_c = kge_c if np.isfinite(kge_c) else np.nan

            # Cal marker (open circle)
            if np.isfinite(kge_c):
                ax_dot.scatter(kge_c, current_y, marker="o", s=50,
                               facecolors="none",
                               edgecolors=GROUP_COLORS.get(gname, BENCHMARK_COLOR),
                               linewidths=1.3, zorder=3)
            # Val marker (filled circle)
            if np.isfinite(kge_v):
                ax_dot.scatter(kge_v, current_y, marker="o", s=50,
                               facecolors=GROUP_COLORS.get(gname, BENCHMARK_COLOR),
                               edgecolors=GROUP_COLORS.get(gname, BENCHMARK_COLOR),
                               linewidths=1.3, zorder=3)
            # Line connecting cal to val
            if np.isfinite(kge_c) and np.isfinite(kge_v):
                ax_dot.plot([kge_c, kge_v], [current_y, current_y],
                            color=GROUP_COLORS.get(gname, BENCHMARK_COLOR),
                            lw=1.0, alpha=0.5, zorder=2)

            y_positions.append(current_y)
            y_labels.append(BENCHMARK_DISPLAY.get(bm, bm.replace("_", " ")))
            current_y += 1

        group_end = current_y - 1
        y_group_spans.append((gname, group_start, group_end))
        current_y += 0.7  # gap between groups

    ax_dot.set_yticks(y_positions)
    ax_dot.set_yticklabels(y_labels, fontsize=8.5)
    ax_dot.set_xlabel("KGE")
    ax_dot.set_title("(b) Benchmark scores by category", loc="left")
    ax_dot.axvline(0, color="black", lw=0.4, alpha=0.4)
    ax_dot.grid(axis="x", alpha=0.25)
    ax_dot.set_xlim(-0.55, 1.02)

    # Group labels on the right margin
    for gname, ystart, yend in y_group_spans:
        ymid = (ystart + yend) / 2
        ax_dot.text(1.05, ymid, gname.replace("\n", " "),
                    transform=ax_dot.get_yaxis_transform(),
                    fontsize=7.5, va="center", ha="left",
                    color=GROUP_COLORS.get(gname, "#555"),
                    fontweight="bold")

    # Legend for cal vs val markers — place below x-axis to match panel (a)
    dot_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markeredgecolor="#555", markersize=7, label="Validation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#555", markeredgewidth=1.3, markersize=7,
               label="Calibration"),
    ]
    ax_dot.legend(handles=dot_legend, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8,
                  frameon=False)

    # Invert y so groups read top-to-bottom
    ax_bar.invert_yaxis()
    ax_dot.invert_yaxis()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_benchmarking.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig_benchmarking")


# ===================================================================
# Figure 2 — Benchmark flow time series
# ===================================================================
def fig_benchmark_flows(flows: pd.DataFrame, scores: pd.DataFrame, output_dir: Path):
    """
    Two-panel time-series figure:
      (a) Evaluation period: observed, best seasonal benchmark, ensemble mean
      (b) Full record: selected benchmarks illustrating temporal resolution
    """
    if not HAS_MATPLOTLIB or flows is None:
        logger.warning("Skipping flow figure (missing data or matplotlib)")
        return

    # Load observed streamflow
    eval_dir = Path(
        "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data"
        "/domain_Bow_at_Banff_lumped_era5/evaluation"
    )
    input_path = eval_dir / "benchmark_input_data.csv"
    if not input_path.exists():
        logger.warning("benchmark_input_data.csv not found — skipping flow figure")
        return

    input_data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    if "streamflow" not in input_data.columns:
        logger.warning("No streamflow column in benchmark_input_data.csv")
        return
    obs = input_data["streamflow"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # ------------------------------------------------------------------
    # Panel (a): Evaluation period (2008-2009) with best seasonal benchmark
    # ------------------------------------------------------------------
    eval_start, eval_end = "2008-01-01", "2009-12-31"

    obs_eval = obs.loc[eval_start:eval_end]
    ax_top.plot(obs_eval.index, obs_eval.values, color="black", lw=1.3,
                ls="--", label="Observed", zorder=5)

    # Best seasonal benchmark = daily_median_flow (KGE_val = 0.804)
    best_bm_col = "bm_daily_median_flow"
    if best_bm_col in flows.columns:
        bm_eval = flows.loc[eval_start:eval_end, best_bm_col]
        ax_top.plot(bm_eval.index, bm_eval.values, color="#74a9cf", lw=1.1,
                    label="Daily median (KGE = 0.80)", zorder=4)

    # Seasonal envelope from the 4 seasonal benchmarks
    seasonal_cols = [f"bm_{b}" for b in BENCHMARK_GROUPS["Seasonal"]
                     if f"bm_{b}" in flows.columns]
    if seasonal_cols:
        sdata = flows.loc[eval_start:eval_end, seasonal_cols].values
        s_min = np.nanmin(sdata, axis=1)
        s_max = np.nanmax(sdata, axis=1)
        ax_top.fill_between(
            flows.loc[eval_start:eval_end].index, s_min, s_max,
            alpha=0.18, color="#74a9cf", label="Seasonal benchmark range",
            zorder=2,
        )

    # Mean flow reference — thin dashed line so it doesn't dominate
    if "bm_mean_flow" in flows.columns:
        mf_eval = flows.loc[eval_start:eval_end, "bm_mean_flow"]
        ax_top.plot(mf_eval.index, mf_eval.values, color="#e41a1c", lw=0.7,
                    ls=":", alpha=0.55, label="Mean flow", zorder=3)

    ax_top.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_top.set_title("(a) Evaluation period (2008\u20132009)", loc="left")
    ax_top.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.85,
                  edgecolor="none")
    ax_top.set_xlim(pd.Timestamp(eval_start), pd.Timestamp(eval_end))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ------------------------------------------------------------------
    # Panel (b): Full record showing temporal resolution hierarchy
    # ------------------------------------------------------------------
    representative = [
        ("bm_mean_flow",          "Mean flow",          "#e41a1c", ":",  0.7, 0.55),
        ("bm_monthly_mean_flow",  "Monthly mean",       "#377eb8", "-",  0.9, 0.8),
        ("bm_daily_mean_flow",    "Daily mean",         "#4daf4a", "-",  0.9, 0.8),
        ("bm_rainfall_runoff_ratio_to_monthly", "R-R ratio (monthly)", "#ff7f00", "-", 0.9, 0.8),
    ]

    ax_bot.plot(obs.index, obs.values, color="black", lw=1.2, ls="--",
                label="Observed", zorder=5)

    for col, label, color, ls, lw, alpha in representative:
        if col in flows.columns:
            ax_bot.plot(flows.index, flows[col].values, color=color, lw=lw,
                        ls=ls, alpha=alpha, label=label, zorder=3)

    ax_bot.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_bot.set_xlabel("Date")
    ax_bot.set_title("(b) Benchmark temporal resolution hierarchy (2004\u20132009)",
                     loc="left")
    ax_bot.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.85,
                  edgecolor="none")
    ax_bot.set_xlim(flows.index[0], flows.index[-1])
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())

    # Cap y-axis at a sensible maximum to avoid spike domination
    obs_p99 = np.nanpercentile(obs.values, 99.5)
    for ax in (ax_top, ax_bot):
        ax.set_ylim(bottom=0, top=max(obs_p99 * 1.3, 300))

    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_benchmark_flows.{fmt}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig_benchmark_flows")


# ===================================================================
# Main
# ===================================================================
def create_all_figures(data_dir: Path, output_dir: Path):
    """Generate all publication figures."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = data_dir / "domain_Bow_at_Banff_lumped_era5" / "evaluation"

    # Load benchmark scores
    scores_path = eval_dir / "benchmark_scores.csv"
    if not scores_path.exists():
        logger.error(f"Benchmark scores not found: {scores_path}")
        sys.exit(1)

    scores = pd.read_csv(scores_path, index_col=0)
    if "benchmarks" in scores.columns:
        scores = scores.set_index("benchmarks")
    logger.info(f"Loaded scores: {scores.shape}")

    # Load benchmark flows
    flows_path = eval_dir / "benchmark_flows.csv"
    flows = None
    if flows_path.exists():
        flows = pd.read_csv(flows_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded flows: {flows.shape}")

    # Generate figures
    fig_benchmarking(scores, output_dir)
    fig_benchmark_flows(flows, scores, output_dir)

    logger.info(f"\nAll figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for Section 4.5 benchmarking"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR
    create_all_figures(Path(args.data_dir), output_dir)


if __name__ == "__main__":
    main()
