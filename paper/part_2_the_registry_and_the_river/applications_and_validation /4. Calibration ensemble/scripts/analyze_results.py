#!/usr/bin/env python3
"""
Analyze results from the 4.4 Calibration Ensemble Study.

Generates comparison plots and summary statistics across optimization
algorithms, including convergence curves, performance metrics, parameter
distributions, and robustness analysis.

Output figures:
  fig1 - Algorithm performance comparison (KGE, NSE, RMSE bar charts)
  fig2 - Convergence curves (objective vs function evaluations)
  fig3 - Calibrated parameter distributions across algorithms
  fig4 - Calibration vs evaluation performance (generalization)
  fig5 - Robustness analysis (multi-seed box plots)
  fig6 - Algorithm family comparison (radar/spider chart)
  figS1 - Hydrograph comparison (calibration period)
  figS2 - Hydrograph comparison (evaluation period)

Usage:
    python analyze_results.py                    # Full analysis
    python analyze_results.py --data-dir /path   # Custom data dir
    python analyze_results.py --skip-hydro       # Skip hydrograph plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/pandas not available.")

# Paths
STUDY_DIR = Path(__file__).parent.parent
CONFIGS_DIR = STUDY_DIR / "configs"
RESULTS_DIR = STUDY_DIR / "results"

SYMFLUENCE_DATA_DIR = Path(
    os.getenv("SYMFLUENCE_DATA_DIR",
              "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
)
DOMAIN_NAME = "Bow_at_Banff_lumped_era5"

# =============================================================================
# Algorithm metadata
# =============================================================================
ALGORITHMS = {
    "dds":           {"label": "DDS",         "family": "Sampling",      "color": "#1f77b4", "marker": "o"},
    "sceua":         {"label": "SCE-UA",      "family": "Evolutionary",  "color": "#ff7f0e", "marker": "s"},
    "de":            {"label": "DE",          "family": "Evolutionary",  "color": "#2ca02c", "marker": "^"},
    "pso":           {"label": "PSO",         "family": "Evolutionary",  "color": "#d62728", "marker": "v"},
    "ga":            {"label": "GA",          "family": "Evolutionary",  "color": "#9467bd", "marker": "D"},
    "cmaes":         {"label": "CMA-ES",      "family": "Evolutionary",  "color": "#8c564b", "marker": "P"},
    "adam":          {"label": "ADAM",         "family": "Gradient",      "color": "#e377c2", "marker": "*"},
    "lbfgs":         {"label": "L-BFGS",      "family": "Gradient",      "color": "#7f7f7f", "marker": "X"},
    "nelder_mead":   {"label": "Nelder-Mead", "family": "Direct Search", "color": "#bcbd22", "marker": "p"},
    "sa":            {"label": "SA",          "family": "Stochastic",    "color": "#17becf", "marker": "h"},
    "basin_hopping": {"label": "Basin Hop.",  "family": "Stochastic",    "color": "#aec7e8", "marker": "H"},
    "bayesian_opt":  {"label": "Bayes. Opt.", "family": "Surrogate",     "color": "#ffbb78", "marker": "d"},
}

FAMILY_COLORS = {
    "Sampling":      "#1f77b4",
    "Evolutionary":  "#2ca02c",
    "Gradient":      "#e377c2",
    "Direct Search": "#bcbd22",
    "Stochastic":    "#17becf",
    "Surrogate":     "#ffbb78",
}

# HBV parameter names
HBV_PARAMS = [
    "tt", "cfmax", "sfcf", "cfr", "cwh",
    "fc", "lp", "beta",
    "k0", "k1", "k2", "uzl", "perc", "maxbas",
]

HBV_PARAM_LABELS = {
    "tt": "TT (C)", "cfmax": "CFMAX\n(mm/C/d)", "sfcf": "SFCF",
    "cfr": "CFR", "cwh": "CWH", "fc": "FC (mm)", "lp": "LP",
    "beta": "Beta", "k0": "K0 (1/d)", "k1": "K1 (1/d)",
    "k2": "K2 (1/d)", "uzl": "UZL (mm)", "perc": "PERC\n(mm/d)",
    "maxbas": "MAXBAS (d)",
}

# Multipliers to convert iteration number to approximate function evaluations.
# Population-based methods evaluate pop_size candidates per iteration.
# SA evaluates steps_per_temp per temperature iteration.
# Basin Hopping runs ~local_steps local evals per hop.
# Gradient methods use ~2 evals/iter (forward + gradient via autodiff).
EVAL_MULTIPLIERS = {
    "dds": 1, "sceua": 145, "de": 20, "pso": 20, "ga": 20, "cmaes": 20,
    "adam": 2, "lbfgs": 8, "nelder_mead": 1, "sa": 10,
    "basin_hopping": 50, "bayesian_opt": 1,
}


# =============================================================================
# Data loading
# =============================================================================
def find_optimization_dir(data_dir: Path, algo_key: str,
                          seed: int = 42) -> Optional[Path]:
    """Locate the optimization output directory for an algorithm run.

    SYMFLUENCE names dirs as: {algorithm}_{experiment_id}
    e.g., dds_cal_ensemble_dds, pso_cal_ensemble_pso
    """
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    experiment_id = f"cal_ensemble_{algo_key}{seed_suffix}"

    # Map algo keys to SYMFLUENCE algorithm directory prefixes
    algo_prefix_map = {
        "dds": "dds", "sceua": "sce-ua", "de": "de", "pso": "pso",
        "ga": "ga", "cmaes": "cma-es", "adam": "adam", "lbfgs": "lbfgs",
        "nelder_mead": "nelder-mead", "sa": "simulated_annealing",
        "basin_hopping": "basin-hopping", "bayesian_opt": "bayesian_opt",
    }

    opt_base = data_dir / f"domain_{DOMAIN_NAME}" / "optimization" / "HBV"
    if not opt_base.exists():
        # Try without domain_ prefix
        opt_base = data_dir / DOMAIN_NAME / "optimization" / "HBV"

    if not opt_base.exists():
        return None

    # Try exact match first: {prefix}_{experiment_id}
    prefix = algo_prefix_map.get(algo_key, algo_key)
    exact = opt_base / f"{prefix}_{experiment_id}"
    if exact.exists():
        return exact

    # Try scanning directory for matches
    try:
        for entry in sorted(opt_base.iterdir()):
            if entry.is_dir() and experiment_id in entry.name:
                return entry
    except OSError:
        pass

    return None


def load_convergence_data(opt_dir: Path) -> Optional[pd.DataFrame]:
    """Load convergence history from optimization output."""
    # Try common convergence log filenames
    for pattern in ["*convergence*.csv", "*history*.csv", "*iteration*.csv",
                    "*calibration_log*.csv"]:
        files = list(opt_dir.glob(pattern))
        if files:
            try:
                df = pd.read_csv(files[0])
                return df
            except Exception:
                continue

    # Try JSON format
    for pattern in ["*convergence*.json", "*history*.json"]:
        files = list(opt_dir.glob(pattern))
        if files:
            try:
                with open(files[0]) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict) and "iterations" in data:
                    return pd.DataFrame(data["iterations"])
            except Exception:
                continue

    return None


def load_best_params(opt_dir: Path) -> Optional[Dict]:
    """Load best calibrated parameters."""
    for pattern in ["*best_params*.json", "*best*.json", "*final*.json"]:
        files = list(opt_dir.glob(pattern))
        if files:
            try:
                with open(files[0]) as f:
                    data = json.load(f)
                return data.get("best_params", data)
            except Exception:
                continue
    return None


def load_final_metrics(opt_dir: Path) -> Optional[Dict]:
    """Load final evaluation metrics."""
    for pattern in ["*final_evaluation*.json", "*evaluation*.json", "*metrics*.json"]:
        files = list(opt_dir.glob(pattern))
        if files:
            try:
                with open(files[0]) as f:
                    data = json.load(f)
                return data
            except Exception:
                continue
    return None


def load_simulated_streamflow(opt_dir: Path) -> Optional[Dict]:
    """Load simulated streamflow from final evaluation."""
    try:
        import xarray as xr
    except ImportError:
        return None

    eval_dir = opt_dir / "final_evaluation"
    if not eval_dir.exists():
        eval_dir = opt_dir

    for pattern in ["*.nc", "*output*.nc"]:
        files = list(eval_dir.glob(pattern))
        if files:
            try:
                ds = xr.open_dataset(files[0])
                for var in ["Q_sim", "streamflow", "discharge", "q_sim"]:
                    if var in ds.data_vars:
                        return {
                            "time": pd.to_datetime(ds["time"].values),
                            "flow": ds[var].values.flatten(),
                        }
            except Exception:
                continue
    return None


def load_observed_streamflow(data_dir: Path) -> Optional[Dict]:
    """Load observed streamflow."""
    # Try with and without domain_ prefix
    candidates = [
        data_dir / f"domain_{DOMAIN_NAME}" / "observations" / "streamflow" / "preprocessed",
        data_dir / DOMAIN_NAME / "observations" / "streamflow" / "preprocessed",
        data_dir / f"domain_{DOMAIN_NAME}" / "observations" / "streamflow",
        data_dir / DOMAIN_NAME / "observations" / "streamflow",
    ]

    for obs_dir in candidates:
        if not obs_dir.exists():
            continue
        files = list(obs_dir.glob("*.csv"))
        if not files:
            continue
        try:
            df = pd.read_csv(files[0], parse_dates=[0])
            df = df.set_index(df.columns[0])
            flow_col = [c for c in df.columns if "flow" in c.lower()
                        or "discharge" in c.lower() or "q" in c.lower()]
            col = flow_col[0] if flow_col else df.columns[0]
            # Resample to daily if data is sub-daily (e.g., hourly)
            series = df[col]
            if hasattr(series.index, 'freq') and series.index.freq is not None:
                freq = series.index.freq
            else:
                # Infer from spacing
                freq = pd.infer_freq(series.index[:100])
            if freq and freq not in ('D', 'B', 'W', 'M', 'MS'):
                series = series.resample('D').mean()
            return {
                "time": series.index,
                "flow": series.values,
            }
        except Exception:
            continue
    return None


# =============================================================================
# Plotting functions
# =============================================================================
def _sort_algos_by_cal_kge(algos, metrics):
    """Return algorithm keys sorted by calibration KGE (descending)."""
    return sorted(algos, key=lambda a: metrics.get(a, {}).get("cal_kge", -999),
                  reverse=True)


def plot_algorithm_performance(
    metrics: Dict[str, Dict],
    output_path: Path,
):
    """Fig 1: Bar chart comparing KGE, NSE, RMSE across algorithms."""
    if not HAS_PLOTTING or not metrics:
        return

    algos = _sort_algos_by_cal_kge(
        [k for k in ALGORITHMS if k in metrics], metrics)
    if not algos:
        print("  No algorithm metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Per-metric y-axis limits: (low, high) — zoomed to show differences
    metric_configs = [
        ("kge", "KGE", "Kling-Gupta Efficiency", axes[0, 0], (0.55, 0.82)),
        ("nse", "NSE", "Nash-Sutcliffe Efficiency", axes[0, 1], (0.50, 0.88)),
        ("rmse", "RMSE (m3/s)", "Root Mean Square Error", axes[1, 0], None),
        ("pbias", "PBIAS (%)", "Percent Bias", axes[1, 1], None),
    ]

    for key, ylabel, title, ax, ylims in metric_configs:
        cal_vals = []
        eval_vals = []
        labels = []
        colors = []

        for algo in algos:
            m = metrics[algo]
            cal_vals.append(m.get(f"cal_{key}", np.nan))
            eval_vals.append(m.get(f"eval_{key}", np.nan))
            labels.append(ALGORITHMS[algo]["label"])
            colors.append(ALGORITHMS[algo]["color"])

        x = np.arange(len(algos))
        width = 0.35

        bars_cal = ax.bar(x - width / 2, cal_vals, width, label="Calibration",
                          color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
        bars_eval = ax.bar(x + width / 2, eval_vals, width, label="Evaluation",
                           color=colors, edgecolor="black", linewidth=0.5, alpha=0.5,
                           hatch="//")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

        if ylims is not None:
            ax.set_ylim(*ylims)

        # Add value labels
        for bar, val in zip(bars_cal, cal_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Calibration Ensemble: Algorithm Performance Comparison\n"
                 "HBV Model, Bow at Banff, ERA5 Forcing",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _get_convergence_xy(df: pd.DataFrame, algo_key: str):
    """Extract (function_evaluations, best_objective) from a convergence df."""
    # Determine raw x column
    x_col = next(
        (c for c in ["function_evaluations", "evaluations", "iteration", "step"]
         if c in df.columns),
        df.columns[0],
    )
    y_col = next(
        (c for c in ["best_objective", "best_value", "best_kge",
                      "objective", "value", "score"]
         if c in df.columns),
        df.columns[1] if len(df.columns) > 1 else df.columns[0],
    )

    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)

    # Convert iterations to approximate function evaluations
    if x_col in ("iteration", "step"):
        multiplier = EVAL_MULTIPLIERS.get(algo_key, 1)
        x = x * multiplier

    # Cumulative best (maximisation)
    if len(y) > 0:
        y = np.maximum.accumulate(y)

    return x, y


def plot_convergence_curves(
    convergence: Dict[str, pd.DataFrame],
    output_path: Path,
):
    """Fig 2: Convergence curves (best objective vs function evaluations)."""
    if not HAS_PLOTTING or not convergence:
        return

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6),
                                            gridspec_kw={"width_ratios": [2, 1]})

    for algo_key, df in convergence.items():
        if algo_key not in ALGORITHMS:
            continue
        meta = ALGORITHMS[algo_key]

        x, y_best = _get_convergence_xy(df, algo_key)
        if len(y_best) == 0:
            continue

        ax_main.plot(x, y_best, color=meta["color"], linewidth=1.5,
                     label=meta["label"], alpha=0.85)
        ax_zoom.plot(x, y_best, color=meta["color"], linewidth=1.5,
                     alpha=0.85)

    ax_main.set_xlabel("Function Evaluations", fontsize=11)
    ax_main.set_ylabel("Best Objective (KGE)", fontsize=11)
    ax_main.set_title("(a) Full Convergence History", fontsize=12, fontweight="bold")
    ax_main.legend(loc="lower right", fontsize=8, ncol=2)
    ax_main.grid(True, alpha=0.2)
    ax_main.set_ylim(bottom=0.3)

    # Zoom: early convergence (first 500 evaluations)
    ax_zoom.set_xlabel("Function Evaluations", fontsize=11)
    ax_zoom.set_title("(b) Early Convergence (0\u2013500 evals)", fontsize=12,
                      fontweight="bold")
    ax_zoom.set_xlim(-10, 520)
    ax_zoom.grid(True, alpha=0.2)
    ax_zoom.legend(loc="lower right", fontsize=7, ncol=2)

    plt.suptitle("Calibration Ensemble: Convergence Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_parameter_distributions(
    all_params: Dict[str, Dict],
    output_path: Path,
):
    """Fig 3: Calibrated parameter values across algorithms (heatmap + strip)."""
    if not HAS_PLOTTING or not all_params:
        return

    algos = [k for k in ALGORITHMS if k in all_params]
    if len(algos) < 2:
        print("  Need at least 2 algorithms for parameter comparison")
        return

    # Build parameter matrix
    params_present = []
    for p in HBV_PARAMS:
        if any(p in all_params.get(a, {}) for a in algos):
            params_present.append(p)

    n_params = len(params_present)
    n_algos = len(algos)

    fig, ax = plt.subplots(figsize=(max(12, n_params * 0.9), max(6, n_algos * 0.5)))

    # Create normalized heatmap data
    data = np.full((n_algos, n_params), np.nan)
    for i, algo in enumerate(algos):
        for j, param in enumerate(params_present):
            if param in all_params.get(algo, {}):
                val = all_params[algo][param]
                if isinstance(val, list):
                    val = val[0]
                data[i, j] = val

    # Normalize columns to [0, 1] for color mapping
    data_norm = np.copy(data)
    for j in range(n_params):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0 and valid.max() > valid.min():
            data_norm[:, j] = (col - valid.min()) / (valid.max() - valid.min())
        else:
            data_norm[:, j] = 0.5

    im = ax.imshow(data_norm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

    # Add text annotations with actual values
    for i in range(n_algos):
        for j in range(n_params):
            val = data[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                text_color = "white" if data_norm[i, j] > 0.7 or data_norm[i, j] < 0.3 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_params))
    ax.set_xticklabels([HBV_PARAM_LABELS.get(p, p) for p in params_present],
                       fontsize=9, ha="center")
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels([ALGORITHMS[a]["label"] for a in algos], fontsize=10)

    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized Parameter Value", fontsize=10)

    # Family grouping: draw horizontal separators between families
    prev_fam = None
    for i, algo in enumerate(algos):
        fam = ALGORITHMS[algo]["family"]
        if prev_fam is not None and fam != prev_fam:
            ax.axhline(y=i - 0.5, color="white", linewidth=2)
        prev_fam = fam

    ax.set_title("Calibrated HBV Parameters Across Optimization Algorithms\n"
                 "(values shown, color = normalized within each parameter)",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_generalization(
    metrics: Dict[str, Dict],
    output_path: Path,
):
    """Fig 4: Calibration vs Evaluation KGE scatter plot."""
    if not HAS_PLOTTING or not metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for algo_key, m in metrics.items():
        if algo_key not in ALGORITHMS:
            continue
        meta = ALGORITHMS[algo_key]
        cal_kge = m.get("cal_kge", np.nan)
        eval_kge = m.get("eval_kge", np.nan)

        if np.isnan(cal_kge) or np.isnan(eval_kge):
            continue

        ax.scatter(cal_kge, eval_kge, c=meta["color"], s=200,
                   marker=meta["marker"], edgecolors="black", linewidth=1,
                   zorder=5, label=meta["label"])

    # 1:1 line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="1:1 line")

    ax.set_xlabel("Calibration KGE", fontsize=12)
    ax.set_ylabel("Evaluation KGE", fontsize=12)
    ax.set_title("Algorithm Generalization:\nCalibration vs Evaluation Performance",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_robustness(
    multi_seed_metrics: Dict[str, List[Dict]],
    output_path: Path,
):
    """Fig 5: Multi-seed robustness box plots."""
    if not HAS_PLOTTING or not multi_seed_metrics:
        return

    algos = [k for k in ALGORITHMS if k in multi_seed_metrics
             and len(multi_seed_metrics[k]) > 1]
    if not algos:
        print("  No multi-seed results for robustness plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metric_configs = [
        ("cal_kge", "Calibration KGE", axes[0]),
        ("eval_kge", "Evaluation KGE", axes[1]),
        ("kge_degradation", "KGE Degradation\n(Cal - Eval)", axes[2]),
    ]

    for key, ylabel, ax in metric_configs:
        box_data = []
        labels = []
        colors = []

        for algo in algos:
            if key == "kge_degradation":
                vals = [
                    m.get("cal_kge", 0) - m.get("eval_kge", 0)
                    for m in multi_seed_metrics[algo]
                    if not np.isnan(m.get("cal_kge", np.nan))
                    and not np.isnan(m.get("eval_kge", np.nan))
                ]
            else:
                vals = [
                    m.get(key, np.nan)
                    for m in multi_seed_metrics[algo]
                    if not np.isnan(m.get(key, np.nan))
                ]

            if vals:
                box_data.append(vals)
                labels.append(ALGORITHMS[algo]["label"])
                colors.append(ALGORITHMS[algo]["color"])

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="white",
                                         markeredgecolor="black", markersize=6))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(axis="x", rotation=45)

        if key == "kge_degradation":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Algorithm Robustness Across Random Seeds\n"
                 "HBV Model, Bow at Banff",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_family_comparison(
    metrics: Dict[str, Dict],
    output_path: Path,
):
    """Fig 6: Algorithm family comparison — mean KGE with individual points."""
    if not HAS_PLOTTING or not metrics:
        return

    # Aggregate by family
    family_metrics = {}
    for algo_key, m in metrics.items():
        if algo_key not in ALGORITHMS:
            continue
        fam = ALGORITHMS[algo_key]["family"]
        if fam not in family_metrics:
            family_metrics[fam] = []
        family_metrics[fam].append((algo_key, m))

    families = list(family_metrics.keys())
    if len(families) < 2:
        return

    criteria = [
        ("cal_kge", "Calibration KGE"),
        ("eval_kge", "Evaluation KGE"),
        ("degradation", "KGE Degradation\n(Cal \u2212 Eval)"),
    ]

    fig, axes = plt.subplots(1, len(criteria), figsize=(15, 5.5))

    for idx, (key, label) in enumerate(criteria):
        ax = axes[idx]

        for i, fam in enumerate(families):
            fam_color = FAMILY_COLORS.get(fam, "#888888")
            pts = []
            for algo_key, m in family_metrics[fam]:
                if key == "degradation":
                    v = m.get("cal_kge", np.nan) - m.get("eval_kge", np.nan)
                else:
                    v = m.get(key, np.nan)
                if not np.isnan(v):
                    pts.append(v)
                    # Plot individual algorithm points
                    ax.scatter(i, v, color=ALGORITHMS[algo_key]["color"],
                               s=60, marker=ALGORITHMS[algo_key]["marker"],
                               edgecolors="black", linewidth=0.5, zorder=5)

            # Family mean bar (semi-transparent behind points)
            if pts:
                mean_v = np.mean(pts)
                ax.bar(i, mean_v, color=fam_color, alpha=0.3,
                       edgecolor="black", linewidth=0.5, width=0.6)
                ax.text(i, max(pts) + 0.005, f"{mean_v:.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

        # Zoom y-axis to data range with padding
        if key == "degradation":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        elif key in ("cal_kge", "eval_kge"):
            ax.set_ylim(0.55, 0.82)

    plt.suptitle("Algorithm Family Comparison\n"
                 "(bars = family mean, markers = individual algorithms)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_hydrograph(
    observed: Optional[Dict],
    simulations: Dict[str, Dict],
    output_path: Path,
    period: str = "calibration",
    date_range: Tuple = None,
    metrics: Dict[str, Dict] = None,
):
    """FigS1/S2: Hydrograph comparison across algorithms."""
    if not HAS_PLOTTING or not simulations:
        return

    fig, (ax_flow, ax_diff) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    t_min = pd.Timestamp(date_range[0]) if date_range else None
    t_max = pd.Timestamp(date_range[1]) if date_range else None

    # Prepare observed (resample to daily if needed)
    obs_s = None
    if observed is not None:
        obs_s = pd.Series(observed["flow"], index=pd.DatetimeIndex(observed["time"]))
        if t_min:
            obs_s = obs_s.loc[t_min:t_max]
        obs_s = obs_s.dropna()

    # Sort algorithms by Cal KGE so best are plotted on top
    algo_keys = [k for k in ALGORITHMS if k in simulations]
    if metrics:
        algo_keys = sorted(algo_keys,
                           key=lambda a: metrics.get(a, {}).get("cal_kge", -999))

    # Plot simulations (worst first, best on top)
    for algo_key in algo_keys:
        sim = simulations[algo_key]
        meta = ALGORITHMS[algo_key]
        sim_s = pd.Series(sim["flow"], index=pd.DatetimeIndex(sim["time"]))
        if t_min:
            sim_s = sim_s.loc[t_min:t_max]
        if len(sim_s) == 0:
            continue

        ax_flow.plot(sim_s.index, sim_s.values, color=meta["color"],
                     linewidth=0.9, label=meta["label"], alpha=0.7)

        # Residuals (align on daily timestamps)
        if obs_s is not None and len(obs_s) > 0:
            common = obs_s.index.intersection(sim_s.index)
            if len(common) > 0:
                residual = sim_s.loc[common] - obs_s.loc[common]
                ax_diff.plot(common, residual.values, color=meta["color"],
                             linewidth=0.7, alpha=0.6)

    # Plot observed last so it's on top
    if obs_s is not None and len(obs_s) > 0:
        ax_flow.plot(obs_s.index, obs_s.values, "k-", linewidth=2.0,
                     label="Observed", zorder=10)

    ax_flow.set_ylabel("Streamflow (m\u00b3/s)", fontsize=11)
    ax_flow.set_title(f"Hydrograph Comparison \u2014 {period.title()} Period\n"
                      f"Bow at Banff (05BB001)", fontsize=13, fontweight="bold")
    ax_flow.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)
    ax_flow.grid(True, alpha=0.2)

    ax_diff.axhline(y=0, color="black", linewidth=0.5)
    ax_diff.set_ylabel("Residual (m\u00b3/s)", fontsize=11)
    ax_diff.set_xlabel("Date", fontsize=11)
    ax_diff.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Summary table generation
# =============================================================================
def generate_summary_table(
    metrics: Dict[str, Dict],
    output_path: Path,
):
    """Generate CSV summary of all algorithm performance metrics."""
    rows = []
    for algo_key in ALGORITHMS:
        if algo_key not in metrics:
            continue
        m = metrics[algo_key]
        meta = ALGORITHMS[algo_key]
        row = {
            "Algorithm": meta["label"],
            "Family": meta["family"],
            "Cal_KGE": m.get("cal_kge", np.nan),
            "Cal_NSE": m.get("cal_nse", np.nan),
            "Cal_RMSE": m.get("cal_rmse", np.nan),
            "Cal_PBIAS": m.get("cal_pbias", np.nan),
            "Eval_KGE": m.get("eval_kge", np.nan),
            "Eval_NSE": m.get("eval_nse", np.nan),
            "Eval_RMSE": m.get("eval_rmse", np.nan),
            "Eval_PBIAS": m.get("eval_pbias", np.nan),
            "KGE_Degradation": (m.get("cal_kge", np.nan) - m.get("eval_kge", np.nan)),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows).round(4)
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name}")
        print("\n  Performance Summary:")
        print(df[["Algorithm", "Family", "Cal_KGE", "Eval_KGE",
                   "KGE_Degradation"]].to_string(index=False))


# =============================================================================
# Main analysis
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze Calibration Ensemble Study results"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=RESULTS_DIR,
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--data-dir", "-d", type=Path, default=SYMFLUENCE_DATA_DIR,
        help="SYMFLUENCE data directory",
    )
    parser.add_argument(
        "--skip-hydro", action="store_true",
        help="Skip hydrograph generation (faster)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds to check for robustness analysis",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Calibration Ensemble Study - Results Analysis")
    print("=" * 60)

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load single-seed results (Part 1) ---
    print("\nLoading algorithm results...")
    metrics = {}
    convergence = {}
    all_params = {}
    simulations = {}

    for algo_key, meta in ALGORITHMS.items():
        opt_dir = find_optimization_dir(args.data_dir, algo_key)
        if opt_dir is None:
            print(f"  {meta['label']}: Not found")
            continue

        print(f"  {meta['label']}: Found at {opt_dir.name}")

        # Load metrics
        final = load_final_metrics(opt_dir)
        if final is not None:
            cm = final.get("calibration_metrics", {})
            em = final.get("evaluation_metrics", {})
            # SYMFLUENCE uses Calib_/Eval_ prefixed keys
            metrics[algo_key] = {
                "cal_kge": float(cm.get("Calib_KGE", cm.get("KGE", np.nan))),
                "cal_nse": float(cm.get("Calib_NSE", cm.get("NSE", np.nan))),
                "cal_rmse": float(cm.get("Calib_RMSE", cm.get("RMSE", np.nan))),
                "cal_pbias": float(cm.get("Calib_PBIAS", cm.get("PBIAS", np.nan))),
                "eval_kge": float(em.get("Eval_KGE", em.get("KGE", np.nan))),
                "eval_nse": float(em.get("Eval_NSE", em.get("NSE", np.nan))),
                "eval_rmse": float(em.get("Eval_RMSE", em.get("RMSE", np.nan))),
                "eval_pbias": float(em.get("Eval_PBIAS", em.get("PBIAS", np.nan))),
            }
            print(f"    Cal KGE: {metrics[algo_key]['cal_kge']:.3f}, "
                  f"Eval KGE: {metrics[algo_key]['eval_kge']:.3f}")

        # Load convergence
        conv = load_convergence_data(opt_dir)
        if conv is not None:
            convergence[algo_key] = conv
            print(f"    Convergence: {len(conv)} records")

        # Load parameters
        params = load_best_params(opt_dir)
        if params is not None:
            all_params[algo_key] = params

        # Load streamflow
        if not args.skip_hydro:
            sim = load_simulated_streamflow(opt_dir)
            if sim is not None:
                simulations[algo_key] = sim

    print(f"\n  Algorithms with results: {len(metrics)}/{len(ALGORITHMS)}")

    # --- Load multi-seed results (Part 2) ---
    print("\nLoading multi-seed results...")
    multi_seed_metrics = {}
    seeds = [42 + i * 1000 for i in range(args.seeds)]

    for algo_key in ["dds", "sceua", "de", "pso", "cmaes", "adam"]:
        seed_results = []
        for seed in seeds:
            opt_dir = find_optimization_dir(args.data_dir, algo_key, seed=seed)
            if opt_dir is not None:
                final = load_final_metrics(opt_dir)
                if final is not None:
                    cm = final.get("calibration_metrics", {})
                    em = final.get("evaluation_metrics", {})
                    seed_results.append({
                        "seed": seed,
                        "cal_kge": float(cm.get("Calib_KGE", cm.get("KGE", np.nan))),
                        "eval_kge": float(em.get("Eval_KGE", em.get("KGE", np.nan))),
                    })
        if seed_results:
            multi_seed_metrics[algo_key] = seed_results
            print(f"  {ALGORITHMS[algo_key]['label']}: {len(seed_results)} seeds")

    # --- Generate plots ---
    print("\nGenerating figures...")

    if metrics:
        plot_algorithm_performance(metrics, plots_dir / "fig1_algorithm_performance.png")
        plot_generalization(metrics, plots_dir / "fig4_generalization.png")
        plot_family_comparison(metrics, plots_dir / "fig6_family_comparison.png")
        generate_summary_table(metrics, args.output_dir / "performance_summary.csv")

    if convergence:
        plot_convergence_curves(convergence, plots_dir / "fig2_convergence.png")

    if all_params:
        plot_parameter_distributions(all_params, plots_dir / "fig3_parameters.png")

    if multi_seed_metrics:
        plot_robustness(multi_seed_metrics, plots_dir / "fig5_robustness.png")

    # Hydrographs
    if not args.skip_hydro and simulations:
        observed = load_observed_streamflow(args.data_dir)
        if observed is not None:
            print(f"  Observed streamflow: {len(observed['flow'])} records")
        else:
            print("  WARNING: Could not load observed streamflow")
        plot_hydrograph(
            observed, simulations,
            plots_dir / "figS1_hydrograph_calibration.png",
            period="calibration",
            date_range=("2004-01-01", "2007-12-31"),
            metrics=metrics,
        )
        plot_hydrograph(
            observed, simulations,
            plots_dir / "figS2_hydrograph_evaluation.png",
            period="evaluation",
            date_range=("2008-01-01", "2009-12-31"),
            metrics=metrics,
        )

    # --- Parameter divergence table ---
    if all_params:
        rows = []
        for algo_key in ALGORITHMS:
            if algo_key not in all_params:
                continue
            row = {"Algorithm": ALGORITHMS[algo_key]["label"],
                   "Family": ALGORITHMS[algo_key]["family"]}
            for p in HBV_PARAMS:
                val = all_params[algo_key].get(p, np.nan)
                if isinstance(val, list):
                    val = val[0]
                row[p] = val
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows).round(4)
            df.to_csv(args.output_dir / "parameter_comparison.csv", index=False)
            print("  Saved: parameter_comparison.csv")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
