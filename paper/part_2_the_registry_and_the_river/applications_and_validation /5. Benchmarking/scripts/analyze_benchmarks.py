#!/usr/bin/env python3
"""
Benchmark Analysis and Comparison Script for SYMFLUENCE Paper Section 4.5

This script loads HydroBM benchmark results and compares them against the
multi-model ensemble performance from Section 4.2. It produces:

1. A benchmark performance summary table
2. Comparison of model KGE vs benchmark KGE (are models better than benchmarks?)
3. Grouped analysis (time-invariant, time-variant, rainfall-runoff, Schaefli-Gupta)
4. Publication-ready figures

Usage:
    python analyze_benchmarks.py [--data-dir DIR] [--output-dir DIR]

Output:
    - benchmark_summary.csv: All benchmark scores
    - benchmark_vs_models.csv: Side-by-side comparison
    - analysis_report.txt: Narrative summary
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

# Directories
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
ENSEMBLE_DIR = BASE_DIR.parent / "2. Model ensemble"

# Section 4.2 model performance (evaluation period KGE from Table 14 in the paper)
# These serve as the authoritative reference values for the 6 published models
SECTION_4_2_MODELS = {
    "SUMMA":  {"Cal_KGE": 0.90, "Eval_KGE": 0.88, "type": "Process-based", "n_params": 11},
    "FUSE":   {"Cal_KGE": 0.90, "Eval_KGE": 0.88, "type": "Conceptual",    "n_params": 13},
    "GR4J":   {"Cal_KGE": 0.92, "Eval_KGE": 0.79, "type": "Conceptual",    "n_params": 4},
    "HBV":    {"Cal_KGE": 0.74, "Eval_KGE": 0.70, "type": "Conceptual",    "n_params": 15},
    "HYPE":   {"Cal_KGE": 0.87, "Eval_KGE": 0.81, "type": "Conceptual",    "n_params": 10},
    "LSTM":   {"Cal_KGE": 0.97, "Eval_KGE": 0.88, "type": "Data-driven",   "n_params": None},
}

# Ensemble performance from Section 4.2.3
# NOTE: These are the KGE of the *combined ensemble output hydrograph* (mean/median
# of all model traces evaluated against observations), NOT the arithmetic mean of
# individual model KGE values. The ensemble hydrograph smooths individual model errors,
# which is why its KGE exceeds even the best individual model.
ENSEMBLE_MEAN_KGE = 0.94
ENSEMBLE_MEDIAN_KGE = 0.92

# Threshold below which a model's KGE is considered pathological (broken run)
PATHOLOGICAL_KGE_THRESHOLD = -10.0

# Benchmark groupings for analysis
BENCHMARK_GROUPS = {
    "Time-invariant": [
        "mean_flow",
        "median_flow",
        "annual_mean_flow",
        "annual_median_flow",
    ],
    "Time-variant (seasonal)": [
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
    ],
    "Rainfall-runoff (long-term)": [
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
        "rainfall_runoff_ratio_to_daily",
        "rainfall_runoff_ratio_to_timestep",
    ],
    "Rainfall-runoff (short-term)": [
        "monthly_rainfall_runoff_ratio_to_monthly",
        "monthly_rainfall_runoff_ratio_to_daily",
        "monthly_rainfall_runoff_ratio_to_timestep",
    ],
    "Schaefli & Gupta (2007)": [
        "scaled_precipitation_benchmark",
        "adjusted_precipitation_benchmark",
        "adjusted_smoothed_precipitation_benchmark",
    ],
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("benchmark_analysis")


def find_kge_val_column(df: pd.DataFrame) -> Optional[str]:
    """Find the validation KGE column in a DataFrame. Prefers 'kge_val' over 'kge'."""
    for col in df.columns:
        if "kge" in col.lower() and "val" in col.lower():
            return col
    for col in df.columns:
        if "kge" in col.lower() and "cal" not in col.lower():
            return col
    for col in df.columns:
        if "kge" in col.lower():
            return col
    return None


def load_benchmark_scores(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load benchmark scores from HydroBM output.

    Args:
        data_dir: SYMFLUENCE data directory

    Returns:
        DataFrame with benchmark names as index and metrics as columns
    """
    eval_dir = data_dir / "domain_Bow_at_Banff_lumped_era5" / "evaluation"
    scores_path = eval_dir / "benchmark_scores.csv"

    if not scores_path.exists():
        logger.error(f"Benchmark scores not found: {scores_path}")
        logger.info("Run 'python run_benchmarking.py' first to generate benchmark results.")
        return None

    scores = pd.read_csv(scores_path, index_col=0)

    # If benchmark names are in a 'benchmarks' column rather than the index, reindex
    if "benchmarks" in scores.columns:
        scores = scores.set_index("benchmarks")

    logger.info(f"Loaded benchmark scores: {len(scores)} benchmarks, {len(scores.columns)} metrics")
    return scores


def load_benchmark_flows(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load benchmark flow time series."""
    eval_dir = data_dir / "domain_Bow_at_Banff_lumped_era5" / "evaluation"
    flows_path = eval_dir / "benchmark_flows.csv"

    if not flows_path.exists():
        logger.warning(f"Benchmark flows not found: {flows_path}")
        return None

    flows = pd.read_csv(flows_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded benchmark flows: {len(flows)} timesteps, {len(flows.columns)} benchmarks")
    return flows


def load_benchmark_metadata(data_dir: Path) -> Optional[Dict]:
    """Load benchmark metadata."""
    eval_dir = data_dir / "domain_Bow_at_Banff_lumped_era5" / "evaluation"
    meta_path = eval_dir / "benchmark_metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        return json.load(f)


def load_ensemble_metrics(ensemble_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the most recent ensemble metrics from Section 4.2 analysis.

    Args:
        ensemble_dir: Path to Section 4.2 directory

    Returns:
        DataFrame with model metrics or None
    """
    analysis_dir = ensemble_dir / "analysis"
    if not analysis_dir.exists():
        logger.warning(f"Ensemble analysis directory not found: {analysis_dir}")
        return None

    metric_files = sorted(analysis_dir.glob("ensemble_metrics_*.csv"), reverse=True)
    if not metric_files:
        logger.warning("No ensemble_metrics CSV files found in Section 4.2 analysis/")
        return None

    metrics = pd.read_csv(metric_files[0], index_col=0)
    logger.info(f"Loaded ensemble metrics from: {metric_files[0].name}")
    return metrics


def build_model_kge_dict(ensemble_metrics: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Build the model KGE dictionary for comparison.

    Strategy: Start with the 6 Table 14 models as the authoritative set.
    For each, check if live ensemble metrics provide a valid update. Exclude
    models with pathological KGE (indicating broken runs).

    Args:
        ensemble_metrics: Live ensemble metrics DataFrame (or None)

    Returns:
        Dictionary mapping model name -> evaluation KGE
    """
    model_kge = {}
    sources = {}

    for model_name, table14 in SECTION_4_2_MODELS.items():
        kge_val = table14["Eval_KGE"]
        source = "Table 14"

        # Check if live data has a valid update for this model
        if ensemble_metrics is not None and "Eval_KGE" in ensemble_metrics.columns:
            if model_name in ensemble_metrics.index:
                live_val = ensemble_metrics.loc[model_name, "Eval_KGE"]
                if pd.notna(live_val) and live_val > PATHOLOGICAL_KGE_THRESHOLD:
                    kge_val = live_val
                    source = "live"
                elif pd.notna(live_val) and live_val <= PATHOLOGICAL_KGE_THRESHOLD:
                    logger.warning(
                        f"Excluding live {model_name} KGE={live_val:.3f} "
                        f"(pathological, below {PATHOLOGICAL_KGE_THRESHOLD}), "
                        f"using Table 14 value {table14['Eval_KGE']:.2f}"
                    )

        model_kge[model_name] = kge_val
        sources[model_name] = source

    # Log summary
    live_count = sum(1 for s in sources.values() if s == "live")
    table_count = sum(1 for s in sources.values() if s == "Table 14")
    logger.info(f"Model KGE sources: {live_count} live, {table_count} Table 14")
    for name, src in sources.items():
        logger.info(f"  {name}: KGE={model_kge[name]:.3f} ({src})")

    return model_kge


def get_valid_benchmark_kge(benchmark_scores: pd.DataFrame) -> pd.Series:
    """
    Extract validation KGE values, filtering out NaN and inf.

    Returns:
        Series of valid benchmark KGE values with benchmark names as index
    """
    kge_col = find_kge_val_column(benchmark_scores)
    if kge_col is None:
        logger.error(f"No KGE column found. Columns: {list(benchmark_scores.columns)}")
        return pd.Series(dtype=float)

    kge = benchmark_scores[kge_col].replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = len(benchmark_scores) - len(kge)
    if n_dropped > 0:
        logger.info(f"Filtered {n_dropped} benchmarks with NaN/inf validation KGE")
    return kge


def compare_benchmarks_to_models(
    benchmark_scores: pd.DataFrame,
    model_kge: Dict[str, float],
) -> pd.DataFrame:
    """
    Compare benchmark KGE scores to model evaluation KGE values.

    Only considers benchmarks with valid (non-NaN, non-inf) validation KGE.
    """
    valid_kge = get_valid_benchmark_kge(benchmark_scores)
    if valid_kge.empty:
        return pd.DataFrame()

    best_benchmark_kge = valid_kge.max()
    best_benchmark_name = valid_kge.idxmax()

    rows = []
    for model_name, eval_kge in model_kge.items():
        n_exceeded = (eval_kge > valid_kge).sum()
        n_total = len(valid_kge)
        margin_over_best = eval_kge - best_benchmark_kge

        rows.append({
            "Model": model_name,
            "Eval_KGE": eval_kge,
            "Best_Benchmark_KGE": best_benchmark_kge,
            "Best_Benchmark": best_benchmark_name,
            "Margin_over_best": margin_over_best,
            "Benchmarks_exceeded": n_exceeded,
            "Valid_benchmarks": n_total,
            "Pct_exceeded": 100 * n_exceeded / n_total if n_total > 0 else 0,
            "Exceeds_all": n_exceeded == n_total,
        })

    df = pd.DataFrame(rows).set_index("Model")
    return df


def analyze_benchmark_groups(benchmark_scores: pd.DataFrame) -> pd.DataFrame:
    """Analyze benchmark performance by group, handling NaN/inf gracefully."""
    kge_col = find_kge_val_column(benchmark_scores)
    if kge_col is None:
        return pd.DataFrame()

    rows = []
    for group_name, benchmarks in BENCHMARK_GROUPS.items():
        available = [b for b in benchmarks if b in benchmark_scores.index]
        if not available:
            continue

        group_kge = benchmark_scores.loc[available, kge_col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if group_kge.empty:
            rows.append({
                "Group": group_name,
                "n_benchmarks": len(available),
                "n_valid": 0,
                "Best_KGE": np.nan,
                "Best_benchmark": "N/A (all NaN/inf)",
                "Mean_KGE": np.nan,
                "Worst_KGE": np.nan,
                "Worst_benchmark": "N/A",
            })
            continue

        rows.append({
            "Group": group_name,
            "n_benchmarks": len(available),
            "n_valid": len(group_kge),
            "Best_KGE": group_kge.max(),
            "Best_benchmark": group_kge.idxmax(),
            "Mean_KGE": group_kge.mean(),
            "Worst_KGE": group_kge.min(),
            "Worst_benchmark": group_kge.idxmin(),
        })

    return pd.DataFrame(rows).set_index("Group")


def generate_report(
    benchmark_scores: pd.DataFrame,
    comparison: pd.DataFrame,
    group_analysis: pd.DataFrame,
    metadata: Optional[Dict],
    output_dir: Path,
):
    """Generate a comprehensive text report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_analysis_report_{timestamp}.txt"

    kge_col = find_kge_val_column(benchmark_scores)
    valid_kge = get_valid_benchmark_kge(benchmark_scores)

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SYMFLUENCE Paper - Section 4.5: Benchmarking Analysis Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        # Domain info
        f.write("Domain: Bow River at Banff (lumped, 2,210 km2)\n")
        f.write("Forcing: ERA5\n")
        f.write("Model ensemble calibration period: 2004-2007 (Section 4.2)\n")
        f.write("Model ensemble evaluation period:  2008-2009 (Section 4.2)\n")
        f.write("\n")
        f.write("NOTE: HydroBM splits the benchmark data at the year midpoint,\n")
        f.write("yielding a 2004-2006 calibration / 2007-2009 validation split.\n")
        f.write("This differs slightly from the model ensemble's 4+2 year split.\n")
        f.write("Benchmark validation KGE values are therefore computed over\n")
        f.write("2007-2009 (3 years) rather than 2008-2009 (2 years).\n")

        if metadata:
            f.write(f"\nData period: {metadata.get('data_period', {}).get('start', 'N/A')} to "
                    f"{metadata.get('data_period', {}).get('end', 'N/A')}\n")
        f.write("\n")

        # Benchmark scores summary
        f.write("-" * 70 + "\n")
        f.write("1. BENCHMARK PERFORMANCE SCORES\n")
        f.write("-" * 70 + "\n\n")

        n_total = len(benchmark_scores)
        n_valid = len(valid_kge)
        n_invalid = n_total - n_valid
        f.write(f"Total benchmarks computed: {n_total}\n")
        f.write(f"Valid validation KGE:      {n_valid}\n")
        if n_invalid > 0:
            invalid_names = set(benchmark_scores.index) - set(valid_kge.index)
            f.write(f"Excluded (NaN/inf):        {n_invalid} ({', '.join(invalid_names)})\n")
        f.write("\n")

        if kge_col is not None and not valid_kge.empty:
            # Show only valid benchmarks, sorted
            valid_scores = benchmark_scores.loc[valid_kge.index].sort_values(
                kge_col, ascending=False
            )
            f.write(f"Valid benchmarks ranked by {kge_col}:\n\n")
            f.write(valid_scores.round(3).to_string())
            f.write("\n\n")

            f.write(f"Best benchmark:  {valid_kge.idxmax()} ({kge_col} = {valid_kge.max():.3f})\n")
            f.write(f"Worst benchmark: {valid_kge.idxmin()} ({kge_col} = {valid_kge.min():.3f})\n\n")

        # Group analysis
        f.write("-" * 70 + "\n")
        f.write("2. BENCHMARK GROUP ANALYSIS\n")
        f.write("-" * 70 + "\n\n")

        if not group_analysis.empty:
            f.write(group_analysis.round(3).to_string())
            f.write("\n\n")

        # Model vs benchmark comparison
        f.write("-" * 70 + "\n")
        f.write("3. MODEL ENSEMBLE vs BENCHMARKS (Section 4.2 Comparison)\n")
        f.write("-" * 70 + "\n\n")

        if not comparison.empty:
            f.write(comparison.round(3).to_string())
            f.write("\n\n")

            all_exceed = comparison["Exceeds_all"].sum()
            n_models = len(comparison)
            f.write(f"Models exceeding ALL valid benchmarks: {all_exceed}/{n_models}\n")

            if not valid_kge.empty:
                best_bm_kge = valid_kge.max()
                f.write(f"Best benchmark KGE:       {best_bm_kge:.3f}\n")
                f.write(f"Ensemble mean KGE*:       {ENSEMBLE_MEAN_KGE:.3f}\n")
                f.write(f"Ensemble median KGE*:     {ENSEMBLE_MEDIAN_KGE:.3f}\n")
                f.write(f"Ensemble mean margin:     +{ENSEMBLE_MEAN_KGE - best_bm_kge:.3f}\n")
                f.write("\n")
                f.write("* Ensemble KGE is computed from the combined ensemble output\n")
                f.write("  hydrograph (mean/median of all model traces), not from the\n")
                f.write("  arithmetic mean of individual model KGE values.\n")
            f.write("\n")

        # Per-model interpretation
        f.write("-" * 70 + "\n")
        f.write("4. INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        if not valid_kge.empty and not comparison.empty:
            best_bm_kge = valid_kge.max()
            best_bm_name = valid_kge.idxmax()

            f.write(f"The best-performing benchmark is '{best_bm_name}' "
                    f"with a validation KGE of {best_bm_kge:.3f}.\n\n")

            exceeding = comparison[comparison["Margin_over_best"] > 0]
            below = comparison[comparison["Margin_over_best"] <= 0]

            if len(exceeding) > 0:
                f.write(f"Models exceeding the best benchmark ({len(exceeding)}/{n_models}):\n")
                for model in exceeding.sort_values("Eval_KGE", ascending=False).index:
                    margin = comparison.loc[model, "Margin_over_best"]
                    f.write(f"  - {model}: KGE = {comparison.loc[model, 'Eval_KGE']:.3f} "
                            f"(+{margin:.3f} above best benchmark)\n")
                f.write("\n")

            if len(below) > 0:
                f.write(f"Models NOT exceeding the best benchmark ({len(below)}/{n_models}):\n")
                for model in below.sort_values("Eval_KGE", ascending=False).index:
                    margin = comparison.loc[model, "Margin_over_best"]
                    f.write(f"  - {model}: KGE = {comparison.loc[model, 'Eval_KGE']:.3f} "
                            f"({margin:.3f} below best benchmark)\n")
                f.write("\n")

            f.write(f"The ensemble mean (KGE = {ENSEMBLE_MEAN_KGE:.3f}) exceeds the best "
                    f"benchmark by {ENSEMBLE_MEAN_KGE - best_bm_kge:.3f}, confirming "
                    f"that the multi-model ensemble provides substantial value beyond "
                    f"simple reference models.\n")

    logger.info(f"Report saved to: {report_path}")
    return report_path


def run_analysis(data_dir: Path, output_dir: Path):
    """Run the complete benchmark analysis pipeline."""
    logger.info("Starting Benchmark Analysis - Section 4.5")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load benchmark results
    benchmark_scores = load_benchmark_scores(data_dir)
    if benchmark_scores is None:
        logger.error("Cannot proceed without benchmark scores.")
        sys.exit(1)

    benchmark_flows = load_benchmark_flows(data_dir)
    metadata = load_benchmark_metadata(data_dir)

    # Load ensemble metrics from Section 4.2 (if available)
    ensemble_metrics = load_ensemble_metrics(ENSEMBLE_DIR)

    # Build model KGE dictionary (merges live data with Table 14 fallbacks)
    model_kge = build_model_kge_dict(ensemble_metrics)

    # Compare benchmarks to models (using only valid benchmark KGE values)
    comparison = compare_benchmarks_to_models(benchmark_scores, model_kge)

    # Grouped benchmark analysis
    group_analysis = analyze_benchmark_groups(benchmark_scores)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"benchmark_summary_{timestamp}.csv"
    benchmark_scores.to_csv(summary_path)
    logger.info(f"Saved benchmark summary: {summary_path}")

    if not comparison.empty:
        comp_path = output_dir / f"benchmark_vs_models_{timestamp}.csv"
        comparison.to_csv(comp_path)
        logger.info(f"Saved comparison table: {comp_path}")

    if not group_analysis.empty:
        group_path = output_dir / f"benchmark_groups_{timestamp}.csv"
        group_analysis.to_csv(group_path)
        logger.info(f"Saved group analysis: {group_path}")

    report_path = generate_report(
        benchmark_scores, comparison, group_analysis, metadata, output_dir
    )

    # Print summary to console
    valid_kge = get_valid_benchmark_kge(benchmark_scores)
    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("=" * 60)

    kge_col = find_kge_val_column(benchmark_scores)
    if kge_col and not valid_kge.empty:
        print(f"\nValid benchmark scores ({len(valid_kge)}/{len(benchmark_scores)} benchmarks):")
        print(valid_kge.sort_values(ascending=False).round(3).to_frame(kge_col).to_string())

    if not comparison.empty:
        print("\nModel vs Best Benchmark:")
        print(comparison[["Eval_KGE", "Best_Benchmark_KGE", "Margin_over_best", "Pct_exceeded"]].round(3).to_string())

    print(f"\nFull report: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze HydroBM benchmarks and compare to Section 4.2 ensemble"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data",
        help="SYMFLUENCE data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR

    run_analysis(data_dir, output_dir)


if __name__ == "__main__":
    main()
