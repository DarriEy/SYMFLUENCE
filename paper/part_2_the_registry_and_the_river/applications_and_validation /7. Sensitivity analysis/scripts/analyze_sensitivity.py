#!/usr/bin/env python3
"""
Multi-Model Sensitivity Screening from Optimization Trajectories
for SYMFLUENCE Paper Section 4.7

This script performs sensitivity screening using calibration iteration data
(DDS/Nelder-Mead trajectories) from each model in the Section 4.2 ensemble.
Parameters are grouped by hydrological process for cross-model comparison.

IMPORTANT METHODOLOGICAL NOTES:
- The sensitivity indices are computed from optimization trajectories, NOT from
  purpose-designed sensitivity analysis samples (e.g., Latin Hypercube, Sobol
  sequences). DDS progressively narrows its search, so samples are correlated
  and biased toward high-performance regions.
- Spearman correlation and RBD-FAST are used as they are more robust to
  non-uniform sampling than variance-based methods like Sobol.
- VISCOUS (pyviscous) total-order indices are included where reliable, but
  values are filtered for numerical stability.
- Results should be interpreted as exploratory sensitivity screening, not
  rigorous variance-based sensitivity analysis.

Usage:
    python analyze_sensitivity.py [--data-dir DIR] [--output-dir DIR]

Output:
    - per_model_sensitivity.csv: Raw sensitivity indices per model
    - process_sensitivity.csv: Aggregated sensitivity by hydrological process
    - cross_model_ranking.csv: Parameter importance rankings per model
    - sensitivity_agreement.csv: Agreement matrix between models (shared processes only)
    - analysis_report.txt: Narrative summary
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"

# Data directory
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = DATA_DIR / "domain_Bow_at_Banff_lumped_era5"

# Models to analyze
MODELS = ["FUSE", "GR4J", "HBV", "HYPE", "SUMMA"]

# Experiment IDs per model (must match config files)
MODEL_EXPERIMENT_IDS = {
    "FUSE": "sensitivity_FUSE",
    "GR4J": "sensitivity_GR4J",
    "HBV": "sensitivity_HBV",
    "HYPE": "sensitivity_HYPE",
    "SUMMA": "sensitivity_SUMMA",
}

# Parameter-to-process mapping for cross-model comparison
# Groups parameters by the hydrological process they control
PROCESS_MAPPING = {
    # FUSE parameters
    "MAXWATR_1": "Soil Storage",
    "MAXWATR_2": "Soil Storage",
    "BASERTE": "Baseflow",
    "QB_POWR": "Baseflow",
    "TIMEDELAY": "Routing",
    "PERCRTE": "Percolation",
    "FRACTEN": "Soil Storage",
    "RTFRAC1": "Surface Runoff",
    "MBASE": "Snow",
    "MFMAX": "Snow",
    "MFMIN": "Snow",
    "PXTEMP": "Snow",
    "LAPSE": "Snow",  # Lapse rate affects snow accumulation; merged into Snow
    # GR4J parameters
    "X1": "Soil Storage",
    "X2": "Groundwater Exchange",
    "X3": "Routing",
    "X4": "Routing",
    # HBV parameters
    "tt": "Snow",
    "cfmax": "Snow",
    "sfcf": "Snow",
    "cfr": "Snow",
    "cwh": "Snow",
    "fc": "Soil Storage",
    "lp": "Evapotranspiration",
    "beta": "Soil Storage",
    "k0": "Surface Runoff",
    "k1": "Baseflow",
    "k2": "Baseflow",
    "uzl": "Surface Runoff",
    "perc": "Percolation",
    "maxbas": "Routing",
    # HYPE parameters
    "ttmp": "Snow",
    "cmlt": "Snow",
    "cevp": "Evapotranspiration",
    "epotdist": "Evapotranspiration",
    "rrcs1": "Baseflow",
    "rrcs2": "Baseflow",
    "rcgrw": "Groundwater Exchange",
    "rivvel": "Routing",
    "damp": "Routing",
    # SUMMA parameters
    "albedoMax": "Snow",
    "albedoMinWinter": "Snow",
    "newSnowDenMin": "Snow",
    "Fcapil": "Soil Storage",
    "k_soil": "Soil Storage",
    "theta_sat": "Soil Storage",
    "critSoilWilting": "Evapotranspiration",
    "theta_res": "Soil Storage",
    "f_impede": "Soil Storage",
    "routingGammaShape": "Routing",
    "routingGammaScale": "Routing",
}

# Canonical process order for consistent display
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sensitivity_analysis")


def find_sensitivity_results(model_name: str) -> Optional[Path]:
    """
    Find the sensitivity analysis results file for a model.

    Each model uses a unique experiment ID (e.g., sensitivity_FUSE) so
    results are stored in separate directories under:
      {project_dir}/reporting/sensitivity_analysis/

    Args:
        model_name: Name of the model

    Returns:
        Path to the all_sensitivity_results.csv file, or None
    """
    exp_id = MODEL_EXPERIMENT_IDS.get(model_name, f"sensitivity_{model_name}")

    search_paths = [
        # Model-specific experiment directory (primary location)
        DOMAIN_DIR / "reporting" / "sensitivity_analysis" / exp_id / "all_sensitivity_results.csv",
        # Model name subdirectory
        DOMAIN_DIR / "reporting" / "sensitivity_analysis" / model_name / "all_sensitivity_results.csv",
        # Flat structure (legacy)
        DOMAIN_DIR / "reporting" / "sensitivity_analysis" / "all_sensitivity_results.csv",
        # Under optimization directory
        DOMAIN_DIR / "optimization" / f"dds_{exp_id}" / "sensitivity_analysis" / "all_sensitivity_results.csv",
        DOMAIN_DIR / "optimization" / model_name / "sensitivity_analysis" / "all_sensitivity_results.csv",
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Found sensitivity results for {model_name}: {path}")
            return path

    logger.warning(f"No sensitivity results found for {model_name}")
    return None


def find_iteration_results(model_name: str) -> Optional[Path]:
    """
    Find calibration iteration results CSV for a model.

    Searches for both the model-specific sensitivity experiment data and
    the original Section 4.2 calibration data (run_1).

    Args:
        model_name: Name of the model

    Returns:
        Path to iteration results CSV, or None
    """
    exp_id = MODEL_EXPERIMENT_IDS.get(model_name, f"sensitivity_{model_name}")
    optimization_dir = DOMAIN_DIR / "optimization"

    # Map model names to their directory names in the optimization folder
    # GR4J is stored under "GR" in SYMFLUENCE
    model_dir_names = {
        "GR4J": "GR",
        "FUSE": "FUSE",
        "HBV": "HBV",
        "HYPE": "HYPE",
        "SUMMA": "SUMMA",
    }
    dir_name = model_dir_names.get(model_name, model_name)

    search_patterns = [
        # Model-specific sensitivity experiment
        optimization_dir / f"dds_{exp_id}" / f"{exp_id}_parallel_iteration_results.csv",
        optimization_dir / exp_id / f"{exp_id}_parallel_iteration_results.csv",
        # Original Section 4.2 calibration data (model-specific subdirectory)
        optimization_dir / dir_name / "dds_run_1" / "run_1_parallel_iteration_results.csv",
        optimization_dir / model_name / "dds_run_1" / "run_1_parallel_iteration_results.csv",
        # Shared optimization directory
        optimization_dir / "dds_run_1" / "run_1_parallel_iteration_results.csv",
        optimization_dir / "run_1_parallel_iteration_results.csv",
    ]

    for path in search_patterns:
        if path.exists():
            logger.info(f"Found iteration results for {model_name}: {path}")
            return path

    # Search model-specific subdirectory for any iteration results
    model_opt_dir = optimization_dir / dir_name
    if model_opt_dir.exists():
        for f in model_opt_dir.rglob("*iteration_results*.csv"):
            logger.info(f"Found iteration results for {model_name}: {f}")
            return f

    logger.warning(f"No iteration results found for {model_name}")
    return None


def load_sensitivity_results(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load SYMFLUENCE sensitivity results for a model.

    The all_sensitivity_results.csv file contains columns for each method:
    pyViscous, Sobol, RBD-FAST, Correlation with parameters as index.

    Args:
        model_name: Name of the model

    Returns:
        DataFrame with sensitivity indices, or None
    """
    results_path = find_sensitivity_results(model_name)
    if results_path is None:
        return None

    try:
        df = pd.read_csv(results_path, index_col=0)
        logger.info(f"Loaded {model_name} sensitivity: {df.shape[0]} parameters, {df.shape[1]} methods")
        return df
    except Exception as e:
        logger.error(f"Failed to load {model_name} sensitivity results: {e}")
        return None


def filter_viscous_values(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """
    Filter VISCOUS sensitivity values for numerical stability.

    VISCOUS can return sentinel values (-999) or extreme outliers when
    the copula estimation fails. Total-order indices should theoretically
    be in [0, 1]. This function replaces unreliable values with NaN and
    rejects the entire method if results are predominantly negative
    (indicating copula estimation failure).

    Args:
        series: Raw VISCOUS sensitivity values
        threshold: Maximum plausible absolute value (default 1.5)

    Returns:
        Filtered Series with outliers replaced by NaN, or None if method failed
    """
    filtered = series.copy()
    # Replace sentinel values
    filtered = filtered.replace(-999, np.nan)
    filtered = filtered.replace(999, np.nan)
    # Replace extreme outliers (VISCOUS total-order indices should be in [0,1])
    mask = filtered.abs() > threshold
    if mask.any():
        logger.info(f"  Filtering {mask.sum()} VISCOUS outlier(s): "
                     f"{series[mask].to_dict()}")
        filtered[mask] = np.nan
    # If all values are NaN after filtering, the method failed for this model
    if filtered.isna().all():
        return None
    # Reject if majority of non-NaN values are negative — total-order indices
    # should be non-negative; predominantly negative values indicate copula failure
    valid = filtered.dropna()
    if len(valid) > 0:
        frac_negative = (valid < 0).sum() / len(valid)
        if frac_negative > 0.5:
            logger.warning(f"  VISCOUS: {frac_negative:.0%} of values are negative "
                           f"(range [{valid.min():.3f}, {valid.max():.3f}]), "
                           f"indicating copula estimation failure — excluding method")
            return None
        # Also reject if max positive value is negligible (near-zero signal)
        max_positive = valid[valid > 0].max() if (valid > 0).any() else 0
        if max_positive < 0.05:
            logger.warning(f"  VISCOUS: max positive value = {max_positive:.4f}, "
                           f"insufficient signal — excluding method")
            return None
    return filtered


def compute_standalone_sensitivity(model_name: str) -> Optional[pd.DataFrame]:
    """
    Compute sensitivity screening from optimization iteration results.

    Uses three methods appropriate for non-designed samples:
    1. Spearman correlation - robust rank-based measure
    2. RBD-FAST - Fourier-based first-order indices
    3. VISCOUS (pyviscous) - copula-based total-order indices (filtered)

    NOTE: Sobol analysis is excluded because the 1D-interpolation surrogate
    approach destroys multi-dimensional structure, producing meaningless indices.

    Args:
        model_name: Name of the model

    Returns:
        DataFrame with sensitivity indices, or None
    """
    iter_path = find_iteration_results(model_name)
    if iter_path is None:
        return None

    try:
        df = pd.read_csv(iter_path).dropna()
        if len(df) < 10:
            logger.warning(f"{model_name}: Insufficient samples ({len(df)}) for sensitivity analysis")
            return None

        # Identify parameter columns (exclude iteration count, metric, and metadata columns)
        non_param_cols = {'iteration', 'Iteration', 'score', 'timestamp',
                          'Calib_RMSE', 'Calib_KGE', 'Calib_KGEp',
                          'Calib_KGEnp', 'Calib_NSE', 'Calib_MAE',
                          'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE',
                          'crash_count', 'crash_rate', 'eval_count'}
        param_cols = [c for c in df.columns if c not in non_param_cols]

        if not param_cols:
            logger.warning(f"{model_name}: No parameter columns found")
            return None

        # Select metric for analysis (check various naming conventions)
        metric = None
        for candidate in ['score', 'Calib_KGEnp', 'Calib_KGE', 'KGE', 'Calib_RMSE', 'RMSE']:
            if candidate in df.columns:
                metric = candidate
                break

        if metric is None:
            logger.warning(f"{model_name}: No metric column found")
            return None

        logger.info(f"{model_name}: Computing sensitivity from {len(df)} samples using metric '{metric}'")

        results = {}

        # Method 1: Spearman correlation (always works, most robust to non-designed samples)
        from scipy.stats import spearmanr
        correlations = []
        for param in param_cols:
            corr, _ = spearmanr(df[param], df[metric])
            correlations.append(abs(corr))
        results['Correlation'] = pd.Series(correlations, index=param_cols)

        # Method 2: RBD-FAST (SALib) - more robust than Sobol to arbitrary samples
        try:
            from SALib.analyze import rbd_fast as rbd_fast_analyze

            problem = {
                'num_vars': len(param_cols),
                'names': param_cols,
                'bounds': [[df[col].min(), df[col].max()] for col in param_cols]
            }
            X = df[param_cols].values
            Y = df[metric].values
            rbd_results = rbd_fast_analyze.analyze(problem, X, Y)
            rbd_series = pd.Series(rbd_results['S1'], index=param_cols)
            # Check if RBD-FAST produced meaningful variation:
            # 1. Standard deviation must be non-trivial
            # 2. Range must span meaningful fraction of [0,1]
            # 3. Max/min ratio must show real differentiation (not all ~same value)
            rbd_range = rbd_series.max() - rbd_series.min()
            rbd_max = rbd_series.abs().max()
            if rbd_series.std() < 1e-3 or rbd_range < 0.05:
                logger.warning(f"{model_name}: RBD-FAST produced near-constant values "
                               f"(std={rbd_series.std():.4f}, range={rbd_range:.4f}), excluding")
            elif rbd_max < 0.05:
                logger.warning(f"{model_name}: RBD-FAST values are all near zero "
                               f"(max abs={rbd_max:.4f}), excluding")
            else:
                results['RBD-FAST'] = rbd_series
        except Exception as e:
            logger.warning(f"{model_name}: RBD-FAST failed: {e}")

        # Method 3: VISCOUS (pyviscous) - with outlier filtering
        try:
            from pyviscous import viscous
            x = df[param_cols].values
            y = df[metric].values.reshape(-1, 1)
            sensitivities = []
            for i in range(len(param_cols)):
                try:
                    sens = viscous(x, y, i, sensType='total')
                    if isinstance(sens, tuple):
                        sens = sens[0]
                    sensitivities.append(sens)
                except Exception:
                    sensitivities.append(np.nan)
            raw_viscous = pd.Series(sensitivities, index=param_cols)
            filtered_viscous = filter_viscous_values(raw_viscous)
            if filtered_viscous is not None:
                results['pyViscous'] = filtered_viscous
            else:
                logger.warning(f"{model_name}: VISCOUS failed (all values filtered out)")
        except ImportError:
            logger.warning(f"{model_name}: pyviscous not installed, skipping VISCOUS method")
        except Exception as e:
            logger.warning(f"{model_name}: VISCOUS analysis failed: {e}")

        if not results:
            return None

        result_df = pd.DataFrame(results)
        logger.info(f"{model_name}: Computed sensitivity for {len(param_cols)} parameters "
                     f"with {len(results)} methods: {list(results.keys())}")
        return result_df

    except Exception as e:
        logger.error(f"Failed to compute sensitivity for {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def normalize_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize sensitivity indices to [0, 1] range per method.

    Takes absolute values (sensitivity magnitude matters, not sign) and
    normalizes each method column by its maximum value within this model.
    This is the ONLY normalization step — no further normalization should
    be applied in visualization.

    Args:
        df: DataFrame with sensitivity indices (already filtered for outliers)

    Returns:
        Normalized DataFrame
    """
    df_abs = df.abs()
    normalized = df_abs.copy()
    for col in df_abs.columns:
        col_max = df_abs[col].max()
        if col_max > 0:
            normalized[col] = df_abs[col] / col_max
    return normalized


def compute_ensemble_sensitivity(df: pd.DataFrame) -> pd.Series:
    """
    Compute an ensemble sensitivity index by averaging across methods.

    Args:
        df: Normalized sensitivity DataFrame

    Returns:
        Series with mean sensitivity per parameter
    """
    return df.mean(axis=1).sort_values(ascending=False)


def aggregate_by_process(
    sensitivity: pd.Series,
    model_name: str,
) -> pd.Series:
    """
    Aggregate parameter sensitivities by hydrological process.

    For each process, takes the MEAN sensitivity among its parameters.
    Mean is preferred over max because max inflates processes that have
    more parameters (e.g., SUMMA has 5 Soil Storage params vs GR4J's 1),
    giving more chances for a high value by noise alone.

    Args:
        sensitivity: Series of parameter sensitivities
        model_name: Model name (for logging)

    Returns:
        Series indexed by process name with aggregated sensitivity
    """
    process_sensitivity = {}

    for param, sens_value in sensitivity.items():
        process = PROCESS_MAPPING.get(param, "Other")
        if process not in process_sensitivity:
            process_sensitivity[process] = []
        process_sensitivity[process].append(sens_value)

    # Aggregate: use mean to fairly represent process importance
    aggregated = {}
    for process, values in process_sensitivity.items():
        aggregated[process] = np.nanmean(values)

    return pd.Series(aggregated).sort_values(ascending=False)


def compute_agreement_matrix(
    process_rankings: Dict[str, pd.Series],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise agreement between models on process importance ranking.

    Uses Spearman rank correlation computed ONLY on processes shared between
    each model pair. Processes that don't exist in a model (NaN) are excluded
    rather than zero-filled, since "not applicable" != "not sensitive."

    Args:
        process_rankings: Dict mapping model names to process sensitivity Series

    Returns:
        Tuple of (agreement_df, overlap_df):
        - agreement_df: Pairwise Spearman rank correlation coefficients
        - overlap_df: Number of shared processes per model pair
    """
    from scipy.stats import spearmanr

    model_names = list(process_rankings.keys())
    n = len(model_names)
    agreement = np.ones((n, n))
    overlap = np.zeros((n, n), dtype=int)

    # Collect all processes
    all_processes = set()
    for rankings in process_rankings.values():
        all_processes.update(rankings.index)
    all_processes = sorted(all_processes)

    # Reindex without fill — keep NaN for missing processes
    aligned = {}
    for model, rankings in process_rankings.items():
        aligned[model] = rankings.reindex(all_processes)

    for i in range(n):
        for j in range(n):
            m1, m2 = model_names[i], model_names[j]
            # Find shared (non-NaN) processes
            shared_mask = aligned[m1].notna() & aligned[m2].notna()
            n_shared = shared_mask.sum()
            overlap[i, j] = n_shared

            if i == j:
                agreement[i, j] = 1.0
            elif n_shared >= 3:  # Need at least 3 shared processes for meaningful correlation
                corr, _ = spearmanr(
                    aligned[m1][shared_mask].values,
                    aligned[m2][shared_mask].values,
                )
                agreement[i, j] = corr
            else:
                agreement[i, j] = np.nan  # Insufficient overlap

    agreement_df = pd.DataFrame(agreement, index=model_names, columns=model_names)
    overlap_df = pd.DataFrame(overlap, index=model_names, columns=model_names)

    return agreement_df, overlap_df


def run_analysis(output_dir: Path) -> None:
    """
    Run the full cross-model sensitivity comparison.

    Args:
        output_dir: Directory to save analysis outputs
    """
    logger.info("Starting Multi-Model Sensitivity Analysis Comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sensitivity results for each model
    all_sensitivities = {}
    for model_name in MODELS:
        logger.info(f"\nProcessing {model_name}...")

        # Try loading pre-computed SYMFLUENCE results first
        df = load_sensitivity_results(model_name)

        # Fall back to computing from iteration results
        if df is None:
            logger.info(f"Computing sensitivity from raw iteration data for {model_name}")
            df = compute_standalone_sensitivity(model_name)

        if df is not None:
            all_sensitivities[model_name] = df

    if not all_sensitivities:
        logger.error("No sensitivity results found for any model!")
        logger.error(
            "Run sensitivity analysis first using: python run_sensitivity.py\n"
            "Or ensure calibration iteration results exist from Section 4.2."
        )
        return

    logger.info(f"\nLoaded sensitivity results for {len(all_sensitivities)} models: "
                f"{list(all_sensitivities.keys())}")

    # 1. Save per-model raw sensitivity results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    for model_name, df in all_sensitivities.items():
        for param in df.index:
            row = {"Model": model_name, "Parameter": param}
            for method in df.columns:
                row[method] = df.loc[param, method]
            all_rows.append(row)

    per_model_df = pd.DataFrame(all_rows)
    per_model_file = output_dir / f"per_model_sensitivity_{timestamp}.csv"
    per_model_df.to_csv(per_model_file, index=False)
    logger.info(f"Saved per-model results: {per_model_file}")

    # 2. Compute normalized ensemble sensitivity per model
    ensemble_sensitivities = {}
    for model_name, df in all_sensitivities.items():
        normalized = normalize_sensitivity(df)
        ensemble = compute_ensemble_sensitivity(normalized)
        ensemble_sensitivities[model_name] = ensemble

    # 3. Aggregate by hydrological process
    process_sensitivities = {}
    for model_name, sensitivity in ensemble_sensitivities.items():
        process_sensitivities[model_name] = aggregate_by_process(sensitivity, model_name)

    # Create process comparison table
    process_df = pd.DataFrame(process_sensitivities)
    # Reindex by canonical process order
    ordered_processes = [p for p in PROCESS_ORDER if p in process_df.index]
    remaining = [p for p in process_df.index if p not in ordered_processes]
    process_df = process_df.reindex(ordered_processes + remaining)

    process_file = output_dir / f"process_sensitivity_{timestamp}.csv"
    process_df.to_csv(process_file)
    logger.info(f"Saved process sensitivity: {process_file}")

    # 4. Create cross-model rankings
    ranking_rows = []
    for model_name, sensitivity in ensemble_sensitivities.items():
        ranked = sensitivity.rank(ascending=False)
        for param, rank in ranked.items():
            ranking_rows.append({
                "Model": model_name,
                "Parameter": param,
                "Sensitivity": sensitivity[param],
                "Rank": int(rank),
                "Process": PROCESS_MAPPING.get(param, "Other"),
            })

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_file = output_dir / f"cross_model_ranking_{timestamp}.csv"
    ranking_df.to_csv(ranking_file, index=False)
    logger.info(f"Saved cross-model rankings: {ranking_file}")

    # 5. Compute model agreement on process importance (shared processes only)
    if len(process_sensitivities) >= 2:
        agreement_df, overlap_df = compute_agreement_matrix(process_sensitivities)
        agreement_file = output_dir / f"sensitivity_agreement_{timestamp}.csv"
        agreement_df.to_csv(agreement_file)
        overlap_file = output_dir / f"process_overlap_{timestamp}.csv"
        overlap_df.to_csv(overlap_file)
        logger.info(f"Saved agreement matrix: {agreement_file}")
        logger.info(f"Saved process overlap counts: {overlap_file}")
    else:
        agreement_df = pd.DataFrame()
        overlap_df = pd.DataFrame()

    # 6. Identify consistently sensitive processes
    if not process_df.empty:
        # A process is "consistently sensitive" if it ranks in the top 3
        # for more than half the models WHERE IT EXISTS (non-NaN)
        top_n = 3
        consistent = {}
        for process in process_df.index:
            n_models_sensitive = 0
            for model in process_df.columns:
                val = process_df.loc[process, model]
                if pd.isna(val):
                    continue  # Skip models that don't have this process
                model_ranked = process_df[model].dropna().rank(ascending=False)
                if model_ranked.get(process, 999) <= top_n:
                    n_models_sensitive += 1
            consistent[process] = n_models_sensitive

        consistent_series = pd.Series(consistent).sort_values(ascending=False)

    # 7. Generate report
    report_file = output_dir / f"analysis_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("Multi-Model Sensitivity Screening Report - Section 4.7\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*70}\n\n")

        f.write("METHODOLOGICAL NOTES:\n")
        f.write("-" * 70 + "\n")
        f.write("  Sensitivity indices are computed from optimization trajectories\n")
        f.write("  (DDS/Nelder-Mead), NOT from purpose-designed sensitivity samples.\n")
        f.write("  Results should be interpreted as exploratory screening.\n")
        f.write("  Process aggregation uses MEAN across parameters (not max).\n")
        f.write("  Agreement matrix uses only processes SHARED between each model pair.\n\n")

        f.write("Models Analyzed:\n")
        for model_name in all_sensitivities:
            n_params = len(all_sensitivities[model_name])
            methods = list(all_sensitivities[model_name].columns)
            f.write(f"  - {model_name}: {n_params} parameters, methods: {methods}\n")
        f.write("\n")

        f.write("Sensitivity Methods:\n")
        f.write("  1. Correlation - Spearman rank correlation (most robust)\n")
        f.write("  2. RBD-FAST    - Fourier amplitude sensitivity test (SALib)\n")
        f.write("  3. pyViscous   - Total-order copula-based indices (where reliable)\n\n")

        f.write("Process Sensitivity Comparison (mean across parameters per process):\n")
        f.write("-" * 70 + "\n")
        if not process_df.empty:
            f.write(process_df.round(3).to_string())
        f.write("\n\n")

        f.write("Top 3 Most Sensitive Parameters per Model:\n")
        f.write("-" * 70 + "\n")
        for model_name, sensitivity in ensemble_sensitivities.items():
            top3 = sensitivity.nlargest(3)
            f.write(f"\n  {model_name}:\n")
            for rank, (param, value) in enumerate(top3.items(), 1):
                process = PROCESS_MAPPING.get(param, "Other")
                f.write(f"    {rank}. {param} ({process}): {value:.3f}\n")
        f.write("\n")

        if not process_df.empty:
            f.write("Consistently Sensitive Processes (top 3 in >50% of models):\n")
            f.write("-" * 70 + "\n")
            # Count only models where the process exists (non-NaN)
            threshold = 0.5
            for process, count in consistent_series.items():
                n_models_with_process = process_df.loc[process].notna().sum() if process in process_df.index else 0
                marker = ""
                if n_models_with_process > 0 and count / n_models_with_process > threshold:
                    marker = " ***"
                f.write(f"  {process}: top-3 in {count}/{n_models_with_process} models that have it{marker}\n")
            f.write("\n")

        if not agreement_df.empty:
            f.write("Model Agreement Matrix (Spearman correlation on SHARED process rankings):\n")
            f.write("-" * 70 + "\n")
            f.write(agreement_df.round(3).to_string())
            f.write("\n\n")
            if not overlap_df.empty:
                f.write("Number of shared processes per model pair:\n")
                f.write(overlap_df.to_string())
                f.write("\n")

    logger.info(f"Saved analysis report: {report_file}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("MULTI-MODEL SENSITIVITY SCREENING SUMMARY")
    print("=" * 70)

    if not process_df.empty:
        print("\nProcess Sensitivity (mean across parameters; higher = more sensitive):")
        print(process_df.round(3).to_string())

    if not agreement_df.empty:
        print("\nModel Agreement (Spearman on shared processes):")
        print(agreement_df.round(3).to_string())
        if not overlap_df.empty:
            print("\nShared process counts:")
            print(overlap_df.to_string())

    print(f"\nFull results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare sensitivity analysis results across models"
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
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR

    run_analysis(output_dir)


if __name__ == "__main__":
    main()
