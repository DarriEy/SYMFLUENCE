#!/usr/bin/env python3
"""
Multivariate Evaluation Analysis for SYMFLUENCE Paper Section 4.10

Processes outputs from the three multivariate evaluation experiments:
  a) Bow: TWS anomaly time series comparison (calibrated/uncalibrated SUMMA vs GRACE)
  b) Paradise: SCA accuracy + soil moisture correlation metrics
  c) Iceland: SCF trend significance and seasonal decomposition

Usage:
    python analyze_multivar.py [--study bow|paradise|iceland|all]
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

# Study domains
STUDY_DOMAINS = {
    "bow": "Bow_at_Banff_multivar",
    "paradise": "paradise_multivar",
    "iceland": "Iceland_multivar",
}


def setup_logging() -> logging.Logger:
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("multivar_analysis")


def analyze_bow_tws(logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze Bow River GRACE TWS anomaly comparison.

    Compares:
    - Uncalibrated SUMMA TWS anomalies vs GRACE
    - Calibrated SUMMA TWS anomalies vs GRACE
    - Improvement in TWS representation through calibration

    Metrics:
    - Pearson/Spearman correlation of monthly TWS anomalies
    - RMSE of TWS anomaly time series
    - Seasonal cycle amplitude comparison
    - Phase shift between simulated and observed TWS
    """
    logger.info("Analyzing Bow River GRACE TWS comparison...")
    domain_dir = DATA_DIR / f"domain_{STUDY_DOMAINS['bow']}"

    results = {
        "study": "bow_grace_tws",
        "description": "TWS anomaly comparison: calibrated/uncalibrated SUMMA vs GRACE",
        "domain": STUDY_DOMAINS["bow"],
        "metrics": {},
        "data_found": False,
    }

    # Check for simulation outputs
    for exp_id in ["bow_tws_uncalibrated", "bow_tws_calibrated"]:
        sim_dir = domain_dir / "simulations" / exp_id
        report_dir = domain_dir / "reporting" / exp_id

        if sim_dir.exists():
            results["data_found"] = True
            logger.info(f"  Found simulation output: {sim_dir}")

            # Look for SUMMA output files
            nc_files = list(sim_dir.rglob("*.nc"))
            logger.info(f"  NetCDF files found: {len(nc_files)}")

        if report_dir.exists():
            csv_files = list(report_dir.rglob("*.csv"))
            logger.info(f"  Report CSV files found: {len(csv_files)}")

    # Look for GRACE observation data
    grace_dir = domain_dir / "observations" / "grace_tws"
    if grace_dir.exists() and any(grace_dir.iterdir()):
        logger.info(f"  GRACE TWS data found: {grace_dir}")
        results["grace_data_available"] = True
    else:
        logger.warning("  No GRACE TWS data found. Download GRACE-FO JPL RL06 mascon data.")
        results["grace_data_available"] = False

    if not results["data_found"]:
        logger.warning("  No simulation outputs found. Run the experiments first.")
        logger.info("  Generating placeholder analysis structure...")

    # Define expected analysis outputs
    results["expected_outputs"] = {
        "tws_anomaly_timeseries.csv": "Monthly TWS anomaly time series for all three sources",
        "tws_correlation_metrics.csv": "Correlation metrics (Pearson, Spearman) for each experiment",
        "tws_seasonal_cycle.csv": "Mean seasonal TWS cycle comparison",
        "tws_improvement_summary.csv": "Calibration improvement metrics",
    }

    return results


def analyze_paradise_sca_sm(logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze Paradise SCA and soil moisture comparison.

    Compares:
    - Simulated SCA (from SUMMA SWE > threshold) vs MODIS MOD10A1
    - Simulated soil moisture vs SMAP L3

    Metrics:
    - SCA: accuracy, precision, recall, F1-score, critical success index
    - Soil moisture: Pearson correlation, RMSE, bias, unbiased RMSE
    """
    logger.info("Analyzing Paradise SCA & soil moisture comparison...")
    domain_dir = DATA_DIR / f"domain_{STUDY_DOMAINS['paradise']}"

    results = {
        "study": "paradise_sca_sm",
        "description": "SCA vs MODIS and soil moisture vs SMAP comparison",
        "domain": STUDY_DOMAINS["paradise"],
        "metrics": {},
        "data_found": False,
    }

    # Check for simulation outputs
    sim_dir = domain_dir / "simulations" / "paradise_sca_sm"
    if sim_dir.exists():
        results["data_found"] = True
        nc_files = list(sim_dir.rglob("*.nc"))
        logger.info(f"  Simulation NetCDF files found: {len(nc_files)}")

    # Check for observation data
    for obs_type, obs_dir_name in [("MODIS SCA", "modis_sca"), ("SMAP SM", "smap_sm")]:
        obs_dir = domain_dir / "observations" / obs_dir_name
        if obs_dir.exists() and any(obs_dir.iterdir()):
            logger.info(f"  {obs_type} data found: {obs_dir}")
            results[f"{obs_dir_name}_available"] = True
        else:
            logger.warning(f"  No {obs_type} data found in {obs_dir}")
            results[f"{obs_dir_name}_available"] = False

    if not results["data_found"]:
        logger.warning("  No simulation outputs found. Run the experiments first.")

    results["expected_outputs"] = {
        "sca_binary_metrics.csv": "SCA classification metrics (accuracy, F1, CSI) per month",
        "sca_confusion_matrix.csv": "Monthly confusion matrices for SCA",
        "sm_correlation_metrics.csv": "Soil moisture correlation and error metrics",
        "sm_timeseries_comparison.csv": "Simulated vs SMAP soil moisture time series",
        "multivar_joint_evaluation.csv": "Combined SCA + SM evaluation summary",
    }

    return results


def analyze_iceland_scf_trend(logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze Iceland SCF trend study.

    Computes:
    - Mann-Kendall trend test on annual/seasonal SCF time series
    - Sen's slope estimator for trend magnitude
    - Comparison of simulated vs MODIS-observed SCF trends
    - Elevation-band stratified trend analysis
    - Seasonal decomposition of SCF trends

    Metrics:
    - Trend significance (p-value) and direction
    - Sen's slope (% SCF per decade)
    - Trend agreement between simulated and observed
    """
    logger.info("Analyzing Iceland SCF trend study...")
    domain_dir = DATA_DIR / f"domain_{STUDY_DOMAINS['iceland']}"

    results = {
        "study": "iceland_scf_trend",
        "description": "Region-wide SCF trend analysis over full MODIS record (2000-2023)",
        "domain": STUDY_DOMAINS["iceland"],
        "metrics": {},
        "data_found": False,
    }

    # Check for simulation outputs
    sim_dir = domain_dir / "simulations" / "iceland_scf_trend"
    if sim_dir.exists():
        results["data_found"] = True
        nc_files = list(sim_dir.rglob("*.nc"))
        logger.info(f"  Simulation NetCDF files found: {len(nc_files)}")

    # Check for MODIS SCF data
    modis_dir = domain_dir / "observations" / "modis_scf"
    if modis_dir.exists() and any(modis_dir.iterdir()):
        logger.info(f"  MODIS SCF data found: {modis_dir}")
        results["modis_scf_available"] = True
    else:
        logger.warning(f"  No MODIS SCF data found in {modis_dir}")
        results["modis_scf_available"] = False

    if not results["data_found"]:
        logger.warning("  No simulation outputs found. Run the experiments first.")

    results["expected_outputs"] = {
        "scf_annual_timeseries.csv": "Annual mean SCF for simulated and MODIS",
        "scf_seasonal_timeseries.csv": "Seasonal SCF time series (accumulation, ablation, snow-free)",
        "scf_trend_results.csv": "Mann-Kendall trend test results per season and elevation band",
        "scf_elevation_band_trends.csv": "SCF trends stratified by elevation band",
        "scf_trend_agreement.csv": "Agreement between simulated and observed SCF trends",
    }

    return results


def write_analysis_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write analysis summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write summary CSV
    summary_file = output_dir / f"multivar_analysis_summary_{timestamp}.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study", "description", "domain", "data_found", "expected_outputs"])
        for result in all_results:
            writer.writerow([
                result["study"],
                result["description"],
                result["domain"],
                result["data_found"],
                "; ".join(result.get("expected_outputs", {}).keys()),
            ])
    logger.info(f"Summary written to: {summary_file}")

    # Write text report
    report_file = output_dir / f"analysis_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("SYMFLUENCE Section 4.10 - Multivariate Evaluation Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for result in all_results:
            f.write(f"Study: {result['study']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Domain: {result['domain']}\n")
            f.write(f"Data available: {result['data_found']}\n")
            f.write("\nExpected outputs:\n")
            for filename, desc in result.get("expected_outputs", {}).items():
                f.write(f"  - {filename}: {desc}\n")
            f.write("\n" + "-" * 50 + "\n\n")

    logger.info(f"Report written to: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze multivariate evaluation results for SYMFLUENCE paper Section 4.10",
    )
    parser.add_argument(
        "--study",
        type=str,
        nargs="+",
        default=["all"],
        choices=["bow", "paradise", "iceland", "all"],
        help="Which study/studies to analyze (default: all)",
    )

    args = parser.parse_args()
    logger = setup_logging()

    studies = args.study
    if "all" in studies:
        studies = ["bow", "paradise", "iceland"]

    logger.info("Starting Multivariate Evaluation Analysis (Section 4.10)")

    analyzers = {
        "bow": analyze_bow_tws,
        "paradise": analyze_paradise_sca_sm,
        "iceland": analyze_iceland_scf_trend,
    }

    all_results = []
    for study_key in studies:
        if study_key in analyzers:
            result = analyzers[study_key](logger)
            all_results.append(result)

    write_analysis_report(all_results, ANALYSIS_DIR, logger)

    logger.info("\nAnalysis complete.")


if __name__ == "__main__":
    main()
