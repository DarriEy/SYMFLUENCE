#!/usr/bin/env python3
"""
Multi-Model Sensitivity Analysis Runner for SYMFLUENCE Paper Section 4.7

This script runs SYMFLUENCE's sensitivity analysis module for each model
from the Section 4.2 ensemble, enabling cross-model comparison of parameter
sensitivity. The analysis uses calibration trial data (DDS iterations) to
compute sensitivity indices via four methods: VISCOUS, Sobol, RBD-FAST,
and Spearman correlation.

Usage:
    python run_sensitivity.py [--models MODEL1,MODEL2,...] [--dry-run]
    python run_sensitivity.py --models FUSE,GR4J       # Run specific models
    python run_sensitivity.py --skip-completed          # Skip models with existing results

Arguments:
    --models: Comma-separated list of models to run (default: all)
    --skip-completed: Skip models that already have sensitivity results
    --dry-run: Print what would be run without executing
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
SYMFLUENCE_CLI = "symfluence"

# Model configurations with unique experiment IDs for isolated sensitivity outputs
MODEL_CONFIGS = {
    "FUSE": "config_Bow_FUSE_sensitivity_era5.yaml",
    "GR4J": "config_Bow_GR4J_sensitivity_era5.yaml",
    "HBV": "config_Bow_HBV_sensitivity_era5.yaml",
    "HYPE": "config_Bow_HYPE_sensitivity_era5.yaml",
    "SUMMA": "config_Bow_SUMMA_sensitivity_era5.yaml",
}

# Mapping from model name to its experiment ID (must match config files)
MODEL_EXPERIMENT_IDS = {
    "FUSE": "sensitivity_FUSE",
    "GR4J": "sensitivity_GR4J",
    "HBV": "sensitivity_HBV",
    "HYPE": "sensitivity_HYPE",
    "SUMMA": "sensitivity_SUMMA",
}

# Sensitivity analysis uses the calibration iteration results
WORKFLOW_STEP = "run_sensitivity_analysis"

# Data directory for checking existing results
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = DATA_DIR / "domain_Bow_at_Banff_lumped_era5"


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the sensitivity experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sensitivity_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("sensitivity_runner")
    logger.info(f"Logging to: {log_file}")
    return logger


def run_command(cmd: List[str], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Execute a shell command."""
    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute command")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        logger.error("Make sure SYMFLUENCE is installed and 'symfluence' is in your PATH")
        return False


def check_calibration_exists(model_name: str) -> bool:
    """Check if calibration iteration results exist for a model.

    Looks in both the original Section 4.2 experiment (run_1) and the
    model-specific sensitivity experiment directory.
    """
    optimization_dir = DOMAIN_DIR / "optimization"
    if not optimization_dir.exists():
        return False

    exp_id = MODEL_EXPERIMENT_IDS.get(model_name, f"sensitivity_{model_name}")

    # Check model-specific sensitivity experiment directory
    search_patterns = [
        optimization_dir / f"dds_{exp_id}" / f"{exp_id}_parallel_iteration_results.csv",
        optimization_dir / exp_id / f"{exp_id}_parallel_iteration_results.csv",
        # Also check original Section 4.2 calibration data (run_1)
        optimization_dir / model_name / "dds_run_1" / "run_1_parallel_iteration_results.csv",
        optimization_dir / "dds_run_1" / "run_1_parallel_iteration_results.csv",
        optimization_dir / "run_1_parallel_iteration_results.csv",
    ]

    for path in search_patterns:
        if path.exists():
            return True

    # Broader glob search
    for f in optimization_dir.rglob("*iteration_results*.csv"):
        return True

    return False


def check_sensitivity_exists(model_name: str) -> bool:
    """Check if sensitivity analysis results already exist for a model.

    Each model has its own experiment ID, so results are stored separately
    under {project_dir}/reporting/sensitivity_analysis/{experiment_id}/.
    """
    exp_id = MODEL_EXPERIMENT_IDS.get(model_name, f"sensitivity_{model_name}")

    # Check model-specific sensitivity output paths
    search_paths = [
        DOMAIN_DIR / "reporting" / "sensitivity_analysis" / exp_id / "all_sensitivity_results.csv",
        DOMAIN_DIR / "reporting" / "sensitivity_analysis" / "all_sensitivity_results.csv",
    ]

    for path in search_paths:
        if path.exists():
            return True

    return False


def run_sensitivity_for_model(
    model_name: str,
    config_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run sensitivity analysis for a single model.

    Args:
        model_name: Name of the model
        config_path: Path to the configuration file
        logger: Logger instance
        dry_run: If True, don't actually run

    Returns:
        Dictionary with run results
    """
    result = {
        "model": model_name,
        "config": str(config_path),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "error": None,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Running sensitivity analysis for: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}\n")

    result["start_time"] = datetime.now()

    try:
        cmd = [
            SYMFLUENCE_CLI,
            "workflow",
            "step",
            WORKFLOW_STEP,
            "--config",
            str(config_path),
        ]

        success = run_command(cmd, logger, dry_run)

        result["end_time"] = datetime.now()
        result["duration_seconds"] = (
            result["end_time"] - result["start_time"]
        ).total_seconds()

        if dry_run:
            result["status"] = "dry_run"
        elif success:
            result["status"] = "completed"
        else:
            result["status"] = "failed"
            result["error"] = "Command returned non-zero exit code"

    except Exception as e:
        result["end_time"] = datetime.now()
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Sensitivity analysis failed for {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return result


def run_sensitivity_ensemble(
    models: Optional[List[str]] = None,
    skip_completed: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Run sensitivity analysis for all models in the ensemble.

    Args:
        models: List of model names to run (default: all)
        skip_completed: Skip models that already have sensitivity results
        dry_run: Print what would be run without executing

    Returns:
        List of result dictionaries for each model
    """
    logger = setup_logging(LOG_DIR)
    logger.info("Starting Multi-Model Sensitivity Analysis Experiment (Section 4.7)")
    logger.info(f"Config directory: {CONFIG_DIR}")

    # Default to all models
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    else:
        invalid = [m for m in models if m not in MODEL_CONFIGS]
        if invalid:
            logger.error(f"Invalid model names: {invalid}")
            logger.error(f"Valid models: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"Models to analyze: {models}")

    # Pre-flight checks
    logger.info("\nPre-flight checks:")
    for model_name in models:
        has_calib = check_calibration_exists(model_name)
        has_sensitivity = check_sensitivity_exists(model_name)
        logger.info(
            f"  {model_name}: calibration_data={'YES' if has_calib else 'NO'}, "
            f"sensitivity_results={'YES' if has_sensitivity else 'NO'}"
        )
        if not has_calib:
            logger.warning(
                f"  WARNING: No calibration iteration results found for {model_name}. "
                f"Sensitivity analysis requires calibration data from Section 4.2."
            )

    # Run sensitivity for each model
    all_results = []
    for model_name in models:
        config_file = MODEL_CONFIGS[model_name]
        config_path = CONFIG_DIR / config_file

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            all_results.append({
                "model": model_name,
                "status": "skipped",
                "error": "Config file not found",
            })
            continue

        if skip_completed and check_sensitivity_exists(model_name):
            logger.info(f"Skipping {model_name}: sensitivity results already exist")
            all_results.append({
                "model": model_name,
                "status": "skipped",
                "error": "Results already exist",
            })
            continue

        result = run_sensitivity_for_model(model_name, config_path, logger, dry_run)
        all_results.append(result)

        if not dry_run and result["status"] == "completed":
            time.sleep(2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SENSITIVITY ANALYSIS RUN SUMMARY")
    logger.info("=" * 60)

    for result in all_results:
        status_icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "skipped": "[SKIP]",
            "dry_run": "[DRY]",
        }.get(result.get("status", "unknown"), "[?]")

        duration_str = ""
        if result.get("duration_seconds"):
            duration_str = f" ({result['duration_seconds']:.1f}s)"

        logger.info(f"{status_icon} {result['model']}{duration_str}")
        if result.get("error"):
            logger.info(f"      Error: {result['error']}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-model sensitivity analysis for SYMFLUENCE paper Section 4.7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models (from Section 4.2 ensemble):
  Conceptual:     GR4J (4 params), HBV (14 params)
  Process-based:  FUSE (13 params), HYPE (10 params), SUMMA (11 params)

Sensitivity Methods (run by SYMFLUENCE):
  1. VISCOUS (pyviscous)  - Total-order sensitivity indices
  2. Sobol (SALib)        - Variance-based decomposition
  3. RBD-FAST (SALib)     - Fourier amplitude sensitivity test
  4. Correlation          - Spearman rank correlation

Examples:
  python run_sensitivity.py                          # Run all models
  python run_sensitivity.py --models FUSE,GR4J       # Run specific models
  python run_sensitivity.py --dry-run                # Preview without running
  python run_sensitivity.py --skip-completed         # Skip already-done models
        """
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to analyze (default: all)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip models that already have sensitivity results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    results = run_sensitivity_ensemble(
        models=models,
        skip_completed=args.skip_completed,
        dry_run=args.dry_run,
    )

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
