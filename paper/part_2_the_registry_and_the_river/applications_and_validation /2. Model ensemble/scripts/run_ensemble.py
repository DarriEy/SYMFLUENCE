#!/usr/bin/env python3
"""
Multi-Model Ensemble Runner for SYMFLUENCE Paper Section 4.2

This script runs all model configurations sequentially for the Bow at Banff
multi-model comparison experiment. Each model (HBV, GR4J, FUSE, jFUSE, SUMMA,
HYPE, RHESSys) is calibrated using the SYMFLUENCE CLI.

Usage:
    python run_ensemble.py [--models MODEL1,MODEL2,...] [--skip-calibrated] [--dry-run]

Arguments:
    --models: Comma-separated list of models to run (default: all)
    --skip-calibrated: Skip models that already have calibration results
    --dry-run: Print what would be run without executing
    --steps: Workflow steps to run (default: calibrate_model)
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
CONFIG_DIR = Path(__file__).parent.parent / "config"
RESULTS_DIR = Path(__file__).parent.parent / "simulations"
LOG_DIR = Path(__file__).parent.parent / "logs"
SYMFLUENCE_CLI = "symfluence"  # Assumes symfluence is in PATH

# Model configurations - Full 10-model ensemble
MODEL_CONFIGS = {
    # Conceptual models
    "HBV": "config_Bow_HBV_era5.yaml",
    "GR4J": "config_Bow_GR4J_era5.yaml",
    "FUSE": "config_Bow_FUSE_era5.yaml",
    "jFUSE": "config_Bow_jFUSE_era5.yaml",
    # Process-based models
    "SUMMA": "config_Bow_SUMMA_era5.yaml",
    "HYPE": "config_Bow_HYPE_era5.yaml",
    "RHESSys": "config_Bow_RHESSys_era5.yaml",
    "MESH": "config_Bow_MESH_era5.yaml",
    # NextGen framework
    "ngen": "config_Bow_ngen_era5.yaml",
    # Machine learning models
    "LSTM": "config_Bow_LSTM_era5.yaml",
}

# Default workflow steps: preprocess, run, then calibrate
DEFAULT_STEPS = [
    "model_specific_preprocessing",
    "run_model",
    "calibrate_model",
]

# Model-specific step overrides (for models that don't use standard calibration)
MODEL_STEPS = {
    # LSTM uses run_model for neural network training (no separate calibrate step)
    "LSTM": ["model_specific_preprocessing", "run_model"],
}


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the ensemble experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ensemble_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("ensemble_runner")
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
            capture_output=False,  # Let output stream to console
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


def run_workflow_step(
    config_path: Path,
    step: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """Run a single workflow step using SYMFLUENCE CLI."""
    cmd = [
        SYMFLUENCE_CLI,
        "workflow",
        "step",
        step,
        "--config",
        str(config_path),
    ]
    logger.info(f"Running step: {step}")
    return run_command(cmd, logger, dry_run)


def run_model_calibration(
    model_name: str,
    config_path: Path,
    steps: List[str],
    logger: logging.Logger,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run calibration for a single model.

    Args:
        model_name: Name of the model (HBV, GR4J, FUSE, etc.)
        config_path: Path to the configuration file
        steps: List of workflow steps to run
        logger: Logger instance
        dry_run: If True, don't actually run calibration

    Returns:
        Dictionary with run results including timing and status
    """
    result = {
        "model": model_name,
        "config": str(config_path),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "error": None,
        "steps_completed": [],
        "steps_failed": [],
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Running calibration for: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Steps: {', '.join(steps)}")
    logger.info(f"{'='*60}\n")

    result["start_time"] = datetime.now()

    try:
        for step in steps:
            success = run_workflow_step(config_path, step, logger, dry_run)
            if success:
                result["steps_completed"].append(step)
            else:
                result["steps_failed"].append(step)
                result["error"] = f"Step '{step}' failed"
                break

        result["end_time"] = datetime.now()
        result["duration_seconds"] = (
            result["end_time"] - result["start_time"]
        ).total_seconds()

        if result["steps_failed"]:
            result["status"] = "failed"
        elif dry_run:
            result["status"] = "dry_run"
        else:
            result["status"] = "completed"

        logger.info(f"Calibration {'completed' if result['status'] == 'completed' else result['status']} for {model_name}")
        if result["duration_seconds"]:
            logger.info(f"Duration: {result['duration_seconds']:.1f} seconds")

    except Exception as e:
        result["end_time"] = datetime.now()
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Calibration failed for {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return result


def check_calibration_exists(model_name: str, data_dir: Path) -> bool:
    """Check if calibration results already exist for a model."""
    # All models use the same experiment ID
    exp_id = "run_1"

    # Check for model-specific optimization results
    # Results are stored with model name prefix
    search_paths = [
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "optimization" / f"{exp_id}_{model_name}*",
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "optimization" / f"{model_name}*",
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "simulations" / exp_id / f"*{model_name}*",
    ]

    for pattern in search_paths:
        if pattern.parent.exists():
            matching = list(pattern.parent.glob(pattern.name))
            if matching:
                return True

    return False


def run_ensemble(
    models: Optional[List[str]] = None,
    steps: Optional[List[str]] = None,
    skip_calibrated: bool = False,
    dry_run: bool = False,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Run the full multi-model ensemble experiment.

    Args:
        models: List of model names to run (default: all)
        steps: List of workflow steps to run
        skip_calibrated: Skip models that already have results
        dry_run: Print what would be run without executing
        data_dir: SYMFLUENCE data directory for checking existing results

    Returns:
        List of result dictionaries for each model
    """
    # Set up logging
    logger = setup_logging(LOG_DIR)
    logger.info("Starting Multi-Model Ensemble Experiment")
    logger.info(f"Config directory: {CONFIG_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")

    # Default to all models
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    else:
        # Validate model names
        invalid = [m for m in models if m not in MODEL_CONFIGS]
        if invalid:
            logger.error(f"Invalid model names: {invalid}")
            logger.error(f"Valid models: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    # Default steps
    if steps is None:
        steps = DEFAULT_STEPS

    logger.info(f"Models to run: {models}")
    logger.info(f"Workflow steps: {steps}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Default data dir
    if data_dir is None:
        data_dir = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

    # Run each model
    all_results = []
    for model_name in models:
        config_file = MODEL_CONFIGS[model_name]
        config_path = CONFIG_DIR / config_file

        # Check if config exists
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            all_results.append({
                "model": model_name,
                "status": "skipped",
                "error": "Config file not found",
            })
            continue

        # Check if already calibrated
        if skip_calibrated and check_calibration_exists(model_name, data_dir):
            logger.info(f"Skipping {model_name}: calibration results already exist")
            all_results.append({
                "model": model_name,
                "status": "skipped",
                "error": "Already calibrated",
            })
            continue

        # Get model-specific steps (or use provided/default steps)
        model_steps = MODEL_STEPS.get(model_name, steps)

        # Run calibration/training
        result = run_model_calibration(model_name, config_path, model_steps, logger, dry_run)
        all_results.append(result)

        # Brief pause between runs
        if not dry_run and result["status"] == "completed":
            time.sleep(5)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE RUN SUMMARY")
    logger.info("="*60)

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

    # Save summary to file
    summary_file = RESULTS_DIR / f"ensemble_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, "w") as f:
        f.write("Multi-Model Ensemble Run Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        for result in all_results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"  Status: {result.get('status', 'unknown')}\n")
            if result.get("duration_seconds"):
                f.write(f"  Duration: {result['duration_seconds']:.1f} seconds\n")
            if result.get("steps_completed"):
                f.write(f"  Steps completed: {', '.join(result['steps_completed'])}\n")
            if result.get("error"):
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")

    logger.info(f"\nSummary saved to: {summary_file}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-model ensemble calibration for SYMFLUENCE paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  Conceptual:     HBV, GR4J, FUSE, jFUSE
  Process-based:  SUMMA, HYPE, RHESSys

Examples:
  python run_ensemble.py                          # Run all models
  python run_ensemble.py --models HBV,GR4J        # Run specific models
  python run_ensemble.py --dry-run                # Preview without running
  python run_ensemble.py --skip-calibrated        # Skip already-done models
        """
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated workflow steps (default: model_specific_preprocessing,run_model,calibrate_model)",
    )
    parser.add_argument(
        "--skip-calibrated",
        action="store_true",
        help="Skip models that already have calibration results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data",
        help="SYMFLUENCE data directory",
    )

    args = parser.parse_args()

    # Parse models list
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    # Parse steps list
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]

    # Run ensemble
    results = run_ensemble(
        models=models,
        steps=steps,
        skip_calibrated=args.skip_calibrated,
        dry_run=args.dry_run,
        data_dir=Path(args.data_dir),
    )

    # Exit with error if any runs failed
    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
