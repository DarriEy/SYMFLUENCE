#!/usr/bin/env python3
"""
Multivariate Evaluation Runner for SYMFLUENCE Paper Section 4.10

This script orchestrates the three multivariate evaluation experiments:
  a) Bow River: GRACE TWS anomaly comparison (calibrated vs uncalibrated SUMMA)
  b) Paradise: Simulated SCA vs MODIS and soil moisture vs SMAP
  c) Iceland: Region-wide SCF trend analysis over the full MODIS record

Usage:
    python run_multivar_evaluation.py [--study bow|paradise|iceland|all] [--dry-run]
    python run_multivar_evaluation.py --study bow          # Run only Bow GRACE TWS study
    python run_multivar_evaluation.py --study paradise     # Run only Paradise SCA/SM study
    python run_multivar_evaluation.py --study iceland      # Run only Iceland SCF trend study
    python run_multivar_evaluation.py --study all          # Run all studies

Arguments:
    --study: Which study to run (default: all)
    --skip-completed: Skip experiments that already have results
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
CONFIG_DIR = BASE_DIR / "configs"
LOG_DIR = BASE_DIR / "logs"
SYMFLUENCE_CLI = "symfluence"

# Data directory
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

# Study configurations
STUDY_CONFIGS = {
    "bow": {
        "name": "Bow River GRACE TWS Anomaly Comparison",
        "description": "Compare TWS anomalies from calibrated/uncalibrated SUMMA against GRACE",
        "domain": "Bow_at_Banff_multivar",
        "configs": {
            "uncalibrated": CONFIG_DIR / "bow_grace_tws" / "config_Bow_SUMMA_uncalibrated_multivar.yaml",
            "calibrated": CONFIG_DIR / "bow_grace_tws" / "config_Bow_SUMMA_calibrated_multivar.yaml",
        },
        "workflow_steps": [
            "setup_project",
            "create_pour_point",
            "acquire_attributes",
            "define_domain",
            "discretize_domain",
            "process_observed_data",
            "acquire_forcings",
            "model_agnostic_preprocessing",
            "model_specific_preprocessing",
            "run_model",
            "postprocess_results",
        ],
    },
    "paradise": {
        "name": "Paradise SCA & Soil Moisture Multivariate Evaluation",
        "description": "Compare simulated SCA vs MODIS and soil moisture vs SMAP",
        "domain": "paradise_multivar",
        "configs": {
            "multivar": CONFIG_DIR / "paradise_sca_sm" / "config_paradise_SUMMA_multivar.yaml",
        },
        "workflow_steps": [
            "setup_project",
            "create_pour_point",
            "acquire_attributes",
            "define_domain",
            "discretize_domain",
            "process_observed_data",
            "acquire_forcings",
            "model_agnostic_preprocessing",
            "model_specific_preprocessing",
            "run_model",
            "postprocess_results",
        ],
    },
    "iceland": {
        "name": "Iceland Region-Wide SCF Trend Study",
        "description": "SCF trend analysis over full MODIS record (2000-2023)",
        "domain": "Iceland_multivar",
        "configs": {
            "scf_trend": CONFIG_DIR / "iceland_scf_trend" / "config_Iceland_SCF_trend_multivar.yaml",
        },
        "workflow_steps": [
            "setup_project",
            "create_pour_point",
            "acquire_attributes",
            "define_domain",
            "discretize_domain",
            "acquire_forcings",
            "model_agnostic_preprocessing",
            "model_specific_preprocessing",
            "run_model",
            "postprocess_results",
        ],
    },
}


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the multivariate evaluation experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"multivar_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("multivar_runner")
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


def check_results_exist(domain_name: str, experiment_id: str) -> bool:
    """Check if results already exist for an experiment."""
    domain_dir = DATA_DIR / f"domain_{domain_name}"
    results_paths = [
        domain_dir / "simulations" / experiment_id,
        domain_dir / "reporting" / experiment_id,
        domain_dir / "results" / experiment_id,
    ]
    return any(p.exists() and any(p.iterdir()) for p in results_paths if p.exists())


def run_study(
    study_key: str,
    logger: logging.Logger,
    skip_completed: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single multivariate evaluation study."""
    study = STUDY_CONFIGS[study_key]
    result = {
        "study": study_key,
        "name": study["name"],
        "status": "pending",
        "sub_results": [],
        "start_time": datetime.now(),
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"Study: {study['name']}")
    logger.info(f"Description: {study['description']}")
    logger.info(f"Domain: {study['domain']}")
    logger.info(f"{'='*70}\n")

    for config_label, config_path in study["configs"].items():
        sub_result = {
            "config": config_label,
            "config_path": str(config_path),
            "status": "pending",
        }

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sub_result["status"] = "failed"
            sub_result["error"] = "Config file not found"
            result["sub_results"].append(sub_result)
            continue

        experiment_id = f"{study_key}_{config_label}"
        if skip_completed and check_results_exist(study["domain"], experiment_id):
            logger.info(f"Skipping {config_label}: results already exist")
            sub_result["status"] = "skipped"
            result["sub_results"].append(sub_result)
            continue

        logger.info(f"\n--- Running configuration: {config_label} ---")

        all_steps_ok = True
        for step in study["workflow_steps"]:
            cmd = [
                SYMFLUENCE_CLI,
                "workflow",
                "step",
                step,
                "--config",
                str(config_path),
            ]

            success = run_command(cmd, logger, dry_run)
            if not success and not dry_run:
                logger.error(f"Step '{step}' failed for {config_label}")
                all_steps_ok = False
                break

        if dry_run:
            sub_result["status"] = "dry_run"
        elif all_steps_ok:
            sub_result["status"] = "completed"
        else:
            sub_result["status"] = "failed"

        result["sub_results"].append(sub_result)

        if not dry_run:
            time.sleep(2)

    # Determine overall study status
    statuses = [sr["status"] for sr in result["sub_results"]]
    if all(s == "completed" for s in statuses):
        result["status"] = "completed"
    elif all(s in ("completed", "skipped", "dry_run") for s in statuses):
        result["status"] = "completed"
    elif any(s == "failed" for s in statuses):
        result["status"] = "failed"
    else:
        result["status"] = "partial"

    result["end_time"] = datetime.now()
    result["duration_seconds"] = (result["end_time"] - result["start_time"]).total_seconds()

    return result


def run_multivar_evaluation(
    studies: Optional[List[str]] = None,
    skip_completed: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Run multivariate evaluation studies."""
    logger = setup_logging(LOG_DIR)
    logger.info("Starting Multivariate Evaluation Experiment (Section 4.10)")
    logger.info(f"Config directory: {CONFIG_DIR}")

    if studies is None or "all" in studies:
        studies = list(STUDY_CONFIGS.keys())
    else:
        invalid = [s for s in studies if s not in STUDY_CONFIGS]
        if invalid:
            logger.error(f"Invalid study names: {invalid}")
            logger.error(f"Valid studies: {list(STUDY_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"Studies to run: {studies}")

    all_results = []
    for study_key in studies:
        result = run_study(study_key, logger, skip_completed, dry_run)
        all_results.append(result)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("MULTIVARIATE EVALUATION RUN SUMMARY")
    logger.info(f"{'='*70}")

    for result in all_results:
        status_icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "partial": "[PART]",
        }.get(result["status"], "[?]")

        duration_str = ""
        if result.get("duration_seconds"):
            duration_str = f" ({result['duration_seconds']:.1f}s)"

        logger.info(f"\n{status_icon} {result['name']}{duration_str}")
        for sr in result["sub_results"]:
            sr_icon = {"completed": "  +", "failed": "  X", "skipped": "  -", "dry_run": "  ~"}.get(
                sr["status"], "  ?"
            )
            logger.info(f"{sr_icon} {sr['config']}: {sr['status']}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multivariate evaluation experiments for SYMFLUENCE paper Section 4.10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Studies:
  bow       - GRACE TWS anomaly comparison (calibrated vs uncalibrated SUMMA)
  paradise  - Simulated SCA vs MODIS + soil moisture vs SMAP
  iceland   - Region-wide SCF trend analysis over full MODIS record (2000-2023)
  all       - Run all studies

Examples:
  python run_multivar_evaluation.py                       # Run all studies
  python run_multivar_evaluation.py --study bow           # Run only Bow TWS study
  python run_multivar_evaluation.py --study paradise      # Run only Paradise study
  python run_multivar_evaluation.py --study iceland       # Run only Iceland SCF study
  python run_multivar_evaluation.py --dry-run             # Preview without running
        """
    )
    parser.add_argument(
        "--study",
        type=str,
        nargs="+",
        default=["all"],
        choices=["bow", "paradise", "iceland", "all"],
        help="Which study/studies to run (default: all)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments that already have results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    results = run_multivar_evaluation(
        studies=args.study,
        skip_completed=args.skip_completed,
        dry_run=args.dry_run,
    )

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
