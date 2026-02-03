#!/usr/bin/env python3
"""
Data Processing Pipeline Experiment Runner for SYMFLUENCE Paper Section 4.12

This script orchestrates the data processing pipeline experiment, running each
pipeline stage individually and profiling execution time, data volumes, and
intermediate outputs. The experiment demonstrates SYMFLUENCE's end-to-end data
processing capabilities across two discretizations (lumped vs semi-distributed).

The experiment profiles six pipeline stages:
  1. Attribute acquisition (DEM, soil, land cover)
  2. Forcing data acquisition (ERA5 retrieval)
  3. Spatial remapping (EASYMORE weight generation + application)
  4. Variable standardization (raw → CFIF → model format)
  5. Observation data processing (streamflow, snow, ET, GRACE)
  6. Geospatial statistics (zonal attribute aggregation)

Usage:
    python run_pipeline_experiment.py [--configs CONFIG1,CONFIG2] [--dry-run]
                                      [--steps STEP1,STEP2,...] [--skip-existing]
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "configs"
ANALYSIS_DIR = BASE_DIR / "analysis"
LOG_DIR = BASE_DIR / "logs"
SYMFLUENCE_CLI = "symfluence"

# Pipeline stage definitions: (step_name, description, category)
# Step names must match `symfluence workflow list-steps` exactly.
PIPELINE_STAGES = [
    ("setup_project", "Project initialisation", "setup"),
    ("create_pour_point", "Pour-point creation", "setup"),
    ("acquire_attributes", "Geospatial attribute acquisition", "acquisition"),
    ("define_domain", "Domain definition", "acquisition"),
    ("discretize_domain", "Domain discretisation", "acquisition"),
    ("process_observed_data", "Observation data processing", "observation"),
    ("acquire_forcings", "Forcing data acquisition", "acquisition"),
    ("model_agnostic_preprocessing", "Spatial remapping & variable standardisation", "preprocessing"),
    ("model_specific_preprocessing", "Model-format conversion", "preprocessing"),
]

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "lumped": "config_Bow_pipeline_era5.yaml",
    "distributed": "config_Bow_pipeline_distributed.yaml",
    "paradise": "config_Paradise_pipeline_era5.yaml",
    "iceland": "config_Iceland_pipeline_era5.yaml",
}


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the pipeline experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_experiment_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("pipeline_experiment")
    logger.info(f"Logging to: {log_file}")
    return logger


def get_directory_size(path: Path) -> int:
    """Compute total size (bytes) of all files under a directory."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def count_files(path: Path, pattern: str = "*") -> int:
    """Count files matching pattern under a directory."""
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def format_bytes(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def run_workflow_step(
    config_path: Path,
    step_name: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Execute a single SYMFLUENCE workflow step and return (success, elapsed_seconds).
    """
    cmd = [SYMFLUENCE_CLI, "workflow", "step", step_name, "--config", str(config_path)]
    logger.info(f"Running: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute command")
        return True, 0.0

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"Step '{step_name}' completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"Step '{step_name}' FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False, elapsed
    except FileNotFoundError:
        logger.error(f"'{SYMFLUENCE_CLI}' not found in PATH")
        return False, 0.0


def profile_data_directory(
    data_dir: Path,
    domain_name: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Scan the SYMFLUENCE data directory for a domain and report sizes and file
    counts for each data category.
    """
    domain_dir = data_dir / f"domain_{domain_name}"
    categories = {
        "settings": domain_dir / "settings",
        "shapefiles": domain_dir / "shapefiles",
        "attributes": domain_dir / "attributes",
        "forcing_raw": domain_dir / "forcing" / "raw_data",
        "forcing_basin_avg": domain_dir / "forcing" / "basin_averaged_data",
        "observations": domain_dir / "observations",
        "parameters": domain_dir / "parameters",
    }

    profile = {}
    for cat_name, cat_path in categories.items():
        size = get_directory_size(cat_path)
        n_nc = count_files(cat_path, "*.nc")
        n_csv = count_files(cat_path, "*.csv")
        n_total = count_files(cat_path)
        profile[cat_name] = {
            "path": str(cat_path),
            "exists": cat_path.exists(),
            "total_bytes": size,
            "total_human": format_bytes(size),
            "n_netcdf": n_nc,
            "n_csv": n_csv,
            "n_files_total": n_total,
        }
        if cat_path.exists():
            logger.info(f"  {cat_name}: {format_bytes(size)} ({n_total} files, {n_nc} NetCDF)")

    return profile


def run_pipeline_for_config(
    config_name: str,
    config_path: Path,
    steps: Optional[List[str]],
    logger: logging.Logger,
    dry_run: bool = False,
    skip_existing: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute all pipeline stages for one configuration and collect profiling data.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Configuration: {config_name} ({config_path.name})")
    logger.info(f"{'='*70}")

    results = {
        "config_name": config_name,
        "config_file": config_path.name,
        "start_time": datetime.now().isoformat(),
        "stages": {},
    }

    # Pre-run directory snapshot
    if data_dir:
        logger.info("Pre-run data directory snapshot:")
        results["pre_profile"] = profile_data_directory(data_dir, config_name, logger)

    # Run each stage
    for step_name, description, category in PIPELINE_STAGES:
        if steps and step_name not in steps:
            logger.info(f"Skipping '{step_name}' (not in requested steps)")
            continue

        logger.info(f"\n--- Stage: {description} ({step_name}) ---")

        # Pre-stage size
        pre_size = get_directory_size(data_dir) if data_dir else 0

        success, elapsed = run_workflow_step(config_path, step_name, logger, dry_run)

        # Post-stage size
        post_size = get_directory_size(data_dir) if data_dir else 0

        results["stages"][step_name] = {
            "description": description,
            "category": category,
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "data_produced_bytes": post_size - pre_size,
            "data_produced_human": format_bytes(max(0, post_size - pre_size)),
        }

    # Post-run directory snapshot
    if data_dir:
        logger.info("\nPost-run data directory snapshot:")
        results["post_profile"] = profile_data_directory(data_dir, config_name, logger)

    results["end_time"] = datetime.now().isoformat()
    return results


def generate_summary_report(
    all_results: Dict[str, Dict],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Write a human-readable summary report alongside the JSON output."""
    lines = []
    lines.append("=" * 70)
    lines.append("SYMFLUENCE Data Processing Pipeline — Experiment Summary")
    lines.append("=" * 70)
    lines.append("")

    for config_name, res in all_results.items():
        lines.append(f"Configuration: {config_name}")
        lines.append(f"  Config file : {res['config_file']}")
        lines.append(f"  Start       : {res['start_time']}")
        lines.append(f"  End         : {res.get('end_time', 'N/A')}")
        lines.append("")

        total_time = 0.0
        lines.append(f"  {'Stage':<40} {'Time (s)':>10} {'Data produced':>15} {'Status':>8}")
        lines.append(f"  {'-'*40} {'-'*10} {'-'*15} {'-'*8}")
        for step_name, stage in res.get("stages", {}).items():
            status = "OK" if stage["success"] else "FAIL"
            total_time += stage["elapsed_seconds"]
            lines.append(
                f"  {stage['description']:<40} {stage['elapsed_seconds']:>10.1f} "
                f"{stage['data_produced_human']:>15} {status:>8}"
            )
        lines.append(f"  {'TOTAL':<40} {total_time:>10.1f}")
        lines.append("")

        # Data directory summary
        post = res.get("post_profile", {})
        if post:
            lines.append("  Data directory contents:")
            for cat, info in post.items():
                if info["exists"]:
                    lines.append(
                        f"    {cat:<25} {info['total_human']:>10}  "
                        f"({info['n_files_total']} files, {info['n_netcdf']} NetCDF)"
                    )
            lines.append("")

    report_path = output_path.with_suffix(".txt")
    report_path.write_text("\n".join(lines))
    logger.info(f"Summary report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SYMFLUENCE data processing pipeline experiment (Section 4.12)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="paradise,lumped,distributed,iceland",
        help="Comma-separated config names to run (default: paradise,lumped,distributed,iceland)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated pipeline steps to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-completed steps")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data",
        help="SYMFLUENCE data directory for profiling",
    )
    args = parser.parse_args()

    logger = setup_logging(LOG_DIR)
    logger.info("SYMFLUENCE Data Processing Pipeline Experiment — Section 4.12")

    requested_configs = [c.strip() for c in args.configs.split(",")]
    requested_steps = [s.strip() for s in args.steps.split(",")] if args.steps else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    all_results = {}
    for config_name in requested_configs:
        config_file = EXPERIMENT_CONFIGS.get(config_name)
        if not config_file:
            logger.warning(f"Unknown config '{config_name}', skipping")
            continue
        config_path = CONFIG_DIR / config_file
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            continue

        all_results[config_name] = run_pipeline_for_config(
            config_name=config_name,
            config_path=config_path,
            steps=requested_steps,
            logger=logger,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            data_dir=data_dir,
        )

    # Save results
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ANALYSIS_DIR / f"pipeline_profile_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Profiling results saved to: {output_path}")

    generate_summary_report(all_results, output_path, logger)
    logger.info("Pipeline experiment complete.")


if __name__ == "__main__":
    main()
