#!/usr/bin/env python3
"""
Attribute Analysis Runner for SYMFLUENCE Paper Section 4.13

This script orchestrates the full attribute analysis experiment across 9 discretization
scenarios on the Bow River at Banff domain. For each scenario, the workflow:
  1. Acquires geospatial attributes (DEM, soil, land cover)
  2. Computes zonal statistics per catchment
  3. Discretizes the domain into HRUs based on scenario-specific attributes
  4. Runs model-agnostic and model-specific preprocessing
  5. Calibrates HBV via DDS (1000 iterations)
  6. Runs the calibrated model on the evaluation period

The experiment isolates the effect of attribute-based discretization on hydrological
model performance by varying only the discretization configuration across scenarios
while keeping all other settings (forcing, model structure, calibration budget) identical.

Usage:
    python run_attribute_analysis.py [--scenarios S1,S2,...] [--dry-run]
    python run_attribute_analysis.py --scenarios elevation_200m,landclass
    python run_attribute_analysis.py --skip-completed
    python run_attribute_analysis.py --preprocessing-only    # Only run attribute/discretization steps

Arguments:
    --scenarios: Comma-separated list of scenarios to run (default: all)
    --skip-completed: Skip scenarios that already have calibration results
    --preprocessing-only: Only run attribute acquisition and discretization (no calibration)
    --dry-run: Print what would be run without executing
"""

import argparse
import json
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
DOMAIN_DIR = DATA_DIR / "domain_Bow_at_Banff_attribute_analysis"

# Scenario configurations mapping scenario name to config filename and experiment ID
SCENARIO_CONFIGS = {
    "lumped_baseline": {
        "config": "config_Bow_attribute_lumped_baseline.yaml",
        "experiment_id": "attr_lumped_baseline",
        "description": "Lumped baseline (1 HRU)",
    },
    "elevation_200m": {
        "config": "config_Bow_attribute_elevation_200m.yaml",
        "experiment_id": "attr_elevation_200m",
        "description": "Elevation bands (200 m)",
    },
    "elevation_400m": {
        "config": "config_Bow_attribute_elevation_400m.yaml",
        "experiment_id": "attr_elevation_400m",
        "description": "Elevation bands (400 m)",
    },
    "landclass": {
        "config": "config_Bow_attribute_landclass.yaml",
        "experiment_id": "attr_landclass",
        "description": "Land cover classes",
    },
    "soilclass": {
        "config": "config_Bow_attribute_soilclass.yaml",
        "experiment_id": "attr_soilclass",
        "description": "Soil type classes",
    },
    "aspect": {
        "config": "config_Bow_attribute_aspect.yaml",
        "experiment_id": "attr_aspect",
        "description": "8-class aspect",
    },
    "radiation": {
        "config": "config_Bow_attribute_radiation.yaml",
        "experiment_id": "attr_radiation",
        "description": "5-class radiation",
    },
    "elev_land": {
        "config": "config_Bow_attribute_elev_land.yaml",
        "experiment_id": "attr_elev_land",
        "description": "Elevation + land cover",
    },
    "elev_soil_land": {
        "config": "config_Bow_attribute_elev_soil_land.yaml",
        "experiment_id": "attr_elev_soil_land",
        "description": "Elevation + soil + land cover",
    },
}

# Workflow steps for full attribute analysis experiment
FULL_WORKFLOW_STEPS = [
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
    "process_observed_data",
    "acquire_forcings",
    "model_agnostic_preprocessing",
    "model_specific_preprocessing",
    "calibrate_model",
    "run_model",
    "postprocess_results",
]

# Preprocessing-only steps (attribute acquisition and discretization)
PREPROCESSING_STEPS = [
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
]


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the attribute analysis experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"attribute_analysis_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("attribute_analysis_runner")
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


def check_results_exist(experiment_id: str) -> bool:
    """Check if calibration/simulation results already exist for a scenario."""
    search_paths = [
        DOMAIN_DIR / "optimization" / experiment_id,
        DOMAIN_DIR / "simulations" / experiment_id,
    ]
    for path in search_paths:
        if path.exists() and any(path.iterdir()):
            return True
    return False


def check_discretization_exists(experiment_id: str) -> bool:
    """Check if HRU shapefiles already exist for a scenario."""
    hru_dir = DOMAIN_DIR / "shapefiles" / "catchment"
    if not hru_dir.exists():
        return False
    for f in hru_dir.rglob(f"*{experiment_id}*HRUs*"):
        return True
    return False


def count_hrus(experiment_id: str) -> Optional[int]:
    """Count the number of HRUs produced by a discretization scenario."""
    hru_dir = DOMAIN_DIR / "shapefiles" / "catchment"
    if not hru_dir.exists():
        return None

    try:
        import geopandas as gpd

        for shp in hru_dir.rglob(f"*{experiment_id}*HRUs*.shp"):
            gdf = gpd.read_file(shp)
            return len(gdf)
        for gpkg in hru_dir.rglob(f"*{experiment_id}*HRUs*.gpkg"):
            gdf = gpd.read_file(gpkg)
            return len(gdf)
    except ImportError:
        pass
    except Exception:
        pass

    return None


def run_scenario(
    scenario_name: str,
    config_path: Path,
    workflow_steps: List[str],
    logger: logging.Logger,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the full workflow for a single attribute analysis scenario."""
    scenario = SCENARIO_CONFIGS[scenario_name]
    result = {
        "scenario": scenario_name,
        "experiment_id": scenario["experiment_id"],
        "description": scenario["description"],
        "config": str(config_path),
        "status": "pending",
        "steps_completed": [],
        "steps_failed": [],
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "n_hrus": None,
        "error": None,
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"Scenario: {scenario_name} - {scenario['description']}")
    logger.info(f"Experiment ID: {scenario['experiment_id']}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Workflow steps: {len(workflow_steps)}")
    logger.info(f"{'='*70}\n")

    result["start_time"] = datetime.now()

    try:
        for step in workflow_steps:
            logger.info(f"--- Step: {step} ---")

            cmd = [
                SYMFLUENCE_CLI,
                "workflow",
                "step",
                step,
                "--config",
                str(config_path),
            ]

            success = run_command(cmd, logger, dry_run)

            if dry_run:
                result["steps_completed"].append(step)
            elif success:
                result["steps_completed"].append(step)
                logger.info(f"Step '{step}' completed successfully")
            else:
                result["steps_failed"].append(step)
                logger.error(f"Step '{step}' FAILED for scenario {scenario_name}")
                result["error"] = f"Step '{step}' failed"
                break

        # Check HRU count after discretization
        result["n_hrus"] = count_hrus(scenario["experiment_id"])
        if result["n_hrus"] is not None:
            logger.info(f"HRU count for {scenario_name}: {result['n_hrus']}")

        result["end_time"] = datetime.now()
        result["duration_seconds"] = (result["end_time"] - result["start_time"]).total_seconds()

        if dry_run:
            result["status"] = "dry_run"
        elif result["steps_failed"]:
            result["status"] = "failed"
        else:
            result["status"] = "completed"

    except Exception as e:
        result["end_time"] = datetime.now()
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Scenario {scenario_name} failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return result


def run_attribute_analysis(
    scenarios: Optional[List[str]] = None,
    skip_completed: bool = False,
    preprocessing_only: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Run the attribute analysis experiment across all scenarios."""
    logger = setup_logging(LOG_DIR)
    logger.info("Starting Attribute Analysis Experiment (Section 4.13)")
    logger.info("Domain: Bow River at Banff")
    logger.info(f"Config directory: {CONFIG_DIR}")
    logger.info(f"Data directory: {DOMAIN_DIR}")

    # Select workflow steps
    workflow_steps = PREPROCESSING_STEPS if preprocessing_only else FULL_WORKFLOW_STEPS
    logger.info(f"Mode: {'preprocessing only' if preprocessing_only else 'full workflow'}")

    # Default to all scenarios
    if scenarios is None:
        scenarios = list(SCENARIO_CONFIGS.keys())
    else:
        invalid = [s for s in scenarios if s not in SCENARIO_CONFIGS]
        if invalid:
            logger.error(f"Invalid scenario names: {invalid}")
            logger.error(f"Valid scenarios: {list(SCENARIO_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"Scenarios to run: {scenarios}")

    # Pre-flight checks
    logger.info("\nPre-flight checks:")
    for scenario_name in scenarios:
        scenario = SCENARIO_CONFIGS[scenario_name]
        config_path = CONFIG_DIR / scenario["config"]
        has_config = config_path.exists()
        has_results = check_results_exist(scenario["experiment_id"])
        has_hrus = check_discretization_exists(scenario["experiment_id"])
        logger.info(
            f"  {scenario_name}: config={'YES' if has_config else 'NO'}, "
            f"discretization={'YES' if has_hrus else 'NO'}, "
            f"results={'YES' if has_results else 'NO'}"
        )
        if not has_config:
            logger.warning(
                f"  WARNING: Config not found: {config_path}. "
                f"Run generate_configs.py first."
            )

    # Run each scenario
    all_results = []
    for scenario_name in scenarios:
        scenario = SCENARIO_CONFIGS[scenario_name]
        config_path = CONFIG_DIR / scenario["config"]

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            all_results.append({
                "scenario": scenario_name,
                "status": "skipped",
                "error": "Config file not found",
            })
            continue

        if skip_completed and check_results_exist(scenario["experiment_id"]):
            logger.info(f"Skipping {scenario_name}: results already exist")
            all_results.append({
                "scenario": scenario_name,
                "status": "skipped",
                "error": "Results already exist",
            })
            continue

        result = run_scenario(scenario_name, config_path, workflow_steps, logger, dry_run)
        all_results.append(result)

        if not dry_run and result["status"] == "completed":
            time.sleep(2)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("ATTRIBUTE ANALYSIS RUN SUMMARY")
    logger.info(f"{'='*70}")

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

        hru_str = ""
        if result.get("n_hrus") is not None:
            hru_str = f" [{result['n_hrus']} HRUs]"

        scenario_desc = SCENARIO_CONFIGS.get(result.get("scenario", ""), {}).get("description", "")
        logger.info(f"{status_icon} {result.get('scenario', '?')} - {scenario_desc}{hru_str}{duration_str}")

        if result.get("error"):
            logger.info(f"      Error: {result['error']}")

    # Save results summary as JSON
    summary_path = BASE_DIR / "analysis" / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in all_results:
        sr = {k: (str(v) if isinstance(v, (datetime, Path)) else v) for k, v in r.items()}
        serializable.append(sr)
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"\nRun summary saved to: {summary_path}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run attribute analysis experiment for SYMFLUENCE paper Section 4.13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Discretization Scenarios:
  lumped_baseline   - No sub-grid discretization (1 HRU = 1 GRU)
  elevation_200m    - Elevation bands at 200 m intervals
  elevation_400m    - Elevation bands at 400 m intervals
  landclass         - Land cover classification only
  soilclass         - Soil type classification only
  aspect            - 8-class aspect (N, NE, E, SE, S, SW, W, NW)
  radiation         - 5-class annual radiation
  elev_land         - Combined elevation (200 m) + land cover
  elev_soil_land    - Combined elevation + soil + land cover

Workflow Steps (full):
  1. setup_project                  5. discretize_domain        9.  model_specific_preprocessing
  2. create_pour_point              6. process_observed_data    10. calibrate_model
  3. acquire_attributes             7. acquire_forcings         11. run_model
  4. define_domain                  8. model_agnostic_preproc   12. postprocess_results

Examples:
  python run_attribute_analysis.py                                  # Run all scenarios
  python run_attribute_analysis.py --scenarios elevation_200m,aspect # Run specific scenarios
  python run_attribute_analysis.py --preprocessing-only              # Only discretize (no model runs)
  python run_attribute_analysis.py --skip-completed                  # Skip finished scenarios
  python run_attribute_analysis.py --dry-run                         # Preview without running
        """
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenarios to run (default: all)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip scenarios that already have results",
    )
    parser.add_argument(
        "--preprocessing-only",
        action="store_true",
        help="Only run attribute acquisition and domain discretization steps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    results = run_attribute_analysis(
        scenarios=scenarios,
        skip_completed=args.skip_completed,
        preprocessing_only=args.preprocessing_only,
        dry_run=args.dry_run,
    )

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
