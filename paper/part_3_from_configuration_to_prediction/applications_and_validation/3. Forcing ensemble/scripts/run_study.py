#!/usr/bin/env python3
"""
Run the 4.3 Forcing Ensemble Study.

This script executes SYMFLUENCE workflows for each forcing dataset configuration,
managing the full pipeline from data acquisition through calibration and evaluation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Study configuration
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Forcing datasets and their configs
FORCING_CONFIGS = {
    'era5': 'config_paradise_era5.yaml',
    'aorc': 'config_paradise_aorc.yaml',
    'hrrr': 'config_paradise_hrrr.yaml',
    'conus404': 'config_paradise_conus404.yaml',
    'rdrs': 'config_paradise_rdrs.yaml',
    # GDDP ensemble members (projections to 2100)
    'gddp_access_cm2': 'config_paradise_gddp_access_cm2.yaml',
    'gddp_gfdl_esm4': 'config_paradise_gddp_gfdl_esm4.yaml',
    'gddp_mri_esm2_0': 'config_paradise_gddp_mri_esm2_0.yaml',
}

# Workflow steps to execute
WORKFLOW_STEPS = [
    'acquire_forcings',
    'model_agnostic_preprocessing',
    'model_specific_preprocessing',
    'calibrate_model',
    'run_benchmarking',
]


def run_workflow_step(config_path: Path, step: str, dry_run: bool = False) -> bool:
    """
    Run a single workflow step for a configuration.

    Args:
        config_path: Path to configuration file
        step: Workflow step name
        dry_run: If True, only print command without executing

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'symfluence', 'workflow', 'step', step,
        '--config', str(config_path)
    ]

    print(f"  Step: {step}")
    if dry_run:
        print(f"    [DRY RUN] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("    [SUCCESS]")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [FAILED] {e.stderr[:200] if e.stderr else 'Unknown error'}")
        return False
    except FileNotFoundError:
        print("    [ERROR] symfluence command not found. Is SYMFLUENCE installed?")
        return False


def run_forcing_experiment(
    forcing: str,
    steps: Optional[List[str]] = None,
    dry_run: bool = False
) -> bool:
    """
    Run the complete experiment for a forcing dataset.

    Args:
        forcing: Forcing dataset name (era5, aorc, hrrr, conus404, rdrs,
                 gddp_access_cm2, gddp_gfdl_esm4, gddp_mri_esm2_0)
        steps: Optional list of specific steps to run
        dry_run: If True, only print commands without executing

    Returns:
        True if all steps successful, False otherwise
    """
    config_file = FORCING_CONFIGS.get(forcing)
    if not config_file:
        print(f"[ERROR] Unknown forcing dataset: {forcing}")
        return False

    config_path = CONFIGS_DIR / config_file
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print("  Run generate_configs.py first!")
        return False

    print(f"\n{'='*60}")
    print(f"Running experiment: {forcing.upper()}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    steps_to_run = steps if steps else WORKFLOW_STEPS
    all_success = True

    for step in steps_to_run:
        success = run_workflow_step(config_path, step, dry_run)
        if not success:
            all_success = False
            if not dry_run:
                print(f"\n[STOPPING] Step '{step}' failed for {forcing}")
                break

    return all_success


def main():
    """Main entry point for the study runner."""
    parser = argparse.ArgumentParser(
        description='Run the 4.3 Forcing Ensemble Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all standard forcing experiments (ERA5, AORC, HRRR, CONUS404, RDRS)
  python run_study.py --forcing all

  # Run specific forcing
  python run_study.py --forcing era5
  python run_study.py --forcing rdrs

  # Run multiple forcings
  python run_study.py --forcing era5 aorc

  # Run all GDDP ensemble members (projections to 2100)
  python run_study.py --forcing all_gddp

  # Run specific GDDP ensemble member
  python run_study.py --forcing gddp_access_cm2

  # Run specific steps only
  python run_study.py --forcing all --steps acquire_forcing calibrate_model

  # Dry run (show commands without executing)
  python run_study.py --forcing all --dry-run
        """
    )

    parser.add_argument(
        '--forcing', '-f',
        nargs='+',
        required=True,
        choices=['all', 'all_gddp', 'era5', 'aorc', 'hrrr', 'conus404', 'rdrs',
                 'gddp_access_cm2', 'gddp_gfdl_esm4', 'gddp_mri_esm2_0'],
        help='Forcing dataset(s) to run'
    )

    parser.add_argument(
        '--steps', '-s',
        nargs='+',
        choices=WORKFLOW_STEPS,
        help='Specific workflow steps to run (default: all steps)'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print commands without executing'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue with next forcing even if current fails'
    )

    args = parser.parse_args()

    # Determine which forcings to run
    # Standard forcings (non-GDDP)
    standard_forcings = ['era5', 'aorc', 'hrrr', 'conus404', 'rdrs']
    # GDDP ensemble members
    gddp_forcings = ['gddp_access_cm2', 'gddp_gfdl_esm4', 'gddp_mri_esm2_0']

    if 'all' in args.forcing:
        # 'all' runs only standard forcings (not GDDP projections)
        forcings = standard_forcings
    elif 'all_gddp' in args.forcing:
        # 'all_gddp' runs all GDDP ensemble members
        forcings = gddp_forcings
    else:
        forcings = args.forcing

    print("=" * 60)
    print("4.3 Forcing Ensemble Study")
    print("=" * 60)
    print(f"\nForcings to run: {', '.join(f.upper() for f in forcings)}")
    print(f"Steps: {', '.join(args.steps) if args.steps else 'all'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Continue on error: {args.continue_on_error}")

    # Run experiments
    results = {}
    for forcing in forcings:
        success = run_forcing_experiment(
            forcing,
            steps=args.steps,
            dry_run=args.dry_run
        )
        results[forcing] = success

        if not success and not args.continue_on_error and not args.dry_run:
            print("\n[STOPPING] Experiment failed. Use --continue-on-error to proceed.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("Study Summary")
    print("=" * 60)
    for forcing, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {forcing.upper():12} : {status}")

    # Exit with appropriate code
    if all(results.values()):
        print("\nAll experiments completed successfully!")
        sys.exit(0)
    else:
        print("\nSome experiments failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
