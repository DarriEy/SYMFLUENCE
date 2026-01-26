#!/usr/bin/env python3
"""
Bow at Banff HBV Study - Main Execution Script

This script orchestrates the complete HBV study including:
1. Daily vs Hourly timestep comparison
2. Optimization algorithm comparison (DDS, PSO, DE, GA, ADAM)
3. Differentiability analysis (smoothing vs non-smoothing)
4. Gradient method comparison (ODE vs Direct AD vs FD)

Usage:
    python run_study.py --part all            # Run all study parts
    python run_study.py --part 1              # Run only timestep comparison
    python run_study.py --part 2              # Run only optimizer comparison
    python run_study.py --part 3              # Run only differentiability analysis
    python run_study.py --dry-run             # Show what would be executed
"""

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime

# Paths
STUDY_DIR = Path(__file__).parent.parent
CONFIG_DIR = STUDY_DIR / "configs"
RESULTS_DIR = STUDY_DIR / "results"
SYMFLUENCE_CLI = "symfluence"  # Assumes symfluence is in PATH

# Study parts configuration
STUDY_PARTS = {
    '1': {
        'name': 'Daily vs Hourly Comparison',
        'description': 'Compare model performance with daily (24h) vs hourly (1h) timesteps',
        'configs': [
            'config_bow_hbv_daily_dds.yaml',
            'config_bow_hbv_hourly_dds.yaml',
        ],
        'steps': ['model_specific_preprocessing', 'calibrate_model', 'run_benchmarking'],
    },
    '2': {
        'name': 'Optimization Algorithm Comparison',
        'description': 'Compare DDS, PSO, DE, GA, and ADAM with 4000 iterations each',
        'configs': [
            'config_bow_hbv_daily_dds.yaml',
            'config_bow_hbv_daily_pso.yaml',
            'config_bow_hbv_daily_de.yaml',
            'config_bow_hbv_daily_ga.yaml',
            'config_bow_hbv_daily_adam.yaml',
        ],
        'steps': ['model_specific_preprocessing', 'calibrate_model', 'run_benchmarking'],
    },
    '3': {
        'name': 'Differentiability Analysis',
        'description': 'Compare smoothing vs non-smoothing and gradient methods',
        'configs': [
            'config_bow_hbv_daily_dds_smooth.yaml',
            'config_bow_hbv_daily_dds_nosmooth.yaml',
            'config_bow_hbv_daily_adam_smooth.yaml',
            'config_bow_hbv_daily_adam_nosmooth.yaml',
        ],
        'steps': ['model_specific_preprocessing', 'calibrate_model', 'run_benchmarking'],
    },
}


class StudyRunner:
    """Orchestrates the HBV study execution."""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Execute a shell command."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  {description}")
            print(f"{'='*60}")
            print(f"Command: {' '.join(cmd)}")

        if self.dry_run:
            print("[DRY RUN] Would execute command")
            return True

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            if self.verbose and result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command failed with exit code {e.returncode}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False

    def run_workflow_step(self, config_file: str, step: str) -> bool:
        """Run a single workflow step."""
        config_path = CONFIG_DIR / config_file
        cmd = [
            SYMFLUENCE_CLI,
            'workflow',
            'step',
            step,
            '--config',
            str(config_path)
        ]
        description = f"Running {step} for {config_file}"
        return self.run_command(cmd, description)

    def run_study_part(self, part_id: str) -> bool:
        """Run a complete study part."""
        if part_id not in STUDY_PARTS:
            print(f"ERROR: Invalid study part '{part_id}'")
            return False

        part = STUDY_PARTS[part_id]
        print(f"\n{'#'*70}")
        print(f"# STUDY PART {part_id}: {part['name']}")
        print(f"# {part['description']}")
        print(f"{'#'*70}")

        success_count = 0
        total_count = len(part['configs'])

        for config_file in part['configs']:
            config_path = CONFIG_DIR / config_file
            if not config_path.exists():
                print(f"WARNING: Config file not found: {config_file}")
                continue

            print(f"\n{'*'*60}")
            print(f"* Processing: {config_file}")
            print(f"{'*'*60}")

            # Run each workflow step
            all_steps_success = True
            for step in part['steps']:
                success = self.run_workflow_step(config_file, step)
                if not success:
                    all_steps_success = False
                    print(f"WARNING: Step {step} failed for {config_file}")
                    break

            if all_steps_success:
                success_count += 1
                print(f"✓ Successfully completed: {config_file}")
            else:
                print(f"✗ Failed: {config_file}")

            # Record result
            self.results.append({
                'part': part_id,
                'config': config_file,
                'success': all_steps_success,
                'timestamp': datetime.now().isoformat()
            })

        # Summary
        print(f"\n{'='*60}")
        print(f"Part {part_id} Summary: {success_count}/{total_count} configs successful")
        print(f"{'='*60}")

        return success_count == total_count

    def run_gradient_comparison(self) -> bool:
        """Run ODE vs Direct AD vs Finite Difference gradient comparison."""
        print(f"\n{'#'*70}")
        print("# GRADIENT COMPARISON: ODE vs Direct AD vs Finite Difference")
        print(f"{'#'*70}")

        # This uses the compare_solvers.py module
        cmd = [
            'python', '-m',
            'symfluence.models.hbv.compare_solvers',
            '--n-days', '365',
            '--timestep', '24',
            '--save-plot', str(RESULTS_DIR / 'gradient_comparison.png')
        ]
        description = "Comparing gradient computation methods"
        return self.run_command(cmd, description)

    def save_results(self):
        """Save study results to CSV."""
        if not self.results:
            return

        results_df = pd.DataFrame(self.results)
        output_file = RESULTS_DIR / f"study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    def generate_summary_report(self):
        """Generate a summary report of all study results."""
        print(f"\n{'='*70}")
        print("STUDY SUMMARY REPORT")
        print(f"{'='*70}\n")

        for part_id, part in STUDY_PARTS.items():
            part_results = [r for r in self.results if r['part'] == part_id]
            if not part_results:
                continue

            success_count = sum(1 for r in part_results if r['success'])
            total_count = len(part_results)

            print(f"Part {part_id}: {part['name']}")
            print(f"  Status: {success_count}/{total_count} successful")
            print("  Configs:")
            for result in part_results:
                status = "✓" if result['success'] else "✗"
                print(f"    {status} {result['config']}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Bow at Banff HBV Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Study Parts:
  1 - Daily vs Hourly Comparison
  2 - Optimization Algorithm Comparison
  3 - Differentiability Analysis
  all - Run all study parts
  gradients - Run gradient comparison only

Examples:
  python run_study.py --part all
  python run_study.py --part 1
  python run_study.py --part 2,3
  python run_study.py --part gradients --dry-run
        """
    )
    parser.add_argument(
        '--part',
        type=str,
        default='all',
        help='Study part(s) to run: 1, 2, 3, gradients, or all (comma-separated for multiple)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing steps (assumes already done)'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = StudyRunner(dry_run=args.dry_run, verbose=not args.quiet)

    # Parse parts to run
    if args.part.lower() == 'all':
        parts_to_run = list(STUDY_PARTS.keys())
        run_gradients = True
    elif args.part.lower() == 'gradients':
        parts_to_run = []
        run_gradients = True
    else:
        parts_to_run = args.part.split(',')
        run_gradients = 'gradients' in [p.lower() for p in parts_to_run]
        parts_to_run = [p for p in parts_to_run if p in STUDY_PARTS]

    # Run study parts
    for part_id in parts_to_run:
        runner.run_study_part(part_id)

    # Run gradient comparison
    if run_gradients:
        runner.run_gradient_comparison()

    # Generate reports
    runner.save_results()
    runner.generate_summary_report()

    print("\n" + "="*70)
    print("Study execution complete!")
    print("="*70)


if __name__ == "__main__":
    main()
