#!/usr/bin/env python3
"""
Run the 4.4 Calibration Ensemble Study.

Orchestrates calibration experiments across 12 optimization algorithms
using the HBV model on the Bow at Banff basin, with a normalized
function evaluation budget for fair comparison.

Study Parts:
  1 - Core algorithm comparison (all 12 algorithms, single seed)
  2 - Robustness analysis (core algorithms x multiple seeds)
  3 - Gradient vs derivative-free comparison (smooth vs non-smooth HBV)

Usage:
    python run_study.py --part all               # Run all parts
    python run_study.py --part 1                  # Core comparison
    python run_study.py --part 2 --seeds 5        # Robustness (5 seeds)
    python run_study.py --part 3                  # Gradient analysis
    python run_study.py --part 1 --dry-run        # Preview commands
    python run_study.py --algorithm dds,pso       # Run specific algorithms
"""

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Paths
STUDY_DIR = Path(__file__).parent.parent
CONFIG_DIR = STUDY_DIR / "configs"
RESULTS_DIR = STUDY_DIR / "results"
SYMFLUENCE_CLI = "symfluence"

# =============================================================================
# Study part definitions
# =============================================================================

# Part 1: Core algorithm comparison (single seed, all algorithms)
PART1_ALGORITHMS = [
    "dds", "sceua", "de", "pso", "ga", "cmaes",
    "adam", "lbfgs",
    "nelder_mead",
    "sa", "basin_hopping",
    "bayesian_opt",
]

# Part 2: Robustness analysis (subset of algorithms x multiple seeds)
PART2_ALGORITHMS = ["dds", "sceua", "de", "pso", "cmaes", "adam"]

# Part 3: Gradient analysis (smooth vs non-smooth)
PART3_CONFIGS = {
    "gradient_analysis": [
        # Gradient methods require smoothing
        "config_bow_hbv_adam.yaml",
        "config_bow_hbv_lbfgs.yaml",
        # DDS with and without smoothing for comparison
        "config_bow_hbv_dds.yaml",
        # We also generate smooth variants of DDS in the study
    ]
}

STUDY_PARTS = {
    "1": {
        "name": "Core Algorithm Comparison",
        "description": (
            "Compare 12 optimization algorithms with normalized function "
            "evaluation budget (~4000 evaluations each) using HBV on Bow at Banff"
        ),
        "configs": [f"config_bow_hbv_{algo}.yaml" for algo in PART1_ALGORITHMS],
        "steps": ["model_specific_preprocessing", "calibrate_model", "run_benchmarking"],
    },
    "2": {
        "name": "Robustness Analysis",
        "description": (
            "Evaluate algorithm robustness across multiple random seeds "
            "for 6 representative algorithms"
        ),
        "configs": [],  # Populated dynamically based on --seeds
        "steps": ["calibrate_model", "run_benchmarking"],
    },
    "3": {
        "name": "Gradient vs Derivative-Free Analysis",
        "description": (
            "Compare gradient-based (ADAM, L-BFGS with smoothing) vs "
            "derivative-free (DDS) methods, and assess impact of smoothing"
        ),
        "configs": PART3_CONFIGS["gradient_analysis"],
        "steps": ["calibrate_model", "run_benchmarking"],
    },
}


class StudyRunner:
    """Orchestrates the calibration ensemble study execution."""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Execute a shell command."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  {description}")
            print(f"{'=' * 60}")
            print(f"Command: {' '.join(cmd)}")

        if self.dry_run:
            print("[DRY RUN] Would execute command")
            return True

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            if self.verbose and result.stdout:
                # Print last 20 lines of output
                lines = result.stdout.strip().split("\n")
                if len(lines) > 20:
                    print(f"... ({len(lines) - 20} lines omitted)")
                for line in lines[-20:]:
                    print(line)
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command failed with exit code {e.returncode}")
            if e.stderr:
                print(f"STDERR: {e.stderr[:500]}")
            return False

    def run_workflow_step(self, config_file: str, step: str) -> bool:
        """Run a single workflow step for a config."""
        config_path = CONFIG_DIR / config_file
        cmd = [
            SYMFLUENCE_CLI, "workflow", "step", step,
            "--config", str(config_path),
        ]
        description = f"Running {step} for {config_file}"
        return self.run_command(cmd, description)

    def run_study_part(self, part_id: str) -> bool:
        """Run a complete study part."""
        if part_id not in STUDY_PARTS:
            print(f"ERROR: Invalid study part '{part_id}'")
            return False

        part = STUDY_PARTS[part_id]
        print(f"\n{'#' * 70}")
        print(f"# STUDY PART {part_id}: {part['name']}")
        print(f"# {part['description']}")
        print(f"{'#' * 70}")

        success_count = 0
        total_count = len(part["configs"])

        for config_file in part["configs"]:
            config_path = CONFIG_DIR / config_file
            if not config_path.exists():
                print(f"WARNING: Config file not found: {config_file}")
                print("  Run generate_configs.py first to create configs")
                continue

            print(f"\n{'*' * 60}")
            print(f"* Processing: {config_file}")
            print(f"{'*' * 60}")

            all_steps_success = True
            for step in part["steps"]:
                success = self.run_workflow_step(config_file, step)
                if not success:
                    all_steps_success = False
                    print(f"WARNING: Step {step} failed for {config_file}")
                    break

            if all_steps_success:
                success_count += 1

            self.results.append({
                "part": part_id,
                "config": config_file,
                "success": all_steps_success,
                "timestamp": datetime.now().isoformat(),
            })

        print(f"\n{'=' * 60}")
        print(f"Part {part_id} Summary: {success_count}/{total_count} configs successful")
        print(f"{'=' * 60}")
        return success_count == total_count

    def save_results(self):
        """Save execution results to CSV."""
        if not self.results:
            return
        if HAS_PANDAS:
            df = pd.DataFrame(self.results)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = RESULTS_DIR / f"execution_log_{ts}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nExecution log saved to: {output_file}")
        else:
            print("\nInstall pandas to save execution logs as CSV")

    def generate_summary_report(self):
        """Print a summary of all execution results."""
        print(f"\n{'=' * 70}")
        print("CALIBRATION ENSEMBLE STUDY - EXECUTION SUMMARY")
        print(f"{'=' * 70}\n")

        for part_id, part in STUDY_PARTS.items():
            part_results = [r for r in self.results if r["part"] == part_id]
            if not part_results:
                continue

            success = sum(1 for r in part_results if r["success"])
            total = len(part_results)

            print(f"Part {part_id}: {part['name']}")
            print(f"  Status: {success}/{total} successful")
            print("  Configs:")
            for result in part_results:
                status = "+" if result["success"] else "x"
                print(f"    [{status}] {result['config']}")
            print()


def populate_robustness_configs(n_seeds: int):
    """Dynamically populate Part 2 configs for multi-seed robustness."""
    configs = []
    seeds = [42 + i * 1000 for i in range(n_seeds)]
    for algo in PART2_ALGORITHMS:
        for seed in seeds:
            seed_suffix = f"_seed{seed}" if seed != 42 else ""
            configs.append(f"config_bow_hbv_{algo}{seed_suffix}.yaml")
    STUDY_PARTS["2"]["configs"] = configs


def main():
    parser = argparse.ArgumentParser(
        description="Run the Calibration Ensemble Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Study Parts:
  1 - Core Algorithm Comparison (12 algorithms)
  2 - Robustness Analysis (6 algorithms x N seeds)
  3 - Gradient vs Derivative-Free Analysis
  all - Run all study parts

Examples:
  python run_study.py --part all
  python run_study.py --part 1 --dry-run
  python run_study.py --part 2 --seeds 5
  python run_study.py --algorithm dds,pso,adam
        """,
    )
    parser.add_argument(
        "--part", type=str, default="all",
        help="Study part(s) to run: 1, 2, 3, or all (comma-separated)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds for Part 2 robustness analysis (default: 5)",
    )
    parser.add_argument(
        "--algorithm", type=str, default=None,
        help="Run specific algorithms only (comma-separated keys)",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing steps (assumes already done)",
    )

    args = parser.parse_args()

    # Populate multi-seed configs for Part 2
    populate_robustness_configs(args.seeds)

    # If specific algorithms requested, override Part 1 configs
    if args.algorithm:
        algo_keys = [k.strip() for k in args.algorithm.split(",")]
        STUDY_PARTS["1"]["configs"] = [
            f"config_bow_hbv_{algo}.yaml" for algo in algo_keys
        ]

    # Remove preprocessing step if requested
    if args.skip_preprocessing:
        for part in STUDY_PARTS.values():
            part["steps"] = [s for s in part["steps"]
                             if s != "model_specific_preprocessing"]

    runner = StudyRunner(dry_run=args.dry_run, verbose=not args.quiet)

    # Parse parts to run
    if args.part.lower() == "all":
        parts_to_run = list(STUDY_PARTS.keys())
    else:
        parts_to_run = [p.strip() for p in args.part.split(",")]

    print("=" * 70)
    print("CALIBRATION ENSEMBLE STUDY")
    print("=" * 70)
    print(f"Parts to run: {', '.join(parts_to_run)}")
    print(f"Dry run: {args.dry_run}")
    if args.algorithm:
        print(f"Algorithms: {args.algorithm}")
    print()

    for part_id in parts_to_run:
        if part_id in STUDY_PARTS:
            runner.run_study_part(part_id)
        else:
            print(f"WARNING: Unknown part '{part_id}', skipping")

    runner.save_results()
    runner.generate_summary_report()

    print("\n" + "=" * 70)
    print("Study execution complete!")
    print("Next step: python analyze_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
