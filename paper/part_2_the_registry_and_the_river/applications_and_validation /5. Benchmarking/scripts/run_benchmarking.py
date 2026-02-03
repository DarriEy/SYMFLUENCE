#!/usr/bin/env python3
"""
Benchmarking Runner for SYMFLUENCE Paper Section 4.5

This script runs HydroBM benchmarks for the Bow at Banff domain to establish
performance baselines (mean flow, climatology, rainfall-runoff ratios, etc.)
against which the multi-model ensemble from Section 4.2 can be compared.

The benchmarking uses observed streamflow and ERA5 forcing data to compute
benchmark flows and performance scores following Schaefli & Gupta (2007)
and Knoben et al. (2020).

Usage:
    python run_benchmarking.py [--dry-run] [--config CONFIG_PATH]

Arguments:
    --config: Path to configuration file (default: ../config/config_Bow_benchmark_era5.yaml)
    --dry-run: Print what would be run without executing
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG_DIR = Path(__file__).parent.parent / "config"
LOG_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_CONFIG = CONFIG_DIR / "config_Bow_benchmark_era5.yaml"
SYMFLUENCE_CLI = "symfluence"


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the benchmarking experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmarking_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("benchmarking_runner")
    logger.info(f"Logging to: {log_file}")
    return logger


def run_command(cmd: list, logger: logging.Logger, dry_run: bool = False) -> bool:
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


def run_benchmarking(config_path: Path, dry_run: bool = False):
    """
    Run the benchmarking workflow for the Bow at Banff domain.

    This executes the SYMFLUENCE benchmarking workflow step, which:
    1. Preprocesses observed streamflow and ERA5 forcing data
    2. Runs HydroBM to compute benchmark flows for 18+ reference models
    3. Computes NSE, KGE, MSE, RMSE for each benchmark
    4. Saves results to evaluation/benchmark_scores.csv

    Args:
        config_path: Path to the YAML configuration file
        dry_run: If True, print commands without executing
    """
    logger = setup_logging(LOG_DIR)
    logger.info("=" * 60)
    logger.info("SYMFLUENCE Benchmarking Experiment - Section 4.5")
    logger.info("Domain: Bow River at Banff (lumped, 2,210 km2)")
    logger.info("Forcing: ERA5")
    logger.info("Reference: Schaefli & Gupta (2007), Knoben et al. (2020)")
    logger.info("=" * 60)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Configuration: {config_path}")

    # Step 1: Run benchmarking
    logger.info("\nStep 1: Running HydroBM benchmarking...")
    cmd = [
        SYMFLUENCE_CLI,
        "workflow",
        "step",
        "run_benchmarking",
        "--config",
        str(config_path),
    ]

    success = run_command(cmd, logger, dry_run)

    if success:
        logger.info("\nBenchmarking completed successfully.")

        # Report expected output locations
        data_dir = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
        eval_dir = data_dir / "domain_Bow_at_Banff_lumped_era5" / "evaluation"
        logger.info("\nExpected output files:")
        logger.info(f"  Benchmark input data: {eval_dir / 'benchmark_input_data.csv'}")
        logger.info(f"  Benchmark flows:      {eval_dir / 'benchmark_flows.csv'}")
        logger.info(f"  Benchmark scores:     {eval_dir / 'benchmark_scores.csv'}")
        logger.info(f"  Metadata:             {eval_dir / 'benchmark_metadata.json'}")

        # Check if files exist
        for fname in ["benchmark_scores.csv", "benchmark_flows.csv", "benchmark_metadata.json"]:
            fpath = eval_dir / fname
            if fpath.exists():
                logger.info(f"  [OK] {fname} exists")
            else:
                logger.warning(f"  [MISSING] {fname} not found")
    else:
        logger.error("\nBenchmarking failed. Check logs for details.")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run HydroBM benchmarking for Bow at Banff (SYMFLUENCE Paper Section 4.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmarks computed:
  Streamflow:         mean_flow, median_flow, annual/monthly/daily mean & median
  Rainfall-Runoff:    long-term and short-term ratios at multiple temporal scales
  Schaefli & Gupta:   scaled, adjusted, and adjusted-smoothed precipitation

Examples:
  python run_benchmarking.py                          # Run benchmarking
  python run_benchmarking.py --dry-run                # Preview without running
  python run_benchmarking.py --config /path/to/cfg    # Use custom config
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()
    run_benchmarking(Path(args.config), args.dry_run)


if __name__ == "__main__":
    main()
