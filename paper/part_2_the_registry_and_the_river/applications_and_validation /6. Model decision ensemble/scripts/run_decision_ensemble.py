#!/usr/bin/env python3
"""
FUSE Decision Ensemble Runner for SYMFLUENCE Paper Section 4.6

This script orchestrates the 64-combination FUSE structural decision ensemble
experiment. It uses the SYMFLUENCE CLI workflow step `run_decision_analysis`,
which internally invokes FuseStructureAnalyzer to iterate through all decision
combinations using a single config. The analyzer modifies the FUSE decisions
file (fuse_zDecisions_*.txt) in-place for each combination, runs FUSE, and
collects metrics into a master CSV.

Usage:
    python run_decision_ensemble.py [--config CONFIG] [--dry-run] [--max-combos N] [--direct]

Arguments:
    --config: Path to configuration file (default: ../config/config_Bow_FUSE_decision_ensemble_era5.yaml)
    --dry-run: Print combinations without executing
    --max-combos: Limit the number of combinations to run (for testing)
    --direct: Use Python API directly instead of CLI
"""

import argparse
import itertools
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
DEFAULT_CONFIG = CONFIG_DIR / "config_Bow_FUSE_decision_ensemble_era5.yaml"
SYMFLUENCE_CLI = "symfluence"

# Add SYMFLUENCE to path for direct API usage
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging for the decision ensemble experiment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"decision_ensemble_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("decision_ensemble")
    logger.info(f"Logging to: {log_file}")
    return logger


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_decision_options(config: Dict) -> Dict[str, List[str]]:
    """Extract FUSE decision options from config."""
    options = config.get("FUSE_DECISION_OPTIONS", {})
    if not options:
        raise ValueError("No FUSE_DECISION_OPTIONS found in configuration")
    return options


def generate_combinations(decision_options: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Generate all combinations of decision options.

    Only varies decisions with more than one option.
    Fixed decisions (single option) are included in every combination.

    Returns:
        List of dictionaries mapping decision name to chosen option.
    """
    decision_names = list(decision_options.keys())
    option_lists = list(decision_options.values())

    combinations = []
    for combo_tuple in itertools.product(*option_lists):
        combo_dict = dict(zip(decision_names, combo_tuple))
        combinations.append(combo_dict)

    return combinations


def print_ensemble_summary(
    decision_options: Dict[str, List[str]],
    combinations: List[Dict[str, str]],
) -> None:
    """Print a summary of the ensemble configuration."""
    print("\n" + "=" * 70)
    print("FUSE DECISION ENSEMBLE CONFIGURATION")
    print("=" * 70)

    varied = {k: v for k, v in decision_options.items() if len(v) > 1}
    fixed = {k: v[0] for k, v in decision_options.items() if len(v) == 1}

    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Varied decisions ({len(varied)}):")
    for name, options in varied.items():
        print(f"  {name}: {options}")

    print(f"\nFixed decisions ({len(fixed)}):")
    for name, value in fixed.items():
        print(f"  {name}: {value}")

    print("=" * 70 + "\n")


def run_via_cli(
    config_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """Run the structure ensemble using the SYMFLUENCE CLI."""
    cmd = [
        SYMFLUENCE_CLI,
        "workflow",
        "step",
        "run_decision_analysis",
        "--config",
        str(config_path),
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute command")
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"CLI command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        logger.error("Make sure SYMFLUENCE is installed and 'symfluence' is in your PATH")
        return False


def run_via_api(
    config_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    max_combos: Optional[int] = None,
) -> bool:
    """Run the structure ensemble using the Python API directly."""
    try:
        from symfluence.models.fuse.structure_analyzer import FuseStructureAnalyzer
    except ImportError:
        logger.error("Could not import FuseStructureAnalyzer. Is SYMFLUENCE installed?")
        return False

    config = load_config(config_path)

    if dry_run:
        decision_options = get_decision_options(config)
        combinations = generate_combinations(decision_options)
        print_ensemble_summary(decision_options, combinations)

        if max_combos:
            combinations = combinations[:max_combos]

        logger.info(f"[DRY RUN] Would run {len(combinations)} combinations:")
        for i, combo in enumerate(combinations, 1):
            varied = {k: v for k, v in combo.items()
                      if len(decision_options[k]) > 1}
            logger.info(f"  Combo {i:3d}: {varied}")
        return True

    analyzer = FuseStructureAnalyzer(config, logger)

    if max_combos:
        original_combinations = analyzer.generate_combinations()
        limited = original_combinations[:max_combos]
        logger.info(f"Limiting to {max_combos} of {len(original_combinations)} combinations")

        # Override generate_combinations to return limited set
        analyzer.generate_combinations = lambda: limited

    results_file, best_combinations = analyzer.run_full_analysis()

    logger.info(f"\nResults saved to: {results_file}")
    logger.info("\nBest combinations:")
    for metric, data in best_combinations.items():
        logger.info(f"  {metric}: {data['score']:.3f}")
        for decision, value in data['combination'].items():
            logger.info(f"    {decision}: {value}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run FUSE decision ensemble experiment (Section 4.6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script explores 64 FUSE structural combinations (6 decisions x 2 options).

Varied decisions:
  ARCH1:  tension1_1 (two-state) vs onestate_1 (single bucket)
  ARCH2:  tens2pll_2 (tension parallel) vs unlimfrc_2 (unlimited frac)
  QSURF:  arno_x_vic (VIC-style) vs prms_varnt (PRMS-style)
  QPERC:  perc_f2sat (frac to sat) vs perc_lower (lower zone)
  ESOIL:  sequential vs rootweight
  QINTF:  intflwnone (no interflow) vs intflwsome (with interflow)

Fixed decisions:
  RFERR:  multiplc_e (multiplicative rainfall error)
  Q_TDH:  rout_gamma (gamma routing)
  SNOWM:  temp_index (temperature index snow)

Examples:
  python run_decision_ensemble.py --dry-run              # Preview combinations
  python run_decision_ensemble.py --max-combos 1         # Test single combo
  python run_decision_ensemble.py                        # Run all 64 via CLI
  python run_decision_ensemble.py --direct               # Run via Python API
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without executing",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Limit the number of combinations to run (for testing)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use Python API directly instead of CLI",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Set up logging
    logger = setup_logging(LOG_DIR)
    logger.info("Starting FUSE Decision Ensemble Experiment (Section 4.6)")
    logger.info(f"Config: {config_path}")

    # Load config and show summary
    config = load_config(config_path)
    decision_options = get_decision_options(config)
    combinations = generate_combinations(decision_options)
    print_ensemble_summary(decision_options, combinations)

    start_time = datetime.now()

    if args.direct or args.max_combos is not None:
        success = run_via_api(config_path, logger, args.dry_run, args.max_combos)
    else:
        success = run_via_cli(config_path, logger, args.dry_run)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nTotal elapsed: {elapsed:.1f}s")

    if not success:
        logger.error("Decision ensemble experiment failed")
        sys.exit(1)

    logger.info("Decision ensemble experiment completed successfully")


if __name__ == "__main__":
    main()
