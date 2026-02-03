#!/usr/bin/env python3
"""Run FUSE on the full Iceland domain for the Large Domain study.

Executes the SYMFLUENCE workflow using the Iceland FUSE+mizuRoute configuration.
"""

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config_Iceland_FUSE_era5.yaml"


def main():
    parser = argparse.ArgumentParser(description="Run FUSE on full Iceland domain")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Path to Iceland FUSE config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific workflow step only")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    cmd = ["symfluence", "workflow", "run", "--config", str(args.config)]
    if args.step:
        cmd = ["symfluence", "workflow", "step", args.step, "--config", str(args.config)]

    print("Large Domain Study - FUSE on Iceland")
    print(f"Config: {args.config}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] Command printed above. Exiting.")
        sys.exit(0)

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
