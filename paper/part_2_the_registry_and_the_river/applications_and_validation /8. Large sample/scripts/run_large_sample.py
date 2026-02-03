#!/usr/bin/env python3
"""Run FUSE across all LamaH-Ice catchments for the Large Sample study.

Iterates over per-catchment FUSE configs and runs the SYMFLUENCE workflow
for each. Skips domains that have already completed optimization.
Logs progress and results to a run log file.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
RUN_LOG = BASE_DIR / "run_log.json"


def get_domain_ids():
    """Discover all configured domain IDs from config files."""
    ids = []
    for f in CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"):
        parts = f.stem.replace("config_lamahice_", "").replace("_FUSE", "")
        try:
            ids.append(int(parts))
        except ValueError:
            continue
    return sorted(ids)


def is_domain_complete(domain_id):
    """Check if a domain has completed DDS optimization."""
    results_file = (
        DATA_DIR / f"domain_{domain_id}" / "optimization" / "FUSE"
        / "dds_run_1" / "run_1_parallel_iteration_results.csv"
    )
    if not results_file.exists():
        return False
    # Check that it has actual iteration results (header + data rows)
    try:
        with open(results_file) as f:
            line_count = sum(1 for _ in f)
        return line_count > 100  # At least some iterations completed
    except Exception:
        return False


def load_run_log():
    """Load the persistent run log."""
    if RUN_LOG.exists():
        with open(RUN_LOG) as f:
            return json.load(f)
    return {"runs": {}}


def save_run_log(log):
    """Save the persistent run log."""
    with open(RUN_LOG, "w") as f:
        json.dump(log, f, indent=2)


def run_domain(domain_id, dry_run=False):
    """Run SYMFLUENCE workflow for a single LamaH-Ice domain."""
    config_path = CONFIGS_DIR / f"config_lamahice_{domain_id}_FUSE.yaml"

    if not config_path.exists():
        print(f"  ERROR: Config not found: {config_path}")
        return 1

    cmd = ["symfluence", "workflow", "run", "--config", str(config_path)]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run FUSE on LamaH-Ice catchments sequentially"
    )
    parser.add_argument(
        "--domain", type=int, default=None,
        help="Run a single domain ID (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop if any domain fails"
    )
    parser.add_argument(
        "--skip-completed", action="store_true", default=True,
        help="Skip domains with existing optimization results (default: True)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run all domains even if already completed"
    )
    parser.add_argument(
        "--start-from", type=int, default=None,
        help="Start from this domain ID (skip earlier domains)"
    )
    args = parser.parse_args()

    if args.force:
        args.skip_completed = False

    all_ids = get_domain_ids()
    if args.domain:
        domains = [args.domain]
    else:
        domains = all_ids

    if args.start_from:
        domains = [d for d in domains if d >= args.start_from]

    # Check completion status
    completed_ids = set()
    pending_ids = []
    if args.skip_completed:
        for d in domains:
            if is_domain_complete(d):
                completed_ids.add(d)
            else:
                pending_ids.append(d)
    else:
        pending_ids = list(domains)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Large Sample Study - FUSE on LamaH-Ice")
    print(f"Started: {timestamp}")
    print(f"Total configured: {len(all_ids)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"To run: {len(pending_ids)}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print("=" * 60)

    if completed_ids:
        print(f"Skipping completed domains: {sorted(completed_ids)}")

    if not pending_ids:
        print("\nAll domains already completed. Use --force to re-run.")
        return

    # Load persistent log
    run_log = load_run_log()

    results = {}
    for i, domain_id in enumerate(pending_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pending_ids)}] Domain {domain_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        start_time = datetime.now()
        rc = run_domain(domain_id, dry_run=args.dry_run)
        elapsed = (datetime.now() - start_time).total_seconds()

        results[domain_id] = rc

        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"\n  Result: {status} ({elapsed:.0f}s)")

        # Update persistent log
        if not args.dry_run:
            run_log["runs"][str(domain_id)] = {
                "status": "success" if rc == 0 else "failed",
                "return_code": rc,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }
            save_run_log(run_log)

        if rc != 0 and args.stop_on_error:
            print(f"\nStopping due to error on domain {domain_id}.")
            print(f"Resume with: python {__file__} --start-from {domain_id}")
            break

    # Final summary
    print(f"\n{'='*60}")
    print("RUN SUMMARY")
    print(f"{'='*60}")

    succeeded = [d for d, rc in results.items() if rc == 0]
    failed = [d for d, rc in results.items() if rc != 0]

    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Previously completed: {len(completed_ids)}")
    print(f"  Total done: {len(succeeded) + len(completed_ids)}/{len(all_ids)}")

    if failed:
        print(f"\n  Failed domains: {failed}")
        print("\n  To retry failed domains:")
        for d in failed:
            print(f"    python {Path(__file__).name} --domain {d}")
        sys.exit(1)
    else:
        print(f"\nAll {len(pending_ids)} domains completed successfully.")


if __name__ == "__main__":
    main()
