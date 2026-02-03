#!/usr/bin/env python3
"""Trim raw ERA5 forcing files to observation overlap + 5yr spinup, and update configs.

For each LamaH-Ice domain:
1. Read observation date range from streamflow CSV
2. Compute overlap between observations and ERA5 availability (1981-01 to 2019-12)
3. Determine calibration/evaluation periods within the overlap
4. Set 5-year spinup before calibration (clamped to ERA5 start)
5. Delete raw ERA5 forcing files outside the needed period
6. Update config YAML files and domain_periods.json

Usage:
    python trim_forcing_update_configs.py --dry-run   # Preview changes
    python trim_forcing_update_configs.py              # Execute trimming and config updates
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path


# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
PERIODS_JSON = CONFIGS_DIR / "domain_periods.json"

# ERA5 availability
ERA5_START_YEAR = 1981
ERA5_START_MONTH = 1
ERA5_END_YEAR = 2019
ERA5_END_MONTH = 12

# Spinup configuration
SPINUP_YEARS = 5
MIN_TOTAL_YEARS = 4  # minimum usable record length

# Forcing subdirectories containing monthly YYYYMM files to trim
FORCING_SUBDIRS = [
    "raw_data",
    "raw_data_em_earth",
    "SUMMA_input",
    "em_earth_remapped",
    "basin_averaged_data",
]

# Regex to extract YYYYMM from any filename
YYYYMM_RE = re.compile(r"(\d{4})(\d{2})\.nc")


def get_all_domain_ids():
    """Find all domain IDs with extracted directories."""
    ids = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.startswith("domain_"):
            did = d.name.replace("domain_", "")
            try:
                ids.append(int(did))
            except ValueError:
                continue
    return sorted(ids)


def read_obs_dates(domain_id):
    """Read first and last observation dates from streamflow CSV."""
    path = (
        DATA_DIR
        / f"domain_{domain_id}"
        / "observations"
        / "streamflow"
        / "raw_data"
        / f"{domain_id}_streamflow.csv"
    )
    if not path.exists():
        return None, None

    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None, None

        first = lines[1].strip().split(";")
        last = lines[-1].strip().split(";")
        return (
            date(int(first[0]), int(first[1]), int(first[2])),
            date(int(last[0]), int(last[1]), int(last[2])),
        )
    except Exception as e:
        print(f"  WARNING: Could not read streamflow for domain {domain_id}: {e}")
        return None, None


def determine_periods_5yr_spinup(obs_start, obs_end):
    """Determine time periods based on obs/ERA5 overlap with 5yr spinup target.

    Periods are allocated sequentially from the available forcing window:
    spinup -> calibration -> evaluation (no overlap allowed).

    When observations start before ERA5, spinup is taken from the beginning
    of the ERA5 period rather than preceding the observations.

    Returns dict with period info and forcing year/month bounds, or None if unusable.
    """
    era5_start = date(ERA5_START_YEAR, ERA5_START_MONTH, 1)
    era5_end = date(ERA5_END_YEAR, ERA5_END_MONTH, 31)

    # Compute overlap between obs and ERA5
    eff_start = max(obs_start, era5_start)
    eff_end = min(obs_end, era5_end)

    if eff_start >= eff_end:
        return None  # No overlap

    # Round overlap start to year boundary (next year if obs starts after June)
    if eff_start.month > 6:
        overlap_start_year = eff_start.year + 1
    else:
        overlap_start_year = eff_start.year

    # Round overlap end to year boundary (previous year if ends before mid-July)
    if eff_end.month < 7 and eff_end.day < 15:
        overlap_end_year = eff_end.year - 1
    else:
        overlap_end_year = eff_end.year

    # Clamp end to ERA5 range
    overlap_end_year = min(overlap_end_year, ERA5_END_YEAR)

    # Determine the full forcing window: try to place spinup before overlap
    window_start = max(ERA5_START_YEAR, overlap_start_year - SPINUP_YEARS)
    window_end = overlap_end_year
    total_years = window_end - window_start + 1

    if total_years < MIN_TOTAL_YEARS:
        return None  # Not enough years for meaningful run

    # Allocate spinup from the start of the window (target 5yr, need >=3yr for cal+eval)
    spinup_years = min(SPINUP_YEARS, max(1, total_years - 3))
    remaining = total_years - spinup_years

    if remaining < 2:
        return None  # Need at least 2 years for cal+eval

    # Split remaining into calibration and evaluation
    if remaining >= 6:
        eval_years = 3
    elif remaining >= 4:
        eval_years = max(1, int(remaining * 0.3))
    else:
        eval_years = 1

    cal_years = remaining - eval_years

    # Assign periods sequentially (no overlap)
    spinup_start = window_start
    spinup_end = window_start + spinup_years - 1
    cal_start = spinup_end + 1
    cal_end = cal_start + cal_years - 1
    eval_start = cal_end + 1
    eval_end_year = window_end

    # Handle end date (use actual obs/ERA5 end if it's in the final year)
    actual_end = min(obs_end, era5_end)
    if actual_end.year == eval_end_year and actual_end < date(eval_end_year, 12, 31):
        exp_end = f"{actual_end.year}-{actual_end.month:02d}-{actual_end.day:02d} 23:00"
        eval_end_str = f"{actual_end.year}-{actual_end.month:02d}-{actual_end.day:02d}"
    else:
        exp_end = f"{eval_end_year}-12-31 23:00"
        eval_end_str = f"{eval_end_year}-12-31"

    # Forcing file range: from spinup start through eval end
    forcing_start_ym = (spinup_start, 1)
    forcing_end_ym = (eval_end_year, 12)

    total_forcing_months = (
        (forcing_end_ym[0] - forcing_start_ym[0]) * 12
        + forcing_end_ym[1]
        - forcing_start_ym[1]
        + 1
    )

    return {
        "start": f"{spinup_start}-01-01 01:00",
        "end": exp_end,
        "spinup": f"{spinup_start}-01-01, {spinup_end}-12-31",
        "calibration": f"{cal_start}-01-01, {cal_end}-12-31",
        "evaluation": f"{eval_start}-01-01, {eval_end_str}",
        "forcing_start_ym": forcing_start_ym,
        "forcing_end_ym": forcing_end_ym,
        "total_forcing_months": total_forcing_months,
        "actual_spinup_years": spinup_years,
        "obs_overlap_years": remaining,
    }


def find_files_to_delete(domain_id, forcing_start_ym, forcing_end_ym):
    """Find forcing files outside the needed date range across all subdirectories."""
    domain_dir = DATA_DIR / f"domain_{domain_id}" / "forcing"
    to_delete = []
    to_keep = 0

    for subdir_name in FORCING_SUBDIRS:
        subdir = domain_dir / subdir_name
        if not subdir.exists():
            continue

        for fpath in sorted(subdir.iterdir()):
            if not fpath.name.endswith(".nc"):
                # Also handle .nc.backup files
                if fpath.name.endswith(".nc.backup"):
                    match = YYYYMM_RE.search(fpath.name.replace(".backup", ""))
                else:
                    continue
            else:
                match = YYYYMM_RE.search(fpath.name)

            if not match:
                continue

            year = int(match.group(1))
            month = int(match.group(2))

            # Check if within range
            ym = (year, month)
            if ym < forcing_start_ym or ym > forcing_end_ym:
                to_delete.append(fpath)
            else:
                to_keep += 1

    return to_delete, to_keep


def update_config_yaml(domain_id, periods, dry_run=False):
    """Update a domain's YAML config with new time periods."""
    config_path = CONFIGS_DIR / f"config_lamahice_{domain_id}_FUSE.yaml"
    if not config_path.exists():
        return False

    content = config_path.read_text()

    # Replace time period fields using regex to handle any existing values
    replacements = {
        r"EXPERIMENT_TIME_START:.*": f"EXPERIMENT_TIME_START: {periods['start']}",
        r"EXPERIMENT_TIME_END:.*": f"EXPERIMENT_TIME_END: {periods['end']}",
        r"CALIBRATION_PERIOD:.*": f"CALIBRATION_PERIOD: {periods['calibration']}",
        r"EVALUATION_PERIOD:.*": f"EVALUATION_PERIOD: {periods['evaluation']}",
        r"SPINUP_PERIOD:.*": f"SPINUP_PERIOD: {periods['spinup']}",
    }

    new_content = content
    for pattern, replacement in replacements.items():
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        if not dry_run:
            config_path.write_text(new_content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Trim forcing files and update configs with 5yr spinup"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without deleting files or modifying configs",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        type=int,
        default=None,
        help="Specific domain IDs to process (default: all)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Update configs but don't delete forcing files",
    )
    args = parser.parse_args()

    domain_ids = args.domains or get_all_domain_ids()
    print(f"Processing {len(domain_ids)} domains")
    print(f"Spinup target: {SPINUP_YEARS} years")
    print(f"ERA5 range: {ERA5_START_YEAR}-{ERA5_START_MONTH:02d} to {ERA5_END_YEAR}-{ERA5_END_MONTH:02d}")
    if args.dry_run:
        print("*** DRY RUN - no files will be modified ***")
    print()

    # Phase 1: Compute new periods for all domains
    print("=" * 70)
    print("Phase 1: Computing observation-ERA5 overlap and new time periods")
    print("=" * 70)

    new_periods = {}
    skipped_no_overlap = []
    skipped_short = []
    skipped_no_data = []

    for did in domain_ids:
        obs_start, obs_end = read_obs_dates(did)
        if obs_start is None:
            skipped_no_data.append(did)
            continue

        periods = determine_periods_5yr_spinup(obs_start, obs_end)
        if periods is None:
            era5_start = date(ERA5_START_YEAR, 1, 1)
            era5_end = date(ERA5_END_YEAR, 12, 31)
            if obs_end < era5_start or obs_start > era5_end:
                skipped_no_overlap.append(did)
                print(f"  Domain {did:>4d}: NO OVERLAP (obs {obs_start} to {obs_end})")
            else:
                skipped_short.append(did)
                print(f"  Domain {did:>4d}: TOO SHORT (obs {obs_start} to {obs_end})")
            continue

        new_periods[did] = periods
        spinup_info = f"{periods['actual_spinup_years']}yr spinup" if periods["actual_spinup_years"] > 0 else "no spinup possible"
        print(
            f"  Domain {did:>4d}: obs {obs_start}-{obs_end} -> "
            f"forcing {periods['forcing_start_ym'][0]}-{periods['forcing_end_ym'][0]} "
            f"({periods['total_forcing_months']} months, {spinup_info}, "
            f"{periods['obs_overlap_years']}yr obs overlap)"
        )

    print("\nSummary:")
    print(f"  Configured: {len(new_periods)}")
    print(f"  Skipped (no ERA5 overlap): {len(skipped_no_overlap)} {skipped_no_overlap}")
    print(f"  Skipped (too short): {len(skipped_short)} {skipped_short}")
    print(f"  Skipped (no obs data): {len(skipped_no_data)} {skipped_no_data}")

    # Phase 2: Identify files to delete
    print()
    print("=" * 70)
    print("Phase 2: Identifying forcing files to trim")
    print("=" * 70)

    total_delete = 0
    total_keep = 0
    total_delete_size_mb = 0

    domain_deletions = {}
    for did in sorted(new_periods.keys()):
        p = new_periods[did]
        to_delete, to_keep = find_files_to_delete(
            did, p["forcing_start_ym"], p["forcing_end_ym"]
        )

        delete_size = sum(f.stat().st_size for f in to_delete) / (1024 * 1024)
        domain_deletions[did] = to_delete
        total_delete += len(to_delete)
        total_keep += to_keep
        total_delete_size_mb += delete_size

        if to_delete:
            print(
                f"  Domain {did:>4d}: DELETE {len(to_delete):>4d} files "
                f"({delete_size:>7.1f} MB), keep {to_keep:>4d} files"
            )

    print("\nTrimming summary:")
    print(f"  Total files to delete: {total_delete:,}")
    print(f"  Total files to keep:   {total_keep:,}")
    print(f"  Disk space to free:    {total_delete_size_mb:,.1f} MB ({total_delete_size_mb/1024:.1f} GB)")

    # Phase 3: Delete files
    if not args.no_delete:
        print()
        print("=" * 70)
        print("Phase 3: Deleting unnecessary forcing files")
        print("=" * 70)

        deleted = 0
        errors = 0
        for did in sorted(domain_deletions.keys()):
            to_delete = domain_deletions[did]
            if not to_delete:
                continue

            if args.dry_run:
                print(f"  Domain {did}: [DRY RUN] would delete {len(to_delete)} files")
                deleted += len(to_delete)
            else:
                for fpath in to_delete:
                    try:
                        fpath.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"  ERROR deleting {fpath}: {e}")
                        errors += 1
                print(f"  Domain {did}: deleted {len(to_delete)} files")

        print(f"\nDeletion summary: {deleted} deleted, {errors} errors")

    # Phase 4: Update configs and domain_periods.json
    print()
    print("=" * 70)
    print("Phase 4: Updating configuration files")
    print("=" * 70)

    configs_updated = 0
    configs_missing = 0

    for did in sorted(new_periods.keys()):
        p = new_periods[did]
        # Prepare periods dict without internal tracking fields
        config_periods = {
            "start": p["start"],
            "end": p["end"],
            "spinup": p["spinup"],
            "calibration": p["calibration"],
            "evaluation": p["evaluation"],
        }

        updated = update_config_yaml(did, config_periods, dry_run=args.dry_run)
        if updated:
            configs_updated += 1
            action = "[DRY RUN] would update" if args.dry_run else "Updated"
            print(f"  Domain {did:>4d}: {action} config")
        else:
            config_path = CONFIGS_DIR / f"config_lamahice_{did}_FUSE.yaml"
            if not config_path.exists():
                configs_missing += 1
            # else: no changes needed

    # Update domain_periods.json
    periods_for_json = {}
    for did, p in new_periods.items():
        periods_for_json[str(did)] = {
            "start": p["start"],
            "end": p["end"],
            "spinup": p["spinup"],
            "calibration": p["calibration"],
            "evaluation": p["evaluation"],
        }

    metadata = {
        "configured": periods_for_json,
        "skipped_no_overlap": skipped_no_overlap,
        "skipped_short": skipped_short,
        "skipped_no_data": skipped_no_data,
    }

    if args.dry_run:
        print(f"\n  [DRY RUN] Would update {PERIODS_JSON}")
    else:
        with open(PERIODS_JSON, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n  Updated {PERIODS_JSON}")

    print(f"\nConfig summary: {configs_updated} updated, {configs_missing} missing configs")

    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Domains processed:      {len(new_periods)}")
    print(f"  Domains skipped:        {len(skipped_no_overlap) + len(skipped_short) + len(skipped_no_data)}")
    print(f"  Forcing files deleted:  {total_delete:,}")
    print(f"  Disk space freed:       {total_delete_size_mb:,.1f} MB ({total_delete_size_mb/1024:.1f} GB)")
    print(f"  Configs updated:        {configs_updated}")
    print(f"  Spinup:                 {SPINUP_YEARS} years (target)")


if __name__ == "__main__":
    main()
