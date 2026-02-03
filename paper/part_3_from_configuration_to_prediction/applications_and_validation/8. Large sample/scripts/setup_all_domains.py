#!/usr/bin/env python3
"""Setup all LamaH-Ice domains: extract archives, determine time periods, generate configs.

This script:
1. Scans for domain tar.gz archives in the SYMFLUENCE data directory
2. Reads streamflow observation data from each archive to determine available date ranges
3. Extracts archives and deletes them to save disk space
4. Generates FUSE config files with appropriate time periods for each domain

Time period logic:
- Default/preferred period: 2005-2014 (spinup 2005-06, cal 2007-11, eval 2012-14)
- If data doesn't cover the default period, uses the best available window:
  - >= 10 years: 2yr spinup, 5yr calibration, 3yr evaluation
  - >= 6 years:  2yr spinup, split remaining ~60/40 cal/eval
  - >= 4 years:  1yr spinup, split remaining ~60/40 cal/eval
  - < 4 years:   skipped (too short for meaningful calibration)
"""

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path


# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
TEMPLATE_CONFIG = CONFIGS_DIR / "config_lamahice_42_FUSE.yaml"

# Default period (from README)
DEFAULT_SPINUP = ("2005-01-01", "2006-12-31")
DEFAULT_CAL = ("2007-01-01", "2011-12-31")
DEFAULT_EVAL = ("2012-01-01", "2014-12-31")
DEFAULT_START = "2005-01-01 01:00"
DEFAULT_END = "2014-12-31 23:00"

# Already-configured domains (preserve their existing configs)
EXISTING_CONFIGS = {42, 45, 98, 100, 102, 103}


def get_domain_ids_from_archives():
    """Find all domain IDs that have tar.gz archives."""
    ids = []
    for f in DATA_DIR.glob("domain_*.tar.gz"):
        name = f.stem.replace(".tar", "")  # domain_X
        did = name.replace("domain_", "")
        try:
            ids.append(int(did))
        except ValueError:
            continue
    return sorted(ids)


def get_domain_ids_from_dirs():
    """Find all domain IDs that have extracted directories."""
    ids = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.startswith("domain_"):
            did = d.name.replace("domain_", "")
            try:
                ids.append(int(did))
            except ValueError:
                continue
    return sorted(ids)


def read_streamflow_dates_from_tar(domain_id):
    """Read first and last streamflow observation dates from a tar.gz archive."""
    tar_path = DATA_DIR / f"domain_{domain_id}.tar.gz"
    streamflow_path = f"domain_{domain_id}/observations/streamflow/raw_data/{domain_id}_streamflow.csv"

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            try:
                member = tf.getmember(streamflow_path)
            except KeyError:
                return None, None
            f = tf.extractfile(member)
            if f is None:
                return None, None
            content = f.read().decode("utf-8")
            lines = content.strip().split("\n")
            if len(lines) < 2:
                return None, None

            # Parse first data line
            first_parts = lines[1].split(";")
            first_date = date(int(first_parts[0]), int(first_parts[1]), int(first_parts[2]))

            # Parse last data line
            last_parts = lines[-1].split(";")
            last_date = date(int(last_parts[0]), int(last_parts[1]), int(last_parts[2]))

            return first_date, last_date
    except Exception as e:
        print(f"  WARNING: Could not read streamflow from tar.gz for domain {domain_id}: {e}")
        return None, None


def read_streamflow_dates_from_dir(domain_id):
    """Read first and last streamflow observation dates from an extracted directory."""
    streamflow_path = DATA_DIR / f"domain_{domain_id}/observations/streamflow/raw_data/{domain_id}_streamflow.csv"

    if not streamflow_path.exists():
        return None, None

    try:
        with open(streamflow_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None, None

        first_parts = lines[1].strip().split(";")
        first_date = date(int(first_parts[0]), int(first_parts[1]), int(first_parts[2]))

        last_parts = lines[-1].strip().split(";")
        last_date = date(int(last_parts[0]), int(last_parts[1]), int(last_parts[2]))

        return first_date, last_date
    except Exception as e:
        print(f"  WARNING: Could not read streamflow from dir for domain {domain_id}: {e}")
        return None, None


def determine_time_periods(first_date, last_date):
    """Determine spinup/calibration/evaluation periods from observation date range.

    Returns dict with keys: start, end, spinup, calibration, evaluation
    or None if record is too short.
    """
    # Check if default period (2005-2014) is covered
    default_start = date(2005, 1, 1)
    default_end = date(2014, 12, 31)

    if first_date <= default_start and last_date >= default_end:
        return {
            "start": DEFAULT_START,
            "end": DEFAULT_END,
            "spinup": f"{DEFAULT_SPINUP[0]}, {DEFAULT_SPINUP[1]}",
            "calibration": f"{DEFAULT_CAL[0]}, {DEFAULT_CAL[1]}",
            "evaluation": f"{DEFAULT_EVAL[0]}, {DEFAULT_EVAL[1]}",
        }

    # Use available data range
    # Round start to Jan 1 of the year (or next year if data starts mid-year after June)
    if first_date.month > 6:
        start_year = first_date.year + 1
    else:
        start_year = first_date.year

    # Round end to Dec 31 of the year (or previous year if data ends before July)
    if last_date.month < 7 and last_date.day < 15:
        end_year = last_date.year - 1
    else:
        end_year = last_date.year

    total_years = end_year - start_year + 1

    if total_years < 4:
        return None  # Too short

    if total_years >= 10:
        spinup_years = 2
        eval_years = 3
        cal_years = total_years - spinup_years - eval_years
    elif total_years >= 6:
        spinup_years = 2
        remaining = total_years - spinup_years
        eval_years = max(2, int(remaining * 0.4))
        cal_years = remaining - eval_years
    else:  # 4-5 years
        spinup_years = 1
        remaining = total_years - spinup_years
        eval_years = max(1, int(remaining * 0.4))
        cal_years = remaining - eval_years

    spinup_start = start_year
    spinup_end = start_year + spinup_years - 1
    cal_start = spinup_end + 1
    cal_end = cal_start + cal_years - 1
    eval_start = cal_end + 1
    eval_end = end_year

    # Handle end date for the experiment (use actual last date if in final year)
    if last_date.year == end_year and last_date.month < 12:
        exp_end = f"{last_date.year}-{last_date.month:02d}-{last_date.day:02d} 23:00"
        eval_end_str = f"{last_date.year}-{last_date.month:02d}-{last_date.day:02d}"
    else:
        exp_end = f"{end_year}-12-31 23:00"
        eval_end_str = f"{end_year}-12-31"

    return {
        "start": f"{start_year}-01-01 01:00",
        "end": exp_end,
        "spinup": f"{spinup_start}-01-01, {spinup_end}-12-31",
        "calibration": f"{cal_start}-01-01, {cal_end}-12-31",
        "evaluation": f"{eval_start}-01-01, {eval_end_str}",
    }


def generate_config(domain_id, periods, template_content):
    """Generate a FUSE config for a single domain with custom time periods."""
    content = template_content
    # Replace domain identifiers
    content = content.replace("Domain 42", f"Domain {domain_id}")
    content = content.replace('DOMAIN_NAME: "42"', f'DOMAIN_NAME: "{domain_id}"')
    content = content.replace("LAMAH_ICE_DOMAIN_ID: 42", f"LAMAH_ICE_DOMAIN_ID: {domain_id}")

    # Replace time periods
    content = content.replace("EXPERIMENT_TIME_START: 1995-01-01 01:00",
                              f"EXPERIMENT_TIME_START: {periods['start']}")
    content = content.replace("EXPERIMENT_TIME_END: 2005-12-31 23:00",
                              f"EXPERIMENT_TIME_END: {periods['end']}")
    content = content.replace("CALIBRATION_PERIOD: 1997-01-01, 2002-12-31",
                              f"CALIBRATION_PERIOD: {periods['calibration']}")
    content = content.replace("EVALUATION_PERIOD: 2003-01-01, 2005-12-31",
                              f"EVALUATION_PERIOD: {periods['evaluation']}")
    content = content.replace("SPINUP_PERIOD: 1995-01-01, 1996-12-31",
                              f"SPINUP_PERIOD: {periods['spinup']}")

    return content


def extract_and_delete(domain_id, dry_run=False):
    """Extract a domain tar.gz archive and delete it."""
    tar_path = DATA_DIR / f"domain_{domain_id}.tar.gz"
    domain_dir = DATA_DIR / f"domain_{domain_id}"

    if domain_dir.exists():
        return True  # Already extracted

    if not tar_path.exists():
        print(f"  WARNING: No tar.gz found for domain {domain_id}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would extract {tar_path.name} and delete archive")
        return True

    print(f"  Extracting {tar_path.name}...")
    result = subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(DATA_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR extracting domain {domain_id}: {result.stderr}")
        return False

    # Verify extraction
    if not domain_dir.exists():
        print(f"  ERROR: Directory not created after extraction for domain {domain_id}")
        return False

    # Delete tar.gz
    print(f"  Deleting {tar_path.name}...")
    tar_path.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup all LamaH-Ice domains")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing")
    parser.add_argument("--configs-only", action="store_true",
                        help="Only generate configs (no extraction)")
    parser.add_argument("--skip-existing-configs", action="store_true", default=True,
                        help="Skip domains that already have configs (default: True)")
    parser.add_argument("--force-all-configs", action="store_true",
                        help="Regenerate configs even for existing domains")
    args = parser.parse_args()

    if args.force_all_configs:
        args.skip_existing_configs = False

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Read template
    if not TEMPLATE_CONFIG.exists():
        print(f"ERROR: Template config not found: {TEMPLATE_CONFIG}")
        sys.exit(1)
    template_content = TEMPLATE_CONFIG.read_text()

    # Find all domains
    archive_ids = set(get_domain_ids_from_archives())
    dir_ids = set(get_domain_ids_from_dirs())
    all_ids = sorted(archive_ids | dir_ids)

    print(f"Found {len(all_ids)} total domains")
    print(f"  Archives: {len(archive_ids)}")
    print(f"  Already extracted: {len(dir_ids)}")
    print(f"  Need extraction: {len(archive_ids - dir_ids)}")
    print()

    # Phase 1: Read streamflow dates and determine time periods
    print("=" * 60)
    print("Phase 1: Reading streamflow data and determining time periods")
    print("=" * 60)

    domain_periods = {}
    skipped_short = []
    skipped_no_data = []

    for domain_id in all_ids:
        # Read dates from extracted dir if available, otherwise from tar.gz
        if domain_id in dir_ids:
            first_date, last_date = read_streamflow_dates_from_dir(domain_id)
        elif domain_id in archive_ids:
            first_date, last_date = read_streamflow_dates_from_tar(domain_id)
        else:
            continue

        if first_date is None or last_date is None:
            print(f"  Domain {domain_id}: NO STREAMFLOW DATA")
            skipped_no_data.append(domain_id)
            continue

        periods = determine_time_periods(first_date, last_date)
        if periods is None:
            print(f"  Domain {domain_id}: Record too short ({first_date} to {last_date})")
            skipped_short.append(domain_id)
            continue

        domain_periods[domain_id] = periods
        total_span = last_date.year - first_date.year
        uses_default = "2005-01-01 01:00" in periods["start"] and "2014-12-31 23:00" in periods["end"]
        label = "DEFAULT" if uses_default else "CUSTOM"
        print(f"  Domain {domain_id}: {first_date} to {last_date} ({total_span}yr) -> {label} {periods['spinup']} | {periods['calibration']} | {periods['evaluation']}")

    print("\nPeriod summary:")
    print(f"  Configured: {len(domain_periods)}")
    print(f"  Skipped (too short): {len(skipped_short)} {skipped_short}")
    print(f"  Skipped (no data): {len(skipped_no_data)} {skipped_no_data}")

    # Phase 2: Extract archives
    if not args.configs_only:
        print()
        print("=" * 60)
        print("Phase 2: Extracting archives and deleting tar.gz files")
        print("=" * 60)

        to_extract = sorted(archive_ids - dir_ids)
        # Also delete tar.gz for already-extracted domains
        for domain_id in sorted(dir_ids & archive_ids):
            tar_path = DATA_DIR / f"domain_{domain_id}.tar.gz"
            if tar_path.exists():
                if args.dry_run:
                    print(f"  [DRY RUN] Would delete existing archive for domain {domain_id}")
                else:
                    print(f"  Deleting existing archive for domain {domain_id}...")
                    tar_path.unlink()

        extract_success = 0
        extract_fail = 0
        for i, domain_id in enumerate(to_extract):
            print(f"\n[{i+1}/{len(to_extract)}] Domain {domain_id}")
            if extract_and_delete(domain_id, dry_run=args.dry_run):
                extract_success += 1
            else:
                extract_fail += 1

        print("\nExtraction summary:")
        print(f"  Success: {extract_success}")
        print(f"  Failed: {extract_fail}")

    # Phase 3: Generate configs
    print()
    print("=" * 60)
    print("Phase 3: Generating configuration files")
    print("=" * 60)

    configs_generated = 0
    configs_skipped = 0

    for domain_id in sorted(domain_periods.keys()):
        config_path = CONFIGS_DIR / f"config_lamahice_{domain_id}_FUSE.yaml"

        if args.skip_existing_configs and domain_id in EXISTING_CONFIGS and config_path.exists():
            print(f"  Domain {domain_id}: SKIPPED (existing config preserved)")
            configs_skipped += 1
            continue

        periods = domain_periods[domain_id]
        content = generate_config(domain_id, periods, template_content)

        if args.dry_run:
            print(f"  [DRY RUN] Would write {config_path.name}")
        else:
            config_path.write_text(content)
            print(f"  Generated: {config_path.name}")
        configs_generated += 1

    print("\nConfig summary:")
    print(f"  Generated: {configs_generated}")
    print(f"  Skipped (existing): {configs_skipped}")

    # Save domain metadata for reference
    metadata_path = BASE_DIR / "configs" / "domain_periods.json"
    metadata = {
        "configured": {str(k): v for k, v in domain_periods.items()},
        "skipped_short": skipped_short,
        "skipped_no_data": skipped_no_data,
    }
    if not args.dry_run:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nDomain metadata saved to: {metadata_path}")

    # Update run_large_sample.py domain list
    print(f"\nTotal domains ready to run: {len(domain_periods)}")
    print(f"Domain IDs: {sorted(domain_periods.keys())}")


if __name__ == "__main__":
    main()
