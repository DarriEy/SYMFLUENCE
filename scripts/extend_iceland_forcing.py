#!/usr/bin/env python3
"""
Normalize and copy new CARRA forcing months (2004-2008) into the Iceland
large-domain raw_data directory so the SYMFLUENCE preprocessing pipeline
can generate extended basin-averaged and FUSE input files.

Fixes three coordinate/variable differences between the new SCF-trend
dataset and the existing domain data:
  1. Longitude: 0-360 → -180/180  (subtract 360)
  2. Latitude:  descending → ascending  (flip)
  3. Variable:  'relhum' → 'r2'  (rename for consistency)

Usage:
    python scripts/extend_iceland_forcing.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

# --- Paths ---
SRC_DIR = Path(
    "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/"
    "domain_Iceland_multivar_scf_trend/forcing/raw_data"
)
DST_DIR = Path(
    "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/"
    "domain_Iceland_multivar/forcing/raw_data"
)


import re

# Only match standard monthly files: *_CARRA_YYYYMM.nc
_MONTHLY_RE = re.compile(r"_CARRA_(\d{6})\.nc$")


def get_new_months(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Return source files whose YYYYMM is not already in dst_dir."""
    existing = {
        m.group(1) for f in dst_dir.glob("*.nc")
        if (m := _MONTHLY_RE.search(f.name))
    }
    new_files = []
    for f in sorted(src_dir.glob("*.nc")):
        match = _MONTHLY_RE.search(f.name)
        if not match:
            continue  # skip temp/analysis artifacts
        if match.group(1) not in existing:
            new_files.append(f)
    return new_files


def normalize_and_copy(src_file: Path, dst_dir: Path, dry_run: bool = False) -> Path:
    """Normalize coordinates/variables and write to dst_dir."""
    # Keep same filename — domain prefix matches already
    dst_file = dst_dir / src_file.name

    if dry_run:
        print(f"  [dry-run] would write {dst_file.name}")
        return dst_file

    ds = xr.open_dataset(src_file)

    # 1. Fix longitude: 0-360 → -180/180
    if float(ds.longitude.values[0]) > 180:
        ds = ds.assign_coords(longitude=ds.longitude.values - 360)

    # 2. Flip latitude to ascending (S → N) if currently descending
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    # 3. Rename relhum → r2 for consistency with existing data
    if "relhum" in ds.data_vars:
        ds = ds.rename({"relhum": "r2"})

    # Write with same encoding as originals
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds.data_vars
    }
    ds.to_netcdf(dst_file, encoding=encoding)
    ds.close()

    return dst_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be copied without writing anything",
    )
    args = parser.parse_args()

    if not SRC_DIR.exists():
        sys.exit(f"Source directory not found: {SRC_DIR}")
    if not DST_DIR.exists():
        sys.exit(f"Destination directory not found: {DST_DIR}")

    new_files = get_new_months(SRC_DIR, DST_DIR)
    if not new_files:
        print("No new months to copy — destination already has all available data.")
        return

    print(f"Found {len(new_files)} new months to normalize and copy:")
    print(f"  Source: {SRC_DIR}")
    print(f"  Dest:   {DST_DIR}")
    print(f"  Range:  {new_files[0].name} → {new_files[-1].name}")
    print()

    for i, src_file in enumerate(new_files, 1):
        dst = normalize_and_copy(src_file, DST_DIR, dry_run=args.dry_run)
        print(f"  [{i:3d}/{len(new_files)}] {dst.name}")

    print()
    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to write files.")
    else:
        print(f"Done. {len(new_files)} files written to {DST_DIR}")
        print()
        print("Next steps:")
        print("  1. Update config EXPERIMENT_TIME_END, CALIBRATION_PERIOD, EVALUATION_PERIOD")
        print("  2. Run: symfluence workflow steps model_agnostic_preprocessing model_specific_preprocessing")
        print("     (basin averaging will only process the 60 new months; weights are cached)")


if __name__ == "__main__":
    main()
