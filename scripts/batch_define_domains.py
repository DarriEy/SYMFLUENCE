#!/usr/bin/env python3
"""Batch script to run define_domain and discretize_domain for all basins."""

import sys
sys.path.insert(0, '/Users/darrieythorsson/compHydro/code/SYMFLUENCE/src')

from pathlib import Path
from symfluence.core import SYMFLUENCE

# Get all config files
config_dir = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE/0_config_files/cyrils_century_basins")
config_files = sorted(config_dir.glob("config_*.yaml"))

# Skip the one currently being processed
skip_basin = "CAN_01AM001_meso"

print(f"Found {len(config_files)} config files", flush=True)
print(f"Skipping {skip_basin} (currently preprocessing)", flush=True)
print(flush=True)

success_count = 0
error_count = 0
skipped_count = 0

for i, config_file in enumerate(config_files, 1):
    basin_name = config_file.stem.replace("config_", "")

    if skip_basin in str(config_file):
        print(f"[{i}/{len(config_files)}] Skipping {basin_name} (currently preprocessing)", flush=True)
        skipped_count += 1
        continue

    print(f"[{i}/{len(config_files)}] Processing {basin_name}...", flush=True)

    try:
        # Initialize SYMFLUENCE with overrides to force domain regeneration
        overrides = {
            'RIVER_BASINS_NAME': 'default',
            'RIVER_NETWORK_SHP_NAME': 'default',
        }

        sf = SYMFLUENCE(config_file, config_overrides=overrides)

        # Run only define_domain and discretize_domain
        sf.run_individual_steps(['define_domain', 'discretize_domain'])

        print("  ✓ Success", flush=True)
        success_count += 1

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}", flush=True)
        error_count += 1

print(flush=True)
print("=" * 50, flush=True)
print("Summary:", flush=True)
print(f"  Success: {success_count}", flush=True)
print(f"  Errors: {error_count}", flush=True)
print(f"  Skipped: {skipped_count}", flush=True)
print(f"  Total: {len(config_files)}", flush=True)
