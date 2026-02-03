#!/usr/bin/env python3
"""Add BOUNDING_BOX_COORDS to all FUSE config files.

Reads catchment shapefiles to compute bounding boxes with a small buffer,
then injects the BOUNDING_BOX_COORDS setting into each config file.
"""

from pathlib import Path

import geopandas as gpd


DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice")
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"

BUFFER_DEG = 0.1  # Buffer in degrees around catchment bounds


def get_bbox_from_shapefile(domain_id):
    """Read catchment shapefile and return bounding box string."""
    # Try wgs84 shapefile first, then regular
    shp_path = DATA_DIR / f"domain_{domain_id}/shapefiles/catchment/{domain_id}_HRUs_GRUs_wgs84.shp"
    if not shp_path.exists():
        shp_path = DATA_DIR / f"domain_{domain_id}/shapefiles/catchment/{domain_id}_HRUs_GRUs.shp"
    if not shp_path.exists():
        return None

    gdf = gpd.read_file(shp_path)
    bounds = gdf.total_bounds  # [lon_min, lat_min, lon_max, lat_max]

    lat_max = bounds[3] + BUFFER_DEG
    lat_min = bounds[1] - BUFFER_DEG
    lon_max = bounds[2] + BUFFER_DEG
    lon_min = bounds[0] - BUFFER_DEG

    return f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"


def add_bbox_to_config(config_path, bbox):
    """Insert BOUNDING_BOX_COORDS into a config file after DOMAIN_NAME."""
    content = config_path.read_text()

    if "BOUNDING_BOX_COORDS" in content:
        return False  # Already has bbox

    # Insert after DOMAIN_NAME line
    lines = content.split("\n")
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if line.startswith("DOMAIN_NAME:") and not inserted:
            new_lines.append(f"BOUNDING_BOX_COORDS: {bbox}")
            inserted = True

    if not inserted:
        # Fallback: insert after first comment header
        new_lines.insert(3, f"BOUNDING_BOX_COORDS: {bbox}")

    config_path.write_text("\n".join(new_lines))
    return True


def main():
    configs = sorted(CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"))
    print(f"Found {len(configs)} config files")

    success = 0
    skipped = 0
    failed = 0

    for config_path in configs:
        # Extract domain ID from filename
        parts = config_path.stem.replace("config_lamahice_", "").replace("_FUSE", "")
        try:
            domain_id = int(parts)
        except ValueError:
            continue

        bbox = get_bbox_from_shapefile(domain_id)
        if bbox is None:
            print(f"  Domain {domain_id}: FAILED - no shapefile found")
            failed += 1
            continue

        if add_bbox_to_config(config_path, bbox):
            print(f"  Domain {domain_id}: Added BBOX {bbox}")
            success += 1
        else:
            print(f"  Domain {domain_id}: SKIPPED (already has BBOX)")
            skipped += 1

    print(f"\nSummary: {success} added, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
