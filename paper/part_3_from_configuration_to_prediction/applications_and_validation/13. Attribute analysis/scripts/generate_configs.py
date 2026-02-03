#!/usr/bin/env python3
"""
Configuration Generator for Attribute Analysis Experiment (Section 4.13)

Generates YAML configuration files for each attribute discretization scenario
from the baseline lumped configuration. Each scenario varies only the
discretization settings to isolate the effect of attribute-based HRU definition.

Scenarios:
  1. lumped_baseline   - No sub-grid discretization (1 HRU = 1 GRU)
  2. elevation_200m    - Elevation bands at 200 m intervals
  3. elevation_400m    - Elevation bands at 400 m intervals
  4. landclass         - Land cover classification only
  5. soilclass         - Soil type classification only
  6. aspect            - 8-class aspect (cardinal + ordinal directions)
  7. radiation         - 5-class radiation discretization
  8. elev_land         - Combined elevation (200 m) + land cover
  9. elev_soil_land    - Combined elevation (200 m) + soil + land cover (full)

Usage:
    python generate_configs.py [--dry-run]
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "configs"
BASELINE_CONFIG = CONFIG_DIR / "config_Bow_attribute_lumped_baseline.yaml"

# Attribute discretization scenarios
# Each scenario overrides specific keys from the baseline configuration
SCENARIOS = {
    "lumped_baseline": {
        "EXPERIMENT_ID": "attr_lumped_baseline",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUS",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Lumped baseline - no sub-grid discretization (1 HRU)",
    },
    "elevation_200m": {
        "EXPERIMENT_ID": "attr_elevation_200m",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "elevation",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Elevation-only discretization at 200 m band intervals",
    },
    "elevation_400m": {
        "EXPERIMENT_ID": "attr_elevation_400m",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "elevation",
        "ELEVATION_BAND_SIZE": 400,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Elevation-only discretization at 400 m band intervals",
    },
    "landclass": {
        "EXPERIMENT_ID": "attr_landclass",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "landclass",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Land cover classification discretization only",
    },
    "soilclass": {
        "EXPERIMENT_ID": "attr_soilclass",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "soilclass",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Soil type classification discretization only",
    },
    "aspect": {
        "EXPERIMENT_ID": "attr_aspect",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "aspect",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 8,
        "description": "8-class aspect discretization (N, NE, E, SE, S, SW, W, NW)",
    },
    "radiation": {
        "EXPERIMENT_ID": "attr_radiation",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "radiation",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 5,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "5-class radiation discretization",
    },
    "elev_land": {
        "EXPERIMENT_ID": "attr_elev_land",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "elevation,landclass",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Combined elevation (200 m) + land cover discretization",
    },
    "elev_soil_land": {
        "EXPERIMENT_ID": "attr_elev_soil_land",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "SUB_GRID_DISCRETIZATION": "elevation,soilclass,landclass",
        "ELEVATION_BAND_SIZE": 200,
        "RADIATION_CLASS_NUMBER": 1,
        "ASPECT_CLASS_NUMBER": 1,
        "description": "Combined elevation + soil + land cover (full multi-attribute)",
    },
}


def load_baseline_config(path: Path) -> dict:
    """Load the baseline YAML configuration, preserving comments where possible."""
    with open(path) as f:
        return yaml.safe_load(f)


def write_config(config: dict, output_path: Path, scenario_name: str, description: str):
    """Write a configuration file with a descriptive header."""
    header = (
        f"### ============================================= SYMFLUENCE configuration file "
        f"============================================================\n"
        f"# Attribute Analysis - {scenario_name}\n"
        f"# Section 4.13: {description}\n"
        f"#\n"
        f"# Auto-generated from baseline config by generate_configs.py\n"
        f"# Varies only discretization settings to isolate the effect of attribute-based HRU definition.\n"
        f"#\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)


def generate_all_configs(dry_run: bool = False):
    """Generate configuration files for all attribute analysis scenarios."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_CONFIG.exists():
        print(f"ERROR: Baseline config not found: {BASELINE_CONFIG}")
        sys.exit(1)

    baseline = load_baseline_config(BASELINE_CONFIG)
    print(f"Loaded baseline config: {BASELINE_CONFIG}")
    print(f"Generating {len(SCENARIOS)} scenario configurations...\n")

    for scenario_name, overrides in SCENARIOS.items():
        description = overrides.pop("description", "")
        config = copy.deepcopy(baseline)

        # Apply overrides
        for key, value in overrides.items():
            config[key] = value

        output_name = f"config_Bow_attribute_{scenario_name}.yaml"
        output_path = CONFIG_DIR / output_name

        # Restore description for display
        overrides["description"] = description

        print(f"  {scenario_name}:")
        print(f"    File: {output_name}")
        print(f"    Experiment ID: {overrides['EXPERIMENT_ID']}")
        print(f"    Discretization: {overrides.get('SUB_GRID_DISCRETIZATION', 'N/A')}")
        print(f"    Description: {description}")

        if not dry_run:
            write_config(config, output_path, scenario_name, description)
            print("    -> Written")
        else:
            print("    -> [DRY RUN] Would write")
        print()

    print(f"Done. {len(SCENARIOS)} configs {'would be ' if dry_run else ''}generated in {CONFIG_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate attribute analysis configuration files for Section 4.13"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    args = parser.parse_args()
    generate_all_configs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
