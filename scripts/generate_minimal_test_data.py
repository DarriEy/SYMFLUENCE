#!/usr/bin/env python3
"""
Script to generate minimal 2-day test datasets for Elliðaár and Fyris.
Run this script to generate the forcing data needed for optimized integration tests.

Usage:
    python scripts/generate_minimal_test_data.py [output_dir]

Environment variables:
    SYMFLUENCE_DATA: Path to existing SYMFLUENCE_data (optional, for copying GIS data)
    OUTPUT_DIR: Output directory for generated data (default: ./SYMFLUENCE_data_minimal)
"""

import os
import sys
import yaml
import shutil
import tempfile
from pathlib import Path
from symfluence import SYMFLUENCE


def find_repo_root():
    """Find SYMFLUENCE repository root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current directory
    return Path.cwd()


def generate_domain_data(domain_name, forcing_dataset, output_root, source_data_root=None):
    print(f"\n{'='*60}")
    print(f"Generating minimal data for {domain_name} ({forcing_dataset})")
    print(f"{'='*60}")

    # Configuration
    data_dir = Path(output_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find repository root
    repo_root = find_repo_root()

    config = {
        'SYMFLUENCE_DATA_DIR': str(data_dir.absolute()),
        'SYMFLUENCE_CODE_DIR': str(repo_root),
        'DOMAIN_NAME': domain_name,
        'FORCING_DATASET': forcing_dataset,
        'DATA_ACCESS': 'cloud',
        'EXPERIMENT_ID': 'minimal_generation',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-03 00:00',
        'CALIBRATION_PERIOD': '2020-01-02, 2020-01-03',
        'SPINUP_PERIOD': '2020-01-01, 2020-01-01',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'DEM_SOURCE': 'copernicus',
        'DOMAIN_DEFINITION_METHOD': 'point',
        'DOMAIN_DISCRETIZATION': 'GRUs',
        'DEM_PATH': 'default',
        'DEM_NAME': 'default',
        'TAUDEM_DIR': 'default',
        'POUR_POINT_SHP_PATH': 'default',
        'POUR_POINT_SHP_NAME': 'default',
        'CATCHMENT_PATH': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'SOIL_CLASS_PATH': 'default',
        'LAND_CLASS_PATH': 'default',
        'RIVER_NETWORK_PATH': 'default',
        'RIVER_BASINS_PATH': 'default',
        'HRU_PATH': 'default',
        'INTERSECT_DEM_PATH': 'default',
        'INTERSECT_DEM_NAME': 'default',
        'INTERSECT_SOIL_PATH': 'default',
        'INTERSECT_SOIL_NAME': 'default',
        'INTERSECT_LAND_PATH': 'default',
        'INTERSECT_LAND_NAME': 'default',
        'ATTRIBUTES_OUTPUT_DIR': 'default',
        'FORCING_PATH': 'default'
    }
    
    if domain_name == 'ellioaar_iceland':
        config.update({
            'BOUNDING_BOX_COORDS': '64.13/-21.94/64.11/-21.96',
            'POUR_POINT_COORDS': '64.12/-21.95',
            'CARRA_DOMAIN': 'west_domain'
        })
    else: # fyris
        config.update({
            'BOUNDING_BOX_COORDS': '59.88/17.59/59.86/17.61',
            'POUR_POINT_COORDS': '59.87/17.60'
        })

    # Write config to a temporary file to avoid polluting repo
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        cfg_path = Path(f.name)
        yaml.dump(config, f)

    try:
        # Setup project
        sym = SYMFLUENCE(cfg_path)
        project_dir = Path(sym.managers['project'].setup_project())

        # Copy existing data from source if available
        if source_data_root:
            source_domain = Path(source_data_root) / f"domain_{domain_name}"
        else:
            # Try to find it in standard locations
            candidates = [
                repo_root.parent / "SYMFLUENCE_data" / f"domain_{domain_name}",
                Path.home() / "SYMFLUENCE_data" / f"domain_{domain_name}",
            ]
            source_domain = None
            for candidate in candidates:
                if candidate.exists():
                    source_domain = candidate
                    break

        if source_domain and source_domain.exists():
            print(f"   Copying existing GIS data from {source_domain}")
            # Copy everything except forcing and simulations
            for item in source_domain.iterdir():
                if item.name not in ['forcing', 'simulations', '_workLog']:
                    dest_item = project_dir / item.name
                    if item.is_dir():
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.copytree(item, dest_item)
                    else:
                        shutil.copy2(item, dest_item)
        else:
            print("   No existing GIS data found, will generate from scratch")

        print("Downloading forcing data...")
        sym.managers['data'].acquire_forcings()

        print("Running model-agnostic preprocessing...")
        sym.managers['data'].run_model_agnostic_preprocessing()

        print("Generating SUMMA settings...")
        sym.managers['model'].preprocess_models()

        print(f"✓ Minimal data generated in {data_dir / f'domain_{domain_name}'}")

    except Exception as e:
        print(f"✗ Error generating data for {domain_name}: {e}")
        raise
    finally:
        # Cleanup config file
        if cfg_path.exists():
            cfg_path.unlink()

if __name__ == "__main__":
    # Get output directory from args or environment
    if len(sys.argv) > 1:
        output_root = Path(sys.argv[1])
    else:
        output_root = Path(os.environ.get("OUTPUT_DIR", "./SYMFLUENCE_data_minimal"))

    # Get source data directory from environment if available
    source_data = os.environ.get("SYMFLUENCE_DATA")

    print(f"Output directory: {output_root.absolute()}")
    if source_data:
        print(f"Source data directory: {source_data}")
    print()

    try:
        generate_domain_data('ellioaar_iceland', 'CARRA', output_root, source_data)
        generate_domain_data('fyris_uppsala', 'CERRA', output_root, source_data)

        print("\n" + '='*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print("The following directories contain the minimal datasets:")
        print(f"1. {output_root / 'domain_ellioaar_iceland'}")
        print(f"2. {output_root / 'domain_fyris_uppsala'}")
        print("\nPlease zip these up and add to the example_data bundle on GitHub release.")

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        sys.exit(1)
