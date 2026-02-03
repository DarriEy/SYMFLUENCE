#!/usr/bin/env python3
"""
Generate configuration files for 4.3 Forcing Ensemble Study.

This script creates configuration files for comparing different atmospheric
forcing datasets (ERA5, AORC, HRRR, CONUS404, RDRS) at the Paradise SNOTEL station,
plus GDDP (NEX-GDDP-CMIP6) ensemble members for climate projections to 2100.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Base configuration template path
BASE_CONFIG = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE/0_config_files/config_paradise.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "configs"

# Forcing dataset configurations
FORCING_DATASETS = {
    'ERA5': {
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    },
    'AORC': {
        'FORCING_DATASET': 'AORC',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    },
    'HRRR': {
        'FORCING_DATASET': 'HRRR',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    },
    'CONUS404': {
        'FORCING_DATASET': 'CONUS404',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    },
    'RDRS': {
        'FORCING_DATASET': 'RDRS',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    },
}

# GDDP ensemble configurations (NEX-GDDP-CMIP6)
# These run from the same start date but extend to 2100
GDDP_ENSEMBLE_MEMBERS = {
    'GDDP_ACCESS-CM2': {
        'FORCING_DATASET': 'GDDP',
        'FORCING_TIME_STEP_SIZE': 86400,  # Daily data
        'DATA_ACCESS': 'cloud',
        'NEX_MODELS': ['ACCESS-CM2'],
        'NEX_SCENARIOS': ['historical', 'ssp245'],
        'NEX_ENSEMBLES': ['r1i1p1f1'],
    },
    'GDDP_GFDL-ESM4': {
        'FORCING_DATASET': 'GDDP',
        'FORCING_TIME_STEP_SIZE': 86400,
        'DATA_ACCESS': 'cloud',
        'NEX_MODELS': ['GFDL-ESM4'],
        'NEX_SCENARIOS': ['historical', 'ssp245'],
        'NEX_ENSEMBLES': ['r1i1p1f1'],
    },
    'GDDP_MRI-ESM2-0': {
        'FORCING_DATASET': 'GDDP',
        'FORCING_TIME_STEP_SIZE': 86400,
        'DATA_ACCESS': 'cloud',
        'NEX_MODELS': ['MRI-ESM2-0'],
        'NEX_SCENARIOS': ['historical', 'ssp245'],
        'NEX_ENSEMBLES': ['r1i1p1f1'],
    },
}

# GDDP-specific time settings (extends to 2100)
GDDP_STUDY_SETTINGS = {
    'EXPERIMENT_TIME_START': '2015-01-01 01:00',
    'EXPERIMENT_TIME_END': '2100-12-31 23:00',
    'CALIBRATION_PERIOD': '2015-10-01, 2018-09-30',
    'EVALUATION_PERIOD': '2018-10-01, 2020-09-30',
    'SPINUP_PERIOD': '2015-01-01, 2015-09-30',
}

# Common study settings
STUDY_SETTINGS = {
    # Experiment identification (2015-2020 to cover all forcing datasets including HRRR)
    'EXPERIMENT_TIME_START': '2015-01-01 01:00',
    'EXPERIMENT_TIME_END': '2020-12-31 23:00',
    'CALIBRATION_PERIOD': '2015-10-01, 2018-09-30',
    'EVALUATION_PERIOD': '2018-10-01, 2020-09-30',
    'SPINUP_PERIOD': '2015-01-01, 2015-09-30',

    # Model settings
    'HYDROLOGICAL_MODEL': 'SUMMA',
    'ROUTING_MODEL': 'mizuRoute',

    # DEM source - use Copernicus for all cases
    'DEM_SOURCE': 'copernicus',

    # Calibration settings
    'OPTIMIZATION_METHODS': ['iteration'],
    'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
    'NUMBER_OF_ITERATIONS': 10,
    'OPTIMIZATION_METRIC': 'RMSE',
    'OPTIMIZATION_TARGET': 'swe',
    'DDS_R': 0.2,

    # Snow-focused calibration parameters
    'PARAMS_TO_CALIBRATE': 'tempCritRain,tempRangeTimestep,frozenPrecipMultip,albedoMax,albedoMinWinter,albedoDecayRate,constSnowDen,mw_exp,k_snow,z0Snow',
    'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaScale',

    # Forcing settings
    'APPLY_LAPSE_RATE': True,
    'LAPSE_RATE': 0.0065,
    'FORCING_MEASUREMENT_HEIGHT': 2,
    'FORCING_VARIABLES': 'default',
}


def load_base_config() -> Dict[str, Any]:
    """Load the base configuration file."""
    with open(BASE_CONFIG, 'r') as f:
        return yaml.safe_load(f)


def create_forcing_config(
    base_config: Dict[str, Any],
    forcing_name: str,
    is_gddp: bool = False
) -> Dict[str, Any]:
    """Create a configuration variant for a specific forcing dataset."""
    config = base_config.copy()

    # Normalize name for filenames (replace problematic characters)
    safe_name = forcing_name.lower().replace('-', '_')

    # Update experiment ID
    exp_id = f"forcing_ensemble_{safe_name}"
    config['EXPERIMENT_ID'] = exp_id

    # Update domain name to reflect forcing
    config['DOMAIN_NAME'] = f"paradise_snotel_wa_{safe_name}"

    # Apply common study settings
    config.update(STUDY_SETTINGS)

    # Apply GDDP-specific time settings if applicable
    if is_gddp:
        config.update(GDDP_STUDY_SETTINGS)
        config.update(GDDP_ENSEMBLE_MEMBERS[forcing_name])
    else:
        config.update(FORCING_DATASETS[forcing_name])

    return config


def save_config(config: Dict[str, Any], filename: str):
    """Save configuration to YAML file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {filename}")


def main():
    """Generate all configuration files for forcing ensemble study."""
    print("=" * 60)
    print("Generating configuration files for 4.3 Forcing Ensemble Study")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Base config: {BASE_CONFIG}")
    print()

    # Load base configuration
    base_config = load_base_config()

    # Generate configs for each forcing dataset
    print("Forcing Dataset Configurations")
    print("-" * 50)
    for forcing_name in FORCING_DATASETS.keys():
        config = create_forcing_config(base_config, forcing_name)
        filename = f"config_paradise_{forcing_name.lower()}.yaml"
        save_config(config, filename)

    # Generate configs for GDDP ensemble members
    print()
    print("GDDP Ensemble Configurations (2015-2100)")
    print("-" * 50)
    for forcing_name in GDDP_ENSEMBLE_MEMBERS.keys():
        config = create_forcing_config(base_config, forcing_name, is_gddp=True)
        # Use safe filename (replace hyphens with underscores)
        safe_name = forcing_name.lower().replace('-', '_')
        filename = f"config_paradise_{safe_name}.yaml"
        save_config(config, filename)

    print()
    print("=" * 60)
    print("Configuration file generation complete!")
    print(f"Total files created: {len(list(OUTPUT_DIR.glob('*.yaml')))}")
    print()
    print("Next steps:")
    print("  1. Review generated configs in configs/")
    print("  2. Run: python run_study.py --forcing all")
    print("  3. For GDDP projections: python run_study.py --forcing gddp_access_cm2")
    print("  4. Analyze: python analyze_results.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
