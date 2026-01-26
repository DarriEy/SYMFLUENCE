#!/usr/bin/env python3
"""
Generate configuration files for Bow at Banff HBV study.

This script creates configuration files for different experimental conditions:
1. Daily vs Hourly timesteps
2. Different optimization algorithms (DDS, PSO, DE, GA, ADAM)
3. Smoothing vs non-smoothing
4. ODE vs direct implementation (configured via HBV_BACKEND)
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Base configuration template path
BASE_CONFIG = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE/0_config_files/config_Bow_lumped_era5_hbv.yaml"
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE/scripts/studies/bow_hbv_study/configs")

# Optimization algorithms to test
OPTIMIZERS = {
    'DDS': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'NUMBER_OF_ITERATIONS': 4000,
        'DDS_R': 0.2,
    },
    'PSO': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'PSO',
        'NUMBER_OF_ITERATIONS': 4000,
        'SWRMSIZE': 20,
        'PSO_COGNITIVE_PARAM': 1.5,
        'PSO_SOCIAL_PARAM': 1.5,
        'PSO_INERTIA_WEIGHT': 0.7,
        'PSO_INERTIA_REDUCTION_RATE': 0.99,
    },
    'DE': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 4000,
        'POPULATION_SIZE': 20,
        'DE_SCALING_FACTOR': 0.7,
        'DE_CROSSOVER_RATE': 0.7,
    },
    'GA': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'GA',
        'NUMBER_OF_ITERATIONS': 4000,
        'POPULATION_SIZE': 20,
    },
    'ADAM': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'ADAM',
        'NUMBER_OF_ITERATIONS': 4000,
        'ADAM_LR': 0.01,
        'ADAM_BETA1': 0.9,
        'ADAM_BETA2': 0.999,
        'ADAM_EPSILON': 1e-8,
    },
}

# Timestep configurations
TIMESTEPS = {
    'daily': 24,
    'hourly': 1,
}

# Smoothing configurations
SMOOTHING = {
    'smooth': {'HBV_SMOOTHING': True, 'HBV_SMOOTHING_FACTOR': 15.0},
    'nosmooth': {'HBV_SMOOTHING': False, 'HBV_SMOOTHING_FACTOR': 15.0},
}

# Backend configurations (for gradient comparison)
BACKENDS = {
    'jax': 'jax',  # Direct JAX implementation with AD
    # Note: ODE backend would be a different study using hbv_ode module
}


def load_base_config() -> Dict[str, Any]:
    """Load the base configuration file."""
    with open(BASE_CONFIG, 'r') as f:
        return yaml.safe_load(f)


def create_config(
    base_config: Dict[str, Any],
    timestep: str,
    optimizer: str,
    smoothing: str = 'nosmooth',
    backend: str = 'jax'
) -> Dict[str, Any]:
    """Create a configuration variant."""
    config = base_config.copy()

    # Update experiment ID
    exp_id = f"study_{timestep}_{optimizer.lower()}_{smoothing}_{backend}"
    config['EXPERIMENT_ID'] = exp_id

    # Update timestep
    config['HBV_TIMESTEP_HOURS'] = TIMESTEPS[timestep]

    # Update optimizer settings
    config.update(OPTIMIZERS[optimizer])

    # Update smoothing settings
    config.update(SMOOTHING[smoothing])

    # Update backend
    config['HBV_BACKEND'] = BACKENDS[backend]

    # Ensure optimization method is iteration
    config['OPTIMIZATION_METHODS'] = ['iteration']

    # Ensure calibration metric is KGE
    config['OPTIMIZATION_METRIC'] = 'KGE'
    config['TARGET_METRIC'] = 'KGE'

    return config


def save_config(config: Dict[str, Any], filename: str):
    """Save configuration to YAML file."""
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {filename}")


def main():
    """Generate all configuration files."""
    print("Generating configuration files for Bow HBV study...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load base configuration
    base_config = load_base_config()

    # Part 1: Daily vs Hourly comparison (using DDS)
    print("Part 1: Daily vs Hourly Comparison")
    print("-" * 50)
    for timestep in ['daily', 'hourly']:
        config = create_config(base_config, timestep, 'DDS')
        filename = f"config_bow_hbv_{timestep}_dds.yaml"
        save_config(config, filename)
    print()

    # Part 2: Optimizer Comparison (using daily timestep)
    print("Part 2: Optimizer Comparison (Daily timestep)")
    print("-" * 50)
    for optimizer in OPTIMIZERS.keys():
        config = create_config(base_config, 'daily', optimizer)
        filename = f"config_bow_hbv_daily_{optimizer.lower()}.yaml"
        save_config(config, filename)
    print()

    # Part 3: Smoothing Comparison (using daily, DDS)
    print("Part 3: Smoothing Comparison")
    print("-" * 50)
    for smoothing in ['smooth', 'nosmooth']:
        config = create_config(base_config, 'daily', 'DDS', smoothing=smoothing)
        filename = f"config_bow_hbv_daily_dds_{smoothing}.yaml"
        save_config(config, filename)
    print()

    # Part 4: Gradient-based optimizer comparison with smoothing
    print("Part 4: Gradient Methods (Daily, Smoothing enabled)")
    print("-" * 50)
    for optimizer in ['DDS', 'ADAM']:
        config = create_config(base_config, 'daily', optimizer, smoothing='smooth')
        filename = f"config_bow_hbv_daily_{optimizer.lower()}_smooth.yaml"
        save_config(config, filename)
    print()

    print("Configuration file generation complete!")
    print(f"\nTotal files created: {len(list(OUTPUT_DIR.glob('*.yaml')))}")


if __name__ == "__main__":
    main()
