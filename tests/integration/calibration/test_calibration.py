#!/usr/bin/env python3
"""
Demonstration of SUMMA calibration with observational data.

This test runs a quick calibration demo for both Elliðaár and Fyris catchments
using observational discharge data.
"""

import platform
import shutil

import pytest
import yaml
from pathlib import Path
from symfluence import SYMFLUENCE


# Skip tests on macOS ARM due to known HDF5/netCDF4 segfault issues with easymore
_MACOS_ARM_SKIP = pytest.mark.skipif(
    platform.system() == 'Darwin' and platform.machine() == 'arm64',
    reason="Skipped on macOS ARM due to HDF5/netCDF4 segfault in easymore with CARRA data"
)

pytestmark = [pytest.mark.integration, pytest.mark.calibration, pytest.mark.requires_data, pytest.mark.slow]

@pytest.mark.slow
@pytest.mark.calibration
@pytest.mark.requires_data
@_MACOS_ARM_SKIP
def test_ellioaar_calibration(ellioaar_domain):
    """Run calibration demo for Elliðaár, Iceland."""
    print("\n" + "="*80)
    print("Elliðaár (Iceland) Calibration Demo")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Configure domain
    config['DOMAIN_NAME'] = 'ellioaar_iceland'
    config['BOUNDING_BOX_COORDS'] = '64.13/-21.94/64.11/-21.96'
    config['POUR_POINT_COORDS'] = '64.12/-21.95'
    config['FORCING_DATASET'] = 'CARRA'
    config['CARRA_DOMAIN'] = 'west_domain'
    config['FORCING_TIME_STEP_SIZE'] = 10800  # 3 hours
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # ULTRA-SHORT period for CI/demo
    config['EXPERIMENT_ID'] = 'calib_demo_short'
    config['EXPERIMENT_TIME_START'] = '2020-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2020-01-03 00:00'  # 2 days
    config['CALIBRATION_PERIOD'] = '2020-01-02, 2020-01-03'
    config['EVALUATION_PERIOD'] = '2020-01-02, 2020-01-03'
    config['SPINUP_PERIOD'] = '2020-01-01, 2020-01-01'

    # Optimization configuration
    config['OPTIMIZATION_METHODS'] = ['iteration']
    config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'
    config['OPTIMIZATION_METRIC'] = 'KGE'
    config['NUMBER_OF_ITERATIONS'] = 5  # Reduced for fast validation
    config['CALIBRATION_TIMESTEP'] = 'daily'

    # Parameters to calibrate
    config['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat'
    config['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'

    # Save config
    temp_config = Path('tests/configs/test_calibration_ellioaar_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    # Check if model needs to be built or forcing data downloaded
    project_dir = Path(config['SYMFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}"

    # Copy data from fixture if project dir doesn't exist
    if not project_dir.exists() and ellioaar_domain and ellioaar_domain.exists():
        print(f"   Using pre-downloaded data from {ellioaar_domain}")
        shutil.copytree(ellioaar_domain, project_dir)

    sym.managers['project'].setup_project()

    # Check if we already have forcing data
    forcing_files = list((project_dir / "forcing" / "raw_data").glob("*CARRA*.nc"))
    if not forcing_files:
        print("\n2. Ensuring data availability...")
        sym.managers['data'].acquire_forcings()
    else:
        print(f"\n2. ✓ Using existing forcing data ({len(forcing_files)} files)")

    sym.managers['data'].run_model_agnostic_preprocessing()
    sym.managers['model'].preprocess_models()

    print(f"\n3. Running calibration (DDS, {config['NUMBER_OF_ITERATIONS']} iterations)...")

    try:
        results_file = sym.managers['optimization'].calibrate_model()
        assert results_file is not None
    except Exception as e:
        pytest.fail(f"Calibration failed: {e}")


@pytest.mark.slow
@pytest.mark.calibration
@pytest.mark.requires_data
@_MACOS_ARM_SKIP
def test_fyris_calibration(fyris_domain):
    """Run calibration demo for Fyris, Uppsala."""
    print("\n" + "="*80)
    print("Fyris (Uppsala, Sweden) Calibration Demo")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Configure domain
    config['DOMAIN_NAME'] = 'fyris_uppsala'
    config['BOUNDING_BOX_COORDS'] = '59.88/17.59/59.86/17.61'
    config['POUR_POINT_COORDS'] = '59.87/17.60'
    config['FORCING_DATASET'] = 'CERRA'
    config['FORCING_TIME_STEP_SIZE'] = 3600  # 1 hour
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # ULTRA-SHORT period for CI/demo
    config['EXPERIMENT_ID'] = 'calib_demo_short'
    config['EXPERIMENT_TIME_START'] = '2020-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2020-01-03 00:00'  # 2 days
    config['CALIBRATION_PERIOD'] = '2020-01-02, 2020-01-03'
    config['EVALUATION_PERIOD'] = '2020-01-02, 2020-01-03'
    config['SPINUP_PERIOD'] = '2020-01-01, 2020-01-01'

    # Optimization configuration
    config['OPTIMIZATION_METHODS'] = ['iteration']
    config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'
    config['OPTIMIZATION_METRIC'] = 'KGE'
    config['NUMBER_OF_ITERATIONS'] = 5
    config['CALIBRATION_TIMESTEP'] = 'daily'

    # Parameters to calibrate
    config['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat'
    config['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'

    # Save config
    temp_config = Path('tests/configs/test_calibration_fyris_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    # Check if model needs to be built or forcing data downloaded
    project_dir = Path(config['SYMFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}"

    # Copy data from fixture if project dir doesn't exist
    if not project_dir.exists() and fyris_domain and fyris_domain.exists():
        print(f"   Using pre-downloaded data from {fyris_domain}")
        shutil.copytree(fyris_domain, project_dir)

    sym.managers['project'].setup_project()

    # Check if we already have forcing data
    forcing_files = list((project_dir / "forcing" / "raw_data").glob("*CERRA*.nc"))
    if not forcing_files:
        print("\n2. Ensuring data availability...")
        sym.managers['data'].acquire_forcings()
    else:
        print(f"\n2. ✓ Using existing forcing data ({len(forcing_files)} files)")

    sym.managers['data'].run_model_agnostic_preprocessing()
    sym.managers['model'].preprocess_models()

    print(f"\n3. Running calibration (DDS, {config['NUMBER_OF_ITERATIONS']} iterations)...")

    try:
        results_file = sym.managers['optimization'].calibrate_model()
        assert results_file is not None
    except Exception as e:
        pytest.fail(f"Calibration failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run calibration demo')
    parser.add_argument('--domain', type=str, choices=['ellioaar', 'fyris', 'both'],
                       default='both', help='Which domain to calibrate')

    args = parser.parse_args()

    results = {}

    if args.domain in ['ellioaar', 'both']:
        results['ellioaar'] = test_ellioaar_calibration()

    if args.domain in ['fyris', 'both']:
        results['fyris'] = test_fyris_calibration()

    print("\n" + "="*80)
    print("Calibration Demo Summary")
    print("="*80)
    for domain, result in results.items():
        status = "✓ Success" if result else "✗ Failed"
        print(f"{domain}: {status}")
        if result:
            print(f"  Results: {result}")
