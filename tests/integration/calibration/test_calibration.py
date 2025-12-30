#!/usr/bin/env python3
"""
Demonstration of SUMMA calibration with observational data.

This test runs a quick calibration demo for both Elliðaár and Fyris catchments
using observational discharge data.
"""

import pytest
import yaml
from pathlib import Path
from symfluence import SYMFLUENCE
import sys



pytestmark = [pytest.mark.integration, pytest.mark.calibration, pytest.mark.requires_data, pytest.mark.slow]

@pytest.mark.slow
@pytest.mark.calibration
@pytest.mark.requires_data
def test_ellioaar_calibration():
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
    config['BOUNDING_BOX_COORDS'] = '64.13/-21.94/64.11/-21.96'  # Fixed: lon_max/lon_min were swapped
    config['POUR_POINT_COORDS'] = '64.12/-21.95'
    config['FORCING_DATASET'] = 'CARRA'
    config['CARRA_DOMAIN'] = 'west_domain'
    config['FORCING_TIME_STEP_SIZE'] = 10800  # 3 hours
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # Calibration settings - using 2-week period for quick demo
    config['EXPERIMENT_ID'] = 'calib_demo'
    config['EXPERIMENT_TIME_START'] = '2020-01-15 00:00'
    config['EXPERIMENT_TIME_END'] = '2020-01-31 21:00'  # 2 weeks for quick demo
    config['CALIBRATION_PERIOD'] = '2020-01-20, 2020-01-31'  # Skip 5-day warmup

    # Optimization configuration
    config['OPTIMIZATION_METHODS'] = ['iteration']
    config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'  # Fast single-objective
    config['OPTIMIZATION_METRIC'] = 'KGE'  # Kling-Gupta Efficiency
    config['NUMBER_OF_ITERATIONS'] = 100  # Quick demo
    config['CALIBRATION_TIMESTEP'] = 'daily'

    # Parameters to calibrate (select most sensitive ones)
    config['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat,fieldCapacity'
    config['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'

    # Observation data path
    config['OBSERVATIONS_PATH'] = 'default'  # Will use observations/streamflow/preprocessed/

    # Save config
    temp_config = Path('tests/configs/test_calibration_ellioaar_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    # Check if model needs to be built or forcing data downloaded
    project_dir = Path(config['SYMFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}"
    summa_settings = project_dir / 'settings' / 'SUMMA'
    forcing_dir = project_dir / 'forcing' / 'SUMMA_input'

    # Check if forcing exists for calibration period (2020)
    needs_rebuild = not summa_settings.exists()
    if forcing_dir.exists():
        forcing_files = list(forcing_dir.glob('*2020*.nc'))
        if not forcing_files or all(f.stat().st_size < 100000 for f in forcing_files):  # Less than 100KB
            print("\n  No adequate forcing data for 2020 found, will rebuild model...")
            needs_rebuild = True
    else:
        needs_rebuild = True

    if needs_rebuild:
        print("\n2. Building SUMMA model and downloading forcing data...")
        print("   This will download CARRA data for Jan 15-31, 2020 (~2-3 min)...")
        sym.managers['data'].acquire_forcings()  # Download raw forcing data from cloud
        sym.managers['data'].run_model_agnostic_preprocessing()  # Process the downloaded data
        sym.managers['model'].preprocess_models()  # Prepare SUMMA settings

    print("\n3. Running calibration (DDS, 100 iterations)...")
    print("   This will take a few minutes...")

    try:
        results_file = sym.managers['optimization'].calibrate_model()

        if results_file:
            print(f"\n✓ Calibration completed!")
            print(f"  Results: {results_file}")
            return results_file
        else:
            print("\n✗ Calibration did not return results file")
            return None

    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@pytest.mark.slow
@pytest.mark.calibration
@pytest.mark.requires_data
def test_fyris_calibration():
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

    # Calibration settings - using 2-week period for quick demo
    config['EXPERIMENT_ID'] = 'calib_demo'
    config['EXPERIMENT_TIME_START'] = '2020-01-15 00:00'
    config['EXPERIMENT_TIME_END'] = '2020-01-31 23:00'  # 2 weeks for quick demo
    config['CALIBRATION_PERIOD'] = '2020-01-20, 2020-01-31'  # Skip 5-day warmup

    # Optimization configuration
    config['OPTIMIZATION_METHODS'] = ['iteration']
    config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'
    config['OPTIMIZATION_METRIC'] = 'KGE'
    config['NUMBER_OF_ITERATIONS'] = 100  # Quick demo
    config['CALIBRATION_TIMESTEP'] = 'daily'

    # Parameters to calibrate
    config['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat,fieldCapacity'
    config['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'

    # Observation data path
    config['OBSERVATIONS_PATH'] = 'default'

    # Save config
    temp_config = Path('tests/configs/test_calibration_fyris_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    # Check if model needs to be built or forcing data downloaded
    project_dir = Path(config['SYMFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}"
    summa_settings = project_dir / 'settings' / 'SUMMA'
    forcing_dir = project_dir / 'forcing' / 'SUMMA_input'

    # Check if forcing exists for calibration period (2020)
    needs_rebuild = not summa_settings.exists()
    if forcing_dir.exists():
        forcing_files = list(forcing_dir.glob('*2020*.nc'))
        if not forcing_files or all(f.stat().st_size < 100000 for f in forcing_files):  # Less than 100KB
            print("\n  No adequate forcing data for 2020 found, will rebuild model...")
            needs_rebuild = True
    else:
        needs_rebuild = True

    if needs_rebuild:
        print("\n2. Building SUMMA model and downloading forcing data...")
        print("   This will download CERRA data for Jan 15-31, 2020 (~2-3 min)...")
        sym.managers['data'].acquire_forcings()  # Download raw forcing data from cloud
        sym.managers['data'].run_model_agnostic_preprocessing()  # Process the downloaded data
        sym.managers['model'].preprocess_models()  # Prepare SUMMA settings

    print("\n3. Running calibration (DDS, 100 iterations)...")
    print("   This will take a few minutes...")

    try:
        results_file = sym.managers['optimization'].calibrate_model()

        if results_file:
            print(f"\n✓ Calibration completed!")
            print(f"  Results: {results_file}")
            return results_file
        else:
            print("\n✗ Calibration did not return results file")
            return None

    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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
