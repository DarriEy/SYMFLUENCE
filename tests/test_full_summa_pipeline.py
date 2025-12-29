"""
Test complete SUMMA pipeline with CARRA and CERRA forcing data.

This test runs the full end-to-end pipeline:
1. Setup domain (if needed)
2. Download forcing data (CARRA or CERRA)
3. Preprocess forcing data (subset to basin HRUs)
4. Build SUMMA model inputs
5. Run SUMMA simulation
6. Verify outputs
"""
import yaml
import pytest
from pathlib import Path
from symfluence import SYMFLUENCE
import xarray as xr


def test_cerra_full_summa_pipeline():
    """
    Test complete SUMMA pipeline with CERRA forcing for Fyrisån, Sweden.

    This test:
    1. Uses existing Fyrisån domain (2km x 2km basin in Uppsala)
    2. Downloads CERRA forcing data (6 hours)
    3. Preprocesses forcing to basin HRUs
    4. Builds SUMMA model
    5. Runs SUMMA simulation
    """
    print("\n" + "="*80)
    print("CERRA → SUMMA Full Pipeline Test")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Configure for Fyrisån basin, Uppsala, Sweden
    config['DOMAIN_NAME'] = 'fyris_uppsala'
    config['BOUNDING_BOX_COORDS'] = '59.87/17.64/59.85/17.66'
    config['POUR_POINT_COORDS'] = '59.86/17.65'
    config['FORCING_DATASET'] = 'CERRA'
    config['EXPERIMENT_ID'] = 'summa_cerra_test'
    config['EXPERIMENT_TIME_START'] = '2010-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2010-01-01 06:00'  # 6 hours (2 CERRA timesteps)
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # Save config
    temp_config = Path('test_cerra_summa_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Initialize SYMFLUENCE
    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    data_root = Path(config['SYMFLUENCE_DATA_DIR'])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

    # Check if domain exists
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    if not hrus_file.exists():
        print("   Domain not found, setting up (first run)...")
        sym.managers["project"].setup_project()
        pour_point_path = sym.managers["project"].create_pour_point()
        sym.managers["data"].acquire_attributes()
        sym.managers["domain"].define_domain()
        sym.managers["domain"].discretize_domain()
        print(f"   ✓ Domain setup complete")
    else:
        print(f"   ✓ Using existing domain")

    # Download forcing data
    print("\n2. Downloading CERRA forcing data...")
    raw_forcing_dir = project_dir / 'forcing' / 'raw_data'
    existing_files = list(raw_forcing_dir.glob('*CERRA*.nc')) if raw_forcing_dir.exists() else []

    if not existing_files:
        sym.managers['data'].acquire_forcings()
        cerra_files = list(raw_forcing_dir.glob('*CERRA*.nc'))
        print(f"   ✓ Downloaded: {cerra_files[0].name}")
    else:
        print(f"   ✓ Using existing: {existing_files[0].name}")
        cerra_files = existing_files

    # Verify raw forcing has required SUMMA variables
    with xr.open_dataset(cerra_files[0]) as ds:
        required_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum']
        optional_vars = ['LWRadAtm']
        missing_required = [v for v in required_vars if v not in ds.data_vars]
        assert not missing_required, f"Missing required SUMMA variables: {missing_required}"
        found_optional = [v for v in optional_vars if v in ds.data_vars]
        print(f"   ✓ Required SUMMA variables present: {required_vars}")
        if found_optional:
            print(f"   ✓ Optional SUMMA variables present: {found_optional}")

    # Preprocess forcing data (subset to basin HRUs)
    print("\n3. Preprocessing forcing data (subsetting to basin HRUs)...")
    try:
        sym.managers['data'].run_model_agnostic_preprocessing()
        print("   ✓ Forcing preprocessing completed")

        # Verify merged forcing file
        merged_files = []
        merged_dir = None
        for candidate in [project_dir / 'forcing' / 'merged_path', project_dir / 'forcing' / 'merged_data']:
            if candidate.exists():
                merged_files = list(candidate.glob('*.nc'))
                if merged_files:
                    merged_dir = candidate
                    break
        if merged_files:
            print(f"   ✓ Merged forcing: {merged_files[0].name} ({merged_dir})")
            with xr.open_dataset(merged_files[0]) as ds:
                n_hrus = ds.dims.get('hru', 0)
                n_time = ds.dims.get('time', 0)
                print(f"   ✓ HRUs: {n_hrus}, Timesteps: {n_time}")
                print(f"   ✓ Variables: {list(ds.data_vars.keys())}")
        else:
            raise FileNotFoundError("No merged forcing file created")
    except Exception as e:
        print(f"   ✗ Preprocessing failed: {e}")
        raise

    # Build SUMMA model
    print("\n4. Building SUMMA model...")
    try:
        sym.managers['model'].preprocess_models()
        print("   ✓ SUMMA model build completed")

        # Verify SUMMA setup files
        settings_dir = project_dir / 'settings' / 'SUMMA'
        summa_manager = settings_dir / 'fileManager.txt'
        if summa_manager.exists():
            print(f"   ✓ SUMMA file manager: {summa_manager}")
        else:
            raise FileNotFoundError("SUMMA file manager not created")
    except Exception as e:
        print(f"   ✗ Model build failed: {e}")
        raise

    # Run SUMMA simulation
    print("\n5. Running SUMMA simulation...")
    try:
        sym.managers['model'].run_models()
        print("   ✓ SUMMA simulation completed")

        # Verify output files
        output_dir = project_dir / 'simulations' / config['EXPERIMENT_ID']
        output_files = list(output_dir.glob('*_timestep.nc')) if output_dir.exists() else []
        if output_files:
            print(f"   ✓ Output file: {output_files[0].name}")
            with xr.open_dataset(output_files[0]) as ds:
                print(f"   ✓ Output variables: {list(ds.data_vars.keys())[:5]}...")  # First 5
        else:
            raise FileNotFoundError("No SUMMA output files found")
    except Exception as e:
        print(f"   ✗ SUMMA run failed: {e}")
        raise

    print("\n" + "="*80)
    print("✓ CERRA → SUMMA pipeline completed successfully!")
    print("="*80)


def test_carra_full_summa_pipeline():
    """
    Test complete SUMMA pipeline with CARRA forcing for Elliðaár, Iceland.

    This test:
    1. Uses existing Elliðaár domain (2km x 2km basin in Reykjavik)
    2. Downloads CARRA forcing data (6 hours)
    3. Preprocesses forcing to basin HRUs
    4. Builds SUMMA model
    5. Runs SUMMA simulation
    """
    print("\n" + "="*80)
    print("CARRA → SUMMA Full Pipeline Test")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Configure for Elliðaár basin, Reykjavik, Iceland
    config['DOMAIN_NAME'] = 'ellioaar_iceland'
    config['BOUNDING_BOX_COORDS'] = '64.13/-21.96/64.11/-21.94'
    config['POUR_POINT_COORDS'] = '64.12/-21.95'
    config['FORCING_DATASET'] = 'CARRA'
    config['CARRA_DOMAIN'] = 'west_domain'
    config['EXPERIMENT_ID'] = 'summa_carra_test'
    config['EXPERIMENT_TIME_START'] = '2010-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2010-01-01 06:00'  # 6 hours
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # Save config
    temp_config = Path('test_carra_summa_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Initialize SYMFLUENCE
    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    data_root = Path(config['SYMFLUENCE_DATA_DIR'])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

    # Check if domain exists
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    if not hrus_file.exists():
        print("   Domain not found, setting up (first run)...")
        sym.managers["project"].setup_project()
        pour_point_path = sym.managers["project"].create_pour_point()
        sym.managers["data"].acquire_attributes()
        sym.managers["domain"].define_domain()
        sym.managers["domain"].discretize_domain()
        print(f"   ✓ Domain setup complete")
    else:
        print(f"   ✓ Using existing domain")

    # Download forcing data
    print("\n2. Downloading CARRA forcing data...")
    raw_forcing_dir = project_dir / 'forcing' / 'raw_data'
    existing_files = list(raw_forcing_dir.glob('*CARRA*.nc')) if raw_forcing_dir.exists() else []

    if not existing_files:
        sym.managers['data'].acquire_forcings()
        carra_files = list(raw_forcing_dir.glob('*CARRA*.nc'))
        print(f"   ✓ Downloaded: {carra_files[0].name}")
    else:
        print(f"   ✓ Using existing: {existing_files[0].name}")
        carra_files = existing_files

    # Verify raw forcing has required SUMMA variables
    with xr.open_dataset(carra_files[0]) as ds:
        required_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum']
        optional_vars = ['LWRadAtm']
        missing_required = [v for v in required_vars if v not in ds.data_vars]
        assert not missing_required, f"Missing required SUMMA variables: {missing_required}"
        found_optional = [v for v in optional_vars if v in ds.data_vars]
        print(f"   ✓ Required SUMMA variables present: {required_vars}")
        if found_optional:
            print(f"   ✓ Optional SUMMA variables present: {found_optional}")

    # Preprocess forcing data (subset to basin HRUs)
    print("\n3. Preprocessing forcing data (subsetting to basin HRUs)...")
    try:
        sym.managers['data'].run_model_agnostic_preprocessing()
        print("   ✓ Forcing preprocessing completed")

        # Verify merged forcing file
        merged_files = []
        merged_dir = None
        for candidate in [project_dir / 'forcing' / 'merged_path', project_dir / 'forcing' / 'merged_data']:
            if candidate.exists():
                merged_files = list(candidate.glob('*.nc'))
                if merged_files:
                    merged_dir = candidate
                    break
        if merged_files:
            print(f"   ✓ Merged forcing: {merged_files[0].name} ({merged_dir})")
            with xr.open_dataset(merged_files[0]) as ds:
                n_hrus = ds.dims.get('hru', 0)
                n_time = ds.dims.get('time', 0)
                print(f"   ✓ HRUs: {n_hrus}, Timesteps: {n_time}")
                print(f"   ✓ Variables: {list(ds.data_vars.keys())}")
        else:
            raise FileNotFoundError("No merged forcing file created")
    except Exception as e:
        print(f"   ✗ Preprocessing failed: {e}")
        raise

    # Build SUMMA model
    print("\n4. Building SUMMA model...")
    try:
        sym.managers['model'].preprocess_models()
        print("   ✓ SUMMA model build completed")

        # Verify SUMMA setup files
        settings_dir = project_dir / 'settings' / 'SUMMA'
        summa_manager = settings_dir / 'fileManager.txt'
        if summa_manager.exists():
            print(f"   ✓ SUMMA file manager: {summa_manager}")
        else:
            raise FileNotFoundError("SUMMA file manager not created")
    except Exception as e:
        print(f"   ✗ Model build failed: {e}")
        raise

    # Run SUMMA simulation
    print("\n5. Running SUMMA simulation...")
    try:
        sym.managers['model'].run_models()
        print("   ✓ SUMMA simulation completed")

        # Verify output files
        output_dir = project_dir / 'simulations' / config['EXPERIMENT_ID']
        output_files = list(output_dir.glob('*_timestep.nc')) if output_dir.exists() else []
        if output_files:
            print(f"   ✓ Output file: {output_files[0].name}")
            with xr.open_dataset(output_files[0]) as ds:
                print(f"   ✓ Output variables: {list(ds.data_vars.keys())[:5]}...")  # First 5
        else:
            raise FileNotFoundError("No SUMMA output files found")
    except Exception as e:
        print(f"   ✗ SUMMA run failed: {e}")
        raise

    print("\n" + "="*80)
    print("✓ CARRA → SUMMA pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    import sys

    print("Regional Reanalysis → SUMMA Full Pipeline Tests")
    print("This will test the complete workflow from data download to SUMMA simulation")
    print()

    # Run CERRA test
    try:
        test_cerra_full_summa_pipeline()
    except Exception as e:
        print(f"\n✗ CERRA test failed: {e}")
        sys.exit(1)

    # Run CARRA test
    try:
        test_carra_full_summa_pipeline()
    except Exception as e:
        print(f"\n✗ CARRA test failed: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
