"""
Quick test to verify SUMMA can build and run with CARRA/CERRA forcing.

Uses pre-existing domains from test_regional_reanalysis.py tests.
Only tests: preprocessing → SUMMA build → SUMMA run
"""
import yaml
from pathlib import Path
from symfluence import SYMFLUENCE
import xarray as xr


def test_cerra_summa_quick():
    """
    Quick SUMMA test with CERRA forcing for pre-existing Fyrisån domain.

    Steps:
    1. Use existing fyris_uppsala domain (from test_regional_reanalysis.py)
    2. Use existing CERRA forcing data
    3. Preprocess forcing (subset to basin)
    4. Build SUMMA model
    5. Run SUMMA simulation
    """
    print("\n" + "="*80)
    print("CERRA → SUMMA Quick Test (Fyrisån, Sweden)")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use existing domain from earlier tests
    config['DOMAIN_NAME'] = 'fyris_uppsala'
    config['BOUNDING_BOX_COORDS'] = '59.87/17.64/59.85/17.66'
    config['POUR_POINT_COORDS'] = '59.86/17.65'
    config['FORCING_DATASET'] = 'CERRA'
    config['FORCING_TIME_STEP_SIZE'] = 10800  # 3 hours to match CERRA 3-hourly data
    config['EXPERIMENT_ID'] = 'summa_test'
    config['EXPERIMENT_TIME_START'] = '2010-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2010-01-01 03:00'  # Changed to 3 hours to align with CERRA 3-hourly data
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # Save config
    temp_config = Path('test_quick_summa_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    data_root = Path(config['SYMFLUENCE_DATA_DIR'])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

    # Check domain exists
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    if not hrus_file.exists():
        raise FileNotFoundError(
            f"Domain {config['DOMAIN_NAME']} not found. "
            f"Run test_regional_reanalysis.py first to create the domain."
        )
    print(f"   ✓ Using existing domain: {project_dir}")

    # Check forcing data exists
    raw_forcing_dir = project_dir / 'forcing' / 'raw_data'
    cerra_files = list(raw_forcing_dir.glob('*CERRA*.nc'))
    if not cerra_files:
        raise FileNotFoundError(
            "No CERRA forcing data found. "
            "Run test_regional_reanalysis.py first to download forcing."
        )
    print(f"   ✓ Using existing forcing: {cerra_files[0].name}")

    # Verify forcing has SUMMA variables
    with xr.open_dataset(cerra_files[0]) as ds:
        expected_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
        found_vars = [v for v in expected_vars if v in ds.data_vars]
        assert len(found_vars) == 7, f"Expected 7 SUMMA variables, found {len(found_vars)}"
        print(f"   ✓ All 7 SUMMA variables present")

    # Check if preprocessing already done
    # The remapped forcing files are in basin_averaged_data after easymore processing
    merged_dir = project_dir / 'forcing' / 'basin_averaged_data'
    merged_files = list(merged_dir.glob('*_remapped_*.nc')) if merged_dir.exists() else []

    if not merged_files:
        print("\n2. Preprocessing forcing data (subsetting to basin HRUs)...")
        print("   This may take a while for the first run...")
        try:
            sym.managers['data'].run_model_agnostic_preprocessing()
            print("   ✓ Forcing preprocessing completed")

            merged_files = list(merged_dir.glob('*.nc'))
            if not merged_files:
                raise FileNotFoundError("No merged forcing file created")

            with xr.open_dataset(merged_files[0]) as ds:
                n_hrus = ds.dims.get('hru', 0)
                n_time = ds.dims.get('time', 0)
                print(f"   ✓ Merged forcing: {merged_files[0].name}")
                print(f"   ✓ Dimensions: {n_hrus} HRUs x {n_time} timesteps")
        except Exception as e:
            print(f"   ✗ Preprocessing failed: {e}")
            raise
    else:
        print(f"\n2. ✓ Using existing preprocessed forcing: {merged_files[0].name}")
        with xr.open_dataset(merged_files[0]) as ds:
            n_hrus = ds.dims.get('hru', 0)
            n_time = ds.dims.get('time', 0)
            print(f"   Dimensions: {n_hrus} HRUs x {n_time} timesteps")

    # Build SUMMA model
    print("\n3. Building SUMMA model...")
    try:
        sym.managers['model'].preprocess_models()
        print("   ✓ SUMMA model build completed")

        # Verify SUMMA files created
        settings_dir = project_dir / 'settings' / 'SUMMA'
        summa_manager = settings_dir / 'fileManager.txt'  # Camel case!
        if not summa_manager.exists():
            raise FileNotFoundError("SUMMA file manager not created")
        print(f"   ✓ SUMMA file manager: {summa_manager}")

        # Check for key SUMMA files
        attributes_file = settings_dir / 'attributes.nc'
        trial_params = settings_dir / 'trialParams.nc'

        if attributes_file.exists():
            print(f"   ✓ Attributes file: {attributes_file.name}")
        if trial_params.exists():
            print(f"   ✓ Trial params: {trial_params.name}")

    except Exception as e:
        print(f"   ✗ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Run SUMMA simulation
    print("\n4. Running SUMMA simulation...")
    try:
        sym.managers['model'].run_models()
        print("   ✓ SUMMA simulation completed")

        # Verify output files
        output_dir = project_dir / 'simulations' / config['EXPERIMENT_ID'] / 'SUMMA'
        output_files = list(output_dir.glob('*_timestep.nc')) if output_dir.exists() else []

        if not output_files:
            raise FileNotFoundError("No SUMMA output files found")

        print(f"   ✓ Output file: {output_files[0].name}")

        # Check output contents
        with xr.open_dataset(output_files[0]) as ds:
            n_vars = len(ds.data_vars)
            n_time = ds.dims.get('time', 0)
            n_hru = ds.dims.get('hru', 0)
            print(f"   ✓ Output dimensions: {n_hru} HRUs x {n_time} timesteps x {n_vars} variables")
            print(f"   ✓ Sample variables: {list(ds.data_vars.keys())[:5]}")

    except Exception as e:
        print(f"   ✗ SUMMA run failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "="*80)
    print("✓ CERRA → SUMMA pipeline PASSED!")
    print("="*80)
    return True


def test_carra_summa_quick():
    """
    Quick SUMMA test with CARRA forcing for pre-existing Elliðaár domain.

    Steps:
    1. Use existing ellioaar_iceland domain (from test_regional_reanalysis.py)
    2. Use existing CARRA forcing data
    3. Preprocess forcing (subset to basin)
    4. Build SUMMA model
    5. Run SUMMA simulation
    """
    print("\n" + "="*80)
    print("CARRA → SUMMA Quick Test (Elliðaár, Iceland)")
    print("="*80)

    # Load template config
    config_path = Path('0_config_files/config_template.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use existing domain from earlier tests
    config['DOMAIN_NAME'] = 'ellioaar_iceland'
    config['BOUNDING_BOX_COORDS'] = '64.13/-21.96/64.11/-21.94'
    config['POUR_POINT_COORDS'] = '64.12/-21.95'
    config['FORCING_DATASET'] = 'CARRA'
    config['CARRA_DOMAIN'] = 'west_domain'
    config['FORCING_TIME_STEP_SIZE'] = 10800  # 3 hours to match CARRA 3-hourly data
    config['EXPERIMENT_ID'] = 'summa_test'
    config['EXPERIMENT_TIME_START'] = '2010-01-01 00:00'
    config['EXPERIMENT_TIME_END'] = '2010-01-01 03:00'  # Changed to 3 hours to align with CARRA 3-hourly data
    config['DATA_ACCESS'] = 'cloud'
    config['DEM_SOURCE'] = 'copernicus'
    config['HYDROLOGICAL_MODEL'] = 'SUMMA'

    # Save config
    temp_config = Path('test_quick_summa_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    print("\n1. Initializing SYMFLUENCE...")
    sym = SYMFLUENCE(temp_config)

    data_root = Path(config['SYMFLUENCE_DATA_DIR'])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

    # Check domain exists
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    if not hrus_file.exists():
        raise FileNotFoundError(
            f"Domain {config['DOMAIN_NAME']} not found. "
            f"Run test_regional_reanalysis.py first to create the domain."
        )
    print(f"   ✓ Using existing domain: {project_dir}")

    # Check forcing data exists
    raw_forcing_dir = project_dir / 'forcing' / 'raw_data'
    carra_files = list(raw_forcing_dir.glob('*CARRA*.nc'))
    if not carra_files:
        raise FileNotFoundError(
            "No CARRA forcing data found. "
            "Run test_regional_reanalysis.py first to download forcing."
        )
    print(f"   ✓ Using existing forcing: {carra_files[0].name}")

    # Verify forcing has SUMMA variables
    with xr.open_dataset(carra_files[0]) as ds:
        expected_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
        found_vars = [v for v in expected_vars if v in ds.data_vars]
        assert len(found_vars) == 7, f"Expected 7 SUMMA variables, found {len(found_vars)}"
        print(f"   ✓ All 7 SUMMA variables present")

    # Check if preprocessing already done
    # The remapped forcing files are in basin_averaged_data after easymore processing
    merged_dir = project_dir / 'forcing' / 'basin_averaged_data'
    merged_files = list(merged_dir.glob('*_remapped_*.nc')) if merged_dir.exists() else []

    if not merged_files:
        print("\n2. Preprocessing forcing data (subsetting to basin HRUs)...")
        print("   This may take a while for the first run...")
        try:
            sym.managers['data'].run_model_agnostic_preprocessing()
            print("   ✓ Forcing preprocessing completed")

            merged_files = list(merged_dir.glob('*.nc'))
            if not merged_files:
                raise FileNotFoundError("No merged forcing file created")

            with xr.open_dataset(merged_files[0]) as ds:
                n_hrus = ds.dims.get('hru', 0)
                n_time = ds.dims.get('time', 0)
                print(f"   ✓ Merged forcing: {merged_files[0].name}")
                print(f"   ✓ Dimensions: {n_hrus} HRUs x {n_time} timesteps")
        except Exception as e:
            print(f"   ✗ Preprocessing failed: {e}")
            raise
    else:
        print(f"\n2. ✓ Using existing preprocessed forcing: {merged_files[0].name}")
        with xr.open_dataset(merged_files[0]) as ds:
            n_hrus = ds.dims.get('hru', 0)
            n_time = ds.dims.get('time', 0)
            print(f"   Dimensions: {n_hrus} HRUs x {n_time} timesteps")

    # Build SUMMA model
    print("\n3. Building SUMMA model...")
    try:
        sym.managers['model'].preprocess_models()
        print("   ✓ SUMMA model build completed")

        # Verify SUMMA files created
        settings_dir = project_dir / 'settings' / 'SUMMA'
        summa_manager = settings_dir / 'fileManager.txt'  # Camel case!
        if not summa_manager.exists():
            raise FileNotFoundError("SUMMA file manager not created")
        print(f"   ✓ SUMMA file manager: {summa_manager}")

        # Check for key SUMMA files
        attributes_file = settings_dir / 'attributes.nc'
        trial_params = settings_dir / 'trialParams.nc'

        if attributes_file.exists():
            print(f"   ✓ Attributes file: {attributes_file.name}")
        if trial_params.exists():
            print(f"   ✓ Trial params: {trial_params.name}")

    except Exception as e:
        print(f"   ✗ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Run SUMMA simulation
    print("\n4. Running SUMMA simulation...")
    try:
        sym.managers['model'].run_models()
        print("   ✓ SUMMA simulation completed")

        # Verify output files
        output_dir = project_dir / 'simulations' / config['EXPERIMENT_ID'] / 'SUMMA'
        output_files = list(output_dir.glob('*_timestep.nc')) if output_dir.exists() else []

        if not output_files:
            raise FileNotFoundError("No SUMMA output files found")

        print(f"   ✓ Output file: {output_files[0].name}")

        # Check output contents
        with xr.open_dataset(output_files[0]) as ds:
            n_vars = len(ds.data_vars)
            n_time = ds.dims.get('time', 0)
            n_hru = ds.dims.get('hru', 0)
            print(f"   ✓ Output dimensions: {n_hru} HRUs x {n_time} timesteps x {n_vars} variables")
            print(f"   ✓ Sample variables: {list(ds.data_vars.keys())[:5]}")

    except Exception as e:
        print(f"   ✗ SUMMA run failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "="*80)
    print("✓ CARRA → SUMMA pipeline PASSED!")
    print("="*80)
    return True


if __name__ == "__main__":
    import sys

    print("Quick SUMMA Tests with Regional Reanalysis Data")
    print("=" * 80)
    print()

    # Test CERRA first (should be faster since domain/forcing already exist)
    try:
        test_cerra_summa_quick()
    except FileNotFoundError as e:
        print(f"\n⚠ CERRA test skipped: {e}")
    except Exception as e:
        print(f"\n✗ CERRA test failed: {e}")
        sys.exit(1)

    # Test CARRA
    try:
        test_carra_summa_quick()
    except FileNotFoundError as e:
        print(f"\n⚠ CARRA test skipped: {e}")
    except Exception as e:
        print(f"\n✗ CARRA test failed: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("✓ All quick SUMMA tests passed!")
    print("="*80)
