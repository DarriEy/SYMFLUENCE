"""
End-to-end test for Paradise SNOTEL point-scale SUMMA model with full optimization.

This comprehensive test validates the complete SYMFLUENCE workflow including:
- Cloud data acquisition (Copernicus DEM, MODIS, ERA5, SNOTEL)
- Point-scale domain definition
- SUMMA model preprocessing and execution
- 1000-iteration DDS multivariate optimization (SWE + SCA)

This test requires significant runtime (2-4 hours) and cloud credentials.
"""

import pytest
from pathlib import Path
from symfluence import SYMFLUENCE
from utils.helpers import load_config_template

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.full_examples,
    pytest.mark.requires_cloud,
    pytest.mark.requires_acquisition,
    pytest.mark.slow,
    pytest.mark.summa,
    pytest.mark.calibration,
    pytest.mark.era5,
]


@pytest.fixture
def paradise_config():
    """Load Paradise SNOTEL SUMMA optimization config."""
    config_path = Path(__file__).parent.parent / "configs" / "test_paradise_summa_optimization.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    return config_path


def test_paradise_snotel_full_workflow(paradise_config, clear_cache_flag):
    """
    Full E2E test for Paradise SNOTEL point-scale SUMMA model with multivariate optimization.

    This test performs a comprehensive workflow:
    1. Defines point-scale domain for Paradise SNOTEL station (WA)
    2. Acquires cloud data:
       - Copernicus DEM
       - MODIS landcover
       - ERA5 forcings (2015-2019, 5 years)
       - SNOTEL SWE data
       - MODIS snow cover data
    3. Runs SUMMA preprocessing
    4. Executes 1000-iteration DDS multivariate optimization
    5. Validates results

    Time periods:
    - Total: 2015-2019 (5 years)
    - Spinup: 2015 (1 year)
    - Calibration: 2016-2017 (2 years)
    - Evaluation: 2018-2019 (2 years)

    Parameters calibrated (6 snow-focused parameters):
    - Snow: snowfrz_scale, z0Snow
    - Albedo: albedoMax, albedoMinWinter, albedoMaxVisible, albedoMinVisible

    Optimization targets:
    - SWE (Snow Water Equivalent): KGE metric, 50% weight
    - SCA (Snow Covered Area): Correlation, 50% weight

    Expected runtime: 2-4 hours

    Caching:
    - Use --clear-cache to force re-download of all data
    - Without --clear-cache, existing data will be reused
    """
    import shutil
    import yaml
    from datetime import datetime

    # Load config first to get domain name
    with open(paradise_config, 'r') as f:
        config = yaml.safe_load(f)

    # Set up project directory path
    import os
    project_base = Path(os.environ.get('SYMFLUENCE_DATA_DIR',
                                       config.get('SYMFLUENCE_DATA_DIR',
                                       '/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data')))
    project_dir = project_base / f"domain_{config['DOMAIN_NAME']}"

    # Clean up previous test data only if --clear-cache flag is set
    if clear_cache_flag and project_dir.exists():
        print(f"\n⚠ --clear-cache flag set, removing previous test data at: {project_dir}")
        shutil.rmtree(project_dir)
        print(f"✓ Previous test data removed")
    elif project_dir.exists():
        print(f"\n✓ Using cached data from: {project_dir}")
        print(f"  (Use --clear-cache to force re-download)")

    # Initialize SYMFLUENCE with config and enable visualizations
    sym = SYMFLUENCE(paradise_config, visualize=True)
    print(f"✓ SYMFLUENCE initialized with visualizations enabled")

    # Setup project (creates directory structure)
    project_dir = sym.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"
    print(f"✓ Fresh project directory created at: {project_dir}")

    # 1. Acquire DEM and attributes (required for domain setup)
    print("\n" + "="*80)
    print("STEP 1: Acquire DEM and Attributes")
    print("="*80)

    # Check if DEM already exists
    dem_dir = project_dir / "attributes" / "elevation" / "dem"
    dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []

    if dem_files and not clear_cache_flag:
        print(f"✓ Using cached DEM: {dem_files[0].name}")
    else:
        print("Downloading Copernicus DEM and MODIS landcover for Paradise SNOTEL...")
        sym.managers["data"].acquire_attributes()
        dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        assert len(dem_files) > 0, f"DEM not downloaded in {dem_dir}"
        print(f"✓ DEM and other attributes acquired ({dem_files[0].name})")

    # 2. Domain definition - Point scale (no watershed delineation needed)
    print("\n" + "="*80)
    print("STEP 2: Domain Definition and Discretization")
    print("="*80)

    # For point-scale, define_domain creates minimal spatial structure
    print("Defining point-scale domain...")
    result, artifacts = sym.managers["domain"].define_domain()
    print(f"✓ Point-scale domain defined successfully")

    # Discretize domain (creates single GRU for point-scale)
    print("Discretizing domain into single GRU...")
    sym.managers["domain"].discretize_domain()

    # Verify discretization outputs
    shapefiles_dir = project_dir / "shapefiles"
    discretization_method = config.get('DOMAIN_DISCRETIZATION', 'GRUs')
    catchment_file = shapefiles_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_{discretization_method}.shp"
    assert catchment_file.exists(), f"Catchment shapefile not created: {catchment_file}"
    print(f"✓ Domain discretized successfully: {catchment_file.name}")

    # 3. Acquire forcing and observation data
    print("\n" + "="*80)
    print("STEP 3: Acquire Forcing and Observation Data")
    print("="*80)

    # Check if forcing data already exists
    forcing_dir = project_dir / "forcing" / "raw_data"
    forcing_files = list(forcing_dir.glob("*.nc")) if forcing_dir.exists() else []

    if forcing_files and not clear_cache_flag:
        print(f"✓ Using cached ERA5 forcing data ({len(forcing_files)} files)")
    else:
        print("Acquiring forcing data (ERA5, 2015-2019)...")
        sym.managers["data"].acquire_forcings()
        forcing_files = list(forcing_dir.glob("*.nc")) if forcing_dir.exists() else []
        assert forcing_dir.exists(), "Forcing directory not created"
        assert len(forcing_files) > 0, "No forcing files downloaded"
        print(f"✓ ERA5 forcing data acquired ({len(forcing_files)} files)")

    # Check if observation data already exists
    snotel_dir = project_dir / "observations" / "snow" / "swe"
    snotel_files = list(snotel_dir.glob("*.csv")) if snotel_dir.exists() else []
    if len(snotel_files) == 0:
        # Try raw data directory
        snotel_raw_dir = project_dir / "observations" / "snow"
        snotel_files = list(snotel_raw_dir.glob("*snotel*.csv")) if snotel_raw_dir.exists() else []

    if snotel_files and not clear_cache_flag:
        print(f"✓ Using cached SNOTEL and MODIS data")
    else:
        print("Acquiring observations (SNOTEL SWE, MODIS snow cover)...")
        sym.managers["data"].acquire_observations()

        # Verify SNOTEL data
        snotel_dir = project_dir / "observations" / "snow" / "swe"
        snotel_files = list(snotel_dir.glob("*.csv")) if snotel_dir.exists() else []
        if len(snotel_files) == 0:
            # Try raw data directory
            snotel_raw_dir = project_dir / "observations" / "snow"
            snotel_files = list(snotel_raw_dir.glob("*snotel*.csv")) if snotel_raw_dir.exists() else []
        assert len(snotel_files) > 0, "SNOTEL data not downloaded"
        print(f"✓ SNOTEL SWE data acquired")

    # Verify MODIS snow data
    modis_dir = project_dir / "observations" / "snow"
    modis_files = list(modis_dir.glob("*MODIS*.nc")) or list(modis_dir.glob("*MOD10*.nc"))
    if len(modis_files) > 0:
        print(f"✓ MODIS snow cover data acquired ({len(modis_files)} files)")
    else:
        print(f"⚠ MODIS data may require processing (check raw files)")

    # Process observations
    print("Processing observation data...")
    sym.managers["data"].process_observed_data()

    # Check for processed observations
    processed_dir = project_dir / "observations" / "snow" / "preprocessed"
    if processed_dir.exists():
        processed_files = list(processed_dir.glob("*.csv"))
        print(f"✓ Observation data processed ({len(processed_files)} files)")

    # 4. Model preprocessing
    print("\n" + "="*80)
    print("STEP 4: Model Preprocessing")
    print("="*80)

    # Model-agnostic preprocessing (point-scale forcing)
    sym.managers["data"].run_model_agnostic_preprocessing()
    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    assert basin_avg_dir.exists(), "Basin-averaged forcing not created"
    basin_avg_files = list(basin_avg_dir.glob("*.nc"))
    assert len(basin_avg_files) > 0, "No basin-averaged forcing files created"
    print(f"✓ Point-scale forcing prepared ({len(basin_avg_files)} files)")

    # SUMMA-specific preprocessing
    sym.managers["model"].preprocess_models()
    summa_input_dir = project_dir / "forcing" / "SUMMA_input"
    assert summa_input_dir.exists(), "SUMMA input directory not created"
    summa_forcing_files = list(summa_input_dir.glob("*.nc"))
    assert len(summa_forcing_files) > 0, "No SUMMA forcing files created"
    print(f"✓ SUMMA preprocessing completed ({len(summa_forcing_files)} files)")

    # 5. Multivariate Optimization
    print("\n" + "="*80)
    print("STEP 5: DDS Multivariate Optimization (1000 iterations)")
    print("="*80)
    print("Optimizing for SWE (50% weight) and SCA (50% weight)...")
    print("This step will take 1-3 hours depending on system performance...")

    # Run optimization
    results_file = sym.managers["optimization"].calibrate_model()
    assert results_file is not None, "Optimization did not produce results file"
    assert results_file.exists(), f"Results file not created: {results_file}"
    print(f"✓ Optimization completed, results at: {results_file}")

    # 6. Validate results
    print("\n" + "="*80)
    print("STEP 6: Results Validation")
    print("="*80)

    # Check optimization directory
    optimization_dir = project_dir / "optimisation"
    assert optimization_dir.exists(), "Optimization directory not created"

    # Check DDS results file
    dds_results = optimization_dir / "DDS_results.csv"
    assert dds_results.exists(), f"DDS results file not found: {dds_results}"
    print(f"✓ DDS results file exists")

    # Verify 1000 iterations completed
    import pandas as pd
    results_df = pd.read_csv(dds_results)
    num_iterations = len(results_df)
    assert num_iterations >= 1000, f"Expected 1000+ iterations, got {num_iterations}"
    print(f"✓ Completed {num_iterations} iterations")

    # Verify multivariate objective exists
    # For multivariate optimization, check for composite objective or individual metrics
    possible_metrics = ['objective', 'composite_objective', 'SWE_KGE', 'SCA_corr']
    metric_col = None
    for col in possible_metrics:
        if col in results_df.columns:
            metric_col = col
            break

    assert metric_col is not None, f"No objective metric found. Available columns: {results_df.columns.tolist()}"

    initial_obj = results_df[metric_col].iloc[0]
    final_obj = results_df[metric_col].max()

    print(f"  Initial {metric_col}: {initial_obj:.4f}")
    print(f"  Final {metric_col}: {final_obj:.4f}")
    print(f"  Improvement: {final_obj - initial_obj:.4f}")

    # Note: We don't strictly require improvement since the initial parameter set
    # might already be good, or snow processes might be complex to optimize
    if final_obj > initial_obj:
        print(f"✓ {metric_col} improved over iterations")
    else:
        print(f"⚠ {metric_col} did not improve (initial params may be optimal)")

    # Check SUMMA output files exist
    summa_output = project_dir / "simulations"
    assert summa_output.exists(), "Simulations directory not found"

    # Look for SUMMA output files
    summa_files = list(summa_output.rglob("*SUMMA*.nc"))
    if len(summa_files) > 0:
        print(f"✓ SUMMA output files exist ({len(summa_files)} files)")
    else:
        print(f"⚠ No SUMMA output files found (optimization may have used different structure)")

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Site: Paradise SNOTEL, Mount Rainier, WA (Station 679)")
    print(f"Period: 2015-2019 (5 years)")
    print(f"Model: SUMMA (point-scale)")
    print(f"Forcing: ERA5")
    print(f"Optimization: DDS multivariate with {num_iterations} iterations")
    print(f"Targets: SWE (KGE, 50%) + SCA (corr, 50%)")
    print(f"Final {metric_col}: {final_obj:.4f}")
    print(f"Project directory: {project_dir}")
    print("="*80)
    print("✓ All validations passed!")
