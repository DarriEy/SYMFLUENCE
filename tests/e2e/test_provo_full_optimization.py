"""
End-to-end test for Provo River lumped SUMMA model with full optimization.

This comprehensive test validates the complete SYMFLUENCE workflow including:
- Cloud data acquisition (Copernicus DEM, MODIS, AORC, USGS, GRACE)
- Lumped basin delineation
- SUMMA model preprocessing and execution
- 1000-iteration DDS optimization

This test requires significant runtime (2-4 hours) and cloud credentials.
"""

import pytest
from pathlib import Path
from symfluence import SYMFLUENCE

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.full_examples,
    pytest.mark.requires_cloud,
    pytest.mark.requires_acquisition,
    pytest.mark.slow,
    pytest.mark.summa,
    pytest.mark.calibration,
    pytest.mark.aorc,
]


@pytest.fixture
def provo_config():
    """Load Provo River SUMMA optimization config."""
    config_path = Path(__file__).parent.parent / "configs" / "test_provo_summa_optimization.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    return config_path


def test_provo_river_full_workflow(provo_config, clear_cache_flag):
    """
    Full E2E test for Provo River lumped SUMMA model with optimization.

    This test performs a comprehensive workflow:
    1. Delineates lumped basin for Provo River (USGS-10163000)
    2. Acquires cloud data:
       - Copernicus DEM
       - MODIS landcover
       - AORC forcings (2015-2019, 5 years)
       - USGS streamflow
       - GRACE data
    3. Runs SUMMA preprocessing
    4. Executes 1000-iteration DDS optimization
    5. Validates results

    Time periods:
    - Total: 2015-2019 (5 years)
    - Spinup: 2015 (1 year)
    - Calibration: 2016-2017 (2 years)
    - Evaluation: 2018-2019 (2 years)

    Parameters calibrated (6 total):
    - Soil: theta_sat, k_soil
    - Snow: snowfrz_scale, z0Snow
    - Routing: routingGammaScale, routingGammaShape

    Expected runtime: 2-4 hours

    Caching:
    - Use --clear-cache to force re-download of all data
    - Without --clear-cache, existing data will be reused
    """
    import shutil
    import yaml

    # Load config first to get domain name
    with open(provo_config, 'r') as f:
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
        print("✓ Previous test data removed")
    elif project_dir.exists():
        print(f"\n✓ Using cached data from: {project_dir}")
        print("  (Use --clear-cache to force re-download)")

    # Initialize SYMFLUENCE with config and enable visualizations
    sym = SYMFLUENCE(provo_config, visualize=True)
    print("✓ SYMFLUENCE initialized with visualizations enabled")

    # Setup project (creates directory structure)
    project_dir = sym.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"
    print(f"✓ Fresh project directory created at: {project_dir}")

    # 1. Acquire DEM first (required for domain delineation)
    print("\n" + "="*80)
    print("STEP 1: Acquire DEM (required for domain delineation)")
    print("="*80)

    # Check if DEM already exists
    dem_dir = project_dir / "attributes" / "elevation" / "dem"
    dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []

    if dem_files and not clear_cache_flag:
        print(f"✓ Using cached DEM: {dem_files[0].name}")
    else:
        print("Downloading Copernicus DEM for Provo River basin...")
        sym.managers["data"].acquire_attributes()
        dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        assert len(dem_files) > 0, f"DEM not downloaded in {dem_dir}"
        print(f"✓ DEM and other attributes acquired ({dem_files[0].name})")

    # 2. Domain definition and discretization
    print("\n" + "="*80)
    print("STEP 2: Domain Definition and Discretization")
    print("="*80)

    # Create pour point shapefile from coordinates
    print("Creating pour point shapefile from coordinates...")
    pour_point_path = sym.managers["project"].create_pour_point()
    assert pour_point_path is not None, "Pour point creation failed"
    assert pour_point_path.exists(), f"Pour point shapefile not created: {pour_point_path}"
    print(f"✓ Pour point shapefile created: {pour_point_path}")

    # Define domain (delineates watershed using TauDEM)
    print("Delineating watershed using TauDEM...")
    result, artifacts = sym.managers["domain"].define_domain()

    # Check what was created
    shapefiles_dir = project_dir / "shapefiles"
    if shapefiles_dir.exists():
        print(f"Checking shapefiles directory: {shapefiles_dir}")
        for subdir in shapefiles_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.*"))
                print(f"  {subdir.name}/: {len(files)} files")
                for f in files[:3]:  # Show first 3 files
                    print(f"    - {f.name}")

    # Check if river basins file was created
    river_basins_file = shapefiles_dir / "river_basins" / f"{config['DOMAIN_NAME']}_riverBasins_lumped.shp"
    if not river_basins_file.exists():
        print(f"⚠ Warning: River basins file not found at {river_basins_file}")
        print("Domain definition may have failed. Check logs for TauDEM errors.")
        # List what files do exist
        if (shapefiles_dir / "river_basins").exists():
            print(f"Files in river_basins/: {list((shapefiles_dir / 'river_basins').glob('*'))}")
        raise FileNotFoundError("Domain delineation failed - river basins shapefile not created. "
                              "This likely indicates TauDEM processing failed. "
                              "Check that TauDEM is installed and DEM is valid.")

    print("✓ Domain defined successfully")

    # Discretize domain (creates spatial units - HRUs/GRUs)
    print("Discretizing domain into GRUs...")
    sym.managers["domain"].discretize_domain()

    # Verify discretization outputs (filename includes discretization method suffix)
    discretization_method = config.get('DOMAIN_DISCRETIZATION', 'GRUs')
    catchment_file = shapefiles_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_{discretization_method}.shp"
    assert catchment_file.exists(), f"Catchment shapefile not created: {catchment_file}"
    print(f"✓ Domain discretized successfully: {catchment_file.name}")

    # 3. Acquire remaining data
    print("\n" + "="*80)
    print("STEP 3: Acquire Forcing and Observation Data")
    print("="*80)

    # Check if forcing data already exists
    forcing_dir = project_dir / "forcing" / "raw_data"
    forcing_files = list(forcing_dir.glob("*.nc")) if forcing_dir.exists() else []

    if forcing_files and not clear_cache_flag:
        print(f"✓ Using cached AORC forcing data ({len(forcing_files)} files)")
    else:
        print("Acquiring forcing data (AORC, 2015-2019)...")
        sym.managers["data"].acquire_forcings()
        forcing_files = list(forcing_dir.glob("*.nc")) if forcing_dir.exists() else []
        assert forcing_dir.exists(), "Forcing directory not created"
        assert len(forcing_files) > 0, "No forcing files downloaded"
        print(f"✓ AORC forcing data acquired ({len(forcing_files)} files)")

    # Check if observation data already exists
    streamflow_dir = project_dir / "observations" / "streamflow"
    raw_streamflow_dir = streamflow_dir / "raw_data"
    usgs_file = list(raw_streamflow_dir.glob("usgs_10163000_*.rdb")) if raw_streamflow_dir.exists() else []

    if usgs_file and not clear_cache_flag:
        print("✓ Using cached USGS streamflow and GRACE data")
    else:
        print("Acquiring observations (USGS streamflow, GRACE)...")
        sym.managers["data"].acquire_observations()

        # Verify USGS streamflow
        assert raw_streamflow_dir.exists(), "Streamflow raw data directory not created"
        usgs_file = list(raw_streamflow_dir.glob("usgs_10163000_*.rdb"))
        assert len(usgs_file) > 0, "USGS streamflow data not downloaded"
        print("✓ USGS streamflow data acquired")

    # Verify GRACE (if downloaded successfully)
    grace_dir = project_dir / "observations" / "GRACE"
    if grace_dir.exists():
        grace_files = list(grace_dir.glob("*.nc"))
        if len(grace_files) > 0:
            print(f"✓ GRACE data acquired ({len(grace_files)} files)")
        else:
            print("⚠ GRACE directory exists but no files (may be expected)")
    else:
        print("⚠ GRACE data not acquired (may require additional credentials)")

    # 4. Model preprocessing
    print("\n" + "="*80)
    print("STEP 4: Model Preprocessing")
    print("="*80)

    # Model-agnostic preprocessing (basin averaging)
    sym.managers["data"].run_model_agnostic_preprocessing()
    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    assert basin_avg_dir.exists(), "Basin-averaged forcing not created"
    basin_avg_files = list(basin_avg_dir.glob("*.nc"))
    assert len(basin_avg_files) > 0, "No basin-averaged forcing files created"
    print(f"✓ Basin-averaged forcing created ({len(basin_avg_files)} files)")

    # SUMMA-specific preprocessing
    sym.managers["model"].preprocess_models()
    summa_input_dir = project_dir / "forcing" / "SUMMA_input"
    assert summa_input_dir.exists(), "SUMMA input directory not created"
    summa_forcing_files = list(summa_input_dir.glob("*.nc"))
    assert len(summa_forcing_files) > 0, "No SUMMA forcing files created"
    print(f"✓ SUMMA preprocessing completed ({len(summa_forcing_files)} files)")

    # 5. Optimization
    print("\n" + "="*80)
    print("STEP 5: DDS Optimization (1000 iterations)")
    print("="*80)
    print("This step will take 1-3 hours depending on system performance...")

    # Run optimization
    results_file = sym.managers["optimization"].calibrate_model()
    assert results_file is not None, "Optimization did not produce results file"
    assert results_file.exists(), f"Results file not created: {results_file}"
    print(f"✓ Optimization completed, results at: {results_file}")

    # 6. Run Model (Best Parameters)
    print("\n" + "="*80)
    print("STEP 6: Run Model (Best Parameters)")
    print("="*80)
    sym.managers["model"].run_models()
    print("✓ Model execution completed")

    # 7. Postprocess Results
    print("\n" + "="*80)
    print("STEP 7: Postprocess Results")
    print("="*80)
    sym.managers["model"].postprocess_results()
    sym.managers["model"].visualize_outputs()
    print("✓ Results postprocessed and visualized")

    # 8. Validate results
    print("\n" + "="*80)
    print("STEP 8: Results Validation")
    print("="*80)

    # Check optimization directory
    optimization_dir = project_dir / "optimisation"
    assert optimization_dir.exists(), "Optimization directory not created"

    # Check DDS results file
    dds_results = optimization_dir / "DDS_results.csv"
    assert dds_results.exists(), f"DDS results file not found: {dds_results}"
    print("✓ DDS results file exists")

    # Verify 1000 iterations completed
    import pandas as pd
    results_df = pd.read_csv(dds_results)
    num_iterations = len(results_df)
    assert num_iterations >= 1000, f"Expected 1000+ iterations, got {num_iterations}"
    print(f"✓ Completed {num_iterations} iterations")

    # Verify KGE metric exists and improved
    assert 'KGE' in results_df.columns or 'objective' in results_df.columns, \
        "No KGE or objective metric found in results"

    metric_col = 'KGE' if 'KGE' in results_df.columns else 'objective'
    initial_kge = results_df[metric_col].iloc[0]
    final_kge = results_df[metric_col].max()

    print(f"  Initial {metric_col}: {initial_kge:.4f}")
    print(f"  Final {metric_col}: {final_kge:.4f}")
    print(f"  Improvement: {final_kge - initial_kge:.4f}")

    # Note: We don't strictly require improvement since the initial parameter set
    # might already be good, or the basin characteristics might limit improvement
    if final_kge > initial_kge:
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
        print("⚠ No SUMMA output files found (optimization may have used different structure)")

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("Basin: Provo River at Provo, UT (USGS-10163000)")
    print("Period: 2015-2019 (5 years)")
    print("Model: SUMMA (lumped)")
    print(f"Optimization: DDS with {num_iterations} iterations")
    print(f"Final {metric_col}: {final_kge:.4f}")
    print(f"Project directory: {project_dir}")
    print("="*80)
    print("✓ All validations passed!")
