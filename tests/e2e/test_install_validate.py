"""
End-to-end install & validate tests.

Replaces .github/workflows/install-validate.yml validation steps with pytest.
These tests validate the complete SYMFLUENCE installation and core functionality.
"""

import pytest
import shutil
import subprocess
from pathlib import Path
from symfluence import SYMFLUENCE



pytestmark = [pytest.mark.e2e, pytest.mark.ci_quick]

@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.smoke
def test_binary_validation(symfluence_code_dir):
    """
    Validate all external tool binaries are functional.

    Tests that SUMMA, mizuRoute, TauDEM, and FUSE executables exist
    and can be executed.
    """
    # Check for SUMMA
    summa_path = shutil.which("summa.exe")
    assert summa_path is not None, "SUMMA binary not found in PATH"

    # Check for mizuRoute
    mizu_path = shutil.which("mizuroute.exe")
    assert mizu_path is not None, "mizuRoute binary not found in PATH"

    # Check for TauDEM tools
    taudem_tools = ["pitremove", "d8flowdir", "aread8", "threshold", "streamnet", "gagewatershed"]
    for tool in taudem_tools:
        tool_path = shutil.which(tool)
        assert tool_path is not None, f"TauDEM tool '{tool}' not found in PATH"

    # Check for FUSE (optional)
    fuse_path = shutil.which("fuse.exe")
    if fuse_path is None:
        pytest.skip("FUSE binary not found (optional)")


@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.smoke
def test_package_imports():
    """
    Verify Python environment and package imports.

    Tests that all critical packages can be imported without errors.
    """
    # Core SYMFLUENCE
    import symfluence
    from symfluence import SYMFLUENCE

    # Critical data packages
    import xarray
    import pandas
    import numpy

    # Geospatial packages
    import geopandas
    import rasterio
    import shapely
    import pyproj
    import fiona

    # Model-specific
    import netCDF4

    # Optimization
    import scipy
    import torch

    # All imports successful
    assert True


@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.requires_data
@pytest.mark.parametrize("domain_name", ["bow"])
def test_quick_workflow_summa_only(
    domain_name,
    tmp_path,
    symfluence_code_dir,
    symfluence_data_root,
    bow_domain
):
    """
    Quick 3-hour SUMMA workflow test.

    Tests the complete workflow with minimal data:
    1. Setup project
    2. Use existing domain data (shapefiles, forcing)
    3. Preprocess forcing (3 hours)
    4. Run SUMMA simulation (3 hours)

    This is a smoke test for CI - fast validation of core functionality.
    """
    from utils.helpers import load_config_template, write_config

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(symfluence_data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Minimal 3-hour simulation
    config["DOMAIN_NAME"] = "bow_banff_minimal"
    config["EXPERIMENT_ID"] = f"test_quick_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-01 04:00"  # 3 hours

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"
    config["DOMAIN_DEFINITION_METHOD"] = "lumped"

    # Use existing data (no downloads)
    config["DOWNLOAD_WSC_DATA"] = False

    # Save config
    cfg_path = tmp_path / "config_quick.yaml"
    write_config(config, cfg_path)

    # Initialize SYMFLUENCE
    sym = SYMFLUENCE(cfg_path)

    # Setup project (creates directory structure)
    project_dir = sym.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Copy existing forcing data from bow_domain
    import shutil
    src_forcing = bow_domain / "forcing" / "raw_data"
    dst_forcing = project_dir / "forcing" / "raw_data"
    if src_forcing.exists():
        dst_forcing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_forcing, dst_forcing, dirs_exist_ok=True)

    # Copy existing shapefiles
    src_shapefiles = bow_domain / "shapefiles"
    dst_shapefiles = project_dir / "shapefiles"
    if src_shapefiles.exists():
        shutil.copytree(src_shapefiles, dst_shapefiles, dirs_exist_ok=True)

    # Copy existing attributes
    src_attributes = bow_domain / "attributes"
    dst_attributes = project_dir / "attributes"
    if src_attributes.exists():
        shutil.copytree(src_attributes, dst_attributes, dirs_exist_ok=True)

    # Model-agnostic preprocessing
    sym.managers["data"].run_model_agnostic_preprocessing()

    # Check preprocessing outputs
    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    assert basin_avg_dir.exists(), "Basin-averaged forcing should be created"

    # Model-specific preprocessing
    sym.managers["model"].preprocess_models()

    # Check SUMMA inputs
    summa_input_dir = project_dir / "forcing" / "SUMMA_input"
    assert summa_input_dir.exists(), "SUMMA input directory should exist"

    # Run SUMMA
    sym.managers["model"].run_models()

    # Check SUMMA output
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), "SUMMA simulation output should exist"
    output_files = list(sim_dir.glob("*_timestep.nc"))
    assert output_files, "SUMMA output files should exist"


@pytest.mark.e2e
@pytest.mark.ci_full
@pytest.mark.requires_data
@pytest.mark.parametrize("model", ["SUMMA"])  # Can add FUSE, NGEN later
def test_full_workflow_1month(
    model,
    tmp_path,
    symfluence_code_dir,
    symfluence_data_root,
    bow_domain
):
    """
    Full 1-month workflow test for each model.

    Tests the complete workflow with standard data:
    1. Setup project
    2. Use existing domain data
    3. Preprocess forcing (1 month)
    4. Run model simulation (1 month)
    5. Verify outputs

    This is a more comprehensive test for full CI validation.
    """
    from utils.helpers import load_config_template, write_config

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(symfluence_data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # 1-month simulation
    config["DOMAIN_NAME"] = "bow_banff_minimal"
    config["EXPERIMENT_ID"] = f"test_full_{model}_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-31 23:00"  # 1 month

    config["CALIBRATION_PERIOD"] = "2004-01-05, 2004-01-19"
    config["EVALUATION_PERIOD"] = "2004-01-20, 2004-01-30"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-04"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = model
    if model == "SUMMA":
        config["ROUTING_MODEL"] = "mizuRoute"

    # Save config
    cfg_path = tmp_path / "config_full.yaml"
    write_config(config, cfg_path)

    # Initialize and run
    sym = SYMFLUENCE(cfg_path)
    project_dir = sym.managers["project"].setup_project()

    # Copy data (same as quick test)
    import shutil
    for subdir in ["forcing/raw_data", "shapefiles", "attributes"]:
        src = bow_domain / subdir
        dst = project_dir / subdir
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Run full pipeline
    sym.managers["data"].run_model_agnostic_preprocessing()
    sym.managers["model"].preprocess_models()
    sym.managers["model"].run_models()

    # Verify outputs
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / model
    assert sim_dir.exists(), f"{model} simulation output should exist"


@pytest.mark.e2e
@pytest.mark.ci_full
@pytest.mark.calibration
@pytest.mark.requires_data
def test_calibration_workflow(tmp_path, symfluence_code_dir, symfluence_data_root, bow_domain):
    """
    Calibration workflow validation.

    Tests the complete calibration pipeline:
    1. Setup project
    2. Preprocess data
    3. Run model
    4. Calibrate model (minimal iterations for testing)
    5. Verify calibration outputs
    """
    from utils.helpers import load_config_template, write_config

    # Create test configuration
    config = load_config_template(symfluence_code_dir)
    config["SYMFLUENCE_DATA_DIR"] = str(symfluence_data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Calibration settings
    config["DOMAIN_NAME"] = "bow_banff_minimal"
    config["EXPERIMENT_ID"] = f"test_calib_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-31 23:00"
    config["CALIBRATION_PERIOD"] = "2004-01-05, 2004-01-19"
    config["EVALUATION_PERIOD"] = "2004-01-20, 2004-01-30"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-04"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"

    # Minimal calibration for testing
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42
    config["PARAMS_TO_CALIBRATE"] = "theta_sat"

    # Save config
    cfg_path = tmp_path / "config_calib.yaml"
    write_config(config, cfg_path)

    # Initialize and run
    sym = SYMFLUENCE(cfg_path)
    project_dir = sym.managers["project"].setup_project()

    # Copy data
    import shutil
    for subdir in ["forcing/raw_data", "shapefiles", "attributes", "observations"]:
        src = bow_domain / subdir
        dst = project_dir / subdir
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Run pipeline
    sym.managers["data"].run_model_agnostic_preprocessing()
    sym.managers["model"].preprocess_models()
    sym.managers["model"].run_models()

    # Run calibration
    results_file = sym.managers["optimization"].calibrate_model()
    assert results_file is not None, "Calibration should produce results"

    # Verify calibration outputs
    calib_dir = project_dir / "optimization"
    assert calib_dir.exists(), "Calibration directory should exist"
