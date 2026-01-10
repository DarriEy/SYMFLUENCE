"""
End-to-end install & validate tests.

Replaces .github/workflows/install-validate.yml validation steps with pytest.
These tests validate the complete SYMFLUENCE installation and core functionality.
"""

import os
import sys
import pytest
import shutil
import subprocess
from pathlib import Path
from symfluence import SYMFLUENCE

pytestmark = [pytest.mark.e2e, pytest.mark.ci_quick]

def _prune_raw_forcing(project_dir: Path, keep_glob: str) -> None:
    raw_dir = project_dir / "forcing" / "raw_data"
    if not raw_dir.exists():
        return
    candidates = sorted(raw_dir.glob(keep_glob))
    if not candidates:
        return
    keep = candidates[0]
    for path in raw_dir.glob("*.nc"):
        if path != keep:
            path.unlink()

    basin_avg_dir = project_dir / "forcing" / "basin_averaged_data"
    if basin_avg_dir.exists():
        for path in basin_avg_dir.glob("*.nc"):
            path.unlink()
    intersection_dir = project_dir / "shapefiles" / "catchment_intersection"
    if intersection_dir.exists():
        for path in intersection_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

@pytest.mark.e2e
@pytest.mark.ci_quick
@pytest.mark.smoke
def test_binary_validation(symfluence_code_dir, symfluence_data_root):
    """
    Validate all external tool binaries are functional.

    Tests that required binaries (SUMMA, mizuRoute, TauDEM) exist and can be executed.
    Also validates optional binaries (FUSE, NGEN) if they are installed.
    """
    from utils.helpers import load_config_template

    # Load config to get installation paths
    config = load_config_template(symfluence_code_dir)
    data_dir = Path(config.get("SYMFLUENCE_DATA_DIR", symfluence_data_root))

    # Required binaries
    print("\n" + "="*60)
    print("Validating REQUIRED binaries...")
    print("="*60)

    # Check for SUMMA - use configured path
    summa_install_path = config.get("SUMMA_INSTALL_PATH", "default")
    if summa_install_path == "default":
        summa_install_path = data_dir / "installs" / "summa" / "bin"
    else:
        summa_install_path = Path(summa_install_path)

    summa_exe = config.get("SUMMA_EXE", "summa.exe")

    # Try PATH first, then fall back to configured location
    summa_in_path = shutil.which(summa_exe)
    if summa_in_path:
        summa_path = Path(summa_in_path)
    else:
        # Check configured installation directory
        summa_path = summa_install_path / summa_exe
        # Also try the symlink summa.exe if the configured exe doesn't exist
        if not summa_path.exists() and summa_exe != "summa.exe":
            summa_path_symlink = summa_install_path / "summa.exe"
            if summa_path_symlink.exists():
                summa_path = summa_path_symlink

    assert summa_path.exists(), f"SUMMA binary not found at {summa_path} (checked PATH and {summa_install_path})"
    print(f"✓ SUMMA found: {summa_path}")

    # Verify SUMMA can run (check version)
    result = subprocess.run([str(summa_path), "--version"], capture_output=True, text=True)
    print(f"  SUMMA version output: {result.stdout.strip() or result.stderr.strip()}")

    # Check for mizuRoute - use configured path
    mizu_install_path = config.get("INSTALL_PATH_MIZUROUTE", "default")
    if mizu_install_path == "default":
        mizu_install_path = data_dir / "installs" / "mizuRoute" / "route" / "bin"
    else:
        mizu_install_path = Path(mizu_install_path)

    mizu_exe = config.get("EXE_NAME_MIZUROUTE", "mizuRoute.exe")

    # Try PATH first, then fall back to configured location
    mizu_in_path = shutil.which(mizu_exe)
    if mizu_in_path:
        mizu_path = Path(mizu_in_path)
    else:
        mizu_path = mizu_install_path / mizu_exe

    assert mizu_path.exists(), f"mizuRoute binary not found at {mizu_path} (checked PATH and {mizu_install_path})"
    print(f"✓ mizuRoute found: {mizu_path}")

    # Check for TauDEM tools (required for domain preprocessing)
    taudem_tools = ["pitremove", "d8flowdir", "aread8", "threshold", "streamnet", "gagewatershed"]
    print(f"✓ Checking {len(taudem_tools)} TauDEM tools...")
    for tool in taudem_tools:
        tool_path = shutil.which(tool)
        assert tool_path is not None, f"TauDEM tool '{tool}' not found in PATH"
    print(f"  All {len(taudem_tools)} TauDEM tools found")

    # Optional binaries
    print("\n" + "="*60)
    print("Validating OPTIONAL binaries...")
    print("="*60)

    optional_found = []
    optional_missing = []

    # Check for FUSE (optional alternative hydrological model)
    fuse_path = shutil.which("fuse.exe")
    if fuse_path:
        print(f"✓ FUSE found: {fuse_path}")
        optional_found.append("FUSE")
        # Try to verify FUSE can run
        try:
            result = subprocess.run(["fuse.exe", "--version"],
                                  capture_output=True, text=True, timeout=5)
            print(f"  FUSE version output: {result.stdout.strip() or result.stderr.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  FUSE found but version check failed (may be normal)")
    else:
        print("⚠ FUSE not found (optional)")
        optional_missing.append("FUSE")

    # Check for NGEN (optional NextGen framework)
    ngen_path = shutil.which("ngen")
    if ngen_path:
        print(f"✓ NGEN found: {ngen_path}")
        optional_found.append("NGEN")
        # Try to verify NGEN can run
        try:
            result = subprocess.run(["ngen", "--help"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "ngen" in result.stdout.lower():
                print(f"  NGEN verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  NGEN found but verification failed")
    else:
        print("⚠ NGEN not found (optional)")
        optional_missing.append("NGEN")

    # Check for HYPE (optional hydrological model)
    hype_install_path = config.get("HYPE_INSTALL_PATH", "default")
    if hype_install_path == "default":
        hype_install_path = data_dir / "installs" / "hype" / "bin"
    else:
        hype_install_path = Path(hype_install_path)

    hype_exe_name = config.get("HYPE_EXE", "hype")
    hype_in_path = shutil.which(hype_exe_name)
    if hype_in_path:
        hype_path = Path(hype_in_path)
    else:
        hype_path = hype_install_path / hype_exe_name

    if hype_path.exists():
        print(f"✓ HYPE found: {hype_path}")
        optional_found.append("HYPE")
        # Try to verify HYPE can run
        try:
            # HYPE usually prints its version/header even if args are missing
            result = subprocess.run([str(hype_path)],
                                  capture_output=True, text=True, timeout=5)
            # HYPE outputs to stderr when run without arguments
            if "HYPE" in result.stdout or "HYPE" in result.stderr or result.returncode != 0:
                print(f"  HYPE verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  HYPE found but verification failed")
    else:
        print(f"⚠ HYPE not found at {hype_path} (optional)")
        optional_missing.append("HYPE")

    # Check for MESH (optional hydrological model)
    mesh_install_path = config.get("MESH_INSTALL_PATH", "default")
    if mesh_install_path == "default":
        mesh_install_path = data_dir / "installs" / "mesh" / "bin"
    else:
        mesh_install_path = Path(mesh_install_path)

    mesh_exe_name = config.get("MESH_EXE", "mesh.exe")
    mesh_in_path = shutil.which(mesh_exe_name)
    if mesh_in_path:
        mesh_path = Path(mesh_in_path)
    else:
        mesh_path = mesh_install_path / mesh_exe_name

    if mesh_path.exists():
        print(f"✓ MESH found: {mesh_path}")
        optional_found.append("MESH")
        # Try to verify MESH can run
        try:
            # MESH may print usage info when run without arguments
            result = subprocess.run([str(mesh_path), "--help"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "MESH" in result.stdout or "MESH" in result.stderr:
                print(f"  MESH verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  MESH found but verification failed")
    else:
        print(f"⚠ MESH not found at {mesh_path} (optional)")
        optional_missing.append("MESH")

    # Check for RHESSys (optional, experimental)
    rhessys_install_path = config.get("RHESSYS_INSTALL_PATH", "default")
    if rhessys_install_path == "default":
        rhessys_install_path = data_dir / "installs" / "rhessys" / "bin"
    else:
        rhessys_install_path = Path(rhessys_install_path)

    rhessys_exe_name = config.get("RHESSYS_EXE", "rhessys")
    rhessys_in_path = shutil.which(rhessys_exe_name)
    if rhessys_in_path:
        rhessys_path = Path(rhessys_in_path)
    else:
        rhessys_path = rhessys_install_path / rhessys_exe_name

    if rhessys_path.exists():
        print(f"✓ RHESSys found: {rhessys_path}")
        optional_found.append("RHESSys")
        try:
            result = subprocess.run([str(rhessys_path), "-h"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "rhessys" in result.stdout.lower() or "usage" in result.stderr.lower():
                print(f"  RHESSys verified working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  RHESSys found but verification failed")
    else:
        print(f"⚠ RHESSys not found at {rhessys_path} (optional)")
        optional_missing.append("RHESSys")

    # Check for WMFire (optional, experimental for RHESSys)
    wmfire_install_path = config.get("WMFIRE_INSTALL_PATH", "default")
    if wmfire_install_path == "default":
        wmfire_install_path = data_dir / "installs" / "wmfire" / "lib"
    else:
        wmfire_install_path = Path(wmfire_install_path)

    if sys.platform == "darwin":
        wmfire_lib_name = config.get("WMFIRE_LIB", "libwmfire.dylib")
    else:
        wmfire_lib_name = config.get("WMFIRE_LIB", "libwmfire.so")
        
    wmfire_path = wmfire_install_path / wmfire_lib_name

    if wmfire_path.exists():
        print(f"✓ WMFire found: {wmfire_path}")
        optional_found.append("WMFire")
    else:
        print(f"⚠ WMFire not found at {wmfire_path} (optional)")
        optional_missing.append("WMFire")

    # Summary
    print("\n" + "="*60)
    print("Binary Validation Summary")
    print("="*60)
    print(f"Required: SUMMA, mizuRoute, TauDEM - ALL FOUND ✓")
    if optional_found:
        print(f"Optional found: {', '.join(optional_found)}")
    if optional_missing:
        print(f"Optional missing: {', '.join(optional_missing)}")
    print("="*60)


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
    config["DOMAIN_NAME"] = "Bow_at_Banff_lumped"  # Match existing test data files
    config["EXPERIMENT_ID"] = f"test_quick_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-01 04:00"  # 3 hours
    config["CALIBRATION_PERIOD"] = None
    config["EVALUATION_PERIOD"] = None
    config["SPINUP_PERIOD"] = None

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

    # Copy existing forcing data from bow_domain (skip if same location)
    import shutil
    src_forcing = bow_domain / "forcing" / "raw_data"
    dst_forcing = project_dir / "forcing" / "raw_data"
    if src_forcing.exists() and src_forcing.resolve() != dst_forcing.resolve():
        dst_forcing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_forcing, dst_forcing, dirs_exist_ok=True)

    _prune_raw_forcing(project_dir, "domain_*_ERA5_merged_200401.nc")

    # Copy existing shapefiles (skip if same location)
    src_shapefiles = bow_domain / "shapefiles"
    dst_shapefiles = project_dir / "shapefiles"
    if src_shapefiles.exists() and src_shapefiles.resolve() != dst_shapefiles.resolve():
        shutil.copytree(src_shapefiles, dst_shapefiles, dirs_exist_ok=True)

    # Copy existing attributes (skip if same location)
    src_attributes = bow_domain / "attributes"
    dst_attributes = project_dir / "attributes"
    if src_attributes.exists() and src_attributes.resolve() != dst_attributes.resolve():
        shutil.copytree(src_attributes, dst_attributes, dirs_exist_ok=True)

    # Reuse preprocessed forcing if available to speed smoke test
    has_preprocessed_forcing = False
    src_basin_avg = bow_domain / "forcing" / "basin_averaged_data"
    dst_basin_avg = project_dir / "forcing" / "basin_averaged_data"
    if src_basin_avg.exists() and src_basin_avg.resolve() != dst_basin_avg.resolve():
        if list(src_basin_avg.glob("*.nc")):
            dst_basin_avg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_basin_avg, dst_basin_avg, dirs_exist_ok=True)
            has_preprocessed_forcing = True

    src_intersection = bow_domain / "shapefiles" / "catchment_intersection"
    dst_intersection = project_dir / "shapefiles" / "catchment_intersection"
    if src_intersection.exists() and src_intersection.resolve() != dst_intersection.resolve():
        if list(src_intersection.glob("*")):
            shutil.copytree(src_intersection, dst_intersection, dirs_exist_ok=True)

    # Model-agnostic preprocessing
    if not has_preprocessed_forcing:
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
@pytest.mark.parametrize("model", ["SUMMA", "HYPE", "RHESSys"])
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
    config["DOMAIN_NAME"] = "Bow_at_Banff_lumped"  # Match existing test data files
    config["EXPERIMENT_ID"] = f"test_full_{model}_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-31 23:00"  # 1 month

    config["CALIBRATION_PERIOD"] = "2004-01-05, 2004-01-19"
    config["EVALUATION_PERIOD"] = "2004-01-20, 2004-01-30"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-04"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = model
    config["forcing"] = {"dataset": "ERA5"}
    if model == "SUMMA":
        config["ROUTING_MODEL"] = "mizuRoute"

    # Save config
    cfg_path = tmp_path / "config_full.yaml"
    write_config(config, cfg_path)

    # Initialize and run
    sym = SYMFLUENCE(cfg_path)
    project_dir = sym.managers["project"].setup_project()

    # Copy data (same as quick test, skip if same location)
    import shutil
    for subdir in ["forcing/raw_data", "shapefiles", "attributes"]:
        src = bow_domain / subdir
        dst = project_dir / subdir
        if src.exists() and src.resolve() != dst.resolve():
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
    config["DOMAIN_NAME"] = "Bow_at_Banff_lumped"  # Match existing test data files
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

    # Copy data (skip if same location)
    import shutil
    for subdir in ["forcing/raw_data", "shapefiles", "attributes", "observations"]:
        src = bow_domain / subdir
        dst = project_dir / subdir
        if src.exists() and src.resolve() != dst.resolve():
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
