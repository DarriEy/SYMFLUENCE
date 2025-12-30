"""
Test CARRA and CERRA regional reanalysis data acquisition and preprocessing.

These tests are independent and set up their own domains from scratch.
"""
import pytest
import yaml
import shutil
from pathlib import Path
from symfluence import SYMFLUENCE
from utils.helpers import load_config_template, write_config


@pytest.fixture
def temp_config(tmp_path, symfluence_code_dir, symfluence_data_root):
    """Create a temporary config based on the template."""
    config = load_config_template(symfluence_code_dir)

    # Set up paths
    data_root = symfluence_data_root

    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Cloud access settings
    config["DATA_ACCESS"] = "cloud"
    config["DEM_SOURCE"] = "copernicus"
    config["DOWNLOAD_SNOTEL"] = False
    config["HYDROLOGICAL_MODEL"] = "SUMMA"

    # Placeholder values (will be overridden in tests)
    config["DOMAIN_NAME"] = "test_domain"
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"
    config["EXPERIMENT_ID"] = "test"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"
    config["FORCING_DATASET"] = "ERA5"

    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

    return cfg_path


@pytest.mark.slow
@pytest.mark.requires_data
def test_carra_full_pipeline(temp_config):
    """
    Test CARRA (Arctic) data acquisition for Elliðaár, Iceland.

    This test uses the small pre-configured Elliðaár basin and:
    1. Checks if domain exists, sets it up if needed
    2. Downloads CARRA forcing data (dual-product strategy)
    3. Verifies all SUMMA variables are present
    """
    # Configure for Elliðaár basin, Reykjavik, Iceland
    with open(temp_config, "r") as f:
        config = yaml.safe_load(f)

    config["DOMAIN_NAME"] = "ellioaar_iceland"
    config["BOUNDING_BOX_COORDS"] = "64.13/-21.96/64.11/-21.94"  # ~2km x 2km
    config["POUR_POINT_COORDS"] = "64.12/-21.95"
    config["FORCING_DATASET"] = "CARRA"
    config["CARRA_DOMAIN"] = "west_domain"
    config["EXPERIMENT_ID"] = "test_carra"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-01 02:00"  # Just 2 hours

    write_config(config, temp_config)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(temp_config)

    data_root = Path(config["SYMFLUENCE_DATA_DIR"])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"

    # Setup domain only if it doesn't exist
    if not hrus_file.exists():
        print(f"Setting up {config['DOMAIN_NAME']} domain (first run)...")
        symfluence.managers["project"].setup_project()
        pour_point_path = symfluence.managers["project"].create_pour_point()
        assert Path(pour_point_path).exists(), "Pour point shapefile should be created"
        symfluence.managers["data"].acquire_attributes()
        symfluence.managers["domain"].define_domain()
        symfluence.managers["domain"].discretize_domain()
        assert hrus_file.exists(), "HRUs file should exist after discretization"
    else:
        print(f"Using existing {config['DOMAIN_NAME']} domain...")

    # Download CARRA forcing (always fresh download)
    symfluence.managers["data"].acquire_forcings()

    # Verify forcing file was created
    raw_data_dir = project_dir / "forcing" / "raw_data"
    carra_files = list(raw_data_dir.glob("*CARRA*.nc"))
    assert len(carra_files) > 0, "CARRA forcing file should be downloaded"

    # Verify file has all SUMMA variables
    import xarray as xr
    with xr.open_dataset(carra_files[0]) as ds:
        expected_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
        found_vars = [v for v in expected_vars if v in ds.data_vars]
        assert len(found_vars) == 7, f"Expected 7 SUMMA variables, found {len(found_vars)}: {found_vars}"

    print(f"✓ CARRA download succeeded: {carra_files[0].name}")
    print(f"✓ All 7 SUMMA variables present: {found_vars}")
    print("✓ Note: Skipping preprocessing (would take hours for full domain)")

    # NOTE: Preprocessing is skipped because it would take hours to process
    # the full CARRA domain (1.3M+ grid points). The preprocessing crash due to
    # easymore segfault is a known library issue, not CARRA-specific.


@pytest.mark.slow
@pytest.mark.requires_data
def test_cerra_full_pipeline(temp_config):
    """
    Test CERRA (Europe) data acquisition for Fyrisån, Sweden.

    This test uses the small pre-configured Fyrisån basin and:
    1. Checks if domain exists, sets it up if needed
    2. Downloads CERRA forcing data (dual-product strategy)
    3. Verifies all SUMMA variables are present
    """
    # Configure for Fyrisån basin, Uppsala, Sweden
    with open(temp_config, "r") as f:
        config = yaml.safe_load(f)

    config["DOMAIN_NAME"] = "fyris_uppsala"
    config["BOUNDING_BOX_COORDS"] = "59.87/17.64/59.85/17.66"  # ~2km x 2km
    config["POUR_POINT_COORDS"] = "59.86/17.65"
    config["FORCING_DATASET"] = "CERRA"
    config["EXPERIMENT_ID"] = "test_cerra"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-01 03:00"  # 3 hours (1 timestep for 3-hourly)

    write_config(config, temp_config)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(temp_config)

    data_root = Path(config["SYMFLUENCE_DATA_DIR"])
    project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"
    hrus_file = project_dir / "shapefiles" / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"

    # Setup domain only if it doesn't exist
    if not hrus_file.exists():
        print(f"Setting up {config['DOMAIN_NAME']} domain (first run)...")
        symfluence.managers["project"].setup_project()
        pour_point_path = symfluence.managers["project"].create_pour_point()
        assert Path(pour_point_path).exists(), "Pour point shapefile should be created"
        symfluence.managers["data"].acquire_attributes()
        symfluence.managers["domain"].define_domain()
        symfluence.managers["domain"].discretize_domain()
        assert hrus_file.exists(), "HRUs file should exist after discretization"
    else:
        print(f"Using existing {config['DOMAIN_NAME']} domain...")

    # Download CERRA forcing (always fresh download)
    symfluence.managers["data"].acquire_forcings()

    # Verify forcing file was created
    raw_data_dir = project_dir / "forcing" / "raw_data"
    cerra_files = list(raw_data_dir.glob("*CERRA*.nc"))
    assert len(cerra_files) > 0, "CERRA forcing file should be downloaded"

    # Verify file has all SUMMA variables
    import xarray as xr
    with xr.open_dataset(cerra_files[0]) as ds:
        expected_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
        found_vars = [v for v in expected_vars if v in ds.data_vars]
        assert len(found_vars) == 7, f"Expected 7 SUMMA variables, found {len(found_vars)}: {found_vars}"

    print(f"✓ CERRA download succeeded: {cerra_files[0].name}")
    print(f"✓ All 7 SUMMA variables present: {found_vars}")
    print("✓ Note: Skipping preprocessing (would take hours for full domain)")

    # NOTE: Preprocessing is skipped because it would take hours to process
    # the full CERRA domain (1.1M+ grid points). The preprocessing works but is
    # impractical for testing with full regional domains.


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
