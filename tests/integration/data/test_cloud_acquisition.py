"""
SYMFLUENCE Cloud Data Acquisition Integration Tests

Uses the Paradise point-scale setup to validate cloud attribute acquisition
and multiple cloud forcing datasets with short time windows.
"""

import pytest
import shutil
import yaml
from pathlib import Path
import os

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from utils.helpers import load_config_template, write_config

# Check if CDS API credentials are available and valid
def _check_cds_credentials():
    """Check if CDS API credentials exist and are valid."""
    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        return False
    try:
        import cdsapi
        # Try to initialize client to validate credentials
        cdsapi.Client()
        return True
    except (AssertionError, Exception):
        # Invalid credentials or other initialization error
        return False

HAS_CDS_CREDENTIALS = _check_cds_credentials()



pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]

@pytest.fixture(scope="module")
def base_config(tmp_path_factory, symfluence_code_dir, symfluence_data_root):
    """Create a base Paradise config for cloud acquisition tests."""
    tmp_path = tmp_path_factory.mktemp("cloud_acq")
    cfg_path = tmp_path / "test_config.yaml"

    # Load template
    config = load_config_template(symfluence_code_dir)
    data_root = symfluence_data_root

    # Base paths
    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Paradise point-scale setup (widen bbox to ensure grid coverage)
    config["DOMAIN_NAME"] = "paradise_cloud"
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "46.9/-121.9/46.6/-121.6"
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    # Cloud access for attributes/forcings
    config["DATA_ACCESS"] = "cloud"
    config["DEM_SOURCE"] = "copernicus"

    # Avoid live SNOTEL downloads in this test
    config["DOWNLOAD_SNOTEL"] = False
    config["SNOTEL_STATION"] = "679"

    # Model setup (point-scale SUMMA)
    config["HYDROLOGICAL_MODEL"] = "SUMMA"  # Ensure it's a string, not a list

    # Placeholder experiment window; per-dataset windows set in tests
    config["EXPERIMENT_ID"] = "cloud_acq"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"

    write_config(config, cfg_path)

    return cfg_path


@pytest.fixture(scope="module")
def prepared_project(base_config):
    """Acquire cloud attributes once and set up the point-scale domain."""
    symfluence = SYMFLUENCE(base_config)

    project_dir = symfluence.managers["project"].setup_project()
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Acquire cloud attributes: Copernicus DEM, MODIS land cover, soil classes
    symfluence.managers["data"].acquire_attributes()

    # Define and discretize the point domain
    symfluence.managers["domain"].define_domain()
    symfluence.managers["domain"].discretize_domain()

    return base_config, project_dir


FORCING_CASES = [
    {
        "dataset": "ERA5",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "domain_paradise_cloud_ERA5_merged_*.nc",
    },
    {
        "dataset": "AORC",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "paradise_cloud_AORC_*.nc",
    },
    {
        "dataset": "NEX-GDDP-CMIP6",
        "start": "2010-01-01 00:00",
        "end": "2010-01-02 00:00",  # 1 day (NEX is daily data)
        "expect_glob": "NEXGDDP_all_*.nc",
        "extras": {
            "NEX_MODELS": ["ACCESS-CM2"],
            "NEX_SCENARIOS": ["historical"],
            "NEX_ENSEMBLES": ["r1i1p1f1"],
            "NEX_VARIABLES": ["pr", "tas", "huss", "rlds", "rsds", "sfcWind"],
        },
    },
    {
        "dataset": "CONUS404",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "paradise_cloud_CONUS404_*.nc",
    },
    {
        "dataset": "HRRR",
        "start": "2020-01-01 00:00",
        "end": "2020-01-01 01:00",  # Just 1 hour
        "expect_glob": "paradise_cloud_HRRR_hourly_*.nc",
    },
    {
        "dataset": "CARRA",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 01:00",  # Just 1 hour
        "expect_glob": "*CARRA*.nc",
        "domain_override": {
            "DOMAIN_NAME": "ellioaar_iceland",
            "BOUNDING_BOX_COORDS": "64.13/-21.96/64.11/-21.94",  # Elliðaár, Reykjavik (very small ~2km x 2km)
            "POUR_POINT_COORDS": "64.12/-21.95",
        },
        "extras": {
            "CARRA_DOMAIN": "west_domain",
        },
    },
    {
        "dataset": "CERRA",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 03:00",  # 3 hours (1 timestep for 3-hourly data)
        "expect_glob": "*CERRA*.nc",
        "domain_override": {
            "DOMAIN_NAME": "fyris_uppsala",
            "BOUNDING_BOX_COORDS": "59.87/17.64/59.85/17.66",  # Fyrisån, Uppsala (very small ~2km x 2km)
            "POUR_POINT_COORDS": "59.86/17.65",
        },
    },
]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("case", FORCING_CASES)
def test_cloud_forcing_acquisition(prepared_project, case):
    """
    Download a short forcing window for each cloud-supported dataset, then
    run the full preprocessing + model pipeline.
    """
    # Skip CARRA/CERRA tests if CDS credentials are not available
    if case["dataset"] in ["CARRA", "CERRA"] and not HAS_CDS_CREDENTIALS:
        pytest.skip(f"Skipping {case['dataset']} test: CDS API credentials not found in ~/.cdsapirc")

    cfg_path, project_dir = prepared_project

    # Load base config and update for this dataset
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle domain override for datasets requiring different geographic locations
    # (e.g., CARRA needs Arctic, CERRA needs Europe)
    if "domain_override" in case:
        for key, value in case["domain_override"].items():
            config[key] = value
        # Update project_dir to match new domain name
        data_root = Path(config["SYMFLUENCE_DATA_DIR"])
        project_dir = data_root / f"domain_{config['DOMAIN_NAME']}"

        # Setup new domain if needed
        symfluence_temp = SYMFLUENCE(cfg_path)
        # Save updated config first
        write_config(config, cfg_path)

        # Re-initialize with new domain
        symfluence_temp = SYMFLUENCE(cfg_path)
        # Check if domain is fully set up (HRUs file exists)
        domain_name = config["DOMAIN_NAME"]
        hrus_file = project_dir / "shapefiles" / "catchment" / f"{domain_name}_HRUs_GRUs.shp"
        if not hrus_file.exists():
            symfluence_temp.managers["project"].setup_project()
            pour_point_path = symfluence_temp.managers["project"].create_pour_point()
            assert Path(pour_point_path).exists(), "Pour point shapefile should be created"
            symfluence_temp.managers["data"].acquire_attributes()
            symfluence_temp.managers["domain"].define_domain()
            symfluence_temp.managers["domain"].discretize_domain()

    config["FORCING_DATASET"] = case["dataset"]
    config["EXPERIMENT_TIME_START"] = case["start"]
    config["EXPERIMENT_TIME_END"] = case["end"]
    config["EXPERIMENT_ID"] = f"cloud_{case['dataset'].lower().replace('-', '_')}"

    # Ensure HYDROLOGICAL_MODEL stays as string (not list)
    if isinstance(config.get("HYDROLOGICAL_MODEL"), list):
        config["HYDROLOGICAL_MODEL"] = config["HYDROLOGICAL_MODEL"][0]
    elif "HYDROLOGICAL_MODEL" not in config:
        config["HYDROLOGICAL_MODEL"] = "SUMMA"

    for key, value in case.get("extras", {}).items():
        config[key] = value

    write_config(config, cfg_path)

    # Clean forcing outputs from prior dataset runs
    for subdir in ["raw_data", "basin_averaged_data", "merged_path"]:
        shutil.rmtree(project_dir / "forcing" / subdir, ignore_errors=True)

    symfluence = SYMFLUENCE(cfg_path)
    symfluence.managers["data"].acquire_forcings()

    raw_data_dir = project_dir / "forcing" / "raw_data"
    matches = list(raw_data_dir.glob(case["expect_glob"]))
    assert matches, f"No forcing output found for {case['dataset']} in {raw_data_dir}"

    # Run full preprocessing and model
    symfluence.managers["data"].run_model_agnostic_preprocessing()
    symfluence.managers["model"].preprocess_models()
    symfluence.managers["model"].run_models()

    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), f"SUMMA simulation output directory missing for {case['dataset']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
