"""
SYMFLUENCE Cloud Data Acquisition Integration Tests

Uses the Paradise point-scale setup to validate cloud attribute acquisition
and multiple cloud forcing datasets with short time windows.
"""

import pytest
import shutil
import yaml
import sys
from pathlib import Path

# Setup path exactly like the notebook does
SYMFLUENCE_CODE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR))

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE


@pytest.fixture(scope="module")
def base_config(tmp_path_factory):
    """Create a base Paradise config for cloud acquisition tests."""
    tmp_path = tmp_path_factory.mktemp("cloud_acq")
    cfg_path = tmp_path / "test_config.yaml"

    # Load template
    template_path = SYMFLUENCE_CODE_DIR / "0_config_files" / "config_template.yaml"
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = SYMFLUENCE_CODE_DIR.parent / "SYMFLUENCE_data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Base paths
    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(SYMFLUENCE_CODE_DIR)

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
    config["HYDROLOGICAL_MODEL"] = "SUMMA"

    # Placeholder experiment window; per-dataset windows set in tests
    config["EXPERIMENT_ID"] = "cloud_acq"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"

    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

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
        "end": "2010-01-02 00:00",
        "expect_glob": "domain_paradise_cloud_ERA5_merged_*.nc",
    },
    {
        "dataset": "AORC",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 03:00",
        "expect_glob": "paradise_cloud_AORC_*.nc",
    },
    {
        "dataset": "NEX-GDDP-CMIP6",
        "start": "2010-01-01 00:00",
        "end": "2010-01-03 00:00",
        "expect_glob": "NEXGDDP_all_*.nc",
        "extras": {
            "NEX_MODELS": ["ACCESS-CM2"],
            "NEX_SCENARIOS": ["historical"],
            "NEX_ENSEMBLES": ["r1i1p1f1"],
            "NEX_VARIABLES": ["pr", "tas", "tasmax", "tasmin"],
        },
    },
    {
        "dataset": "HRRR",
        "start": "2020-01-01 00:00",
        "end": "2020-01-01 03:00",
        "expect_glob": "paradise_cloud_HRRR_hourly_*.nc",
    },
    {
        "dataset": "CONUS404",
        "start": "2010-01-01 00:00",
        "end": "2010-01-01 03:00",
        "expect_glob": "paradise_cloud_CONUS404_*.nc",
    },
]


@pytest.mark.parametrize("case", FORCING_CASES)
def test_cloud_forcing_acquisition(prepared_project, case):
    """
    Download a short forcing window for each cloud-supported dataset, then
    run the full preprocessing + model pipeline.
    """
    cfg_path, project_dir = prepared_project

    # Load base config and update for this dataset
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    config["FORCING_DATASET"] = case["dataset"]
    config["EXPERIMENT_TIME_START"] = case["start"]
    config["EXPERIMENT_TIME_END"] = case["end"]
    config["EXPERIMENT_ID"] = f"cloud_{case['dataset'].lower().replace('-', '_')}"

    for key, value in case.get("extras", {}).items():
        config[key] = value

    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

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
