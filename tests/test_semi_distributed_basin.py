"""
SYMFLUENCE Semi-Distributed Basin Integration Tests

Tests the semi-distributed basin workflow from notebook 02b for supported models.
Downloads example data and reuses the lumped domain assets for a short simulation.
"""

import pytest
import requests
import shutil
import zipfile
import yaml
import sys
from pathlib import Path

# Setup path exactly like the notebook does
SYMFLUENCE_CODE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR))

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-v0.2/example_data_v0.2.zip"


@pytest.fixture(scope="module")
def test_data_dir():
    """
    Download and extract example data to ../SYMFLUENCE_data/ for testing.
    """
    # Use ../SYMFLUENCE_data/ parallel to the code directory
    data_root = SYMFLUENCE_CODE_DIR.parent / "SYMFLUENCE_data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Check if example domain already exists
    example_domain = "domain_Bow_at_Banff_lumped"
    example_domain_path = data_root / example_domain

    # Download if it doesn't exist
    if not example_domain_path.exists():
        print(f"\nDownloading example data to {data_root}...")
        zip_path = data_root / "example_data_v0.2.zip"

        # Download
        response = requests.get(EXAMPLE_DATA_URL, stream=True, timeout=600)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting example data...")
        # Extract to a temp location
        extract_dir = data_root / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Move the domain to data root
        example_data_dir = extract_dir / "example_data_v0.2"
        src_domain = example_data_dir / example_domain

        if src_domain.exists():
            src_domain.rename(example_domain_path)
            print(f"Created domain: {example_domain}")
        else:
            raise FileNotFoundError(f"{example_domain} not found in downloaded data")

        # Cleanup
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        print(f"Test data ready at {example_domain_path}")
    else:
        print(f"Using existing test data at {example_domain_path}")

    yield data_root


def _copy_with_name_adaptation(src: Path, dst: Path, old_name: str, new_name: str) -> bool:
    """Copy directory or file and adapt filenames containing the old domain name."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for file in dst.rglob("*"):
        if file.is_file() and old_name in file.name:
            new_file = file.parent / file.name.replace(old_name, new_name)
            file.rename(new_file)
    return True


@pytest.fixture(scope="function")
def config_path(test_data_dir, tmp_path):
    """Create test configuration based on config_template.yaml."""
    # Load template
    template_path = SYMFLUENCE_CODE_DIR / "0_config_files" / "config_template.yaml"
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)

    # Update paths
    config["SYMFLUENCE_DATA_DIR"] = str(test_data_dir)
    config["SYMFLUENCE_CODE_DIR"] = str(SYMFLUENCE_CODE_DIR)

    # Domain settings from notebook 02b
    config["DOMAIN_NAME"] = "Bow_at_Banff_semi_distributed"
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Semi-distributed basin settings
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["STREAM_THRESHOLD"] = 5000
    config["DOMAIN_DISCRETIZATION"] = "GRUs"

    # Short 1-month period for testing
    config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-31 23:00"
    config["CALIBRATION_PERIOD"] = "2004-01-05, 2004-01-19"
    config["EVALUATION_PERIOD"] = "2004-01-20, 2004-01-30"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-04"

    # Streamflow
    config["STATION_ID"] = "05BB001"
    config["DOWNLOAD_WSC_DATA"] = False

    # Minimal calibration for testing
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return cfg_path, config


MODELS = ["SUMMA", "FUSE", "NGEN"]


@pytest.mark.parametrize("model", MODELS)
def test_semi_distributed_basin_workflow(config_path, test_data_dir, model):
    """
    Test semi-distributed basin workflow for each model.

    Follows notebook 02b workflow:
    1. Setup project
    2. Reuse data from lumped basin example
    3. Define domain (watershed delineation)
    4. Discretize domain
    5. Model-agnostic preprocessing
    6. Model-specific preprocessing
    7. Run model
    8. Calibrate model
    """
    cfg_path, config = config_path

    # Update model in config
    config["HYDROLOGICAL_MODEL"] = model
    if model == "SUMMA":
        config["ROUTING_MODEL"] = "mizuRoute"
        config["MIZU_FROM_MODEL"] = "SUMMA"
        config["SETTINGS_MIZU_ROUTING_VAR"] = "averageRoutedRunoff"
        config["SETTINGS_MIZU_ROUTING_UNITS"] = "m/s"
        config["SETTINGS_MIZU_ROUTING_DT"] = "3600"
        config["PARAMS_TO_CALIBRATE"] = "k_soil,theta_sat"
        config["BASIN_PARAMS_TO_CALIBRATE"] = "routingGammaScale"
    elif model == "FUSE":
        config["FUSE_SPATIAL_MODE"] = "semi_distributed"
        config["SETTINGS_FUSE_PARAMS_TO_CALIBRATE"] = "MAXWATR_1,MAXWATR_2,BASERTE"
    elif model == "NGEN":
        config["NGEN_MODULES_TO_CALIBRATE"] = "CFE"
        config["NGEN_CFE_PARAMS_TO_CALIBRATE"] = "smcmax,satdk,bb"
        config["NGEN_INSTALL_PATH"] = str(
            Path(config["SYMFLUENCE_DATA_DIR"]) / "installs" / "ngen" / "cmake_build"
        )

    # Save updated config
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Reuse data from the lumped example domain
    lumped_domain = "Bow_at_Banff_lumped"
    lumped_data_dir = test_data_dir / f"domain_{lumped_domain}"
    reusable_data = {
        "Elevation": lumped_data_dir / "attributes" / "elevation",
        "Land Cover": lumped_data_dir / "attributes" / "landclass",
        "Soils": lumped_data_dir / "attributes" / "soilclass",
        "Forcing": lumped_data_dir / "forcing" / "raw_data",
        "Streamflow": lumped_data_dir / "observations" / "streamflow",
    }
    for _, src_path in reusable_data.items():
        if src_path.exists():
            rel_path = src_path.relative_to(lumped_data_dir)
            dst_path = project_dir / rel_path
            _copy_with_name_adaptation(
                src_path, dst_path, lumped_domain, config["DOMAIN_NAME"]
            )

    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define domain (watershed delineation)
    watershed_path = symfluence.managers["domain"].define_domain()
    # watershed_path can be None for workflows that use existing data

    # Step 4: Discretize domain
    hru_path = symfluence.managers["domain"].discretize_domain()

    # Step 5: Model-agnostic preprocessing
    symfluence.managers["data"].run_model_agnostic_preprocessing()

    # Step 6: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 7: Run model
    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 8: Calibrate model
    results_file = symfluence.managers["optimization"].calibrate_model()
    assert results_file is not None, "Calibration should produce results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
