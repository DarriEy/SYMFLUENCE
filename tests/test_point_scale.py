"""
SYMFLUENCE Point-Scale Integration Tests

Tests the point-scale workflow from notebook 01a (Paradise SNOTEL example).
Runs a short SUMMA simulation for a point domain.
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
    """Ensure example data exists in a writable SYMFLUENCE_data directory."""
    data_root = SYMFLUENCE_CODE_DIR.parent / "SYMFLUENCE_data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Check if example domain already exists
    example_domain = "domain_paradise"
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

    return data_root


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

    # Point-scale settings from notebook 01a
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "46.781/-121.751/46.779/-121.749"
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    # Data sources
    config["DOWNLOAD_SNOTEL"] = False
    config["SNOTEL_STATION"] = "679"

    # Model and forcing
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["FORCING_DATASET"] = "ERA5"

    # Short 1-month period for testing
    config["EXPERIMENT_TIME_START"] = "2000-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2000-01-31 23:00"
    config["CALIBRATION_PERIOD"] = "2000-01-05, 2000-01-19"
    config["EVALUATION_PERIOD"] = "2000-01-20, 2000-01-30"
    config["SPINUP_PERIOD"] = "2000-01-01, 2000-01-04"

    # Domain and experiment ids
    config["DOMAIN_NAME"] = "paradise"
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return cfg_path, config


def test_point_scale_workflow(config_path):
    """
    Test point-scale workflow for SUMMA.

    Follows notebook 01a workflow:
    1. Setup project
    2. Create pour point
    3. Define domain (point)
    4. Discretize domain
    5. Model-agnostic preprocessing
    6. Model-specific preprocessing
    7. Run model
    """
    cfg_path, config = config_path

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Create pour point
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define domain (point)
    domain_path = symfluence.managers["domain"].define_domain()

    # Step 4: Discretize domain
    hru_path = symfluence.managers["domain"].discretize_domain()

    # Step 5: Model-agnostic preprocessing
    symfluence.managers["data"].run_model_agnostic_preprocessing()

    # Step 6: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 7: Run model
    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"] / "SUMMA"
    assert sim_dir.exists(), "SUMMA simulation output directory should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
