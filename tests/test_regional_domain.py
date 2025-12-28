"""
SYMFLUENCE Regional Domain Integration Tests

Tests the regional workflow from notebook 03a (Iceland example) for SUMMA.
Builds and runs the model without calibration.
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
    example_domain = "domain_Iceland"
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

    # Regional Iceland settings from notebook 03a
    config["DOMAIN_NAME"] = "Iceland"
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["DELINEATE_COASTAL_WATERSHEDS"] = True
    config["DELINEATE_BY_POURPOINT"] = False
    config["CLEANUP_INTERMEDIATE_FILES"] = False

    config["BOUNDING_BOX_COORDS"] = "66.5/-25.0/63.0/-13.0"
    config["POUR_POINT_COORDS"] = "64.01/-16.01"
    config["STREAM_THRESHOLD"] = 2000

    # Experiment settings (shortened for testing)
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-31 23:00"

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return cfg_path, config


def test_regional_domain_workflow(config_path):
    """
    Test regional domain workflow for SUMMA (no calibration).

    Follows notebook 03a workflow:
    1. Setup project
    2. Create pour point
    3. Define regional domain
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

    # Step 3: Define regional domain
    watershed_path = symfluence.managers["domain"].define_domain()

    # Step 4: Discretize domain
    hru_path = symfluence.managers["domain"].discretize_domain()

    # Step 5: Model-agnostic preprocessing
    symfluence.managers["data"].run_model_agnostic_preprocessing()

    # Step 6: Model-specific preprocessing
    symfluence.managers["model"].preprocess_models()

    # Step 7: Run model
    symfluence.managers["model"].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config["EXPERIMENT_ID"]
    summa_dir = sim_dir / "SUMMA"
    routing_dir = sim_dir / "mizuRoute"
    assert summa_dir.exists(), "SUMMA simulation output directory should exist"
    assert routing_dir.exists(), "mizuRoute simulation output directory should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
