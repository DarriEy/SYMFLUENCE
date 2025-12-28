"""
SYMFLUENCE Lumped Basin Integration Tests

Tests the lumped basin workflow from notebook 02a for all supported models.
Downloads test data from GitHub release and runs a short 1-month simulation.
"""

import pytest
import requests
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

    Downloads fresh data and renames domain to domain_Bow_at_Banff_lumped_test
    to keep it separate from any existing work.
    """
    import shutil

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

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting example data...")
        # Extract to a temp location
        extract_dir = data_root / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
    template_path = SYMFLUENCE_CODE_DIR / '0_config_files' / 'config_template.yaml'
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths
    config['SYMFLUENCE_DATA_DIR'] = str(test_data_dir)
    config['SYMFLUENCE_CODE_DIR'] = str(SYMFLUENCE_CODE_DIR)

    # Domain settings from notebook 02a
    config['DOMAIN_NAME'] = 'Bow_at_Banff_lumped'
    config['EXPERIMENT_ID'] = f'test_{tmp_path.name}'  # Unique test experiment ID
    config['POUR_POINT_COORDS'] = '51.1722/-115.5717'

    # Lumped basin settings
    config['DOMAIN_DEFINITION_METHOD'] = 'lumped'
    config['DOMAIN_DISCRETIZATION'] = 'GRUs'

    # Short 1-month period for testing
    config['EXPERIMENT_TIME_START'] = '2004-01-01 01:00'
    config['EXPERIMENT_TIME_END'] = '2004-01-31 23:00'
    config['CALIBRATION_PERIOD'] = '2004-01-05, 2004-01-19'
    config['EVALUATION_PERIOD'] = '2004-01-20, 2004-01-30'
    config['SPINUP_PERIOD'] = '2004-01-01, 2004-01-04'

    # Streamflow
    config['STATION_ID'] = '05BB001'
    config['DOWNLOAD_WSC_DATA'] = False

    # Minimal calibration for testing
    config['NUMBER_OF_ITERATIONS'] = 3
    config['RANDOM_SEED'] = 42  # Fixed seed for reproducibility

    # Save config
    cfg_path = tmp_path / 'test_config.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return cfg_path, config


MODELS = ['SUMMA', 'FUSE', 'GR', 'NGEN']


@pytest.mark.parametrize("model", MODELS)
def test_lumped_basin_workflow(config_path, model):
    """
    Test lumped basin workflow for each model.

    Follows notebook 02a workflow:
    1. Setup project
    2. Define domain (watershed delineation)
    3. Discretize domain
    4. Model-agnostic preprocessing
    5. Model-specific preprocessing
    6. Run model
    7. Calibrate model
    """
    cfg_path, config = config_path

    # Update model in config
    config['HYDROLOGICAL_MODEL'] = model
    if model == 'SUMMA':
        config['ROUTING_MODEL'] = 'mizuRoute'
        config['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat'
        config['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'
    elif model == 'FUSE':
        config['FUSE_SPATIAL_MODE'] = 'lumped'
        config['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = 'MAXWATR_1,MAXWATR_2,BASERTE'
    elif model == 'GR':
        config['GR_spatial'] = 'lumped'
    elif model == 'NGEN':
        config['NGEN_MODULES_TO_CALIBRATE'] = 'CFE'
        config['NGEN_CFE_PARAMS_TO_CALIBRATE'] = 'smcmax,satdk,bb'
        # Point to ngen install in data directory
        config['NGEN_INSTALL_PATH'] = str(Path(config['SYMFLUENCE_DATA_DIR']) / 'installs' / 'ngen' / 'cmake_build')

    # Save updated config
    with open(cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers['project'].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    pour_point_path = symfluence.managers['project'].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 2: Define domain (watershed delineation)
    watershed_path = symfluence.managers['domain'].define_domain()
    # watershed_path can be None for lumped domains that use existing data

    # Step 3: Discretize domain
    hru_path = symfluence.managers['domain'].discretize_domain()

    # Step 4: Model-agnostic preprocessing
    symfluence.managers['data'].run_model_agnostic_preprocessing()

    # Step 5: Model-specific preprocessing
    symfluence.managers['model'].preprocess_models()

    # Step 6: Run model
    symfluence.managers['model'].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config['EXPERIMENT_ID'] / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 7: Calibrate model
    results_file = symfluence.managers['optimization'].calibrate_model()
    if model == 'GR':
        assert results_file is None, "GR calibration is not implemented and should return None"
    else:
        assert results_file is not None, "Calibration should produce results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
