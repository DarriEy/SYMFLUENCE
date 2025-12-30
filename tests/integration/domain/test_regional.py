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
from pathlib import Path

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from tests.utils.helpers import load_config_template, write_config
from tests.utils.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-v0.5.5/example_data_v0.5.5.zip"


@pytest.fixture(scope="module")
def test_data_dir(symfluence_data_root):
    """
    Download and extract example data to ../SYMFLUENCE_data/ for testing.
    """
    # Use ../SYMFLUENCE_data/ parallel to the code directory
    data_root = symfluence_data_root

    # Check if example domain already exists
    example_domain = "domain_Iceland"
    example_domain_path = data_root / example_domain

    # Download if it doesn't exist
    if not example_domain_path.exists():
        print(f"\nDownloading example data to {data_root}...")
        zip_path = data_root / "example_data_v0.5.5.zip"

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
        example_data_dir = extract_dir / "example_data_v0.5.5"
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
def config_path(test_data_dir, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    config = load_config_template(symfluence_code_dir)

    # Update paths
    config["SYMFLUENCE_DATA_DIR"] = str(test_data_dir)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Regional Iceland settings from notebook 03a
    config["DOMAIN_NAME"] = "Iceland"
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["DELINEATE_COASTAL_WATERSHEDS"] = False
    config["DELINEATE_BY_POURPOINT"] = False
    config["CLEANUP_INTERMEDIATE_FILES"] = False

    config["BOUNDING_BOX_COORDS"] = "66.5/-25.0/63.0/-13.0"
    config["POUR_POINT_COORDS"] = "64.01/-16.01"
    config["STREAM_THRESHOLD"] = 2000

    # Experiment settings (shortened for testing)
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 01:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-01 23:00"

    # Limit forcing remapping to a single monthly file
    source_forcing_dir = (
        Path(config["SYMFLUENCE_DATA_DIR"])
        / f"domain_{config['DOMAIN_NAME']}"
        / "forcing"
        / "raw_data"
    )
    subset_dir = tmp_path / "forcing_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    forcing_candidates = sorted(source_forcing_dir.glob("*.nc"))
    if not forcing_candidates:
        raise FileNotFoundError(f"No forcing files found in {source_forcing_dir}")
    shutil.copy2(forcing_candidates[0], subset_dir / forcing_candidates[0].name)
    config["FORCING_PATH"] = str(subset_dir)

    # Model settings
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.mark.slow
@pytest.mark.requires_data
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

    baseline_dir = (
        Path(config["SYMFLUENCE_DATA_DIR"])
        / f"domain_{config['DOMAIN_NAME']}"
        / "shapefiles"
    )
    baseline_river_basins = (
        baseline_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_delineate.shp"
    )
    baseline_river_network = (
        baseline_dir
        / "river_network"
        / f"{config['DOMAIN_NAME']}_riverNetwork_delineate.shp"
    )
    baseline_hrus = (
        baseline_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert baseline_river_basins.exists(), "Baseline river basins shapefile missing"
    assert baseline_river_network.exists(), "Baseline river network shapefile missing"
    assert baseline_hrus.exists(), "Baseline HRU shapefile missing"
    expected_river_basins = load_shapefile_signature(baseline_river_basins)
    expected_river_network = load_shapefile_signature(baseline_river_network)
    expected_hrus = load_shapefile_signature(baseline_hrus)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers["project"].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    # Step 2: Create pour point
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define regional domain
    watershed_path, delineation_artifacts = symfluence.managers["domain"].define_domain()
    assert (
        delineation_artifacts.method == config["DOMAIN_DEFINITION_METHOD"]
    ), "Delineation method mismatch"

    # Step 4: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers["domain"].discretize_domain()
    assert (
        discretization_artifacts.method == config["DOMAIN_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (03a)
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_delineate.shp"
    )
    river_network_path = delineation_artifacts.river_network_path or (
        shapefile_dir
        / "river_network"
        / f"{config['DOMAIN_NAME']}_riverNetwork_delineate.shp"
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert river_basins_path.exists()
    assert river_network_path.exists()
    assert hrus_path.exists()
    assert_shapefile_signature_matches(river_basins_path, expected_river_basins)
    assert_shapefile_signature_matches(river_network_path, expected_river_network)
    assert_shapefile_signature_matches(hrus_path, expected_hrus)

    # Step 5: Model-agnostic preprocessing
    for subdir in ["SUMMA_input", "basin_averaged_data", "merged_path"]:
        shutil.rmtree(project_dir / "forcing" / subdir, ignore_errors=True)
    weights_dir = project_dir / "shapefiles" / "catchment_intersection" / "with_forcing"
    for weight_file in weights_dir.glob("*_HRU_ID_remapping.nc"):
        weight_file.unlink(missing_ok=True)
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
