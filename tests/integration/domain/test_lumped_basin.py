"""
SYMFLUENCE Lumped Basin Integration Tests

Tests the lumped basin workflow from notebook 02a for all supported models.
Downloads test data from GitHub release and runs a short 1-month simulation.
"""

import pytest
import requests
import zipfile
import yaml
from pathlib import Path

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from utils.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)
from utils.helpers import write_config

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-v0.2/example_data_v0.2.zip"



pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]

@pytest.fixture(scope="module")
def test_data_dir(symfluence_code_dir, symfluence_data_root):
    """
    Download and extract example data to ../SYMFLUENCE_data/ for testing.

    Downloads fresh data and renames domain to domain_Bow_at_Banff_lumped_test
    to keep it separate from any existing work.
    """
    import shutil

    # Use ../SYMFLUENCE_data/ parallel to the code directory
    data_root = symfluence_data_root
    read_only_root = symfluence_code_dir.parent / "SYMFLUENCE_data"

    # Check if example domain already exists
    example_domain = "domain_Bow_at_Banff_lumped"
    example_domain_path = data_root / example_domain
    read_only_domain_path = read_only_root / example_domain

    # Download if it doesn't exist
    if not example_domain_path.exists():
        if read_only_domain_path.exists():
            print(f"Copying existing test data from {read_only_domain_path} to {data_root}")
            shutil.copytree(read_only_domain_path, example_domain_path, dirs_exist_ok=True)
            yield data_root
            return
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
def config_path(test_data_dir, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    from utils.helpers import load_config_template

    config = load_config_template(symfluence_code_dir)

    # Update paths
    config['SYMFLUENCE_DATA_DIR'] = str(test_data_dir)
    config['SYMFLUENCE_CODE_DIR'] = str(symfluence_code_dir)

    # Domain settings from notebook 02a
    config['DOMAIN_NAME'] = 'Bow_at_Banff_lumped'
    config['EXPERIMENT_ID'] = f'test_{tmp_path.name}'  # Unique test experiment ID
    config['POUR_POINT_COORDS'] = '51.1722/-115.5717'

    # Lumped basin settings
    config['DOMAIN_DEFINITION_METHOD'] = 'lumped'
    config['DOMAIN_DISCRETIZATION'] = 'GRUs'
    config['RIVER_BASINS_NAME'] = 'Bow_at_Banff_lumped_riverBasins_lumped.shp'

    # Optimized: 5-day period for faster testing (was 31 days)
    config['EXPERIMENT_TIME_START'] = '2004-01-01 01:00'
    config['EXPERIMENT_TIME_END'] = '2004-01-05 23:00'
    config['CALIBRATION_PERIOD'] = '2004-01-02, 2004-01-04'
    config['EVALUATION_PERIOD'] = '2004-01-05, 2004-01-05'
    config['SPINUP_PERIOD'] = '2004-01-01, 2004-01-01'

    # Streamflow
    config['STATION_ID'] = '05BB001'
    config['DOWNLOAD_WSC_DATA'] = False

    # Minimal calibration for testing (1 iteration)
    config['NUMBER_OF_ITERATIONS'] = 1
    config['RANDOM_SEED'] = 42  # Fixed seed for reproducibility

    # Save config
    cfg_path = tmp_path / 'test_config.yaml'
    write_config(config, cfg_path)

    return cfg_path, config


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


MODELS = [
    'SUMMA',
    'FUSE',
    pytest.param('GR', marks=pytest.mark.full),
    pytest.param('NGEN', marks=pytest.mark.full),
]


@pytest.mark.slow
@pytest.mark.requires_data
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
        config['GR_SPATIAL_MODE'] = 'lumped'
        config['GR_SKIP_CALIBRATION'] = True
    elif model == 'NGEN':
        config['NGEN_MODULES_TO_CALIBRATE'] = 'CFE'
        config['NGEN_CFE_PARAMS_TO_CALIBRATE'] = 'smcmax,satdk,bb'
        # Point to ngen install in data directory
        config['NGEN_INSTALL_PATH'] = str(Path(config['SYMFLUENCE_DATA_DIR']) / 'installs' / 'ngen' / 'cmake_build')

    # Save updated config
    with open(cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    baseline_dir = (
        Path(config["SYMFLUENCE_DATA_DIR"])
        / f"domain_{config['DOMAIN_NAME']}"
        / "shapefiles"
    )
    baseline_river_basins = (
        baseline_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_lumped.shp"
    )
    baseline_hrus = (
        baseline_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert baseline_river_basins.exists(), "Baseline river basins shapefile missing"
    assert baseline_hrus.exists(), "Baseline HRU shapefile missing"
    expected_river_basins = load_shapefile_signature(baseline_river_basins)
    expected_hrus = load_shapefile_signature(baseline_hrus)

    # Initialize SYMFLUENCE
    symfluence = SYMFLUENCE(cfg_path)

    # Step 1: Setup project
    project_dir = symfluence.managers['project'].setup_project()
    assert project_dir.exists(), "Project directory should be created"

    pour_point_path = symfluence.managers['project'].create_pour_point()
    if pour_point_path is None:
        existing_pour_point = (
            project_dir / "shapefiles" / "pour_point" / f"{config['DOMAIN_NAME']}_pourPoint.shp"
        )
        assert existing_pour_point.exists(), "Pour point shapefile should be created"
    else:
        assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    _prune_raw_forcing(project_dir, "domain_*_ERA5_merged_200401.nc")

    # Step 2: Define domain (watershed delineation)
    watershed_path, delineation_artifacts = symfluence.managers['domain'].define_domain()
    assert (
        delineation_artifacts.method == config["DOMAIN_DEFINITION_METHOD"]
    ), "Delineation method mismatch"
    # watershed_path can be None for lumped domains that use existing data

    # Step 3: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers['domain'].discretize_domain()
    assert (
        discretization_artifacts.method == config["DOMAIN_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (02a)
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir
        / "river_basins"
        / f"{config['DOMAIN_NAME']}_riverBasins_lumped.shp"
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp"
    )
    assert river_basins_path.exists()
    assert hrus_path.exists()
    assert_shapefile_signature_matches(river_basins_path, expected_river_basins)
    assert_shapefile_signature_matches(hrus_path, expected_hrus)

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
