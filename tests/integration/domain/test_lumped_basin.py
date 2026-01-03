"""
SYMFLUENCE Lumped Basin Integration Tests

Tests the lumped basin workflow from notebook 02a for all supported models.
Downloads test data from GitHub release and runs a short 1-month simulation.
"""

import pytest
import yaml
from pathlib import Path

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from utils.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)
from utils.helpers import write_config


pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


@pytest.fixture(scope="function")
def config_path(bow_domain, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    from utils.helpers import load_config_template

    config = load_config_template(symfluence_code_dir)

    # Update paths
    # bow_domain is the path to the domain directory (e.g. .../domain_Bow_at_Banff_lumped)
    config['SYMFLUENCE_DATA_DIR'] = str(bow_domain.parent)
    config['SYMFLUENCE_CODE_DIR'] = str(symfluence_code_dir)

    # Domain settings from notebook 02a
    # Extract domain name from directory name (remove 'domain_' prefix)
    domain_name = bow_domain.name.replace("domain_", "")
    config['DOMAIN_NAME'] = domain_name
    config['EXPERIMENT_ID'] = f'test_{tmp_path.name}'  # Unique test experiment ID
    config['POUR_POINT_COORDS'] = '51.1722/-115.5717'

    # Lumped basin settings
    config['DOMAIN_DEFINITION_METHOD'] = 'lumped'
    config['DOMAIN_DISCRETIZATION'] = 'GRUs'
    
    # Handle different naming conventions in data bundles
    # River basins
    river_basins_name = f'{domain_name}_riverBasins_lumped.shp'
    river_basins_dir = bow_domain / "shapefiles" / "river_basins"
    if not (river_basins_dir / river_basins_name).exists():
        if river_basins_dir.exists():
            shps = list(river_basins_dir.glob("*.shp"))
            if shps:
                river_basins_name = shps[0].name
    config['RIVER_BASINS_NAME'] = river_basins_name

    # Catchment/HRUs
    catchment_name = f'{domain_name}_HRUs_GRUs.shp'
    catchment_dir = bow_domain / "shapefiles" / "catchment"
    if not (catchment_dir / catchment_name).exists():
        if catchment_dir.exists():
            shps = list(catchment_dir.glob("*.shp"))
            if shps:
                catchment_name = shps[0].name
    config['CATCHMENT_SHP_NAME'] = catchment_name

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
    'LSTM',
    'HYPE',
    pytest.param('MESH', marks=pytest.mark.full),
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
    elif model == 'LSTM':
        config['LSTM_EPOCHS'] = 1
        config['LSTM_LOOKBACK'] = 24
        config['LSTM_HIDDEN_SIZE'] = 16
        config['LSTM_BATCH_SIZE'] = 4
    elif model == 'HYPE':
        config['HYPE_TIMESHIFT'] = 0
        config['HYPE_SPINUP_DAYS'] = 0
        config['HYPE_FRAC_THRESHOLD'] = 0.1
        # HYPE calibration not yet fully implemented in this test
        config['HYPE_SKIP_CALIBRATION'] = True
    elif model == 'MESH':
        # MESH configuration
        config['MESH_SKIP_CALIBRATION'] = True
        # Point to MESH install in data directory
        config['MESH_INSTALL_PATH'] = str(Path(config['SYMFLUENCE_DATA_DIR']) / 'installs' / 'MESH-Dev')
        config['MESH_EXE'] = 'sa_mesh'

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
        / config.get("RIVER_BASINS_NAME", f"{config['DOMAIN_NAME']}_riverBasins_lumped.shp")
    )
    baseline_hrus = (
        baseline_dir / "catchment" / config.get("CATCHMENT_SHP_NAME", f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp")
    )
    assert baseline_river_basins.exists(), f"Baseline river basins shapefile missing: {baseline_river_basins}"
    assert baseline_hrus.exists(), f"Baseline HRU shapefile missing: {baseline_hrus}"
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
        / config.get("RIVER_BASINS_NAME", f"{config['DOMAIN_NAME']}_riverBasins_lumped.shp")
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / config.get("CATCHMENT_SHP_NAME", f"{config['DOMAIN_NAME']}_HRUs_GRUs.shp")
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
    # Check for HYPE binary if needed
    if model == 'HYPE':
        hype_exe = Path(config.get('SYMFLUENCE_DATA_DIR')) / 'installs' / 'hype' / config.get('HYPE_EXE', 'hype')
        if not hype_exe.exists():
            pytest.skip(f"HYPE binary not found at {hype_exe}, skipping run and calibration")
    elif model == 'MESH':
        mesh_exe = Path(config.get('MESH_INSTALL_PATH', '')) / config.get('MESH_EXE', 'sa_mesh')
        if not mesh_exe.exists():
            pytest.skip(f"MESH binary not found at {mesh_exe}, skipping run and calibration")

    symfluence.managers['model'].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config['EXPERIMENT_ID'] / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 7: Calibrate model
    # Skip calibration for LSTM/GR/HYPE/MESH as they either auto-calibrate or are not supported
    if model in ['GR', 'LSTM', 'HYPE', 'MESH']:
        pass
    else:
        results_file = symfluence.managers['optimization'].calibrate_model()
        assert results_file is not None, "Calibration should produce results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
