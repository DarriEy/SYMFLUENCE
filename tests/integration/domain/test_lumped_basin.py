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


from symfluence.core.config.models import SymfluenceConfig


pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


@pytest.fixture(scope="function")
def config_path(bow_domain, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    from utils.helpers import load_config_template

    config_dict = load_config_template(symfluence_code_dir)

    # Update paths
    # bow_domain is the path to the domain directory (e.g. .../domain_Bow_at_Banff_lumped)
    config_dict['SYMFLUENCE_DATA_DIR'] = str(bow_domain.parent)
    config_dict['SYMFLUENCE_CODE_DIR'] = str(symfluence_code_dir)

    # Domain settings from notebook 02a
    # Extract domain name from directory name (remove 'domain_' prefix)
    domain_name = bow_domain.name.replace("domain_", "")
    config_dict['DOMAIN_NAME'] = domain_name
    config_dict['EXPERIMENT_ID'] = f'test_{tmp_path.name}'  # Unique test experiment ID
    config_dict['POUR_POINT_COORDS'] = '51.1722/-115.5717'

    # Lumped basin settings
    config_dict['DOMAIN_DEFINITION_METHOD'] = 'lumped'
    config_dict['DOMAIN_DISCRETIZATION'] = 'GRUs'
    
    # Handle different naming conventions in data bundles
    # River basins
    river_basins_name = f'{domain_name}_riverBasins_lumped.shp'
    river_basins_dir = bow_domain / "shapefiles" / "river_basins"
    if not (river_basins_dir / river_basins_name).exists():
        if river_basins_dir.exists():
            shps = list(river_basins_dir.glob("*.shp"))
            if shps:
                river_basins_name = shps[0].name
    config_dict['RIVER_BASINS_NAME'] = river_basins_name

    # River network
    river_network_name = f'{domain_name}_riverNetwork_lumped.shp'
    river_network_dir = bow_domain / "shapefiles" / "river_network"
    if not (river_network_dir / river_network_name).exists():
        if river_network_dir.exists():
            shps = list(river_network_dir.glob("*.shp"))
            if shps:
                river_network_name = shps[0].name
    config_dict['RIVER_NETWORK_NAME'] = river_network_name

    # Catchment/HRUs
    catchment_name = f'{domain_name}_HRUs_GRUs.shp'
    catchment_dir = bow_domain / "shapefiles" / "catchment"
    if not (catchment_dir / catchment_name).exists():
        if catchment_dir.exists():
            shps = list(catchment_dir.glob("*.shp"))
            if shps:
                catchment_name = shps[0].name
    config_dict['CATCHMENT_SHP_NAME'] = catchment_name

    # Optimized: 5-day period for faster testing (was 31 days)
    config_dict['EXPERIMENT_TIME_START'] = '2004-01-01 01:00'
    config_dict['EXPERIMENT_TIME_END'] = '2004-01-05 23:00'
    config_dict['CALIBRATION_PERIOD'] = '2004-01-02, 2004-01-04'
    config_dict['EVALUATION_PERIOD'] = '2004-01-05, 2004-01-05'
    config_dict['SPINUP_PERIOD'] = '2004-01-01, 2004-01-01'

    # Streamflow
    config_dict['STATION_ID'] = '05BB001'
    config_dict['DOWNLOAD_WSC_DATA'] = False

    # Minimal calibration for testing (1 iteration)
    config_dict['NUMBER_OF_ITERATIONS'] = 1
    config_dict['RANDOM_SEED'] = 42  # Fixed seed for reproducibility

    # Ensure required fields for SymfluenceConfig are present
    if 'HYDROLOGICAL_MODEL' not in config_dict:
        config_dict['HYDROLOGICAL_MODEL'] = 'SUMMA'
    if 'FORCING_DATASET' not in config_dict:
        config_dict['FORCING_DATASET'] = 'ERA5'

    # Create and validate SymfluenceConfig
    config = SymfluenceConfig(**config_dict)

    # Save config back to YAML for SYMFLUENCE initialization
    cfg_path = tmp_path / 'test_config.yaml'
    write_config(config.to_dict(flatten=True), cfg_path)

    return cfg_path, config


def _prune_raw_forcing(project_dir: Path, keep_glob: str) -> None:
    raw_dir = project_dir / "forcing" / "raw_data"
    if not raw_dir.exists():
        return
    
    # Try multiple globs to support both old and new names
    candidates = sorted(raw_dir.glob(keep_glob))
    if not candidates:
        # Try a more generic pattern if the specific one fails
        if "ERA5" in keep_glob:
            candidates = sorted(raw_dir.glob("*ERA5*1month*.nc"))
    
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
    pytest.param('RHESSys', marks=pytest.mark.full),
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
    cfg_path, typed_config = config_path
    config_dict = typed_config.to_dict(flatten=True)

    # Update model in config
    config_dict['HYDROLOGICAL_MODEL'] = model
    if model == 'SUMMA':
        config_dict['ROUTING_MODEL'] = 'mizuRoute'
        config_dict['PARAMS_TO_CALIBRATE'] = 'k_soil,theta_sat'
        config_dict['BASIN_PARAMS_TO_CALIBRATE'] = 'routingGammaScale'
    elif model == 'FUSE':
        config_dict['FUSE_SPATIAL_MODE'] = 'lumped'
        config_dict['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = 'MAXWATR_1,MAXWATR_2,BASERTE'
    elif model == 'GR':
        config_dict['GR_SPATIAL_MODE'] = 'lumped'
        config_dict['GR_SKIP_CALIBRATION'] = True
    elif model == 'NGEN':
        config_dict['NGEN_MODULES_TO_CALIBRATE'] = 'CFE'
        config_dict['NGEN_CFE_PARAMS_TO_CALIBRATE'] = 'smcmax,satdk,bb'
        # Point to ngen install in data directory
        config_dict['NGEN_INSTALL_PATH'] = str(Path(config_dict['SYMFLUENCE_DATA_DIR']) / 'installs' / 'ngen' / 'cmake_build')
    elif model == 'LSTM':
        config_dict['LSTM_EPOCHS'] = 1
        config_dict['LSTM_LOOKBACK'] = 24
        config_dict['LSTM_HIDDEN_SIZE'] = 16
        config_dict['LSTM_BATCH_SIZE'] = 4
    elif model == 'HYPE':
        config_dict['HYPE_TIMESHIFT'] = 0
        config_dict['HYPE_SPINUP_DAYS'] = 0
        config_dict['HYPE_FRAC_THRESHOLD'] = 0.1
        # HYPE calibration not yet fully implemented in this test
        config_dict['HYPE_SKIP_CALIBRATION'] = True
    elif model == 'MESH':
        # MESH configuration
        config_dict['MESH_SKIP_CALIBRATION'] = True
        # Point to MESH install in data directory
        config_dict['MESH_INSTALL_PATH'] = str(Path(config_dict['SYMFLUENCE_DATA_DIR']) / 'installs' / 'mesh' / 'bin')
        config_dict['MESH_EXE'] = 'mesh.exe'
    elif model == 'RHESSys':
        # RHESSys configuration
        config_dict['RHESSYS_SKIP_CALIBRATION'] = True
        # Point to RHESSys install in data directory
        config_dict['RHESSYS_INSTALL_PATH'] = str(Path(config_dict['SYMFLUENCE_DATA_DIR']) / 'installs' / 'rhessys' / 'bin')
        config_dict['RHESSYS_EXE'] = 'rhessys'
        # VMFire settings
        config_dict['RHESSYS_USE_VMFIRE'] = True
        config_dict['VMFIRE_INSTALL_PATH'] = str(Path(config_dict['SYMFLUENCE_DATA_DIR']) / 'installs' / 'vmfire' / 'bin')
        config_dict['VMFIRE_EXE'] = 'vmfire'

    # Create new validated typed config with model-specific overrides
    config = SymfluenceConfig(**config_dict)

    # Save updated config to YAML
    with open(cfg_path, 'w') as f:
        yaml.dump(config.to_dict(flatten=True), f, default_flow_style=False, sort_keys=False)

    baseline_dir = (
        Path(config.system.data_dir)
        / f"domain_{config.domain.name}"
        / "shapefiles"
    )
    baseline_river_basins = (
        baseline_dir
        / "river_basins"
        / config.paths.river_basins_name
    )
    baseline_hrus = (
        baseline_dir / "catchment" / config.paths.catchment_name
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
            project_dir / "shapefiles" / "pour_point" / f"{config.domain.name}_pourPoint.shp"
        )
        assert existing_pour_point.exists(), "Pour point shapefile should be created"
    else:
        assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    _prune_raw_forcing(project_dir, "domain_*_ERA5_merged_200401.nc")

    # Step 2: Define domain (watershed delineation)
    watershed_path, delineation_artifacts = symfluence.managers['domain'].define_domain()
    assert (
        delineation_artifacts.method == config.domain.definition_method
    ), "Delineation method mismatch"
    # watershed_path can be None for lumped domains that use existing data

    # Step 3: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers['domain'].discretize_domain()
    assert (
        discretization_artifacts.method == config.domain.discretization
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (02a)
    shapefile_dir = project_dir / "shapefiles"
    river_basins_path = delineation_artifacts.river_basins_path or (
        shapefile_dir
        / "river_basins"
        / config.paths.river_basins_name
    )
    hrus_path = (
        discretization_artifacts.hru_paths
        if isinstance(discretization_artifacts.hru_paths, Path)
        else shapefile_dir / "catchment" / config.paths.catchment_name
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
    # Check for binary if needed
    if model == 'HYPE':
        hype_exe = Path(config.system.data_dir) / 'installs' / 'hype' / 'bin' / 'hype'
        if not hype_exe.exists():
            pytest.skip(f"HYPE binary not found at {hype_exe}, skipping run and calibration")
    elif model == 'MESH':
        mesh_exe = Path(config.system.data_dir) / 'installs' / 'mesh' / 'bin' / 'mesh.exe'
        if not mesh_exe.exists():
            pytest.skip(f"MESH binary not found at {mesh_exe}, skipping run and calibration")
    elif model == 'RHESSys':
        rhessys_exe = Path(config.system.data_dir) / 'installs' / 'rhessys' / 'bin' / 'rhessys'
        if not rhessys_exe.exists():
            pytest.skip(f"RHESSys binary not found at {rhessys_exe}, skipping run and calibration")
        if config_dict.get('RHESSYS_USE_VMFIRE'):
            vmfire_exe = Path(config_dict['SYMFLUENCE_DATA_DIR']) / 'installs' / 'vmfire' / 'bin' / 'vmfire'
            if not vmfire_exe.exists():
                pytest.skip(f"VMFire binary not found at {vmfire_exe}, skipping run and calibration")

    symfluence.managers['model'].run_models()

    # Check model output exists
    sim_dir = project_dir / "simulations" / config.domain.experiment_id / model
    assert sim_dir.exists(), f"{model} simulation output directory should exist"

    # Step 7: Calibrate model
    # Skip calibration for LSTM/GR/HYPE/MESH as they either auto-calibrate or are not supported
    if model in ['GR', 'LSTM', 'HYPE', 'MESH', 'RHESSys']:
        pass
    else:
        results_file = symfluence.managers['optimization'].calibrate_model()
        assert results_file is not None, "Calibration should produce results"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
