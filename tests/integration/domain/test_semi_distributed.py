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
from pathlib import Path

# Import SYMFLUENCE - this should work now since we added the path
from symfluence import SYMFLUENCE
from utils.helpers import load_config_template, write_config
from utils.geospatial import (
    assert_shapefile_signature_matches,
    load_shapefile_signature,
)

# GitHub release URL for example data
EXAMPLE_DATA_URL = "https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-v0.5.5/example_data_v0.5.5.zip"



pytestmark = [pytest.mark.integration, pytest.mark.domain, pytest.mark.requires_data, pytest.mark.slow]


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


@pytest.fixture(scope="function")
def config_path(example_data_bundle, tmp_path, symfluence_code_dir):
    """Create test configuration based on config_template.yaml."""
    # Load template
    config = load_config_template(symfluence_code_dir)

    # Update paths
    config["SYMFLUENCE_DATA_DIR"] = str(example_data_bundle)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    # Domain settings from notebook 02b
    config["DOMAIN_NAME"] = "Bow_at_Banff_semi_distributed"
    config["EXPERIMENT_ID"] = f"test_{tmp_path.name}"
    config["POUR_POINT_COORDS"] = "51.1722/-115.5717"

    # Semi-distributed basin settings
    config["DELINEATION_METHOD"] = "stream_threshold"
    config["DOMAIN_DEFINITION_METHOD"] = "delineate"
    config["STREAM_THRESHOLD"] = 10000
    config["DOMAIN_DISCRETIZATION"] = "GRUs"

    # Optimized: 1-day period for faster testing (was 5 days)
    config["EXPERIMENT_TIME_START"] = "2004-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2004-01-01 23:00"
    config["CALIBRATION_PERIOD"] = "2004-01-01, 2004-01-01"
    config["EVALUATION_PERIOD"] = "2004-01-01, 2004-01-01"
    config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-01"

    # Streamflow
    config["STATION_ID"] = "05BB001"
    config["DOWNLOAD_WSC_DATA"] = False

    # Minimal calibration for testing (optimized: 1 iteration, was 3)
    config["NUMBER_OF_ITERATIONS"] = 1
    config["RANDOM_SEED"] = 42

    # Save config
    cfg_path = tmp_path / "test_config.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


MODELS = [
    "SUMMA",
    "FUSE",
    pytest.param("NGEN", marks=pytest.mark.full),
    "HYPE",
]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("model", MODELS)
def test_semi_distributed_basin_workflow(config_path, example_data_bundle, model):
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
    write_config(config, cfg_path)

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
    signature_strict = config.get("STREAM_THRESHOLD") == 5000
    if signature_strict:
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

    # Step 2: Reuse data from the lumped example domain
    lumped_domain = "Bow_at_Banff_lumped"
    lumped_data_dir = example_data_bundle / f"domain_{lumped_domain}"
    
    # Fallback for v0.6.0 bundle structure
    if not lumped_data_dir.exists():
        lumped_data_dir = example_data_bundle / "domain_bow_banff_minimal"
        lumped_domain = "Bow_at_Banff_lumped" # Name used inside files in minimal bundle
        print(f"  Note: Using {lumped_data_dir.name} as data source for reuse.")

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

    _prune_raw_forcing(project_dir, "domain_*_ERA5_merged_200401.nc")

    # Clear processed forcing outputs so HRU counts stay consistent after threshold changes.
    forcing_dir = project_dir / "forcing"
    if forcing_dir.exists():
        for subdir in ["basin_averaged_data", "merged_path", "SUMMA_input", "GR_input", "NGEN_input"]:
            shutil.rmtree(forcing_dir / subdir, ignore_errors=True)
        for temp_dir in forcing_dir.glob("temp_*"):
            shutil.rmtree(temp_dir, ignore_errors=True)

    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    # Step 3: Define domain (watershed delineation)
    watershed_path, delineation_artifacts = symfluence.managers["domain"].define_domain()
    assert (
        delineation_artifacts.method == config["DOMAIN_DEFINITION_METHOD"]
    ), "Delineation method mismatch"
    # watershed_path can be None for workflows that use existing data

    # Step 4: Discretize domain
    hru_path, discretization_artifacts = symfluence.managers["domain"].discretize_domain()
    assert (
        discretization_artifacts.method == config["DOMAIN_DISCRETIZATION"]
    ), "Discretization method mismatch"

    # Verify geospatial artifacts (02b)
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
    if signature_strict:
        assert_shapefile_signature_matches(river_basins_path, expected_river_basins)
        assert_shapefile_signature_matches(river_network_path, expected_river_network)
        assert_shapefile_signature_matches(hrus_path, expected_hrus)

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
