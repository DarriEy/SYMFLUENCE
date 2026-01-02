"""
SYMFLUENCE Cloud Attribute Acquisition Integration Tests

Validates cloud attribute acquisition for DEM, land cover, and soil classes.
"""

from pathlib import Path
import pytest

from symfluence import SYMFLUENCE
from utils.helpers import load_config_template, write_config


pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]


@pytest.fixture(scope="module")
def attr_config(tmp_path_factory, symfluence_code_dir, symfluence_data_root):
    """Create a base config for cloud attribute acquisition tests."""
    tmp_path = tmp_path_factory.mktemp("cloud_attr_acq")
    cfg_path = tmp_path / "test_config.yaml"

    config = load_config_template(symfluence_code_dir)
    data_root = symfluence_data_root

    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    config["DOMAIN_NAME"] = "paradise_cloud_attrs"
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["DOMAIN_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "47.2/-122.4/46.3/-121.3"
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    config["DATA_ACCESS"] = "cloud"
    config["DEM_SOURCE"] = "copernicus"

    config["DOWNLOAD_SNOTEL"] = False
    config["SNOTEL_STATION"] = "679"

    config["HYDROLOGICAL_MODEL"] = "SUMMA"

    config["EXPERIMENT_ID"] = "cloud_attr_acq"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"

    write_config(config, cfg_path)

    return cfg_path


@pytest.fixture(scope="module")
def attr_paths(attr_config):
    """Acquire cloud attributes once and return expected output paths."""
    symfluence = SYMFLUENCE(attr_config)

    project_dir = Path(symfluence.managers["project"].setup_project())
    pour_point_path = symfluence.managers["project"].create_pour_point()
    assert Path(pour_point_path).exists(), "Pour point shapefile should be created"

    symfluence.managers["data"].acquire_attributes()

    domain_name = symfluence.config.get("DOMAIN_NAME")
    dem_path = project_dir / "attributes" / "elevation" / "dem" / f"domain_{domain_name}_elv.tif"
    soil_path = project_dir / "attributes" / "soilclass" / f"domain_{domain_name}_soil_classes.tif"
    land_path = project_dir / "attributes" / "landclass" / f"domain_{domain_name}_land_classes.tif"

    return {"dem": dem_path, "soil": soil_path, "land": land_path}


def test_cloud_attribute_dem(attr_paths):
    """Verify DEM acquisition from cloud source."""
    dem_path = attr_paths["dem"]
    assert dem_path.exists(), f"DEM file missing: {dem_path}"


def test_cloud_attribute_soilclass(attr_paths):
    """Verify soil class acquisition from cloud source."""
    soil_path = attr_paths["soil"]
    assert soil_path.exists(), f"Soil class file missing: {soil_path}"


def test_cloud_attribute_landclass(attr_paths):
    """Verify land class acquisition from cloud source."""
    land_path = attr_paths["land"]
    assert land_path.exists(), f"Land class file missing: {land_path}"
