"""
SYMFLUENCE Cloud Attribute Acquisition Integration Tests

Validates cloud attribute acquisition for DEM, land cover, and soil classes.
"""

from pathlib import Path

import pytest

from symfluence import SYMFLUENCE
from test_helpers.helpers import load_config_template, write_config

pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]


def _alos_deps_available() -> bool:
    """Check if ALOS optional dependencies are installed."""
    try:
        import planetary_computer  # noqa: F401
        import pystac_client  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def base_attr_config(symfluence_code_dir, symfluence_data_root):
    """Create a base config for cloud attribute acquisition tests."""
    config = load_config_template(symfluence_code_dir)
    data_root = symfluence_data_root

    config["SYMFLUENCE_DATA_DIR"] = str(data_root)
    config["SYMFLUENCE_CODE_DIR"] = str(symfluence_code_dir)

    config["DOMAIN_NAME"] = "paradise_cloud_attrs"
    config["DOMAIN_DEFINITION_METHOD"] = "point"
    config["SUB_GRID_DISCRETIZATION"] = "GRUs"
    config["BOUNDING_BOX_COORDS"] = "47.2/-122.4/46.3/-121.3"
    config["POUR_POINT_COORDS"] = "46.78/-121.75"

    config["DATA_ACCESS"] = "cloud"
    config["DEM_SOURCE"] = "copernicus"

    config["DOWNLOAD_SNOTEL"] = False
    config["HYDROLOGICAL_MODEL"] = "SUMMA"

    config["EXPERIMENT_ID"] = "cloud_attr_acq"
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-02 00:00"
    config["CALIBRATION_PERIOD"] = None
    config["EVALUATION_PERIOD"] = None
    config["SPINUP_PERIOD"] = None

    # Disable all by default for single dataset testing
    config["DOWNLOAD_DEM"] = False
    config["DOWNLOAD_SOIL"] = False
    config["DOWNLOAD_LAND_COVER"] = False

    return config


def test_cloud_attribute_dem(tmp_path, base_attr_config):
    """Verify DEM acquisition from cloud source."""
    config = base_attr_config.copy()
    config["DOWNLOAD_DEM"] = True
    config["DOMAIN_NAME"] = "test_dem_only"

    cfg_path = tmp_path / "test_config_dem.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    dem_path = project_dir / "attributes" / "elevation" / "dem" / f"domain_{config['DOMAIN_NAME']}_elv.tif"
    assert dem_path.exists(), f"DEM file missing: {dem_path}"


def test_cloud_attribute_soilclass(tmp_path, base_attr_config):
    """Verify soil class acquisition from cloud source."""
    config = base_attr_config.copy()
    config["DOWNLOAD_SOIL"] = True
    config["DOMAIN_NAME"] = "test_soil_only"

    cfg_path = tmp_path / "test_config_soil.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    soil_path = project_dir / "attributes" / "soilclass" / f"domain_{config['DOMAIN_NAME']}_soil_classes.tif"
    assert soil_path.exists(), f"Soil class file missing: {soil_path}"


def test_cloud_attribute_landclass_modis(tmp_path, base_attr_config):
    """Verify MODIS land class acquisition from cloud source."""
    config = base_attr_config.copy()
    config["DOWNLOAD_LAND_COVER"] = True
    config["LAND_CLASS_SOURCE"] = "modis"
    config["DOMAIN_NAME"] = "test_land_modis_only"

    cfg_path = tmp_path / "test_config_land_modis.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    land_path = project_dir / "attributes" / "landclass" / f"domain_{config['DOMAIN_NAME']}_land_classes.tif"
    assert land_path.exists(), f"Land class file missing: {land_path}"


def test_cloud_attribute_landclass_usgs(tmp_path, base_attr_config):
    """Verify USGS NLCD land class acquisition from cloud source."""
    config = base_attr_config.copy()
    config["DOWNLOAD_LAND_COVER"] = True
    config["LAND_CLASS_SOURCE"] = "usgs_nlcd"
    config["DOMAIN_NAME"] = "test_land_usgs_only"

    cfg_path = tmp_path / "test_config_land_usgs.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    land_path = project_dir / "attributes" / "landclass" / f"domain_{config['DOMAIN_NAME']}_land_classes.tif"
    assert land_path.exists(), f"Land class file missing: {land_path}"


# =============================================================================
# New DEM Source Integration Tests
# =============================================================================

@pytest.mark.parametrize("dem_source", ["copernicus_90", "srtm", "etopo", "mapzen"])
def test_cloud_attribute_dem_sources(tmp_path, base_attr_config, dem_source):
    """Verify DEM acquisition from new cloud DEM sources."""
    config = base_attr_config.copy()
    config["DOWNLOAD_DEM"] = True
    config["DEM_SOURCE"] = dem_source
    config["DOMAIN_NAME"] = f"test_dem_{dem_source}"

    cfg_path = tmp_path / f"test_config_dem_{dem_source}.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    dem_path = project_dir / "attributes" / "elevation" / "dem" / f"domain_{config['DOMAIN_NAME']}_elv.tif"
    assert dem_path.exists(), f"DEM file missing for source '{dem_source}': {dem_path}"


@pytest.mark.skipif(
    not _alos_deps_available(),
    reason="planetary-computer and pystac-client not installed"
)
def test_cloud_attribute_dem_alos(tmp_path, base_attr_config):
    """Verify ALOS DEM acquisition (requires optional dependencies)."""
    config = base_attr_config.copy()
    config["DOWNLOAD_DEM"] = True
    config["DEM_SOURCE"] = "alos"
    config["DOMAIN_NAME"] = "test_dem_alos"

    cfg_path = tmp_path / "test_config_dem_alos.yaml"
    write_config(config, cfg_path)

    symfluence = SYMFLUENCE(cfg_path)
    project_dir = Path(symfluence.managers["project"].setup_project())
    symfluence.managers["data"].acquire_attributes()

    dem_path = project_dir / "attributes" / "elevation" / "dem" / f"domain_{config['DOMAIN_NAME']}_elv.tif"
    assert dem_path.exists(), f"DEM file missing for ALOS: {dem_path}"
