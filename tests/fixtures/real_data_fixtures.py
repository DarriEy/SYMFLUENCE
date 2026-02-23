"""
Real data fixtures using tests/data/domain_Bow_at_Banff/.

These fixtures provide lightweight real data for unit tests,
replacing file I/O mocks with actual file structures.
"""
from pathlib import Path

import pytest


def _get_test_data_root():
    """Get the path to tests/data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def test_data_root():
    """Root directory for all test data."""
    root = _get_test_data_root()
    assert root.exists(), f"Test data directory not found: {root}"
    return root


@pytest.fixture(scope="session")
def bow_test_data(test_data_root):
    """Complete Bow at Banff test domain."""
    domain = test_data_root / "domain_Bow_at_Banff"
    assert domain.exists(), f"Bow domain not found: {domain}"
    return domain


@pytest.fixture(scope="session")
def real_forcing_nc(bow_test_data):
    """Real ERA5 forcing NetCDF (Jan 2004, 737KB)."""
    path = bow_test_data / "forcing" / "raw_data" / "domain_Bow_at_Banff_ERA5_merged_200401.nc"
    assert path.exists(), f"Forcing file not found: {path}"
    return path


@pytest.fixture(scope="session")
def real_dem_tif(bow_test_data):
    """Real DEM GeoTIFF (3.4MB)."""
    path = bow_test_data / "attributes" / "elevation" / "dem" / "domain_Bow_at_Banff_elv.tif"
    assert path.exists(), f"DEM not found: {path}"
    return path


@pytest.fixture(scope="session")
def real_landclass_tif(bow_test_data):
    """Real land classification GeoTIFF (478KB)."""
    path = bow_test_data / "attributes" / "landclass" / "domain_Bow_at_Banff_land_classes.tif"
    assert path.exists(), f"Land class not found: {path}"
    return path


@pytest.fixture(scope="session")
def real_soilclass_tif(bow_test_data):
    """Real soil classification GeoTIFF (28KB)."""
    path = bow_test_data / "attributes" / "soilclass" / "domain_Bow_at_Banff_soil_classes.tif"
    assert path.exists(), f"Soil class not found: {path}"
    return path


@pytest.fixture(scope="session")
def real_streamflow_csv(bow_test_data):
    """Real streamflow observations (2004, 253KB)."""
    path = bow_test_data / "observations" / "streamflow" / "preprocessed" / "Bow_at_Banff_streamflow_processed.csv"
    assert path.exists(), f"Streamflow not found: {path}"
    return path
