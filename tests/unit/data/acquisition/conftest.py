"""
Shared Fixtures for Acquisition Handler Tests.

This conftest.py imports and exposes fixtures from the fixtures modules,
making them available to all acquisition tests.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import xarray as xr

# Import fixtures from fixture modules
from fixtures.acquisition_fixtures import (
    MockConfigFactory,
    MockResponse,
    MockResponseFactory,
    MockSessionFactory,
    create_mock_logger,
    create_capturing_logger,
    patch_requests_session,
    patch_robust_session,
)

from fixtures.synthetic_data import (
    era5_dataset,
    grace_dataset,
    soil_moisture_dataset,
    dem_array,
    dem_dataset,
    forcing_dataset,
    evapotranspiration_dataset,
    snow_cover_dataset,
)


# =============================================================================
# Mock Config Fixtures
# =============================================================================

@pytest.fixture
def mock_config_factory():
    """Factory for creating test configurations."""
    return MockConfigFactory


@pytest.fixture
def mock_config():
    """Standard mock configuration for handler tests."""
    return MockConfigFactory.create()


@pytest.fixture
def mock_config_with_credentials():
    """Mock configuration including credential settings."""
    return MockConfigFactory.create_with_credentials()


@pytest.fixture
def minimal_config():
    """Minimal configuration with only required fields."""
    return MockConfigFactory.create_minimal()


# =============================================================================
# Mock Response Fixtures
# =============================================================================

@pytest.fixture
def mock_response_factory():
    """Factory for creating mock HTTP responses."""
    return MockResponseFactory


@pytest.fixture
def mock_session_factory():
    """Factory for creating mock sessions."""
    return MockSessionFactory


@pytest.fixture
def mock_session():
    """Basic mock session with success responses."""
    return MockSessionFactory.create()


# =============================================================================
# Logger Fixtures
# =============================================================================

@pytest.fixture
def mock_logger():
    """Mock logger that suppresses output."""
    return create_mock_logger("test_acquisition")


@pytest.fixture
def capturing_logger():
    """Logger that captures messages for assertion."""
    return create_capturing_logger("test_acquisition_capture")


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for handler tests."""
    output_dir = tmp_path / "acquisition_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory structure matching SYMFLUENCE layout."""
    data_dir = tmp_path / "test_data"
    domain_dir = data_dir / "domain_test_domain"
    attrs_dir = domain_dir / "attributes"

    for d in [data_dir, domain_dir, attrs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return data_dir


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================

@pytest.fixture
def synthetic_era5():
    """Synthetic ERA5 dataset for testing."""
    return era5_dataset()


@pytest.fixture
def synthetic_era5_small():
    """Small synthetic ERA5 dataset for fast tests."""
    return era5_dataset(
        lat_range=(46.0, 46.5),
        lon_range=(8.0, 8.5),
        time_range=("2020-01-01", "2020-01-07"),
    )


@pytest.fixture
def synthetic_grace():
    """Synthetic GRACE TWS dataset."""
    return grace_dataset()


@pytest.fixture
def synthetic_soil_moisture():
    """Synthetic soil moisture dataset."""
    return soil_moisture_dataset()


@pytest.fixture
def synthetic_dem():
    """Synthetic DEM dataset."""
    return dem_dataset()


@pytest.fixture
def synthetic_forcing():
    """Generic synthetic forcing dataset."""
    return forcing_dataset()


@pytest.fixture
def synthetic_forcing_descending_lat():
    """Forcing data with descending latitude (ERA5 style)."""
    return forcing_dataset(lat_descending=True, lat_name="latitude", lon_name="longitude")


@pytest.fixture
def synthetic_forcing_lon_360():
    """Forcing data with 0-360 longitude convention."""
    return forcing_dataset(lon_0_360=True, lon_range=(-10.0, 10.0))


# =============================================================================
# Bounding Box Fixtures
# =============================================================================

@pytest.fixture
def standard_bbox() -> Dict[str, float]:
    """Standard bounding box for tests (Switzerland area)."""
    return {
        "lat_min": 46.0,
        "lat_max": 47.0,
        "lon_min": 8.0,
        "lon_max": 9.0,
    }


@pytest.fixture
def small_bbox() -> Dict[str, float]:
    """Small bounding box for fast tests."""
    return {
        "lat_min": 46.0,
        "lat_max": 46.5,
        "lon_min": 8.0,
        "lon_max": 8.5,
    }


@pytest.fixture
def negative_lon_bbox() -> Dict[str, float]:
    """Bounding box with negative longitude (Western Europe)."""
    return {
        "lat_min": 50.0,
        "lat_max": 51.0,
        "lon_min": -5.0,
        "lon_max": -4.0,
    }


@pytest.fixture
def dateline_crossing_bbox() -> Dict[str, float]:
    """Bounding box crossing the dateline."""
    return {
        "lat_min": 40.0,
        "lat_max": 45.0,
        "lon_min": 170.0,
        "lon_max": -170.0,
    }


# =============================================================================
# Handler Instance Fixtures
# =============================================================================

@pytest.fixture
def base_handler_instance(mock_config, mock_logger):
    """
    Create a concrete implementation of BaseAcquisitionHandler for testing.

    Since BaseAcquisitionHandler is abstract, we create a minimal concrete class.
    """
    from symfluence.data.acquisition.base import BaseAcquisitionHandler

    class TestableHandler(BaseAcquisitionHandler):
        """Concrete handler for testing base class functionality."""

        def download(self, output_dir: Path) -> Path:
            """Minimal implementation that returns a dummy path."""
            output_file = output_dir / "test_output.nc"
            output_file.touch()
            return output_file

    return TestableHandler(mock_config, mock_logger)


@pytest.fixture
def retry_mixin_instance(mock_logger):
    """Create an instance of RetryMixin for testing."""
    from symfluence.data.acquisition.mixins.retry import RetryMixin

    class TestableRetryMixin(RetryMixin):
        def __init__(self, logger):
            self.logger = logger

    return TestableRetryMixin(mock_logger)


@pytest.fixture
def chunked_mixin_instance(mock_logger):
    """Create an instance of ChunkedDownloadMixin for testing."""
    from symfluence.data.acquisition.mixins.chunked import ChunkedDownloadMixin

    class TestableChunkedMixin(ChunkedDownloadMixin):
        def __init__(self, logger):
            self.logger = logger

    return TestableChunkedMixin(mock_logger)


@pytest.fixture
def spatial_mixin_instance(mock_logger, standard_bbox):
    """Create an instance of SpatialSubsetMixin for testing."""
    from symfluence.data.acquisition.mixins.spatial import SpatialSubsetMixin

    class TestableSpatialMixin(SpatialSubsetMixin):
        def __init__(self, logger, bbox):
            self.logger = logger
            self.bbox = bbox

    return TestableSpatialMixin(mock_logger, standard_bbox)


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def clean_environment():
    """
    Fixture that removes credential environment variables during test.

    Restores original environment after test completes.
    """
    # Save original values
    original_env = {}
    env_vars = [
        "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
        "CDSAPI_URL", "CDSAPI_KEY",
    ]

    for var in env_vars:
        original_env[var] = os.environ.pop(var, None)

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is not None:
            os.environ[var] = value


@pytest.fixture
def mock_earthdata_env():
    """Fixture that sets mock Earthdata credentials in environment."""
    original_user = os.environ.get("EARTHDATA_USERNAME")
    original_pass = os.environ.get("EARTHDATA_PASSWORD")

    os.environ["EARTHDATA_USERNAME"] = "test_user"
    os.environ["EARTHDATA_PASSWORD"] = "test_password"

    yield

    # Restore
    if original_user:
        os.environ["EARTHDATA_USERNAME"] = original_user
    else:
        os.environ.pop("EARTHDATA_USERNAME", None)

    if original_pass:
        os.environ["EARTHDATA_PASSWORD"] = original_pass
    else:
        os.environ.pop("EARTHDATA_PASSWORD", None)


@pytest.fixture
def mock_cds_env():
    """Fixture that sets mock CDS credentials in environment."""
    original_url = os.environ.get("CDSAPI_URL")
    original_key = os.environ.get("CDSAPI_KEY")

    os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
    os.environ["CDSAPI_KEY"] = "12345:abcdef-1234-5678"

    yield

    # Restore
    if original_url:
        os.environ["CDSAPI_URL"] = original_url
    else:
        os.environ.pop("CDSAPI_URL", None)

    if original_key:
        os.environ["CDSAPI_KEY"] = original_key
    else:
        os.environ.pop("CDSAPI_KEY", None)


# =============================================================================
# NetCDF File Fixtures
# =============================================================================

@pytest.fixture
def temp_netcdf_file(tmp_path, synthetic_era5_small):
    """Create a temporary NetCDF file for testing."""
    nc_path = tmp_path / "test_data.nc"
    synthetic_era5_small.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def temp_netcdf_chunks(tmp_path, synthetic_era5_small):
    """Create multiple temporary NetCDF chunks for merge testing."""
    chunk_files = []
    ds = synthetic_era5_small

    # Split into 3 time chunks
    time_chunks = [
        ds.isel(time=slice(0, 56)),    # First ~2 days
        ds.isel(time=slice(56, 112)),  # Days 3-4
        ds.isel(time=slice(112, 168)), # Days 5-7
    ]

    for i, chunk in enumerate(time_chunks):
        chunk_path = tmp_path / f"chunk_{i:02d}.nc"
        chunk.to_netcdf(chunk_path)
        chunk_files.append(chunk_path)

    return chunk_files


# =============================================================================
# Utility Functions
# =============================================================================

def assert_dataset_has_coords(ds: xr.Dataset, expected_coords: list):
    """Assert that a dataset has the expected coordinates."""
    for coord in expected_coords:
        assert coord in ds.coords, f"Missing coordinate: {coord}"


def assert_bbox_within(ds: xr.Dataset, bbox: Dict[str, float], lat_name: str = "lat", lon_name: str = "lon"):
    """Assert that a dataset is within the specified bounding box."""
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    assert lat_vals.min() >= bbox["lat_min"] - 1e-6, "Latitude below minimum"
    assert lat_vals.max() <= bbox["lat_max"] + 1e-6, "Latitude above maximum"
    assert lon_vals.min() >= bbox["lon_min"] - 1e-6, "Longitude below minimum"
    assert lon_vals.max() <= bbox["lon_max"] + 1e-6, "Longitude above maximum"
