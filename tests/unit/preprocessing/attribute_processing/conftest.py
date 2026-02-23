"""
Shared fixtures for attribute_processing tests.

Provides mock data, test configurations, and utilities for testing
the attributeProcessor class and its methods.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory structure."""
    project_dir = tmp_path / "test_domain"

    # Create subdirectories (under data/ for new layout)
    (project_dir / "shapefiles" / "catchment").mkdir(parents=True)
    (project_dir / "data" / "attributes" / "elevation" / "slope").mkdir(parents=True)
    (project_dir / "data" / "attributes" / "elevation" / "aspect").mkdir(parents=True)
    (project_dir / "data" / "attributes" / "dem").mkdir(parents=True)
    (project_dir / "data" / "forcing").mkdir(parents=True)
    (project_dir / "data" / "observations" / "streamflow" / "preprocessed").mkdir(parents=True)
    (project_dir / "cache" / "glhymps").mkdir(parents=True)
    (project_dir / "cache" / "landcover").mkdir(parents=True)

    return project_dir


@pytest.fixture
def base_config(temp_project_dir):
    """Basic configuration for attributeProcessor."""
    return {
        "DOMAIN_NAME": "test_domain",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "CATCHMENT_PATH": str(temp_project_dir / "shapefiles" / "catchment"),
        "CATCHMENT_SHP_NAME": "test_domain_catchment.shp",
        "CATCHMENT_SHP_HRUID": "hru_id",
        "DEM_SOURCE": "test",
        "FORCING_TIME_STEP_SIZE": 86400,
        "SYMFLUENCE_DATA_DIR": str(temp_project_dir.parent),
    }


@pytest.fixture
def distributed_config(base_config):
    """Configuration for distributed HRU processing."""
    config = base_config.copy()
    config["DOMAIN_DEFINITION_METHOD"] = "distributed"
    return config


@pytest.fixture
def test_logger():
    """Test logger instance."""
    logger = logging.getLogger("test_attribute_processor")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def lumped_catchment_shapefile(temp_project_dir, base_config):
    """Create a simple lumped catchment shapefile."""
    catchment_path = Path(base_config["CATCHMENT_PATH"])
    shp_name = base_config["CATCHMENT_SHP_NAME"]

    # Create a simple rectangular polygon
    geom = box(-121.8, 46.7, -121.7, 46.8)  # Small bbox near Paradise
    gdf = gpd.GeoDataFrame(
        {"basin_id": [1], "area_km2": [10.5], "geometry": [geom]},
        crs="EPSG:4326"
    )

    output_path = catchment_path / shp_name
    gdf.to_file(output_path)

    return output_path


@pytest.fixture
def distributed_catchment_shapefile(temp_project_dir, distributed_config):
    """Create a distributed catchment shapefile with 5 HRUs."""
    catchment_path = Path(distributed_config["CATCHMENT_PATH"])
    shp_name = distributed_config["CATCHMENT_SHP_NAME"]

    # Create 5 small HRUs in a grid
    hrus = []
    for i in range(5):
        lon_offset = (i % 3) * 0.03
        lat_offset = (i // 3) * 0.03
        geom = box(-121.8 + lon_offset, 46.7 + lat_offset,
                   -121.77 + lon_offset, 46.73 + lat_offset)
        hrus.append({
            "basin_id": 1,
            "hru_id": i + 1,
            "area_km2": 2.1,
            "geometry": geom
        })

    gdf = gpd.GeoDataFrame(hrus, crs="EPSG:4326")

    output_path = catchment_path / shp_name
    gdf.to_file(output_path)

    return output_path


@pytest.fixture
def mock_dem_file(temp_project_dir):
    """Create a small synthetic DEM file."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not available")

    dem_dir = temp_project_dir / "data" / "attributes" / "dem"
    dem_file = dem_dir / "test_dem.tif"

    # Create a simple 10x10 DEM with some elevation gradient
    width, height = 10, 10
    elevation = np.arange(100, 100 + width * height, dtype=np.float32).reshape(height, width)

    # Bounds matching the lumped catchment
    transform = from_bounds(-121.8, 46.7, -121.7, 46.8, width, height)

    with rasterio.open(
        dem_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(elevation, 1)

    return dem_file


@pytest.fixture
def mock_streamflow_data(temp_project_dir):
    """Create synthetic streamflow data CSV."""
    obs_dir = temp_project_dir / "data" / "observations" / "streamflow" / "preprocessed"
    csv_file = obs_dir / "test_domain_streamflow_processed.csv"

    # Create 1 year of daily data with realistic patterns
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # Synthetic flow: baseflow + seasonal variation + random noise
    baseflow = 5.0
    seasonal = 10.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 1, len(dates))
    flow = baseflow + seasonal + noise
    flow = np.maximum(flow, 0.1)  # Ensure positive flows

    df = pd.DataFrame({
        "date": dates,
        "flow_cms": flow
    })

    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_temperature_data(temp_project_dir):
    """Create synthetic temperature data CSV."""
    forcing_dir = temp_project_dir / "data" / "forcing"
    csv_file = forcing_dir / "test_domain_temperature.csv"

    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # Synthetic temperature: annual cycle
    mean_temp = 10.0
    amplitude = 15.0
    temp = mean_temp + amplitude * np.sin(2 * np.pi * np.arange(len(dates)) / 365 - np.pi/2)

    df = pd.DataFrame({
        "date": dates,
        "temperature_C": temp
    })

    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_precipitation_data(temp_project_dir):
    """Create synthetic precipitation data CSV."""
    forcing_dir = temp_project_dir / "data" / "forcing"
    csv_file = forcing_dir / "test_domain_precipitation.csv"

    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # Synthetic precipitation: seasonal + random events
    seasonal = 2.0 * (1 + np.sin(2 * np.pi * np.arange(len(dates)) / 365))
    events = np.random.exponential(1.0, len(dates)) * (np.random.rand(len(dates)) > 0.7)
    precip = seasonal + events

    df = pd.DataFrame({
        "date": dates,
        "precipitation_mm": precip
    })

    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_zonal_stats_result():
    """Mock output from rasterstats.zonal_stats."""
    return [{
        "min": 100.0,
        "max": 200.0,
        "mean": 150.0,
        "median": 148.0,
        "std": 25.0,
        "count": 100
    }]


@pytest.fixture
def mock_zonal_stats_aspect():
    """Mock zonal stats for aspect (circular statistics required)."""
    # Aspect values in degrees (0-360)
    return [{
        "min": 0.0,
        "max": 359.0,
        "mean": 180.0,  # Not valid for circular data
        "median": 175.0,
        "std": 90.0,
        "count": 100,
        # Mock the actual pixel values for circular stats
        "_values": np.array([0, 45, 90, 135, 180, 225, 270, 315] * 12 + [0, 45, 90, 135])
    }]


@pytest.fixture
def mock_empty_zonal_stats():
    """Mock zonal stats with no data (masked/nodata pixels)."""
    return [{
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "count": 0
    }]


# Helper functions for test data generation

def create_test_raster(output_path, data, bounds, crs="EPSG:4326", nodata=-9999):
    """Helper to create a test raster file."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not available")

    height, width = data.shape
    transform = from_bounds(*bounds, width, height)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)

    return output_path


def create_test_shapefile(output_path, geometries, attributes, crs="EPSG:4326"):
    """Helper to create a test shapefile."""
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
    gdf.to_file(output_path)
    return output_path
