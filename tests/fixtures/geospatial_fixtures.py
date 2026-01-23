"""
Geospatial Test Fixtures for Domain Delineation Tests.

Provides domain-specific synthetic data generators for testing geospatial operations:
- synthetic_watershed_gdf(): Basin polygon GeoDataFrames
- synthetic_river_network_gdf(): River network polyline GeoDataFrames
- synthetic_pour_point_gdf(): Pour point GeoDataFrames
- synthetic_forcing_netcdf_with_grid(): Forcing netCDF with grid coordinates

Following the pattern established in synthetic_data.py.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import geopandas and shapely with graceful fallback
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString, box
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    gpd = None


# =============================================================================
# Watershed GeoDataFrame Generator
# =============================================================================

def synthetic_watershed_gdf(
    n_basins: int = 5,
    lat_center: float = 46.5,
    lon_center: float = 8.5,
    basin_size_km: float = 10.0,
    crs: str = "EPSG:4326",
    include_attributes: bool = True,
) -> "gpd.GeoDataFrame":
    """
    Generate synthetic watershed basin GeoDataFrame.

    Creates a set of adjacent basin polygons suitable for testing delineation
    and spatial operations. Basins are arranged in a grid pattern.

    Args:
        n_basins: Number of basins to create (default: 5)
        lat_center: Center latitude for basin grid (default: 46.5)
        lon_center: Center longitude for basin grid (default: 8.5)
        basin_size_km: Approximate basin size in kilometers (default: 10.0)
        crs: Coordinate reference system (default: EPSG:4326)
        include_attributes: Include GRU_ID, GRU_area, etc. (default: True)

    Returns:
        GeoDataFrame with basin polygon geometries and attributes:
            - GRU_ID: Unique basin identifier (1-indexed)
            - GRU_area: Basin area in square meters
            - gru_to_seg: Segment mapping (same as GRU_ID)
            - center_lat: Basin centroid latitude
            - center_lon: Basin centroid longitude

    Example:
        >>> basins = synthetic_watershed_gdf(n_basins=5)
        >>> basins.columns.tolist()
        ['GRU_ID', 'GRU_area', 'gru_to_seg', 'center_lat', 'center_lon', 'geometry']
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for geospatial fixtures")

    # Convert km to degrees (approximate at mid-latitudes)
    # 1 degree latitude ~ 111 km, 1 degree longitude ~ 111 * cos(lat) km
    deg_lat = basin_size_km / 111.0
    deg_lon = basin_size_km / (111.0 * np.cos(np.radians(lat_center)))

    # Arrange basins in a roughly square grid
    n_cols = int(np.ceil(np.sqrt(n_basins)))
    n_rows = int(np.ceil(n_basins / n_cols))

    # Start from the center and expand outward
    start_lat = lat_center - (n_rows / 2) * deg_lat
    start_lon = lon_center - (n_cols / 2) * deg_lon

    geometries = []
    attributes = []

    for i in range(n_basins):
        row = i // n_cols
        col = i % n_cols

        # Create basin polygon
        min_lat = start_lat + row * deg_lat
        max_lat = start_lat + (row + 1) * deg_lat
        min_lon = start_lon + col * deg_lon
        max_lon = start_lon + (col + 1) * deg_lon

        polygon = box(min_lon, min_lat, max_lon, max_lat)
        geometries.append(polygon)

        if include_attributes:
            # Calculate approximate area in square meters
            # Using rough approximation: 1 deg^2 at 45 lat ~ 8.5e9 m^2
            area_deg2 = (max_lat - min_lat) * (max_lon - min_lon)
            area_m2 = area_deg2 * 8.5e9 * np.cos(np.radians(lat_center))

            attributes.append({
                'GRU_ID': i + 1,
                'GRU_area': area_m2,
                'gru_to_seg': i + 1,
                'center_lat': (min_lat + max_lat) / 2,
                'center_lon': (min_lon + max_lon) / 2,
            })

    if include_attributes:
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

    return gdf


# =============================================================================
# River Network GeoDataFrame Generator
# =============================================================================

def synthetic_river_network_gdf(
    basins_gdf: Optional["gpd.GeoDataFrame"] = None,
    n_segments: int = 5,
    lat_center: float = 46.5,
    lon_center: float = 8.5,
    crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """
    Generate synthetic river network GeoDataFrame.

    Creates river segments either based on provided basins (connecting centroids)
    or as a standalone dendritic network.

    Args:
        basins_gdf: Optional basin GeoDataFrame to derive network from
        n_segments: Number of river segments if basins not provided (default: 5)
        lat_center: Center latitude if basins not provided (default: 46.5)
        lon_center: Center longitude if basins not provided (default: 8.5)
        crs: Coordinate reference system (default: EPSG:4326)

    Returns:
        GeoDataFrame with river segment geometries and attributes:
            - LINKNO: Unique segment identifier (1-indexed)
            - DSLINKNO: Downstream segment ID (0 for outlet)
            - Length: Segment length in meters (approximate)
            - Slope: Segment slope (placeholder 0.01)
            - GRU_ID: Associated basin ID

    Example:
        >>> basins = synthetic_watershed_gdf(n_basins=5)
        >>> rivers = synthetic_river_network_gdf(basins)
        >>> rivers.columns.tolist()
        ['LINKNO', 'DSLINKNO', 'Length', 'Slope', 'GRU_ID', 'geometry']
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for geospatial fixtures")

    if basins_gdf is not None:
        # Create network based on basin centroids
        centroids = basins_gdf.geometry.centroid
        n_segments = len(basins_gdf)

        # Simple linear downstream topology: 1 -> 2 -> 3 -> ... -> outlet
        segments = []
        for i in range(n_segments):
            gru_id = basins_gdf.iloc[i].get('GRU_ID', i + 1)

            # For simplicity, create point geometry at centroid
            # (matching lumped delineator behavior)
            point = centroids.iloc[i]

            # Downstream link: next segment, or 0 for last (outlet)
            downstream_id = gru_id + 1 if i < n_segments - 1 else 0

            segments.append({
                'LINKNO': gru_id,
                'DSLINKNO': downstream_id if downstream_id <= n_segments else 0,
                'Length': 1000.0,  # Placeholder length in meters
                'Slope': 0.01,     # Placeholder slope
                'GRU_ID': gru_id,
                'geometry': point,
            })

        gdf = gpd.GeoDataFrame(segments, crs=crs)

    else:
        # Create standalone network
        deg_step = 0.01  # Approximately 1 km step

        segments = []
        for i in range(n_segments):
            lat = lat_center + i * deg_step
            lon = lon_center + i * deg_step * 0.5  # Slight eastward trend

            point = Point(lon, lat)
            downstream_id = i + 2 if i < n_segments - 1 else 0

            segments.append({
                'LINKNO': i + 1,
                'DSLINKNO': downstream_id,
                'Length': 1000.0,
                'Slope': 0.01,
                'GRU_ID': i + 1,
                'geometry': point,
            })

        gdf = gpd.GeoDataFrame(segments, crs=crs)

    return gdf


# =============================================================================
# Pour Point GeoDataFrame Generator
# =============================================================================

def synthetic_pour_point_gdf(
    lat: float = 46.5,
    lon: float = 8.5,
    crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """
    Generate synthetic pour point (outlet) GeoDataFrame.

    Creates a single-point GeoDataFrame representing a watershed outlet,
    suitable for delineation testing.

    Args:
        lat: Latitude of pour point (default: 46.5)
        lon: Longitude of pour point (default: 8.5)
        crs: Coordinate reference system (default: EPSG:4326)

    Returns:
        GeoDataFrame with single point geometry:
            - lat: Latitude value
            - lon: Longitude value
            - geometry: Point geometry

    Example:
        >>> pour_point = synthetic_pour_point_gdf(lat=46.5, lon=8.5)
        >>> pour_point.geometry[0].coords[0]
        (8.5, 46.5)
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for geospatial fixtures")

    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame(
        {'lat': [lat], 'lon': [lon]},
        geometry=[point],
        crs=crs,
    )

    return gdf


# =============================================================================
# Forcing NetCDF with Grid Generator
# =============================================================================

def synthetic_forcing_netcdf_with_grid(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    resolution: float = 0.25,
    time_range: Tuple[str, str] = ("2020-01-01", "2020-01-31"),
    dataset_type: str = "era5",
    output_path: Optional[Path] = None,
) -> Tuple[Optional[Path], xr.Dataset]:
    """
    Generate synthetic forcing netCDF with grid coordinates.

    Creates ERA5-like or other forcing data with explicit lat/lon coordinates
    suitable for testing native grid creation.

    Args:
        lat_range: (lat_min, lat_max) in degrees (default: (46.0, 47.0))
        lon_range: (lon_min, lon_max) in degrees (default: (8.0, 9.0))
        resolution: Grid resolution in degrees (default: 0.25)
        time_range: (start, end) date strings (default: ("2020-01-01", "2020-01-31"))
        dataset_type: Type of forcing data to mimic (default: "era5")
            - "era5": ERA5-like with latitude/longitude coords (lat descending)
            - "cerra": CERRA-like with lat/lon coords
            - "carra": CARRA-like with lat/lon coords
        output_path: Optional path to write netCDF file (default: None, no file written)

    Returns:
        Tuple of (output_path, xr.Dataset):
            - output_path: Path to written file (or None if not written)
            - dataset: xarray Dataset with forcing data

    Example:
        >>> path, ds = synthetic_forcing_netcdf_with_grid()
        >>> list(ds.coords)
        ['time', 'latitude', 'longitude']
        >>> ds.dims
        Frozen({'time': 31, 'latitude': 5, 'longitude': 5})
    """
    # Create coordinate arrays
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    time = pd.date_range(time_range[0], time_range[1], freq="D")

    # Dataset type-specific coordinate names and ordering
    if dataset_type == "era5":
        lat_name = "latitude"
        lon_name = "longitude"
        lat = lat[::-1]  # ERA5 uses descending latitude
    else:
        lat_name = "lat"
        lon_name = "lon"

    shape = (len(time), len(lat), len(lon))

    # Generate synthetic temperature data with diurnal variation
    t2m = 280 + 10 * np.sin(2 * np.pi * np.arange(len(time)) / 365)
    t2m = t2m[:, np.newaxis, np.newaxis] + np.random.normal(0, 2, shape)

    # Generate synthetic precipitation (mostly zeros with some events)
    tp = np.maximum(0, np.random.exponential(0.001, shape))

    ds = xr.Dataset(
        {
            "t2m": (["time", lat_name, lon_name], t2m.astype(np.float32)),
            "tp": (["time", lat_name, lon_name], tp.astype(np.float32)),
        },
        coords={
            "time": time,
            lat_name: lat,
            lon_name: lon,
        }
    )

    # Add attributes
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["history"] = f"Synthetic {dataset_type.upper()} test data"
    ds.attrs["source"] = "SYMFLUENCE test fixtures"
    ds["t2m"].attrs["units"] = "K"
    ds["t2m"].attrs["long_name"] = "2 metre temperature"
    ds["tp"].attrs["units"] = "m"
    ds["tp"].attrs["long_name"] = "Total precipitation"

    # Write to file if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)

    return output_path, ds


# =============================================================================
# DEM Raster Generator (for geospatial tests)
# =============================================================================

def synthetic_dem_raster(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    resolution: float = 0.001,  # ~100m
    elevation_range: Tuple[float, float] = (500, 2000),
    output_path: Optional[Path] = None,
) -> Tuple[Optional[Path], np.ndarray, Dict]:
    """
    Generate synthetic DEM raster for geospatial tests.

    Creates terrain-like elevation data suitable for testing TauDEM operations
    and watershed delineation.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        resolution: Grid resolution in degrees (default: 0.001 ~ 100m)
        elevation_range: (min, max) elevation in meters
        output_path: Optional path to write GeoTIFF file

    Returns:
        Tuple of (output_path, elevation_array, metadata):
            - output_path: Path to written file (or None)
            - elevation_array: 2D numpy array of elevations
            - metadata: Dictionary with raster metadata for rasterio

    Example:
        >>> path, elev, meta = synthetic_dem_raster()
        >>> elev.shape
        (1000, 1000)
        >>> meta['crs']
        'EPSG:4326'
    """
    height = int((lat_range[1] - lat_range[0]) / resolution)
    width = int((lon_range[1] - lon_range[0]) / resolution)

    # Create synthetic terrain with stream-like patterns
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Base terrain with valley structure (for realistic flow directions)
    # Valley runs from top-left to bottom-right
    valley_depth = 0.3 * (elevation_range[1] - elevation_range[0])
    valley = valley_depth * np.exp(-((xx - yy) ** 2) / 10)

    # Combine sine waves for ridge-like structure
    elevation = (
        elevation_range[0] +
        (elevation_range[1] - elevation_range[0]) * (
            0.5 +
            0.3 * np.sin(xx) * np.sin(yy) +
            0.2 * np.sin(2 * xx + 0.5) * np.cos(yy)
        ) - valley
    )

    # Add noise for realistic texture
    elevation += np.random.normal(0, 10, (height, width))

    # Ensure minimum elevation
    elevation = np.maximum(elevation, elevation_range[0])

    metadata = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": (resolution, 0, lon_range[0], 0, -resolution, lat_range[1]),
    }

    # Write to file if path provided and rasterio is available
    if output_path is not None:
        try:
            import rasterio
            from rasterio.transform import from_bounds

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            transform = from_bounds(
                lon_range[0], lat_range[0],
                lon_range[1], lat_range[1],
                width, height
            )

            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs='EPSG:4326',
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(elevation.astype(np.float32), 1)

        except ImportError:
            output_path = None

    return output_path, elevation.astype(np.float32), metadata


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def synthetic_watershed():
    """Fixture providing a synthetic watershed GeoDataFrame with 5 basins."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")
    return synthetic_watershed_gdf(n_basins=5)


@pytest.fixture
def synthetic_river_network(synthetic_watershed):
    """Fixture providing a synthetic river network derived from watershed basins."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")
    return synthetic_river_network_gdf(synthetic_watershed)


@pytest.fixture
def synthetic_pour_point():
    """Fixture providing a synthetic pour point GeoDataFrame."""
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")
    return synthetic_pour_point_gdf()


@pytest.fixture
def geospatial_test_domain(tmp_path):
    """
    Fixture providing a complete test domain with all geospatial data.

    Creates a temporary directory structure with:
    - DEM raster (if rasterio available)
    - Pour point shapefile
    - Config dictionary
    - Output directories

    Returns:
        Dictionary with:
            - 'tmp_path': Base temporary directory
            - 'dem_path': Path to DEM (or None)
            - 'pour_point_path': Path to pour point shapefile
            - 'config': Mock configuration dictionary
            - 'shapefiles_dir': Path to shapefiles directory
    """
    if not GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")

    # Create directory structure
    shapefiles_dir = tmp_path / "shapefiles"
    pour_point_dir = shapefiles_dir / "pour_point"
    river_basins_dir = shapefiles_dir / "river_basins"
    river_network_dir = shapefiles_dir / "river_network"
    dem_dir = tmp_path / "attributes" / "dem"
    forcing_dir = tmp_path / "forcing"

    for d in [pour_point_dir, river_basins_dir, river_network_dir, dem_dir, forcing_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create pour point shapefile
    pour_point = synthetic_pour_point_gdf(lat=46.5, lon=8.5)
    pour_point_path = pour_point_dir / "test_domain_pourPoint.shp"
    pour_point.to_file(pour_point_path)

    # Create DEM if rasterio available
    dem_path = dem_dir / "test_domain_dem.tif"
    dem_path, _, _ = synthetic_dem_raster(
        lat_range=(46.0, 47.0),
        lon_range=(8.0, 9.0),
        resolution=0.01,  # Coarser for faster tests
        output_path=dem_path,
    )

    # Create forcing data
    forcing_path = forcing_dir / "era5_test.nc"
    _, forcing_ds = synthetic_forcing_netcdf_with_grid(
        lat_range=(46.0, 47.0),
        lon_range=(8.0, 9.0),
        resolution=0.25,
        output_path=forcing_path,
    )

    # Create mock configuration
    config = {
        'DOMAIN_NAME': 'test_domain',
        'PROJECT_DIR': str(tmp_path),
        'DEM_PATH': str(dem_path) if dem_path else 'default',
        'POUR_POINT_SHP_PATH': str(pour_point_dir),
        'POUR_POINT_SHP_NAME': 'test_domain_pourPoint.shp',
        'BOUNDING_BOX_COORDS': '47.0/8.0/46.0/9.0',  # lat_max/lon_min/lat_min/lon_max
        'GRID_CELL_SIZE': 1000.0,
        'CLIP_GRID_TO_WATERSHED': True,
        'GRID_SOURCE': 'generate',
        'NATIVE_GRID_DATASET': 'era5',
        'NUM_PROCESSES': 1,
        'CLEANUP_INTERMEDIATE_FILES': True,
    }

    return {
        'tmp_path': tmp_path,
        'dem_path': dem_path,
        'pour_point_path': pour_point_path,
        'forcing_path': forcing_path,
        'config': config,
        'shapefiles_dir': shapefiles_dir,
    }


@pytest.fixture
def mock_config_dict():
    """
    Fixture providing a minimal mock configuration dictionary.

    Suitable for unit tests that don't need actual file paths.
    """
    return {
        'DOMAIN_NAME': 'test_domain',
        'PROJECT_DIR': '/tmp/test_project',
        'BOUNDING_BOX_COORDS': '47.0/8.0/46.0/9.0',
        'GRID_CELL_SIZE': 1000.0,
        'CLIP_GRID_TO_WATERSHED': True,
        'GRID_SOURCE': 'generate',
        'NATIVE_GRID_DATASET': 'era5',
        'NUM_PROCESSES': 1,
        'CLEANUP_INTERMEDIATE_FILES': True,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "synthetic_watershed_gdf",
    "synthetic_river_network_gdf",
    "synthetic_pour_point_gdf",
    "synthetic_forcing_netcdf_with_grid",
    "synthetic_dem_raster",
    "GEOPANDAS_AVAILABLE",
]
