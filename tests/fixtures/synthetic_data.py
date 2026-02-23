"""
Synthetic Data Generators for Acquisition Handler Tests.

Provides domain-specific synthetic data generators for testing:
- era5_dataset(): 4D forcing data matching ERA5 schema
- grace_dataset(): GRACE TWS anomaly data
- soil_moisture_dataset(): SMAP/ESA CCI style data
- dem_array(): Elevation raster data
- forcing_dataset(): Generic meteorological forcing data
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# =============================================================================
# ERA5 Synthetic Data
# =============================================================================

def era5_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-01-31"),
    resolution: float = 0.25,
    hourly: bool = True,
    variables: List[str] = None
) -> xr.Dataset:
    """
    Generate synthetic ERA5-like forcing dataset.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees (default: 0.25)
        hourly: If True, hourly timesteps; if False, daily
        variables: List of variable names (default: ERA5 forcing vars)

    Returns:
        xarray Dataset with ERA5-like structure
    """
    if variables is None:
        variables = [
            "t2m",      # 2m temperature (K)
            "d2m",      # 2m dewpoint temperature (K)
            "sp",       # Surface pressure (Pa)
            "u10",      # 10m u-wind (m/s)
            "v10",      # 10m v-wind (m/s)
            "tp",       # Total precipitation (m)
            "ssrd",     # Surface solar radiation downward (J/m²)
            "strd",     # Surface thermal radiation downward (J/m²)
        ]

    # Create coordinate arrays
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    freq = "H" if hourly else "D"
    time = pd.date_range(time_range[0], time_range[1], freq=freq)

    # Generate synthetic data for each variable
    data_vars = {}
    for var in variables:
        # Create realistic-ish synthetic data based on variable
        shape = (len(time), len(lat), len(lon))

        if var == "t2m":
            # Temperature: ~280K with diurnal cycle
            base = 280 + 10 * np.sin(2 * np.pi * np.arange(len(time)) / (24 if hourly else 1))
            data = base[:, np.newaxis, np.newaxis] + np.random.normal(0, 2, shape)
        elif var == "d2m":
            # Dewpoint: slightly lower than T2m
            base = 275 + 8 * np.sin(2 * np.pi * np.arange(len(time)) / (24 if hourly else 1))
            data = base[:, np.newaxis, np.newaxis] + np.random.normal(0, 2, shape)
        elif var == "sp":
            # Surface pressure: ~101000 Pa
            data = 101000 + np.random.normal(0, 500, shape)
        elif var in ("u10", "v10"):
            # Wind components: small values
            data = np.random.normal(0, 3, shape)
        elif var == "tp":
            # Precipitation: mostly zeros with some events
            data = np.maximum(0, np.random.exponential(0.001, shape))
        elif var in ("ssrd", "strd"):
            # Radiation: positive values with diurnal cycle
            if var == "ssrd":
                cycle = np.maximum(0, np.sin(2 * np.pi * (np.arange(len(time)) - 6) / 24))
                base = 800 * cycle if hourly else 400
            else:
                base = 300
            data = base * np.ones(shape) * (0.8 + 0.4 * np.random.random(shape))
        else:
            # Generic variable
            data = np.random.random(shape)

        data_vars[var] = (["time", "latitude", "longitude"], data.astype(np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "latitude": lat[::-1],  # ERA5 uses descending latitude
            "longitude": lon,
        }
    )

    # Add attributes matching ERA5 style
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["history"] = "Synthetic ERA5 test data"

    return ds


# =============================================================================
# GRACE Synthetic Data
# =============================================================================

def grace_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-12-31"),
    resolution: float = 0.5,
) -> xr.Dataset:
    """
    Generate synthetic GRACE-like TWS anomaly dataset.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees (default: 0.5)

    Returns:
        xarray Dataset with GRACE-like structure
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    time = pd.date_range(time_range[0], time_range[1], freq="MS")

    shape = (len(time), len(lat), len(lon))

    # TWS anomaly with seasonal cycle (cm)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(time)) / 12)
    tws = seasonal[:, np.newaxis, np.newaxis] + np.random.normal(0, 2, shape)

    # Uncertainty
    uncertainty = 1 + 0.5 * np.random.random(shape)

    ds = xr.Dataset(
        {
            "lwe_thickness": (["time", "lat", "lon"], tws.astype(np.float32)),
            "uncertainty": (["time", "lat", "lon"], uncertainty.astype(np.float32)),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["institution"] = "Synthetic GRACE test data"
    ds["lwe_thickness"].attrs["units"] = "cm"
    ds["lwe_thickness"].attrs["long_name"] = "Liquid Water Equivalent Thickness"

    return ds


# =============================================================================
# Soil Moisture Synthetic Data
# =============================================================================

def soil_moisture_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-01-31"),
    resolution: float = 0.1,
    dataset_type: str = "SMAP",  # SMAP or ESA_CCI
) -> xr.Dataset:
    """
    Generate synthetic soil moisture dataset (SMAP or ESA CCI style).

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees
        dataset_type: "SMAP" or "ESA_CCI"

    Returns:
        xarray Dataset with soil moisture data
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    time = pd.date_range(time_range[0], time_range[1], freq="D")

    shape = (len(time), len(lat), len(lon))

    # Soil moisture: 0-0.5 m³/m³ typical range
    sm = 0.25 + 0.1 * np.sin(2 * np.pi * np.arange(len(time)) / 30)
    soil_moisture = np.clip(
        sm[:, np.newaxis, np.newaxis] + np.random.normal(0, 0.05, shape),
        0, 0.5
    )

    if dataset_type == "SMAP":
        var_name = "soil_moisture"
        coord_names = {"lat": "latitude", "lon": "longitude"}
    else:
        var_name = "sm"
        coord_names = {"lat": "lat", "lon": "lon"}

    ds = xr.Dataset(
        {
            var_name: (["time", coord_names["lat"], coord_names["lon"]], soil_moisture.astype(np.float32)),
            "sm_uncertainty": (["time", coord_names["lat"], coord_names["lon"]],
                              (0.02 + 0.01 * np.random.random(shape)).astype(np.float32)),
        },
        coords={
            "time": time,
            coord_names["lat"]: lat,
            coord_names["lon"]: lon,
        }
    )

    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["source"] = f"Synthetic {dataset_type} test data"
    ds[var_name].attrs["units"] = "m3 m-3"
    ds[var_name].attrs["long_name"] = "Volumetric soil moisture"

    return ds


# =============================================================================
# DEM Synthetic Data
# =============================================================================

def dem_array(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    resolution: float = 0.001,  # ~100m
    elevation_range: Tuple[float, float] = (500, 2000),
) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic DEM elevation data.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        resolution: Grid resolution in degrees
        elevation_range: (min, max) elevation in meters

    Returns:
        Tuple of (elevation_array, metadata_dict)
    """
    height = int((lat_range[1] - lat_range[0]) / resolution)
    width = int((lon_range[1] - lon_range[0]) / resolution)

    # Create synthetic terrain with some structure
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Combine sine waves for terrain-like structure
    elevation = (
        elevation_range[0] +
        (elevation_range[1] - elevation_range[0]) * (
            0.5 +
            0.3 * np.sin(xx) * np.sin(yy) +
            0.2 * np.sin(2 * xx + 0.5) * np.cos(yy)
        )
    )

    # Add noise
    elevation += np.random.normal(0, 10, (height, width))

    metadata = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": (resolution, 0, lon_range[0], 0, -resolution, lat_range[1]),
    }

    return elevation.astype(np.float32), metadata


def dem_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    resolution: float = 0.01,
    elevation_range: Tuple[float, float] = (500, 2000),
) -> xr.Dataset:
    """
    Generate synthetic DEM as xarray Dataset.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        resolution: Grid resolution in degrees
        elevation_range: (min, max) elevation in meters

    Returns:
        xarray Dataset with elevation data
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)

    # Create synthetic terrain
    xx, yy = np.meshgrid(lon, lat)
    elevation = (
        elevation_range[0] +
        (elevation_range[1] - elevation_range[0]) * (
            0.5 + 0.3 * np.sin(4 * np.pi * (xx - lon_range[0]) / (lon_range[1] - lon_range[0]))
        )
    )
    elevation += np.random.normal(0, 10, elevation.shape)

    ds = xr.Dataset(
        {
            "elevation": (["lat", "lon"], elevation.astype(np.float32)),
        },
        coords={
            "lat": lat,
            "lon": lon,
        }
    )

    ds.attrs["Conventions"] = "CF-1.6"
    ds["elevation"].attrs["units"] = "m"
    ds["elevation"].attrs["long_name"] = "Surface elevation"

    return ds


# =============================================================================
# Generic Forcing Dataset
# =============================================================================

def forcing_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-01-31"),
    resolution: float = 0.25,
    freq: str = "H",
    lat_name: str = "lat",
    lon_name: str = "lon",
    lat_descending: bool = False,
    lon_0_360: bool = False,
) -> xr.Dataset:
    """
    Generate generic forcing dataset with configurable conventions.

    Useful for testing spatial subset operations with various coordinate conventions.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees
        freq: Time frequency (H=hourly, D=daily)
        lat_name: Name for latitude coordinate
        lon_name: Name for longitude coordinate
        lat_descending: If True, latitude in descending order
        lon_0_360: If True, convert longitude to 0-360 range

    Returns:
        xarray Dataset
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    time = pd.date_range(time_range[0], time_range[1], freq=freq)

    if lat_descending:
        lat = lat[::-1]

    if lon_0_360:
        lon = np.where(lon < 0, lon + 360, lon)

    shape = (len(time), len(lat), len(lon))

    # Generic forcing variable
    data = np.random.random(shape).astype(np.float32)

    ds = xr.Dataset(
        {
            "var": (["time", lat_name, lon_name], data),
        },
        coords={
            "time": time,
            lat_name: lat,
            lon_name: lon,
        }
    )

    return ds


# =============================================================================
# Evapotranspiration Synthetic Data
# =============================================================================

def evapotranspiration_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-12-31"),
    resolution: float = 0.05,
    dataset_type: str = "FLUXCOM",  # FLUXCOM or MODIS
) -> xr.Dataset:
    """
    Generate synthetic evapotranspiration dataset.

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees
        dataset_type: "FLUXCOM" or "MODIS"

    Returns:
        xarray Dataset with ET data
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)

    if dataset_type == "MODIS":
        # MODIS MOD16: 8-day composites
        time = pd.date_range(time_range[0], time_range[1], freq="8D")
        var_name = "ET"
        units = "kg/m²/8day"
        scale_factor = 10.0
    else:
        # FLUXCOM: monthly
        time = pd.date_range(time_range[0], time_range[1], freq="MS")
        var_name = "LE"
        units = "W/m²"
        scale_factor = 50.0

    shape = (len(time), len(lat), len(lon))

    # ET with seasonal cycle
    doy = np.arange(len(time)) / len(time) * 365
    seasonal = scale_factor * (1 + 0.8 * np.sin(2 * np.pi * doy / 365))
    et_data = np.maximum(0, seasonal[:, np.newaxis, np.newaxis] + np.random.normal(0, 5, shape))

    ds = xr.Dataset(
        {
            var_name: (["time", "lat", "lon"], et_data.astype(np.float32)),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    ds.attrs["source"] = f"Synthetic {dataset_type} test data"
    ds[var_name].attrs["units"] = units
    ds[var_name].attrs["long_name"] = "Evapotranspiration"

    return ds


# =============================================================================
# Snow Cover Synthetic Data
# =============================================================================

def snow_cover_dataset(
    lat_range: Tuple[float, float] = (46.0, 47.0),
    lon_range: Tuple[float, float] = (8.0, 9.0),
    time_range: Tuple[str, str] = ("2020-01-01", "2020-03-31"),
    resolution: float = 0.005,  # ~500m for MODIS
) -> xr.Dataset:
    """
    Generate synthetic snow cover fraction dataset (MODIS-like).

    Args:
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        time_range: (start, end) date strings
        resolution: Grid resolution in degrees

    Returns:
        xarray Dataset with snow cover fraction
    """
    lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    time = pd.date_range(time_range[0], time_range[1], freq="D")

    shape = (len(time), len(lat), len(lon))

    # Snow cover fraction with elevation dependence and seasonal melt
    base_elevation_effect = 0.5  # Higher = more snow
    melt_progress = np.linspace(1, 0, len(time))  # Progressive melt

    sca = np.clip(
        melt_progress[:, np.newaxis, np.newaxis] * base_elevation_effect +
        np.random.normal(0, 0.1, shape),
        0, 1
    )

    ds = xr.Dataset(
        {
            "NDSI_Snow_Cover": (["time", "lat", "lon"], (sca * 100).astype(np.int8)),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    ds.attrs["source"] = "Synthetic MODIS snow test data"
    ds["NDSI_Snow_Cover"].attrs["units"] = "%"
    ds["NDSI_Snow_Cover"].attrs["long_name"] = "NDSI Snow Cover"
    ds["NDSI_Snow_Cover"].attrs["valid_range"] = [0, 100]

    return ds


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def synthetic_era5():
    """Fixture providing a synthetic ERA5 dataset."""
    return era5_dataset()


@pytest.fixture
def synthetic_era5_daily():
    """Fixture providing a synthetic ERA5 daily dataset."""
    return era5_dataset(hourly=False)


@pytest.fixture
def synthetic_grace():
    """Fixture providing a synthetic GRACE dataset."""
    return grace_dataset()


@pytest.fixture
def synthetic_soil_moisture_smap():
    """Fixture providing a synthetic SMAP dataset."""
    return soil_moisture_dataset(dataset_type="SMAP")


@pytest.fixture
def synthetic_soil_moisture_esa():
    """Fixture providing a synthetic ESA CCI SM dataset."""
    return soil_moisture_dataset(dataset_type="ESA_CCI")


@pytest.fixture
def synthetic_dem():
    """Fixture providing a synthetic DEM dataset."""
    return dem_dataset()


@pytest.fixture
def synthetic_forcing_descending_lat():
    """Fixture providing forcing data with descending latitude."""
    return forcing_dataset(lat_descending=True)


@pytest.fixture
def synthetic_forcing_lon_360():
    """Fixture providing forcing data with 0-360 longitude."""
    return forcing_dataset(lon_0_360=True, lon_range=(-10.0, 10.0))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "era5_dataset",
    "grace_dataset",
    "soil_moisture_dataset",
    "dem_array",
    "dem_dataset",
    "forcing_dataset",
    "evapotranspiration_dataset",
    "snow_cover_dataset",
]
