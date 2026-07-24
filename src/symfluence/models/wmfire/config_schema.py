# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for WMFIRE.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from symfluence.core.config.models.base import FROZEN_CONFIG


class WMFireConfig(BaseModel):
    """WMFire wildfire spread module configuration for RHESSys.

    WMFire is a fire spread model that couples with RHESSys to simulate
    wildfire spread based on fuel loads, moisture, wind, and topography.

    Reference:
        Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. 2017.
        Balancing uncertainty and complexity to incorporate fire spread in
        an eco-hydrological model. International Journal of Wildland Fire.
    """
    model_config = FROZEN_CONFIG

    # Grid resolution and timestep
    grid_resolution: int = Field(
        default=30,
        alias='WMFIRE_GRID_RESOLUTION',
        ge=10,
        le=200,
        description='Fire grid cell resolution in meters (30, 60, or 90 recommended)'
    )
    timestep_hours: int = Field(
        default=24,
        alias='WMFIRE_TIMESTEP_HOURS',
        ge=1,
        le=24,
        description='Fire spread timestep in hours (1-24)'
    )

    # Fuel and moisture configuration
    ndays_average: float = Field(
        default=30.0,
        alias='WMFIRE_NDAYS_AVERAGE',
        ge=1.0,
        le=365.0,
        description='Fuel moisture averaging window in days'
    )
    fuel_source: Literal['static', 'rhessys_litter'] = Field(
        default='static',
        alias='WMFIRE_FUEL_SOURCE',
        description='Source of fuel load data: static values or RHESSys litter pools'
    )
    moisture_source: Literal['static', 'rhessys_soil'] = Field(
        default='static',
        alias='WMFIRE_MOISTURE_SOURCE',
        description='Source of moisture data: static values or RHESSys soil moisture'
    )
    carbon_to_fuel_ratio: float = Field(
        default=2.0,
        alias='WMFIRE_CARBON_TO_FUEL_RATIO',
        ge=1.0,
        le=5.0,
        description='Conversion factor from kg carbon to kg fuel'
    )

    # Ignition configuration
    ignition_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_SHAPEFILE',
        description='Path to ignition point shapefile (overrides ignition_point if set)'
    )
    ignition_point: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_POINT',
        description='Ignition point as "lat/lon" (e.g., "51.2096/-115.7539")'
    )
    ignition_date: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_DATE',
        description='Ignition date as "YYYY-MM-DD" for fire simulation start'
    )
    ignition_name: Optional[str] = Field(
        default='ignition',
        alias='WMFIRE_IGNITION_NAME',
        description='Name for the ignition point (used in output shapefile)'
    )

    # Fire perimeter validation
    perimeter_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_SHAPEFILE',
        description='Path to observed fire perimeter shapefile for validation'
    )
    perimeter_dir: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_DIR',
        description='Directory containing fire perimeter shapefiles for comparison'
    )

    # Output options
    write_geotiff: bool = Field(
        default=True,
        alias='WMFIRE_WRITE_GEOTIFF',
        description='Write georeferenced GeoTIFF outputs for visualization'
    )

    # Optional coefficient overrides (None = use defaults from fire.def)
    load_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K1',
        description='Fuel load coefficient k1 (default 3.9)'
    )
    load_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K2',
        description='Fuel load coefficient k2 (default 0.07)'
    )
    moisture_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K1',
        description='Moisture coefficient k1 (default 3.8)'
    )
    moisture_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K2',
        description='Moisture coefficient k2 (default 0.27)'
    )
    ign_def_mod: Optional[float] = Field(
        default=None,
        alias='WMFIRE_IGN_DEF_MOD',
        description='Ignition probability modifier (default 1.0, increase for more fires)'
    )
    mean_ign: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MEAN_IGN',
        description='Mean ignition events per timestep (default 1.0)'
    )
    windmax: Optional[float] = Field(
        default=None,
        alias='WMFIRE_WINDMAX',
        description='Maximum wind speed multiplier (default 1.0)'
    )
    slope_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_SLOPE_K1',
        description='Slope effect coefficient k1 (default 0.91)'
    )

    @field_validator('grid_resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate grid resolution is reasonable."""
        recommended = [30, 60, 90]
        if v not in recommended:
            import warnings
            warnings.warn(
                f"Grid resolution {v}m is not standard. "
                f"Recommended values: {recommended}"
            )
        return v

    @field_validator('ignition_point')
    @classmethod
    def validate_ignition_point(cls, v):
        """Validate ignition point format."""
        if v is not None:
            parts = v.split('/')
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid ignition_point format: {v}. "
                    f"Expected 'lat/lon' (e.g., '51.2096/-115.7539')"
                )
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude {lat} out of range [-90, 90]")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude {lon} out of range [-180, 180]")
            except ValueError as e:
                raise ValueError(f"Invalid ignition_point coordinates: {e}") from e
        return v
