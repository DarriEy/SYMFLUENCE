"""
Input validation utilities for geofabric processing.

Provides validation functions for configuration and file inputs.

Refactored from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Dict, Any


class GeofabricValidator:
    """
    Validates inputs for geofabric delineation.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def validate_dem_exists(dem_path: Path):
        """
        Validate DEM file exists and has correct format.

        Args:
            dem_path: Path to DEM file

        Raises:
            FileNotFoundError: If DEM file doesn't exist
            ValueError: If DEM format is not GeoTIFF
        """
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        if not dem_path.suffix.lower() in ['.tif', '.tiff']:
            raise ValueError(f"DEM must be GeoTIFF format, got: {dem_path.suffix}")

    @staticmethod
    def validate_pour_point_exists(pour_point_path: Path):
        """
        Validate pour point shapefile exists.

        Args:
            pour_point_path: Path to pour point shapefile

        Raises:
            FileNotFoundError: If pour point file doesn't exist
        """
        if not pour_point_path.exists():
            raise FileNotFoundError(f"Pour point file not found: {pour_point_path}")

    @staticmethod
    def validate_hydrofabric_type(hydrofabric_type: str):
        """
        Validate hydrofabric type is supported.

        Args:
            hydrofabric_type: Type of hydrofabric (MERIT, TDX, NWS)

        Raises:
            ValueError: If hydrofabric type is not supported
        """
        valid_types = ['MERIT', 'TDX', 'NWS']
        if hydrofabric_type.upper() not in valid_types:
            raise ValueError(
                f"Unsupported hydrofabric type: {hydrofabric_type}. "
                f"Valid types: {valid_types}"
            )

    @staticmethod
    def validate_delineation_method(method: str):
        """
        Validate delineation method is supported.

        Args:
            method: Delineation method name

        Raises:
            ValueError: If delineation method is not supported
        """
        valid_methods = ['stream_threshold', 'curvature', 'slope_area', 'multi_scale']
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Unsupported delineation method: {method}. "
                f"Valid methods: {valid_methods}"
            )
