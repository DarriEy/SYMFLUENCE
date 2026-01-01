"""
Geospatial utility functions for hydrological modeling.

Provides common geospatial operations used across model preprocessors:
- Catchment centroid calculation with proper CRS handling
- Area calculations with automatic CRS detection
- Spatial aggregation utilities
"""

from pathlib import Path
from typing import Tuple, Optional
import logging

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class GeospatialUtilsMixin:
    """
    Mixin providing geospatial utility methods.

    Requires the following attributes in the class:
        - self.logger: logging.Logger (optional, but recommended)

    Provides:
        - Centroid calculation with proper CRS handling
        - Area calculations with automatic UTM projection
        - CRS validation and conversion utilities
    """

    def calculate_catchment_centroid(
        self,
        catchment_gdf: 'gpd.GeoDataFrame'
    ) -> Tuple[float, float]:
        """
        Calculate catchment centroid with proper CRS handling.

        Ensures accurate centroid calculation by:
        1. Detecting or assuming geographic CRS (EPSG:4326)
        2. Calculating appropriate UTM zone from bounds
        3. Projecting to UTM for accurate centroid
        4. Converting back to geographic coordinates (lon, lat)

        Args:
            catchment_gdf: GeoDataFrame of catchment polygon(s)

        Returns:
            Tuple of (longitude, latitude) in decimal degrees

        Raises:
            ImportError: If geopandas is not available

        Example:
            >>> centroid_lon, centroid_lat = self.calculate_catchment_centroid(catchment)
            >>> print(f"Centroid: {centroid_lon:.6f}°E, {centroid_lat:.6f}°N")
        """
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required for centroid calculation")

        # Ensure CRS is defined
        if catchment_gdf.crs is None:
            if hasattr(self, 'logger'):
                self.logger.warning("Catchment CRS not defined, assuming EPSG:4326")
            catchment_gdf = catchment_gdf.set_crs(epsg=4326)

        # Convert to geographic coordinates if not already
        catchment_geo = catchment_gdf.to_crs(epsg=4326)

        # Get approximate center from bounds (for UTM zone calculation)
        bounds = catchment_geo.total_bounds  # (minx, miny, maxx, maxy)
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

        # Calculate appropriate UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1

        # Northern hemisphere: 326xx, Southern: 327xx
        epsg_code = f"326{utm_zone:02d}" if center_lat >= 0 else f"327{utm_zone:02d}"

        # Project to UTM for accurate centroid calculation
        catchment_utm = catchment_geo.to_crs(f"EPSG:{epsg_code}")

        # Calculate centroid in projected coordinates
        centroid_utm = catchment_utm.geometry.centroid.iloc[0]

        # Create GeoDataFrame for reprojection
        centroid_gdf = gpd.GeoDataFrame(
            geometry=[centroid_utm],
            crs=f"EPSG:{epsg_code}"
        )

        # Convert back to geographic coordinates
        centroid_geo = centroid_gdf.to_crs(epsg=4326)

        # Extract coordinates
        lon = centroid_geo.geometry.x[0]
        lat = centroid_geo.geometry.y[0]

        if hasattr(self, 'logger'):
            self.logger.info(
                f"Calculated catchment centroid: {lon:.6f}°E, {lat:.6f}°N "
                f"(UTM Zone {utm_zone})"
            )

        return lon, lat

    def calculate_catchment_area_km2(
        self,
        catchment_gdf: 'gpd.GeoDataFrame'
    ) -> float:
        """
        Calculate total catchment area in km².

        Automatically detects appropriate UTM projection for accurate area calculation.

        Args:
            catchment_gdf: GeoDataFrame of catchment polygon(s)

        Returns:
            Total area in square kilometers

        Raises:
            ImportError: If geopandas is not available

        Example:
            >>> area = self.calculate_catchment_area_km2(catchment)
            >>> print(f"Catchment area: {area:.2f} km²")
        """
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required for area calculation")

        # Ensure CRS is defined
        if catchment_gdf.crs is None:
            if hasattr(self, 'logger'):
                self.logger.warning("Catchment CRS not defined, assuming EPSG:4326")
            catchment_gdf = catchment_gdf.set_crs(epsg=4326)

        # Use geopandas estimate_utm_crs() for automatic UTM selection
        try:
            utm_crs = catchment_gdf.estimate_utm_crs()
            catchment_proj = catchment_gdf.to_crs(utm_crs)
        except AttributeError:
            # Fallback for older geopandas versions
            catchment_geo = catchment_gdf.to_crs(epsg=4326)
            bounds = catchment_geo.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            epsg_code = f"326{utm_zone:02d}" if center_lat >= 0 else f"327{utm_zone:02d}"
            catchment_proj = catchment_geo.to_crs(f"EPSG:{epsg_code}")

        # Calculate area in m² and convert to km²
        area_km2 = catchment_proj.geometry.area.sum() / 1e6

        if hasattr(self, 'logger'):
            self.logger.info(f"Calculated catchment area: {area_km2:.2f} km²")

        return area_km2

    def validate_and_fix_crs(
        self,
        gdf: 'gpd.GeoDataFrame',
        assumed_epsg: int = 4326
    ) -> 'gpd.GeoDataFrame':
        """
        Validate GeoDataFrame CRS and assign default if missing.

        Args:
            gdf: GeoDataFrame to validate
            assumed_epsg: EPSG code to assume if CRS is missing (default: 4326)

        Returns:
            GeoDataFrame with valid CRS

        Example:
            >>> gdf = self.validate_and_fix_crs(my_geodataframe)
        """
        if gdf.crs is None:
            if hasattr(self, 'logger'):
                self.logger.warning(
                    f"CRS not defined, assuming EPSG:{assumed_epsg}"
                )
            return gdf.set_crs(epsg=assumed_epsg)
        return gdf
