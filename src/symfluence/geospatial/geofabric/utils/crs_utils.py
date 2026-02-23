"""
CRS (Coordinate Reference System) utilities for geofabric processing.

Provides CRS consistency checking and pour point location finding.
Eliminates code duplication across GeofabricDelineator and GeofabricSubsetter.

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any, Tuple

import geopandas as gpd


class CRSUtils:
    """
    Coordinate reference system utilities.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def ensure_crs_consistency(
        basins: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame,
        pour_point: gpd.GeoDataFrame,
        logger: Any
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Ensure CRS consistency across all GeoDataFrames.

        Transforms all GeoDataFrames to a common CRS. Priority order:
        1. Basins CRS
        2. Rivers CRS
        3. Pour point CRS
        4. Default to EPSG:4326

        Args:
            basins: Basin GeoDataFrame
            rivers: River network GeoDataFrame
            pour_point: Pour point GeoDataFrame
            logger: Logger instance for info messages

        Returns:
            Tuple of (basins, rivers, pour_point) with consistent CRS
        """
        # Determine target CRS
        target_crs = basins.crs or rivers.crs or pour_point.crs or "EPSG:4326"
        logger.info(f"Ensuring CRS consistency. Target CRS: {target_crs}")

        # Transform all to target CRS
        if basins.crs != target_crs:
            basins = basins.to_crs(target_crs)
        if rivers.crs != target_crs:
            rivers = rivers.to_crs(target_crs)
        if pour_point.crs != target_crs:
            pour_point = pour_point.to_crs(target_crs)

        return basins, rivers, pour_point

    @staticmethod
    def find_basin_for_pour_point(
        pour_point: gpd.GeoDataFrame,
        basins: gpd.GeoDataFrame,
        id_col: str = 'GRU_ID'
    ) -> Any:
        """
        Find the basin containing the pour point.

        Uses spatial join to find which basin polygon contains the pour point.

        Args:
            pour_point: Pour point GeoDataFrame (single point)
            basins: Basin GeoDataFrame
            id_col: Column name for basin ID (default: 'GRU_ID')

        Returns:
            Basin ID containing the pour point

        Raises:
            ValueError: If no basin contains the pour point
        """
        # Spatial join to find containing basin
        containing_basin = gpd.sjoin(pour_point, basins, how='left', predicate='within')

        if containing_basin.empty:
            raise ValueError("No basin contains the given pour point.")

        # Return the basin ID from the first match
        return containing_basin.iloc[0][id_col]
