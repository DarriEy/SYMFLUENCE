# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRS (Coordinate Reference System) utilities for geofabric processing.

Provides CRS consistency checking and pour point location finding.
Eliminates code duplication across GeofabricDelineator and GeofabricSubsetter.

Refactored from geofabric_utils.py (2026-01-01)
"""

import warnings
from typing import Any, Optional, Tuple

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
        id_col: str = 'GRU_ID',
        logger: Optional[Any] = None,
    ) -> Any:
        """
        Find the basin containing the pour point.

        Uses spatial join to find which basin polygon contains the pour point.
        Falls back to nearest-basin matching if the pour point does not fall
        strictly within any basin polygon (e.g., pour point on a boundary,
        small geometric precision issues, or very small basins).

        Args:
            pour_point: Pour point GeoDataFrame (single point)
            basins: Basin GeoDataFrame
            id_col: Column name for basin ID (default: 'GRU_ID')
            logger: Optional logger for diagnostic messages

        Returns:
            Basin ID containing (or nearest to) the pour point

        Raises:
            ValueError: If no basin contains or is near the pour point
        """
        # Primary: spatial join with 'within' predicate
        containing_basin = gpd.sjoin(pour_point, basins, how='left', predicate='within')

        if not containing_basin.empty:
            basin_id = containing_basin.iloc[0][id_col]
            # Check for NaN (sjoin returns NaN when no match found with how='left')
            if basin_id == basin_id:  # NaN != NaN
                return basin_id

        # Fallback: find the nearest basin polygon
        if logger:
            logger.warning(
                "Pour point does not fall within any basin polygon. "
                "Falling back to nearest-basin matching."
            )

        pour_point_geom = pour_point.geometry.iloc[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*geographic CRS.*")
            distances = basins.geometry.distance(pour_point_geom)
        nearest_idx = distances.idxmin()
        nearest_distance = distances[nearest_idx]
        basin_id = basins.loc[nearest_idx, id_col]

        if logger:
            logger.info(
                f"Nearest basin {basin_id} is {nearest_distance:.6f} degrees "
                f"from pour point (approx. {nearest_distance * 111_000:.0f} m)."
            )

        return basin_id
