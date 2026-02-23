"""
I/O utilities for geofabric processing.

Provides file loading and saving operations for geofabric data.
Eliminates code duplication across GeofabricDelineator, GeofabricSubsetter,
and LumpedWatershedDelineator classes.

Refactored from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Any

import geopandas as gpd


class GeofabricIOUtils:
    """
    File I/O operations for geofabric data.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def load_geopandas(path: Path, logger: Any) -> gpd.GeoDataFrame:
        """
        Load geospatial data into a GeoDataFrame with CRS validation.

        Supports shapefiles (.shp), GeoPackage (.gpkg), GeoParquet (.parquet),
        and any other format supported by geopandas.read_file().

        Automatically sets CRS to EPSG:4326 if undefined.

        Args:
            path: Path to geospatial file
            logger: Logger instance for warnings

        Returns:
            GeoDataFrame with validated CRS

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Geospatial file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == '.parquet':
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path)

        if gdf.crs is None:
            logger.warning(f"CRS is not defined for {path}. Setting to EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")

        return gdf

    @staticmethod
    def save_geofabric(
        basins: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame,
        basins_path: Path,
        rivers_path: Path,
        logger: Any
    ):
        """
        Save basin and river shapefiles.

        Creates parent directories if they don't exist.

        Args:
            basins: Basin GeoDataFrame
            rivers: River network GeoDataFrame
            basins_path: Output path for basins shapefile
            rivers_path: Output path for rivers shapefile
            logger: Logger instance for info messages
        """
        # Create parent directories
        basins_path.parent.mkdir(parents=True, exist_ok=True)
        rivers_path.parent.mkdir(parents=True, exist_ok=True)

        # Save files
        basins.to_file(basins_path)
        rivers.to_file(rivers_path)

        logger.info(f"Saved basins shapefile: {basins_path}")
        logger.info(f"Saved rivers shapefile: {rivers_path}")
