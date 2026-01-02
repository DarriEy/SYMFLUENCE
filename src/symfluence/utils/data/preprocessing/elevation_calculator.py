"""
ElevationCalculator - DEM-based elevation statistics for forcing grids.

This module handles:
- Safe CRS-aligned elevation extraction from DEMs
- Batched processing for memory efficiency
- Zonal statistics using rasterstats

Extracted from ForcingResampler to improve testability and reduce coupling.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import geopandas as gpd
import rasterio
import rasterstats


class ElevationCalculator:
    """
    Calculates elevation statistics for geometries using DEM data.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize ElevationCalculator.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self._default_nodata = -9999

    def calculate_mean_elevation(
        self,
        gdf: gpd.GeoDataFrame,
        dem_path: Path,
        batch_size: int = 50,
        nodata_value: float = -9999
    ) -> List[float]:
        """
        Calculate mean elevation for each geometry in a GeoDataFrame.

        Handles CRS alignment and processes in batches for memory efficiency.

        Args:
            gdf: GeoDataFrame containing geometries
            dem_path: Path to DEM raster file
            batch_size: Number of geometries to process per batch
            nodata_value: Value to use for failed calculations

        Returns:
            List of elevation values corresponding to each geometry
        """
        self.logger.info(f"Calculating elevation statistics for {len(gdf)} geometries")

        elevations = [nodata_value] * len(gdf)

        try:
            # Get and align CRS
            gdf_projected = self._align_crs(gdf, dem_path)

            # Process in batches
            num_batches = (len(gdf_projected) + batch_size - 1) // batch_size
            self.logger.info(f"Processing elevation in {num_batches} batches of {batch_size}")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gdf_projected))

                self.logger.info(
                    f"Processing elevation batch {batch_idx+1}/{num_batches} "
                    f"({start_idx} to {end_idx-1})"
                )

                try:
                    batch_gdf = gdf_projected.iloc[start_idx:end_idx]

                    zs = rasterstats.zonal_stats(
                        batch_gdf.geometry,
                        str(dem_path),
                        stats=['mean'],
                        nodata=nodata_value
                    )

                    for i, item in enumerate(zs):
                        idx = start_idx + i
                        elevations[idx] = item['mean'] if item['mean'] is not None else nodata_value

                except Exception as e:
                    self.logger.warning(
                        f"Error calculating elevations for batch {batch_idx+1}: {str(e)}"
                    )

            valid_count = sum(1 for elev in elevations if elev != nodata_value)
            self.logger.info(
                f"Successfully calculated elevation for {valid_count}/{len(elevations)} geometries"
            )

        except Exception as e:
            self.logger.error(f"Error in elevation calculation: {str(e)}")
            elevations = [nodata_value] * len(gdf)

        return elevations

    def _align_crs(self, gdf: gpd.GeoDataFrame, dem_path: Path) -> gpd.GeoDataFrame:
        """
        Align GeoDataFrame CRS with DEM CRS.

        Args:
            gdf: Input GeoDataFrame
            dem_path: Path to DEM raster

        Returns:
            GeoDataFrame with CRS aligned to DEM
        """
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            self.logger.info(f"DEM CRS: {dem_crs}")

        shapefile_crs = gdf.crs
        self.logger.info(f"Shapefile CRS: {shapefile_crs}")

        if dem_crs != shapefile_crs:
            self.logger.info(
                f"CRS mismatch detected. Reprojecting from {shapefile_crs} to {dem_crs}"
            )
            try:
                gdf_projected = gdf.to_crs(dem_crs)
                self.logger.info("CRS reprojection successful")
                return gdf_projected
            except Exception as e:
                self.logger.error(f"Failed to reproject CRS: {str(e)}")
                self.logger.warning("Using original CRS - elevation calculation may fail")
                return gdf.copy()
        else:
            self.logger.info("CRS match - no reprojection needed")
            return gdf.copy()

    def get_nodata_value(self, raster_path: Path) -> float:
        """
        Get the nodata value from a raster file.

        Args:
            raster_path: Path to raster file

        Returns:
            Nodata value or default (-9999) if not specified
        """
        try:
            with rasterio.open(raster_path) as src:
                nodata = src.nodatavals[0]
                if nodata is None:
                    return self._default_nodata
                return nodata
        except Exception as e:
            self.logger.warning(f"Could not read nodata value: {e}")
            return self._default_nodata


def create_elevation_calculator(logger: logging.Logger) -> ElevationCalculator:
    """
    Factory function to create an ElevationCalculator.

    Args:
        logger: Logger instance

    Returns:
        Configured ElevationCalculator instance
    """
    return ElevationCalculator(logger)
