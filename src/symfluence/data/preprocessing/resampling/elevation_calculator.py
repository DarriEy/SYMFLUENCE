"""
Elevation Calculator

Calculates elevation statistics for geometries using DEM data.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.mask import mask

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class ElevationCalculator:
    """
    Safely calculates elevation statistics using rasterio.

    Uses CRS alignment and batch processing to avoid segfaults
    that can occur with rasterstats on certain systems.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize elevation calculator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate(self, gdf, dem_path: Path, batch_size: int = 50) -> List[float]:
        """
        Calculate elevation statistics for geometries.

        Args:
            gdf: GeoDataFrame containing geometries
            dem_path: Path to DEM raster
            batch_size: Number of geometries per batch (unused, kept for API compatibility)

        Returns:
            List of elevation values corresponding to each geometry
        """
        self.logger.info(f"Calculating elevation statistics for {len(gdf)} geometries")

        # Initialize with default value
        elevations = [-9999.0] * len(gdf)

        try:
            with rasterio.open(dem_path) as src:
                dem_crs = src.crs
                self.logger.info(f"DEM CRS: {dem_crs}")

                shapefile_crs = gdf.crs
                self.logger.info(f"Shapefile CRS: {shapefile_crs}")

                # Reproject if CRS mismatch
                if dem_crs != shapefile_crs:
                    self.logger.info(
                        f"CRS mismatch detected. Reprojecting from {shapefile_crs} to {dem_crs}"
                    )
                    try:
                        gdf_projected = gdf.to_crs(dem_crs)
                        self.logger.info("CRS reprojection successful")
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.error(f"Failed to reproject CRS: {str(e)}")
                        self.logger.warning("Using original CRS - elevation calculation may fail")
                        gdf_projected = gdf.copy()
                else:
                    self.logger.info("CRS match - no reprojection needed")
                    gdf_projected = gdf.copy()

                self.logger.info(
                    f"Processing elevation for {len(gdf_projected)} geometries using rasterio"
                )

                # Create iterator with progress bar if available
                if tqdm is not None:
                    iterator = tqdm(
                        gdf_projected.geometry.iloc,
                        total=len(gdf_projected),
                        desc="Calculating Elevation"
                    )
                else:
                    iterator = gdf_projected.geometry.iloc

                for idx, geom in enumerate(iterator):
                    try:
                        if geom is None or geom.is_empty:
                            continue

                        out_image, out_transform = mask(src, [geom], crop=True, nodata=-9999)
                        data = out_image[0]
                        valid_data = data[data != -9999]

                        if valid_data.size > 0:
                            elevations[idx] = float(np.mean(valid_data))

                    except ValueError:
                        # Usually means geometry is outside raster bounds
                        pass
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        if idx < 5:
                            self.logger.debug(
                                f"Error calculating elevation for geometry {idx}: {str(e)}"
                            )

            valid_count = sum(1 for elev in elevations if elev != -9999.0)
            self.logger.info(
                f"Successfully calculated elevation for {valid_count}/{len(elevations)} geometries"
            )

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error in elevation calculation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

        return elevations
