"""
GRU-based domain discretization using whole watershed units.

Uses Grouped Response Units directly as Hydrologic Response Units without
further subdivision, suitable for lumped or semi-distributed modeling.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import rasterio  # type: ignore
import rasterstats  # type: ignore
from pyproj import CRS  # type: ignore

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Use Grouped Response Units (GRUs) as Hydrologic Response Units (HRUs) without further discretization.

    Returns:
        Path: Path to the output HRU shapefile.
    """
    # Determine default name based on method
    default_name = f"{discretizer.domain_name}_riverBasins_{discretizer.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
    if discretizer.config.get("DELINEATE_COASTAL_WATERSHEDS") == True:
        default_name = f"{discretizer.domain_name}_riverBasins_with_coastal.shp"
    elif discretizer.config.get("DOMAIN_DEFINITION_METHOD") == "point":
        default_name = f"{discretizer.domain_name}_riverBasins_point.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    hru_output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath="shapefiles/catchment",
        default_name=f"{discretizer.domain_name}_HRUs_GRUs.shp",
    )

    gru_gdf = discretizer._read_shapefile(gru_shapefile)
    gru_gdf["HRU_ID"] = range(1, len(gru_gdf) + 1)
    gru_gdf["hru_type"] = "GRU"

    discretizer.logger.debug("Calculating elevation and centroid statistics for HRUs")

    # Get CRS information
    with rasterio.open(discretizer.dem_path) as src:
        dem_crs = src.crs
        discretizer.logger.debug(f"DEM CRS: {dem_crs}")

    shapefile_crs = gru_gdf.crs
    discretizer.logger.debug(f"Shapefile CRS: {shapefile_crs}")

    # Check if CRS match
    if dem_crs != shapefile_crs:
        discretizer.logger.debug(
            f"CRS mismatch detected. Reprojecting shapefile from {shapefile_crs} to {dem_crs}"
        )
        gru_gdf_projected = gru_gdf.to_crs(dem_crs)
    else:
        discretizer.logger.debug("CRS match - no reprojection needed")
        gru_gdf_projected = gru_gdf.copy()

    # Use rasterstats with the raster array and transform
    try:
        with rasterio.open(discretizer.dem_path) as src:
            dem_array = src.read(1)
            dem_transform = src.transform
            dem_nodata = src.nodata

        zs = rasterstats.zonal_stats(
            gru_gdf_projected.geometry,
            dem_array,
            affine=dem_transform,
            stats=["mean"],
            nodata=dem_nodata if dem_nodata is not None else -9999,
        )
        gru_gdf["elev_mean"] = [
            item["mean"] if item["mean"] is not None else -9999 for item in zs
        ]
        discretizer.logger.debug(
            f"Successfully calculated elevation statistics for {len(gru_gdf)} HRUs"
        )

    except Exception as e:
        discretizer.logger.error(f"Error calculating zonal statistics: {str(e)}")
        # Fallback: set all elevation means to -9999
        gru_gdf["elev_mean"] = -9999
        discretizer.logger.warning(
            "Setting all elevation means to -9999 due to calculation error"
        )

    # Calculate centroids in projected CRS for accuracy
    # Project to UTM for accurate centroid calculation if not already in UTM
    try:
        if gru_gdf.crs.is_geographic:
            utm_crs = gru_gdf.estimate_utm_crs()
            gru_gdf_utm = gru_gdf.to_crs(utm_crs)
        else:
            # Already in projected coordinate system
            gru_gdf_utm = gru_gdf.copy()
            utm_crs = gru_gdf.crs

        centroids_utm = gru_gdf_utm.geometry.centroid
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))

        gru_gdf["center_lon"] = centroids_wgs84.x
        gru_gdf["center_lat"] = centroids_wgs84.y

        discretizer.logger.debug(
            "Calculated centroids in WGS84: "
            f"lat range {centroids_wgs84.y.min():.6f} to {centroids_wgs84.y.max():.6f}, "
            f"lon range {centroids_wgs84.x.min():.6f} to {centroids_wgs84.x.max():.6f}"
        )

    except Exception as e:
        discretizer.logger.error(f"Error calculating centroids: {str(e)}")
        # Fallback: try to use existing center_lat/center_lon if they exist and look reasonable
        if "center_lat" in gru_gdf.columns and "center_lon" in gru_gdf.columns:
            # Check if existing values look like actual lat/lon (rough check)
            if (
                gru_gdf["center_lat"].between(-90, 90).all()
                and gru_gdf["center_lon"].between(-180, 180).all()
            ):
                discretizer.logger.debug("Using existing center_lat/center_lon coordinates")
            else:
                discretizer.logger.warning(
                    "Existing center_lat/center_lon appear to be in projected coordinates, setting to default values"
                )
                gru_gdf["center_lat"] = 0.0
                gru_gdf["center_lon"] = 0.0
        else:
            gru_gdf["center_lat"] = 0.0
            gru_gdf["center_lon"] = 0.0

    if "COMID" in gru_gdf.columns:
        gru_gdf["GRU_ID"] = gru_gdf["COMID"]
    elif "fid" in gru_gdf.columns:
        gru_gdf["GRU_ID"] = gru_gdf["fid"]

    gru_gdf["HRU_area"] = gru_gdf["GRU_area"]
    gru_gdf["HRU_ID"] = gru_gdf["GRU_ID"]

    gru_gdf.to_file(hru_output_shapefile)
    discretizer.logger.debug(f"GRUs saved as HRUs to {hru_output_shapefile}")

    return hru_output_shapefile
