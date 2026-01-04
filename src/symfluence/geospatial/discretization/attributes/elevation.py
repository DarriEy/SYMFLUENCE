from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on elevation within each GRU.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    # Determine default name based on method
    default_name = f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp"
    if discretizer.config.get("DELINEATE_COASTAL_WATERSHEDS") == True:
        default_name = f"{discretizer.domain_name}_riverBasins__with_coastal.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    # Note: DEM path is already resolved in discretizer.dem_path, but re-resolving here as per original code pattern
    dem_raster = discretizer._get_file_path(
        path_key="DEM_PATH", 
        name_key="DEM_NAME",
        default_subpath="attributes/elevation/dem", 
        default_name=f"domain_{discretizer.config.get('DOMAIN_NAME')}_elv.tif"
    )
    
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath="shapefiles/catchment",
        default_name=f"{discretizer.domain_name}_HRUs_elevation.shp",
    )

    elevation_band_size = float(discretizer.config.get("ELEVATION_BAND_SIZE"))
    gru_gdf, elevation_thresholds = discretizer._read_and_prepare_data(
        gru_shapefile, dem_raster, elevation_band_size
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, dem_raster, elevation_thresholds, "elevClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Elevation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
