from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from symfluence.geospatial.raster_utils import calculate_annual_radiation

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on radiation properties using MultiPolygon HRUs.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp",
    )
    
    dem_raster = discretizer._get_file_path(
        path_key="DEM_PATH", 
        name_key="DEM_NAME",
        default_subpath="attributes/elevation/dem", 
        default_name=f"domain_{discretizer.config.get('DOMAIN_NAME')}_elv.tif"
    )
    
    radiation_raster = discretizer._get_file_path(
        path_key="RADIATION_PATH",
        name_key="RADIATION_NAME",
        default_subpath="attributes/radiation",
        default_name="annual_radiation.tif",
    )
    
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath="shapefiles/catchment",
        default_name=f"{discretizer.domain_name}_HRUs_radiation.shp",
    )

    radiation_class_number = int(discretizer.config.get("RADIATION_CLASS_NUMBER"))

    if not radiation_raster.exists():
        discretizer.logger.info(
            "Annual radiation raster not found. Calculating radiation..."
        )
        radiation_raster = calculate_annual_radiation(
            dem_raster, radiation_raster, discretizer.logger
        )
        if radiation_raster is None:
            raise ValueError("Failed to calculate annual radiation")

    gru_gdf, radiation_thresholds = discretizer._read_and_prepare_data(
        gru_shapefile, radiation_raster, radiation_class_number
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, radiation_raster, radiation_thresholds, "radiationClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Radiation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
