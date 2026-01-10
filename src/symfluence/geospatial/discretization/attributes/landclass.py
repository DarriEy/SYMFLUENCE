"""
Land cover-based domain discretization for vegetation classification.

Creates HRUs based on land cover categories from classification rasters,
enabling land-use-aware parameterization of hydrological processes.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on land cover classifications using MultiPolygon HRUs.

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

    land_raster = discretizer._get_file_path(
        path_key="LAND_CLASS_PATH",
        name_key="LAND_CLASS_NAME",
        default_subpath="attributes/landclass",
        default_name=f"domain_{discretizer.config.get('DOMAIN_NAME')}_land_classes.tif",
    )
    
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath="shapefiles/catchment",
        default_name=f"{discretizer.domain_name}_HRUs_landclass.shp",
    )

    gru_gdf, land_classes = discretizer._read_and_prepare_data(gru_shapefile, land_raster)
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, land_raster, land_classes, "landClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
