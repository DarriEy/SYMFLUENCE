"""
Point domain delineator.

Creates minimal square basin shapefiles for point-scale modeling where
watershed delineation is not applicable (e.g., flux tower sites, lysimeters).
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import geopandas as gpd # type: ignore
from shapely.geometry import Polygon # type: ignore


from ..base.base_delineator import BaseGeofabricDelineator
from ....geospatial.delineation_registry import DelineationRegistry


@DelineationRegistry.register('point')
class PointDelineator(BaseGeofabricDelineator):
    """
    Handles point-scale domain delineation by creating a small square basin
    from bounding box coordinates.
    """

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files."""
        return "point"

    def create_point_domain_shapefile(self) -> Optional[Path]:
        """
        Create a square basin shapefile from bounding box coordinates for point modeling.

        Returns:
            Path to the created shapefile or None if failed
        """
        try:
            self.logger.info("Creating point domain shapefile from bounding box coordinates")

            bbox_coords = self._get_config_value(lambda: self.config.domain.bounding_box_coords, default="", dict_key='BOUNDING_BOX_COORDS')
            if not bbox_coords:
                self.logger.error("BOUNDING_BOX_COORDS not found in configuration")
                return None

            try:
                lat_max, lon_min, lat_min, lon_max = map(float, bbox_coords.split("/"))
            except ValueError:
                self.logger.error(
                    f"Invalid bounding box format: {bbox_coords}. Expected format: lat_max/lon_min/lat_min/lon_max"
                )
                return None

            coords = [
                (lon_min, lat_min),
                (lon_max, lat_min),
                (lon_max, lat_max),
                (lon_min, lat_max),
                (lon_min, lat_min),
            ]
            polygon = Polygon(coords)
            area_deg2 = polygon.area

            gdf = gpd.GeoDataFrame(
                {
                    "GRU_ID": [1],
                    "GRU_area": [area_deg2],
                    "basin_name": [self.domain_name],
                    "method": ["point"],
                },
                geometry=[polygon],
                crs="EPSG:4326",
            )

            output_dir = self.project_dir / "shapefiles" / "river_basins"
            output_dir.mkdir(parents=True, exist_ok=True)
            method_suffix = self._get_method_suffix()
            output_path = output_dir / f"{self.domain_name}_riverBasins_{method_suffix}.shp"

            gdf.to_file(output_path)

            self.logger.info(f"Point domain shapefile created successfully: {output_path}")
            self.logger.info(
                f"Bounding box: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}"
            )
            self.logger.info(f"Area: {area_deg2:.6f} square degrees")

            return output_path
        except Exception as exc:
            self.logger.error(f"Error creating point domain shapefile: {str(exc)}")
            self.logger.error(traceback.format_exc())
            return None
