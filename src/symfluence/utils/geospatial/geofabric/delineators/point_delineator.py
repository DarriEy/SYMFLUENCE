from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd # type: ignore
from shapely.geometry import Polygon # type: ignore


class PointDelineator:
    """
    Handles point-scale domain delineation by creating a small square basin
    from bounding box coordinates.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get("SYMFLUENCE_DATA_DIR"))
        self.domain_name = self.config.get("DOMAIN_NAME")
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

    def create_point_domain_shapefile(self) -> Optional[Path]:
        """
        Create a square basin shapefile from bounding box coordinates for point modeling.
        
        Returns:
            Path to the created shapefile or None if failed
        """
        try:
            self.logger.info("Creating point domain shapefile from bounding box coordinates")

            bbox_coords = self.config.get("BOUNDING_BOX_COORDS", "")
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
            output_path = output_dir / f"{self.domain_name}_riverBasins_point.shp"

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
