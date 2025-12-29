from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Polygon

from utils.geospatial.artifacts import DelineationArtifacts
from utils.geospatial.geofabric_utils import (
    GeofabricDelineator,
    GeofabricSubsetter,
    LumpedWatershedDelineator,
)


def create_point_domain_shapefile(
    config: Dict[str, Any],
    logger: Any,
) -> Optional[Path]:
    """
    Create a square basin shapefile from bounding box coordinates for point modeling.
    """
    try:
        logger.info("Creating point domain shapefile from bounding box coordinates")

        bbox_coords = config.get("BOUNDING_BOX_COORDS", "")
        if not bbox_coords:
            logger.error("BOUNDING_BOX_COORDS not found in configuration")
            return None

        try:
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_coords.split("/"))
        except ValueError:
            logger.error(
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

        domain_name = config.get("DOMAIN_NAME")
        data_dir = Path(config.get("SYMFLUENCE_DATA_DIR"))
        project_dir = data_dir / f"domain_{domain_name}"

        gdf = gpd.GeoDataFrame(
            {
                "GRU_ID": [1],
                "GRU_area": [area_deg2],
                "basin_name": [domain_name],
                "method": ["point"],
            },
            geometry=[polygon],
            crs="EPSG:4326",
        )

        output_dir = project_dir / "shapefiles" / "river_basins"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{domain_name}_riverBasins_point.shp"

        gdf.to_file(output_path)

        logger.info(f"Point domain shapefile created successfully: {output_path}")
        logger.info(
            f"Bounding box: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}"
        )
        logger.info(f"Area: {area_deg2:.6f} square degrees")

        return output_path
    except Exception as exc:
        logger.error(f"Error creating point domain shapefile: {str(exc)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


class DomainDelineator:
    """
    Handles domain delineation with explicit artifact tracking.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get("SYMFLUENCE_DATA_DIR"))
        self.domain_name = self.config.get("DOMAIN_NAME")
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.delineator = GeofabricDelineator(self.config, self.logger)
        self.lumped_delineator = LumpedWatershedDelineator(self.config, self.logger)
        self.subsetter = GeofabricSubsetter(self.config, self.logger)

    def _get_pour_point_path(self) -> Optional[Path]:
        pour_point_path = self.config.get("POUR_POINT_SHP_PATH")
        if pour_point_path == "default":
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(pour_point_path)

        pour_point_name = self.config.get("POUR_POINT_SHP_NAME", "default")
        if pour_point_name == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"
        else:
            pour_point_path = pour_point_path / pour_point_name

        return pour_point_path

    def _get_subset_paths(self) -> Tuple[Path, Path]:
        if self.config.get("OUTPUT_BASINS_PATH") == "default":
            basins_path = (
                self.project_dir
                / "shapefiles"
                / "river_basins"
                / f"{self.domain_name}_riverBasins_subset_{self.config.get('GEOFABRIC_TYPE')}.shp"
            )
        else:
            basins_path = Path(self.config.get("OUTPUT_BASINS_PATH"))

        if self.config.get("OUTPUT_RIVERS_PATH") == "default":
            rivers_path = (
                self.project_dir
                / "shapefiles"
                / "river_network"
                / f"{self.domain_name}_riverNetwork_subset_{self.config.get('GEOFABRIC_TYPE')}.shp"
            )
        else:
            rivers_path = Path(self.config.get("OUTPUT_RIVERS_PATH"))

        return basins_path, rivers_path

    def define_domain(self) -> Tuple[Optional[object], DelineationArtifacts]:
        domain_method = self.config.get("DOMAIN_DEFINITION_METHOD")
        artifacts = DelineationArtifacts(method=domain_method)

        if self.config.get("RIVER_BASINS_NAME") != "default":
            self.logger.info("Shapefile provided, skipping domain definition")
            return None, artifacts

        if domain_method == "point":
            output_path = create_point_domain_shapefile(self.config, self.logger)
            artifacts.river_basins_path = output_path
            artifacts.pour_point_path = self._get_pour_point_path()
            return output_path, artifacts

        if domain_method == "subset":
            result = self.subsetter.subset_geofabric()
            basins_path, rivers_path = self._get_subset_paths()
            artifacts.river_basins_path = basins_path
            artifacts.river_network_path = rivers_path
            artifacts.pour_point_path = self._get_pour_point_path()
            return result, artifacts

        if domain_method == "lumped":
            river_network_path, river_basins_path = (
                self.lumped_delineator.delineate_lumped_watershed()
            )
            artifacts.river_network_path = river_network_path
            artifacts.river_basins_path = river_basins_path
            artifacts.pour_point_path = self._get_pour_point_path()
            return (river_network_path, river_basins_path), artifacts

        if domain_method == "delineate":
            river_network_path, river_basins_path = self.delineator.delineate_geofabric()
            if self.config.get("DELINEATE_COASTAL_WATERSHEDS"):
                coastal_result = self.delineator.delineate_coastal()
                if coastal_result and all(coastal_result):
                    river_network_path, river_basins_path = coastal_result
                    self.logger.info(f"Coastal delineation completed: {coastal_result}")

            artifacts.river_network_path = river_network_path
            artifacts.river_basins_path = river_basins_path
            artifacts.pour_point_path = self._get_pour_point_path()
            return (river_network_path, river_basins_path), artifacts

        self.logger.error(f"Unknown domain definition method: {domain_method}")
        return None, artifacts
