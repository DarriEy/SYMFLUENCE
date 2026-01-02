from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from symfluence.utils.geospatial.geofabric import (
    GeofabricDelineator,
    GeofabricSubsetter,
    LumpedWatershedDelineator,
    PointDelineator,
)


@dataclass
class DelineationArtifacts:
    method: str
    river_basins_path: Optional[Path] = None
    river_network_path: Optional[Path] = None
    pour_point_path: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def create_point_domain_shapefile(
    config: Dict[str, Any],
    logger: Any,
) -> Optional[Path]:
    """
    Create a square basin shapefile from bounding box coordinates for point modeling.
    """
    delineator = PointDelineator(config, logger)
    return delineator.create_point_domain_shapefile()


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
        self.point_delineator = PointDelineator(self.config, self.logger)

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
            output_path = self.point_delineator.create_point_domain_shapefile()
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
