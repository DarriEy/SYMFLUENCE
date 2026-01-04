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
from symfluence.utils.common.path_resolver import PathResolverMixin


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


class DomainDelineator(PathResolverMixin):
    """
    Handles domain delineation with explicit artifact tracking.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager
        
        # properties from mixins: self.project_dir, self.domain_name, self.data_dir are available

        self.delineator = GeofabricDelineator(self.config, self.logger, self.reporting_manager)
        self.lumped_delineator = LumpedWatershedDelineator(self.config, self.logger)
        self.subsetter = GeofabricSubsetter(self.config, self.logger)
        self.point_delineator = PointDelineator(self.config, self.logger)

    def _get_pour_point_path(self) -> Optional[Path]:
        return self._get_file_path(
            path_key="POUR_POINT_SHP_PATH",
            name_key="POUR_POINT_SHP_NAME",
            default_subpath="shapefiles/pour_point",
            default_name=f"{self.domain_name}_pourPoint.shp"
        )

    def _get_subset_paths(self) -> Tuple[Path, Path]:
        geofabric_type = self.config.get('GEOFABRIC_TYPE')
        
        basins_path = self._get_default_path(
            config_key="OUTPUT_BASINS_PATH",
            default_subpath=f"shapefiles/river_basins/{self.domain_name}_riverBasins_subset_{geofabric_type}.shp"
        )

        rivers_path = self._get_default_path(
            config_key="OUTPUT_RIVERS_PATH",
            default_subpath=f"shapefiles/river_network/{self.domain_name}_riverNetwork_subset_{geofabric_type}.shp"
        )

        return basins_path, rivers_path

    def define_domain(self) -> Tuple[Optional[object], DelineationArtifacts]:
        with self.time_limit("Domain Definition"):
            domain_method = self._get_config_value("DOMAIN_DEFINITION_METHOD")
            artifacts = DelineationArtifacts(method=domain_method)

            if self._get_config_value("RIVER_BASINS_NAME") != "default":
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