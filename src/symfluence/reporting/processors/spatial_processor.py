"""
Spatial processor for reporting and configuration.

Handles spatial operations related to reporting, such as finding reach IDs
based on pour points.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import traceback

from symfluence.core.mixins import ConfigMixin
from symfluence.core.constants import ConfigKeys


class SpatialProcessor(ConfigMixin):
    """
    Handles spatial processing tasks for reporting and configuration.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the spatial processor.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: Logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key=ConfigKeys.SYMFLUENCE_DATA_DIR)) / f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}"

    def update_sim_reach_id(self, config_path: Optional[str] = None) -> Optional[int]:
        """
        Update the SIM_REACH_ID in both the config object and YAML file by finding the
        nearest river segment to the pour point.

        Args:
            config_path: Either a path to the config file or None. If None, will only update the in-memory config.

        Returns:
            The found reach ID, or None if failed.
        """
        import geopandas as gpd  # type: ignore
        try:
            # Load the pour point shapefile
            pour_point_name = self._get_config_value(lambda: self.config.paths.pour_point_name, dict_key=ConfigKeys.POUR_POINT_SHP_NAME)
            if pour_point_name == 'default':
                pour_point_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_pourPoint.shp"

            pour_point_path = self._get_file_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', pour_point_name)

            if not pour_point_path.exists():
                self.logger.error(f"Pour point shapefile not found: {pour_point_path}")
                return None

            pour_point_gdf = gpd.read_file(pour_point_path)

            # Load the river network shapefile
            river_network_name = self._get_config_value(lambda: self.config.paths.river_network_name, dict_key=ConfigKeys.RIVER_NETWORK_SHP_NAME)
            if river_network_name == 'default':
                river_network_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_riverNetwork_{self._get_config_value(lambda: self.config.domain.definition_method, dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD)}.shp"

            river_network_path = self._get_file_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', river_network_name)

            if not river_network_path.exists():
                self.logger.error(f"River network shapefile not found: {river_network_path}")
                return None

            river_network_gdf = gpd.read_file(river_network_path)

            # Ensure both GeoDataFrames have the same CRS
            if pour_point_gdf.crs != river_network_gdf.crs:
                pour_point_gdf = pour_point_gdf.to_crs(river_network_gdf.crs)

            # Get the pour point coordinates
            pour_point = pour_point_gdf.geometry.iloc[0]

            # Convert to a projected CRS for accurate distance calculation
            # Use UTM zone based on the data's centroid
            center_lon = pour_point.centroid.x
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_crs = f"EPSG:326{utm_zone if pour_point.centroid.y >= 0 else utm_zone+30}"

            # Reproject both geometries to UTM
            pour_point_utm = pour_point_gdf.to_crs(utm_crs).geometry.iloc[0]
            river_network_utm = river_network_gdf.to_crs(utm_crs)

            # Find the nearest stream segment to the pour point
            nearest_segment = river_network_utm.iloc[river_network_utm.distance(pour_point_utm).idxmin()]

            # Get the ID of the nearest segment
            seg_id_col = self._get_config_value(lambda: self.config.paths.river_network_segid, default='seg_id', dict_key=ConfigKeys.RIVER_NETWORK_SHP_SEGID)
            if seg_id_col not in nearest_segment:
                 # Try common alternatives if configured one is missing
                 for alt_col in ['seg_id', 'segId', 'SEG_ID', 'COMID', 'feature_id']:
                     if alt_col in nearest_segment:
                         seg_id_col = alt_col
                         break

            reach_id = nearest_segment[seg_id_col]

            # Update the config object (only if it's a dict, not a frozen Pydantic model)
            if isinstance(self.config, dict):
                self.config['SIM_REACH_ID'] = reach_id

            # Update the YAML config file if a file path was provided
            if config_path is not None and isinstance(config_path, (str, Path)):
                config_file_path = Path(config_path)

                if not config_file_path.exists():
                    self.logger.error(f"Config file not found at {config_file_path}")
                    return None

                # Read the current config file
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config_lines = f.readlines()

                # Find and update the SIM_REACH_ID line
                updated = False
                for i, line in enumerate(config_lines):
                    if line.strip().startswith('SIM_REACH_ID:'):
                        config_lines[i] = f"SIM_REACH_ID: {reach_id}                                              # River reach ID used for streamflow evaluation and optimization\n"
                        updated = True
                        break

                if not updated:
                    # If SIM_REACH_ID line not found, add it in the Simulation settings section
                    for i, line in enumerate(config_lines):
                        if line.strip().startswith('## Simulation settings'):
                            config_lines.insert(i + 1, f"SIM_REACH_ID: {reach_id}                                              # River reach ID used for streamflow evaluation and optimization\n")
                            updated = True
                            break

                if not updated:
                     # Append if section not found
                     config_lines.append(f"\nSIM_REACH_ID: {reach_id}\n")

                # Write the updated config back to file
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(config_lines)

                self.logger.info(f"Updated SIM_REACH_ID to {reach_id} in both config object and file: {config_file_path}")
            else:
                self.logger.info(f"Updated SIM_REACH_ID to {reach_id} in config object only")

            return reach_id

        except Exception as e:
            self.logger.error(f"Error updating SIM_REACH_ID: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """
        Get absolute path for a file type, handling defaults.

        Args:
            file_type: Config key for the file path (e.g. 'POUR_POINT_SHP_PATH')
            file_def_path: Default subdirectory relative to project dir
            file_name: Name of the file

        Returns:
            Path object to the file
        """
        path_val = self.config.get(file_type)
        if path_val == 'default' or path_val is None:
            return self.project_dir / file_def_path / file_name
        else:
            return Path(path_val)
