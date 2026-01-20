"""
Subset existing geofabric data based on pour points.

Supports MERIT, TDX, and NWS hydrofabric formats.
Uses graph-based upstream tracing to subset basins and rivers.

Refactored from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import geopandas as gpd

from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.graph_processor import RiverGraphProcessor
from ..utils.io_utils import GeofabricIOUtils
from ..utils.crs_utils import CRSUtils


class GeofabricSubsetter(BaseGeofabricDelineator):
    """
    Subsets geofabric data based on pour points and upstream basins.

    Supports three hydrofabric formats with different column naming conventions:
    - MERIT: COMID-based with up1, up2, up3 upstream columns
    - TDX: streamID/LINKNO with USLINKNO1, USLINKNO2 upstream columns
    - NWS: COMID-based with toCOMID (reverse direction)
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize geofabric subsetter.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Hydrofabric type configurations
        self.hydrofabric_types = {
            'MERIT': {
                'basin_id_col': 'COMID',
                'river_id_col': 'COMID',
                'upstream_cols': ['up1', 'up2', 'up3'],
                'upstream_default': -9999
            },
            'TDX': {
                'basin_id_col': 'streamID',
                'river_id_col': 'LINKNO',
                'upstream_cols': ['USLINKNO1', 'USLINKNO2'],
                'upstream_default': -9999
            },
            'NWS': {
                'basin_id_col': 'COMID',
                'river_id_col': 'COMID',
                'upstream_cols': ['toCOMID'],
                'upstream_default': 0
            }
        }

        # Initialize graph processor
        self.graph = RiverGraphProcessor()

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files.

        Uses the new naming convention based on definition_method and subset_from_geofabric.
        """
        return self._get_method_suffix()

    def subset_geofabric(self) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Subset the geofabric based on configuration.

        Returns:
            Tuple of (subset_basins, subset_rivers) GeoDataFrames
        """
        hydrofabric_type = self._get_config_value(lambda: self.config.domain.delineation.geofabric_type, dict_key='GEOFABRIC_TYPE').upper()
        if hydrofabric_type not in self.hydrofabric_types:
            self.logger.error(f"Unknown hydrofabric type: {hydrofabric_type}")
            return None, None

        fabric_config = self.hydrofabric_types[hydrofabric_type]

        # Load data using shared utility
        basins = GeofabricIOUtils.load_geopandas(
            Path(self._get_config_value(lambda: self.config.paths.source_geofabric_basins_path, dict_key='SOURCE_GEOFABRIC_BASINS_PATH')),
            self.logger
        )
        rivers = GeofabricIOUtils.load_geopandas(
            Path(self._get_config_value(lambda: self.config.paths.source_geofabric_rivers_path, dict_key='SOURCE_GEOFABRIC_RIVERS_PATH')),
            self.logger
        )
        pour_point = GeofabricIOUtils.load_geopandas(
            self._get_pour_point_path(),
            self.logger
        )

        # Ensure CRS consistency using shared utility
        basins, rivers, pour_point = CRSUtils.ensure_crs_consistency(
            basins, rivers, pour_point, self.logger
        )

        # Find downstream basin using shared utility
        downstream_basin_id = CRSUtils.find_basin_for_pour_point(
            pour_point, basins, fabric_config['basin_id_col']
        )

        # Build graph and find upstream using shared processor
        river_graph = self.graph.build_river_graph(rivers, fabric_config)
        upstream_basin_ids = self.graph.find_upstream_basins(
            downstream_basin_id, river_graph, self.logger
        )

        # Subset basins and rivers
        subset_basins = basins[basins[fabric_config['basin_id_col']].isin(upstream_basin_ids)].copy()
        subset_rivers = rivers[rivers[fabric_config['river_id_col']].isin(upstream_basin_ids)].copy()

        # Add SYMFLUENCE-specific columns
        self._add_symfluence_columns(subset_basins, subset_rivers, hydrofabric_type)

        # Save using custom paths
        basins_path, rivers_path = self._get_output_paths()
        GeofabricIOUtils.save_geofabric(
            subset_basins, subset_rivers,
            basins_path, rivers_path,
            self.logger
        )

        return subset_basins, subset_rivers

    def _add_symfluence_columns(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, hydrofabric_type: str):
        """
        Add SYMFLUENCE-specific columns based on hydrofabric type.

        Modifies GeoDataFrames in place.

        Args:
            basins: Basin GeoDataFrame to modify
            rivers: River GeoDataFrame to modify
            hydrofabric_type: Type of hydrofabric (NWS, TDX, Merit)
        """
        if hydrofabric_type == 'NWS':
            basins['GRU_ID'] = basins['COMID']
            basins['gru_to_seg'] = basins['COMID']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area
            # Rivers
            rivers['LINKNO'] = rivers['COMID']
            rivers['DSLINKNO'] = rivers['toCOMID']

        elif hydrofabric_type == 'TDX':
            basins['GRU_ID'] = basins['fid']
            basins['gru_to_seg'] = basins['streamID']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area

        elif hydrofabric_type in ['Merit', 'MERIT']:
            basins['GRU_ID'] = basins['COMID']
            basins['gru_to_seg'] = basins['COMID']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area
            # Rivers
            rivers['LINKNO'] = rivers['COMID']
            rivers['DSLINKNO'] = rivers['NextDownID']
            rivers_metric = rivers.to_crs('epsg:3763')
            rivers['Length'] = rivers_metric.geometry.length
            rivers.rename(columns={'slope': 'Slope'}, inplace=True)

    def aggregate_to_lumped(
        self,
        basins: gpd.GeoDataFrame,
        preserve_path: Path
    ) -> gpd.GeoDataFrame:
        """
        Aggregate subset basins to single lumped polygon.

        This method dissolves multiple subset basins into a single polygon,
        preserving the original basins for use in remap files.

        Args:
            basins: Subset basins GeoDataFrame
            preserve_path: Path to save original (unaggregated) basins

        Returns:
            GeoDataFrame with single dissolved polygon
        """
        # Save original basins for remap files
        basins.to_file(preserve_path)
        self.logger.info(f"Preserved original basins to: {preserve_path}")

        # Dissolve to single polygon
        dissolved = basins.dissolve()

        # Set lumped attributes
        dissolved = dissolved.reset_index(drop=True)
        dissolved['GRU_ID'] = 1
        dissolved['gru_to_seg'] = 1

        # Calculate area in metric CRS
        dissolved_metric = dissolved.to_crs('EPSG:3763')
        dissolved['GRU_area'] = dissolved_metric.geometry.area.values[0]

        self.logger.info(f"Aggregated {len(basins)} basins to single lumped polygon")
        return dissolved

    def _get_output_paths(self) -> Tuple[Path, Path]:
        """
        Get output paths for subset shapefiles.

        Returns:
            Tuple of (basins_path, rivers_path)
        """
        method_suffix = self._get_method_suffix()

        if self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH') == 'default':
            basins_path = (
                self.project_dir / "shapefiles" / "river_basins" /
                f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            )
        else:
            basins_path = Path(self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH'))

        if self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH') == 'default':
            rivers_path = (
                self.project_dir / "shapefiles" / "river_network" /
                f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
            )
        else:
            rivers_path = Path(self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH'))

        return basins_path, rivers_path
