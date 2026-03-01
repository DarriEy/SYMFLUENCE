# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Lumped watershed delineation using TauDEM.

Delineates a single lumped watershed based on DEM and pour point.
Uses TauDEM for watershed delineation and creates simplified river network.

Refactored from geofabric_utils.py (2026-01-01)
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

from ....geospatial.delineation_registry import DelineationRegistry
from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.gdal_processor import GDALProcessor
from ..processors.taudem_executor import TauDEMExecutor


@DelineationRegistry.register('lumped')
class LumpedWatershedDelineator(BaseGeofabricDelineator):
    """
    Delineates lumped watersheds using TauDEM.

    A lumped watershed is a single basin with a single pour point,
    suitable for simple hydrological modeling.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize lumped watershed delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Specific paths for lumped delineation
        self.output_dir = self.project_dir / "shapefiles/tempdir"
        self.interim_dir = self.project_dir / "taudem-interim-files" / "lumped"
        self.delineation_method = 'TauDEM'

        # Initialize processors
        self.taudem = TauDEMExecutor(config, logger, self.taudem_dir)
        self.gdal = GDALProcessor(logger)

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files."""
        return "lumped"

    def delineate_lumped_watershed(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate a lumped watershed.

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        self.logger.info(f"Delineating lumped watershed: {self.domain_name}")

        # Get pour point path
        pour_point_path = self._get_pour_point_path()
        self.pour_point_path = pour_point_path

        # Define output paths
        method_suffix = self._get_method_suffix()
        river_basins_path = (
            self.project_dir / "shapefiles" / "river_basins" /
            f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        )
        river_network_path = (
            self.project_dir / "shapefiles" / "river_network" /
            f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        )

        # Create directories if they don't exist
        river_basins_path.parent.mkdir(parents=True, exist_ok=True)
        river_network_path.parent.mkdir(parents=True, exist_ok=True)

        # Delineate watershed
        self._delineate_with_taudem()

        # Create river network shapefile from pour point
        self._create_river_network(pour_point_path, river_network_path)

        # Ensure required fields are present in both shapefiles
        self._ensure_required_fields(river_basins_path, river_network_path)

        return river_network_path, river_basins_path

    def _create_river_network(self, pour_point_path: Path, river_network_path: Path) -> None:
        """
        Create a simple river network shapefile based on the pour point.

        Args:
            pour_point_path: Path to the pour point shapefile
            river_network_path: Path to save the river network shapefile
        """
        try:
            # Load pour point
            pour_point_gdf = gpd.read_file(pour_point_path)

            # Create river network from pour point
            river_network = pour_point_gdf.copy()

            # Add required fields for river network
            river_network['LINKNO'] = 1
            river_network['DSLINKNO'] = 0  # Outlet has no downstream link
            river_network['Length'] = 100.0  # Placeholder length in meters
            river_network['Slope'] = 0.01   # Placeholder slope
            river_network['GRU_ID'] = 1

            # Save river network shapefile
            river_network.to_file(river_network_path)
            self.logger.debug(f"Created river network shapefile at: {river_network_path}")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating river network: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _ensure_required_fields(self, river_basins_path: Path, river_network_path: Path) -> None:
        """
        Ensure that all required fields are present in both shapefiles.

        Args:
            river_basins_path: Path to the river basins shapefile
            river_network_path: Path to the river network shapefile
        """
        try:
            # Load and check river basins
            basins_gdf = gpd.read_file(river_basins_path)

            # Add required fields for basins if missing
            required_basin_fields = {
                'GRU_ID': 1,
                'gru_to_seg': 1
            }

            # Calculate area in square meters
            if 'GRU_area' not in basins_gdf.columns:
                utm_crs = basins_gdf.estimate_utm_crs()
                basins_utm = basins_gdf.to_crs(utm_crs)
                basins_gdf['GRU_area'] = basins_utm.geometry.area
                # Convert back to original CRS
                basins_gdf = basins_gdf.to_crs(basins_gdf.crs)

            # Add any missing fields
            for field, default_value in required_basin_fields.items():
                if field not in basins_gdf.columns:
                    basins_gdf[field] = default_value

            # Save updated basin shapefile
            basins_gdf.to_file(river_basins_path)
            self.logger.debug(f"Updated river basins shapefile with required fields at: {river_basins_path}")

            # Load and check river network if it exists
            if river_network_path.exists():
                network_gdf = gpd.read_file(river_network_path)

                # Add required fields for network if missing
                required_network_fields = {
                    'LINKNO': 1,
                    'DSLINKNO': 0,
                    'Length': 100.0,
                    'Slope': 0.01,
                    'GRU_ID': 1
                }

                # Add any missing fields
                for field, default_value in required_network_fields.items():
                    if field not in network_gdf.columns:
                        network_gdf[field] = default_value

                # Save updated network shapefile
                network_gdf.to_file(river_network_path)
                self.logger.debug(f"Updated river network shapefile with required fields at: {river_network_path}")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error ensuring required fields: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _delineate_with_taudem(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using TauDEM.

        Returns:
            Path to the delineated watershed shapefile, or None if delineation fails
        """
        try:
            if not self.pour_point_path.is_file():
                self.logger.error(f"Pour point file not found: {self.pour_point_path}")
                return None

            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Apply DEM conditioning (stream burning) if configured
            dem_input = self._condition_dem()

            # TauDEM processing steps for lumped watershed delineation
            steps = [
                f"{self.taudem_dir}/pitremove -z {dem_input} -fel {self.output_dir}/fel.tif",
                f"{self.taudem_dir}/d8flowdir -fel {self.output_dir}/fel.tif -p {self.output_dir}/p.tif -sd8 {self.output_dir}/sd8.tif",
                f"{self.taudem_dir}/aread8 -p {self.output_dir}/p.tif -ad8 {self.output_dir}/ad8.tif",
                f"{self.taudem_dir}/threshold -ssa {self.output_dir}/ad8.tif -src {self.output_dir}/src.tif -thresh 100",
                f"{self.taudem_dir}/moveoutletstostreams -p {self.output_dir}/p.tif -src {self.output_dir}/src.tif -o {self.pour_point_path} -om {self.output_dir}/om.shp",
                f"{self.taudem_dir}/gagewatershed -p {self.output_dir}/p.tif -o {self.output_dir}/om.shp -gw {self.output_dir}/watershed.tif -id {self.output_dir}/watershed_id.txt"
            ]

            for step in steps:
                self.taudem.run_command(step)
                self.logger.debug(f"Completed TauDEM step: {step}")

            # Convert the watershed raster to polygon
            method_suffix = self._get_method_suffix()
            watershed_shp_path = (
                self.project_dir / "shapefiles" / "river_basins" /
                f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            )
            watershed_shp_path.parent.mkdir(parents=True, exist_ok=True)

            self.gdal.raster_to_polygon(self.output_dir / "watershed.tif", watershed_shp_path)

            # Add required attributes if they don't exist
            watershed_gdf = gpd.read_file(watershed_shp_path)

            # For lumped basins, dissolve all polygons into a single feature
            # This handles artifacts from raster-to-polygon conversion
            if len(watershed_gdf) > 1:
                self.logger.info(f"Dissolving {len(watershed_gdf)} polygons into single lumped basin")
                # Dissolve all features into one
                watershed_gdf['dissolve_key'] = 1
                watershed_gdf = watershed_gdf.dissolve(by='dissolve_key').reset_index(drop=True)
                watershed_gdf = watershed_gdf.drop(columns=['dissolve_key'], errors='ignore')
                self.logger.info(f"Dissolved to {len(watershed_gdf)} feature(s)")

            if 'GRU_ID' not in watershed_gdf.columns:
                watershed_gdf['GRU_ID'] = 1

            if 'gru_to_seg' not in watershed_gdf.columns:
                watershed_gdf['gru_to_seg'] = 1

            # Calculate area in square meters if it doesn't exist
            if 'GRU_area' not in watershed_gdf.columns:
                utm_crs = watershed_gdf.estimate_utm_crs()
                watershed_utm = watershed_gdf.to_crs(utm_crs)
                watershed_gdf['GRU_area'] = watershed_utm.geometry.area
                watershed_gdf = watershed_gdf.to_crs('EPSG:4326')

            # Save updated watershed shapefile
            watershed_gdf.to_file(watershed_shp_path)

            self.logger.debug(f"Updated watershed shapefile at: {watershed_shp_path}")

            # Clean up temporary files if requested
            if self._get_config_value(lambda: self.config.domain.delineation.cleanup_intermediate_files, default=True, dict_key='CLEANUP_INTERMEDIATE_FILES'):
                shutil.rmtree(self.output_dir, ignore_errors=True)
                self.logger.debug(f"Cleaned up intermediate files: {self.output_dir}")

            return watershed_shp_path

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error during TauDEM watershed delineation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
