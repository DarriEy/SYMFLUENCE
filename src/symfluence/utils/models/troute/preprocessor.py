"""
TRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the t-route routing model.
"""

import os
import yaml
import netCDF4 as nc4
import geopandas as gpd
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from typing import Dict, Any

from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPreProcessor
from symfluence.utils.geospatial.geometry_utils import GeospatialUtilsMixin


@ModelRegistry.register_preprocessor('TROUTE')
class TRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin):
    """
    A standalone preprocessor for t-route within the SYMFLUENCE framework.

    This class creates all necessary input and configuration files for a
    t-route run without any dependency on other routing model utilities.
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "troute"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

    def run_preprocessing(self):
        """Main entry point for running all t-route preprocessing steps."""
        self.logger.info("--- Starting t-route Preprocessing ---")
        self.copy_base_settings()
        self.create_troute_topology_file()
        self.create_troute_yaml_config()
        self.logger.info("--- t-route Preprocessing Completed Successfully ---")

    def copy_base_settings(self):
        """Copies base settings for t-route from package data."""
        self.logger.info("Copying t-route base settings...")
        from symfluence.utils.resources import get_base_settings_dir

        try:
            base_settings_path = get_base_settings_dir('troute')
        except FileNotFoundError:
            self.logger.warning("Base settings for t-route not found in package. Skipping copy.")
            return

        self.setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.setup_dir / file)
        self.logger.info("t-route base settings copied.")

    def create_troute_topology_file(self):
        """
        Creates the NetCDF network topology file using t-route's expected NWM variable names.
        """
        self.logger.info("Creating t-route specific network topology file...")

        # Define paths using SYMFLUENCE conventions
        river_network_path = self.project_dir / 'shapefiles/river_network'
        river_network_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverNetwork_{self.config_dict.get('DOMAIN_DEFINITION_METHOD','delineate')}.shp"
        river_basin_path = self.project_dir / 'shapefiles/river_basins'
        river_basin_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_{self.config_dict.get('DOMAIN_DEFINITION_METHOD')}.shp"
        topology_name = self.config_dict.get('SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc')
        topology_filepath = self.setup_dir / topology_name

        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)

        with nc4.Dataset(topology_filepath, 'w', format='NETCDF4') as ncid:
            # Set global attributes
            ncid.setncattr('Author', "Created by SYMFLUENCE workflow for t-route")
            ncid.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            ncid.setncattr('Conventions', "CF-1.6")

            # Create dimensions based on t-route standards
            ncid.createDimension('link', len(shp_river))
            ncid.createDimension('nhru', len(shp_basin))
            ncid.createDimension('gages', None) # Unlimited dimension for gages

            # Map SYMFLUENCE shapefile columns to t-route's required variable names
            self._create_and_fill_nc_var(ncid, 'comid', 'i4', 'link', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SEGID')], 'Unique segment ID')
            self._create_and_fill_nc_var(ncid, 'to_node', 'i4', 'link', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')], 'Downstream segment ID')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'link', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_LENGTH')], 'Segment length', 'meters')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'link', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SLOPE')], 'Segment slope', 'm/m')
            self._create_and_fill_nc_var(ncid, 'link_id_hru', 'i4', 'nhru', shp_basin[self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')], 'Segment ID for HRU discharge')
            self._create_and_fill_nc_var(ncid, 'hru_area_m2', 'f8', 'nhru', shp_basin[self.config_dict.get('RIVER_BASIN_SHP_AREA')], 'HRU area', 'm^2')

            # Add required placeholder variables with sensible defaults
            centroids = self.calculate_feature_centroids(shp_river)

            shp_river['lat'] = centroids.y
            shp_river['lon'] = centroids.x
            self._create_and_fill_nc_var(ncid, 'lat', 'f8', 'link', shp_river['lat'], 'Latitude of segment midpoint', 'degrees_north')
            self._create_and_fill_nc_var(ncid, 'lon', 'f8', 'link', shp_river['lon'], 'Longitude of segment midpoint', 'degrees_east')
            self._create_and_fill_nc_var(ncid, 'alt', 'f8', 'link', [0.0] * len(shp_river), 'Mean elevation of segment', 'meters')
            self._create_and_fill_nc_var(ncid, 'from_node', 'i4', 'link', [0] * len(shp_river), 'Upstream node ID')
            self._create_and_fill_nc_var(ncid, 'n', 'f8', 'link', [0.035] * len(shp_river), 'Mannings roughness coefficient')

        self.logger.info(f"t-route topology file created at {topology_filepath}")

    def create_troute_yaml_config(self):
        """Creates the t-route YAML configuration file from SYMFLUENCE config settings."""
        self.logger.info("Creating t-route YAML configuration file...")

        # Determine paths and parameters from config
        source_model = self.config_dict.get('TROUTE_FROM_MODEL', 'SUMMA').upper()
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        input_dir = self.project_dir / f"simulations/{experiment_id}" / source_model
        output_dir = self.project_dir / f"simulations/{experiment_id}" / 'troute'
        topology_name = self.config_dict.get('SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc')

        # Calculate nts (Number of Timesteps)
        start_dt = datetime.fromisoformat(self.config_dict.get('EXPERIMENT_TIME_START'))
        end_dt = datetime.fromisoformat(self.config_dict.get('EXPERIMENT_TIME_END'))
        time_step_seconds = int(self.config_dict.get('SETTINGS_TROUTE_DT_SECONDS', 3600))
        total_seconds = (end_dt - start_dt).total_seconds() + time_step_seconds
        nts = int(total_seconds / time_step_seconds)

        # Build configuration dictionary matching t-route's schema
        config_dict = {
            'log_parameters': {'showtiming': True, 'log_level': 'DEBUG'},
            'network_topology_parameters': {'supernetwork_parameters': {'geo_file_path': str(self.setup_dir / topology_name)}},
            'compute_parameters': {
                'restart_parameters': {'start_datetime': self.config_dict.get('EXPERIMENT_TIME_START')},
                'forcing_parameters': {
                    'nts': nts,
                    'qlat_input_folder': str(input_dir),
                    'qlat_file_pattern_filter': f"{experiment_id}_timestep.nc"
                }
            },
            'output_parameters': {'stream_output': {'stream_output_directory': str(output_dir)}}
        }

        # Write dictionary to YAML file
        yaml_filename = self.config_dict.get('SETTINGS_TROUTE_CONFIG_FILE', 'troute_config.yml')
        yaml_filepath = self.setup_dir / yaml_filename
        with open(yaml_filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        self.logger.info(f"t-route YAML config written to {yaml_filepath}")

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, data, long_name, units='-'):
        """Helper to create and fill a NetCDF variable."""
        var = ncid.createVariable(var_name, var_type, (dim,))
        var[:] = data.values if hasattr(data, 'values') else data
        var.long_name = long_name
        var.units = units
