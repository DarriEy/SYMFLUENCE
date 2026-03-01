# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the t-route routing model.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import yaml

from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin
from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_preprocessor('TROUTE')
class TRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin):  # type: ignore[misc]
    """
    A standalone preprocessor for t-route within the SYMFLUENCE framework.

    This class creates all necessary input and configuration files for a
    t-route run without any dependency on other routing model utilities.
    """


    MODEL_NAME = "troute"
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

    def run_preprocessing(self):
        """Main entry point for running all t-route preprocessing steps."""
        self.logger.info("--- Starting t-route Preprocessing ---")
        self.copy_base_settings()
        self.create_troute_topology_file()
        self.create_troute_yaml_config()
        self.logger.info("--- t-route Preprocessing Completed Successfully ---")

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """Copies base settings for t-route from package data."""
        self.logger.info("Copying t-route base settings...")
        from symfluence.resources import get_base_settings_dir

        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

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
        river_network_name = f"{self.domain_name}_riverNetwork_{self.domain_definition_method}.shp"
        river_basin_path = self.project_dir / 'shapefiles/river_basins'
        river_basin_name = f"{self.domain_name}_riverBasins_{self.domain_definition_method}.shp"
        topology_name = self._get_config_value(
            lambda: self.config.model.troute.topology_file, default='troute_topology.nc'
        )
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
            seg_id_col = self._get_config_value(lambda: self.config.paths.river_network_segid, default='LINKNO')
            downseg_id_col = self._get_config_value(lambda: self.config.paths.river_network_downsegid, default='DSLINKNO')
            length_col = self._get_config_value(lambda: self.config.paths.river_network_length, default='Length')
            slope_col = self._get_config_value(lambda: self.config.paths.river_network_slope, default='Slope')
            hru_to_seg_col = self._get_config_value(lambda: self.config.paths.river_basin_hru_to_seg, default='gru_to_seg')
            area_col = self._get_config_value(lambda: self.config.paths.river_basin_area, default='GRU_area')

            self._create_and_fill_nc_var(ncid, 'comid', 'i4', 'link', shp_river[seg_id_col], 'Unique segment ID')
            self._create_and_fill_nc_var(ncid, 'to_node', 'i4', 'link', shp_river[downseg_id_col], 'Downstream segment ID')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'link', shp_river[length_col], 'Segment length', 'meters')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'link', shp_river[slope_col], 'Segment slope', 'm/m')
            self._create_and_fill_nc_var(ncid, 'link_id_hru', 'i4', 'nhru', shp_basin[hru_to_seg_col], 'Segment ID for HRU discharge')
            self._create_and_fill_nc_var(ncid, 'hru_area_m2', 'f8', 'nhru', shp_basin[area_col], 'HRU area', 'm^2')

            # Add required placeholder variables with sensible defaults
            centroids = self.calculate_feature_centroids(shp_river)

            shp_river['lat'] = centroids.y
            shp_river['lon'] = centroids.x
            self._create_and_fill_nc_var(ncid, 'lat', 'f8', 'link', shp_river['lat'], 'Latitude of segment midpoint', 'degrees_north')
            self._create_and_fill_nc_var(ncid, 'lon', 'f8', 'link', shp_river['lon'], 'Longitude of segment midpoint', 'degrees_east')
            self._create_and_fill_nc_var(ncid, 'alt', 'f8', 'link', [0.0] * len(shp_river), 'Mean elevation of segment', 'meters')
            self._create_and_fill_nc_var(ncid, 'from_node', 'i4', 'link', [0] * len(shp_river), 'Upstream node ID')
            mannings_n = float(self._get_config_value(
                lambda: self.config.model.troute.mannings_n, default=0.035
            ))
            self._create_and_fill_nc_var(ncid, 'n', 'f8', 'link', [mannings_n] * len(shp_river), 'Mannings roughness coefficient')

            # Add drainage area from river network shapefile (DSContArea is in m²)
            if 'DSContArea' in shp_river.columns:
                da_km2 = shp_river['DSContArea'].values.astype(np.float64) / 1e6
                self._create_and_fill_nc_var(ncid, 'drainage_area_km2', 'f8', 'link', da_km2, 'Downstream contributing area', 'km^2')
                self.logger.info(f"Drainage area: {da_km2.min():.1f}-{da_km2.max():.1f} km²")

                # Compute channel width from hydraulic geometry: W = a * A^b
                hg_a = float(self._get_config_value(
                    lambda: self.config.model.troute.hg_width_coeff, default=2.71
                ))
                hg_b = float(self._get_config_value(
                    lambda: self.config.model.troute.hg_width_exp, default=0.557
                ))
                da_clamped = np.maximum(da_km2, 0.01)
                channel_width = np.maximum(hg_a * da_clamped ** hg_b, 1.0)
                self._create_and_fill_nc_var(ncid, 'channel_width', 'f8', 'link', channel_width, 'Channel width from hydraulic geometry', 'meters')
                self.logger.info(f"Channel width (W={hg_a}*A^{hg_b}): {channel_width.min():.1f}-{channel_width.max():.1f} m")
            else:
                self.logger.warning("DSContArea not found in river shapefile — channel width will use fallback")

            # Add stream order if available
            if 'strmOrder' in shp_river.columns:
                self._create_and_fill_nc_var(ncid, 'stream_order', 'i4', 'link', shp_river['strmOrder'], 'Strahler stream order')

        self.logger.info(f"t-route topology file created at {topology_filepath}")

    def create_troute_yaml_config(self):
        """Creates the t-route YAML configuration file from SYMFLUENCE config settings."""
        self.logger.info("Creating t-route YAML configuration file...")

        # Determine paths and parameters from config
        source_model = self._get_config_value(
            lambda: self.config.model.troute.from_model, default='SUMMA'
        ).upper()
        input_dir = self.project_dir / f"simulations/{self.experiment_id}" / source_model
        output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'troute'
        topology_name = self._get_config_value(
            lambda: self.config.model.troute.topology_file, default='troute_topology.nc'
        )

        # Calculate nts (Number of Timesteps)
        start_dt = datetime.fromisoformat(self.time_start)
        end_dt = datetime.fromisoformat(self.time_end)
        time_step_seconds = int(self._get_config_value(
            lambda: self.config.model.troute.dt_seconds, default=3600
        ))
        total_seconds = (end_dt - start_dt).total_seconds() + time_step_seconds
        nts = int(total_seconds / time_step_seconds)

        # Build configuration dictionary matching t-route's schema
        config_dict = {
            'log_parameters': {'showtiming': True, 'log_level': 'DEBUG'},
            'network_topology_parameters': {'supernetwork_parameters': {'geo_file_path': str(self.setup_dir / topology_name)}},
            'compute_parameters': {
                'restart_parameters': {'start_datetime': self.time_start},
                'forcing_parameters': {
                    'nts': nts,
                    'qlat_input_folder': str(input_dir),
                    'qlat_file_pattern_filter': f"{self.experiment_id}_timestep.nc"
                }
            },
            'output_parameters': {'stream_output': {'stream_output_directory': str(output_dir)}}
        }

        # Write dictionary to YAML file
        yaml_filename = self._get_config_value(
            lambda: self.config.model.troute.config_file, default='troute_config.yml'
        )
        yaml_filepath = self.setup_dir / yaml_filename
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        self.logger.info(f"t-route YAML config written to {yaml_filepath}")

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, data, long_name, units='-'):
        """Helper to create and fill a NetCDF variable."""
        var = ncid.createVariable(var_name, var_type, (dim,))
        var[:] = data.values if hasattr(data, 'values') else data
        var.long_name = long_name
        var.units = units
