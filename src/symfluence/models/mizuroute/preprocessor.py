"""
MizuRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the mizuRoute routing model.
"""

import os
import sys
import pandas as pd
import netCDF4 as nc4
import geopandas as gpd
import numpy as np
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from typing import Dict, Any
import easymore
import xarray as xr

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPreProcessor
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin

def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")


@ModelRegistry.register_preprocessor('MIZUROUTE')
class MizuRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin):
    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "mizuRoute"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)
        
        self.logger.debug(f"MizuRoutePreProcessor initialized. Default setup_dir: {self.setup_dir}")
        
        # Override setup_dir if SETTINGS_MIZU_PATH is provided (for isolated parallel runs)
        mizu_settings_path = self.config_dict.get('SETTINGS_MIZU_PATH')
        if mizu_settings_path and mizu_settings_path != 'default':
            self.setup_dir = Path(mizu_settings_path)
            self.logger.debug(f"MizuRoutePreProcessor using custom setup_dir from SETTINGS_MIZU_PATH: {self.setup_dir}")
        
        # Ensure setup directory exists
        if not self.setup_dir.exists():
            self.logger.info(f"Creating mizuRoute setup directory: {self.setup_dir}")
            self.setup_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logger.debug(f"mizuRoute setup directory already exists: {self.setup_dir}")


    def run_preprocessing(self):
        self.logger.debug("Starting mizuRoute spatial preprocessing")
        self.copy_base_settings()
        self.create_network_topology_file()

        # Phase 3: Use typed config when available
        if self.config:
            needs_remap = self.config.model.mizuroute.needs_remap if self.config.model.mizuroute else False
            from_model = self.config.model.mizuroute.from_model if self.config.model.mizuroute else None
            fuse_routing = self._resolve_config_value(
                lambda: self.config.model.fuse.routing_integration if self.config.model.fuse else None,
                'FUSE_ROUTING_INTEGRATION'
            )
            gr_routing = self._resolve_config_value(
                lambda: self.config.model.gr.routing_integration if self.config.model.gr else None,
                'GR_ROUTING_INTEGRATION'
            )
        else:
            needs_remap = self.config_dict.get('SETTINGS_MIZU_NEEDS_REMAP')
            from_model = self.config_dict.get('MIZU_FROM_MODEL')
            fuse_routing = self.config_dict.get('FUSE_ROUTING_INTEGRATION')
            gr_routing = self.config_dict.get('GR_ROUTING_INTEGRATION')

        # Check if lumped-to-distributed remapping is needed (set during topology creation)
        if getattr(self, 'needs_remap_lumped_distributed', False):
            self.logger.info("Creating area-weighted remap file for lumped-to-distributed routing")
            self.create_area_weighted_remap_file()
            needs_remap = True  # Override to enable remapping in control file

        self.logger.info(f"Should we remap?: {needs_remap}")
        if needs_remap and not getattr(self, 'needs_remap_lumped_distributed', False):
            self.remap_summa_catchments_to_routing()

        # Choose control writer based on source model
        if from_model == 'FUSE' or fuse_routing == 'mizuRoute':
            self.create_fuse_control_file()
        elif from_model == 'GR' or gr_routing == 'mizuRoute':
            self.create_gr_control_file()
        else:
            self.create_control_file()

        self.logger.info("mizuRoute spatial preprocessing completed")


    def copy_base_settings(self):
        self.logger.info("Copying mizuRoute base settings")
        from symfluence.resources import get_base_settings_dir
        base_settings_path = get_base_settings_dir('mizuRoute')
        self.setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.setup_dir / file)
        self.logger.info("mizuRoute base settings copied")

    def create_area_weighted_remap_file(self):
        """Create remapping file with area-based weights from delineated catchments"""
        self.logger.info("Creating area-weighted remapping file")
        
        # Load topology to get HRU information
        topology_file = self.setup_dir / self.config_dict.get('SETTINGS_MIZU_TOPOLOGY')
        with xr.open_dataset(topology_file) as topo:
            hru_ids = topo['hruId'].values
        
        n_hrus = len(hru_ids)
        
        # Use the weights stored during topology creation
        if hasattr(self, 'subcatchment_weights') and hasattr(self, 'subcatchment_gru_ids'):
            weights = self.subcatchment_weights
            gru_ids = self.subcatchment_gru_ids
        else:
            # Fallback: load from delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.config_dict.get('DOMAIN_NAME')}_catchment_delineated.shp"
            shp_catchments = gpd.read_file(catchment_path)
            weights = shp_catchments['avg_subbas'].values
            gru_ids = shp_catchments['GRU_ID'].values.astype(int)
        
        remap_name = self.config_dict.get('SETTINGS_MIZU_REMAP')
        
        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Area-weighted remapping for lumped to distributed routing')
            
            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU
            
            # Create variables
            # RN_hruId: The routing HRU IDs (from delineated catchments)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = gru_ids
            rn_hru.long_name = 'River network HRU ID'
            
            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'
            
            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'
            
            # weight: Area-based weights from delineated catchments
            weight_var = ncid.createVariable('weight', 'f8', ('data',))
            weight_var[:] = weights
            weight_var.long_name = 'Areal weights based on delineated subcatchment areas'
        
        self.logger.info(f"Area-weighted remapping file created with {n_hrus} HRUs")
        self.logger.info(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
        self.logger.info(f"Weight sum: {weights.sum():.4f}")



    def create_gr_control_file(self):
        """Create mizuRoute control file specifically for GR4J input"""
        self.logger.debug("Creating mizuRoute control file for GR4J")
        
        control_name = self.config_dict.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuRoute_control_GR.txt')
        
        with open(self.setup_dir / control_name, 'w') as cf:
            self._write_control_file_header(cf)
            self._write_gr_control_file_directories(cf)
            self._write_control_file_parameters(cf)
            self._write_control_file_simulation_controls(cf)
            self._write_control_file_topology(cf)
            self._write_gr_control_file_runoff(cf)
            self._write_control_file_remapping(cf)
            self._write_control_file_miscellaneous(cf)

    def _write_gr_control_file_directories(self, cf):
        """Write GR-specific directory paths for mizuRoute control"""
        experiment_output_gr = self.config_dict.get('EXPERIMENT_OUTPUT_GR', 'default')
        experiment_output_mizuroute = self.config_dict.get('EXPERIMENT_OUTPUT_MIZUROUTE')

        if experiment_output_gr == 'default':
            experiment_output_gr = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'GR'
        else:
            experiment_output_gr = Path(experiment_output_gr)

        if experiment_output_mizuroute == 'default' or not experiment_output_mizuroute:
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)
            
        # Ensure output directory exists
        experiment_output_mizuroute.mkdir(parents=True, exist_ok=True)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_gr}/    ! Folder that contains runoff data from GR4J \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")

    def _write_gr_control_file_runoff(self, cf):
        """Write GR-specific runoff file settings"""
        # Handle 'default' values - use actual defaults for mizuRoute compatibility
        routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        if routing_var in ('default', None, ''):
            routing_var = 'q_routed'

        routing_units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
        if routing_units in ('default', None, ''):
            routing_units = 'm/s'

        # GR output from airGR is currently daily (86400s)
        # We force this here to ensure mizuRoute time matching works correctly
        routing_dt = '86400'

        # GR saves output as {domain_name}_{experiment_id}_runs_def.nc
        domain_name = self.config_dict.get('DOMAIN_NAME')
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        gr_output_file = f"{domain_name}_{experiment_id}_runs_def.nc"

        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {gr_output_file}    ! netCDF name for GR4J runoff \n")
        cf.write(f"<vname_qsim>            {routing_var}    ! Variable name for GR4J runoff \n")
        cf.write(f"<units_qsim>            {routing_units}    ! Units of input runoff \n")
        cf.write(f"<dt_qsim>               {routing_dt}    ! Time interval of input runoff in seconds \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        cf.write("<dname_hruid>           gru     ! Dimension name for HM_HRU ID \n")
        cf.write("<vname_hruid>           gruId   ! Variable name for HM_HRU ID \n")
        cf.write("<calendar>              standard    ! Calendar of the nc file \n")



    def _check_if_headwater_basin(self, shp_river):
        """
        Check if this is a headwater basin with None/invalid river network data.
        
        Args:
            shp_river: GeoDataFrame of river network
            
        Returns:
            bool: True if this appears to be a headwater basin with invalid network data
        """
        # Check for critical None values in key columns
        seg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_SEGID')
        downseg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')
        
        if seg_id_col in shp_river.columns and downseg_id_col in shp_river.columns:
            # Check if all segment IDs are None/null
            seg_ids_null = shp_river[seg_id_col].isna().all()
            downseg_ids_null = shp_river[downseg_id_col].isna().all()
            
            if seg_ids_null and downseg_ids_null:
                self.logger.info("Detected headwater basin: all river network IDs are None/null")
                return True
                
            # Also check for string 'None' values (sometimes shapefiles store None as string)
            if shp_river[seg_id_col].dtype == 'object':
                seg_ids_none_str = (shp_river[seg_id_col] == 'None').all()
                downseg_ids_none_str = (shp_river[downseg_id_col] == 'None').all()
                
                if seg_ids_none_str and downseg_ids_none_str:
                    self.logger.info("Detected headwater basin: all river network IDs are 'None' strings")
                    return True
        
        return False

    def _create_synthetic_river_network(self, shp_river, hru_ids):
        """
        Create a synthetic single-segment river network for headwater basins.
        
        Args:
            shp_river: Original GeoDataFrame (with None values)
            hru_ids: Array of HRU IDs from delineated catchments
            
        Returns:
            GeoDataFrame: Modified river network with synthetic single segment
        """
        self.logger.info("Creating synthetic river network for headwater basin")
        
        # Use the first HRU ID as the segment ID (should be reasonable identifier)
        synthetic_seg_id = int(hru_ids[0]) if len(hru_ids) > 0 else 1
        
        # Create synthetic values for the single segment
        synthetic_data = {
            self.config_dict.get('RIVER_NETWORK_SHP_SEGID'): synthetic_seg_id,
            self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID'): 0,  # Outlet (downstream ID = 0)
            self.config_dict.get('RIVER_NETWORK_SHP_LENGTH'): 1000.0,  # Default 1 km length
            self.config_dict.get('RIVER_NETWORK_SHP_SLOPE'): 0.001,  # Default 0.1% slope
        }
        
        # Get the geometry column name (usually 'geometry')
        geom_col = shp_river.geometry.name
        
        # Create a simple point geometry at the centroid of the original (if it exists)
        if not shp_river.empty and shp_river.geometry.iloc[0] is not None:
            # Use the centroid of the first geometry, handling CRS projection via mixin
            synthetic_geom = self.calculate_feature_centroids(shp_river.iloc[[0]]).iloc[0]
        else:
            # Create a default point geometry (this won't be used for actual routing)
            from shapely.geometry import Point
            synthetic_geom = Point(0, 0)
        
        synthetic_data[geom_col] = synthetic_geom
        
        # Create new GeoDataFrame with single row
        synthetic_gdf = gpd.GeoDataFrame([synthetic_data], crs=shp_river.crs)
        
        self.logger.info(f"Created synthetic river network: segment ID {synthetic_seg_id} (outlet)")
        
        return synthetic_gdf

    def create_network_topology_file(self):
        self.logger.info("Creating network topology file")

        # Check for grid-based distribute mode
        is_grid_distribute = self.config_dict.get('DOMAIN_DEFINITION_METHOD') == 'distribute'
        if is_grid_distribute:
            self._create_grid_topology_file()
            return

        river_network_path = self.config_dict.get('RIVER_NETWORK_SHP_PATH')
        river_network_name = self.config_dict.get('RIVER_NETWORK_SHP_NAME')
        method_suffix = self._get_method_suffix()

        # Check if this is lumped domain with distributed routing
        # If so, use the delineated river network (from distributed delineation)
        is_lumped_to_distributed = (
            self.config_dict.get('DOMAIN_DEFINITION_METHOD') == 'lumped' and
            self.config_dict.get('ROUTING_DELINEATION', 'river_network') == 'river_network'
        )

        # For lumped-to-distributed, use delineated river network and catchments
        routing_suffix = 'delineate' if is_lumped_to_distributed else method_suffix

        if river_network_name == 'default':
            river_network_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverNetwork_{routing_suffix}.shp"

        if river_network_path == 'default':
            river_network_path = self.project_dir / 'shapefiles/river_network'
        else:
            river_network_path = Path(river_network_path)

        river_basin_path = self.config_dict.get('RIVER_BASINS_PATH')
        river_basin_name = self.config_dict.get('RIVER_BASINS_NAME')

        if river_basin_name == 'default':
            river_basin_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_{routing_suffix}.shp"

        if river_basin_path == 'default':
            river_basin_path = self.project_dir / 'shapefiles/river_basins'
        else:
            river_basin_path = Path(river_basin_path)        

        topology_name = self.config_dict.get('SETTINGS_MIZU_TOPOLOGY')
        
        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)

        if is_lumped_to_distributed:
            self.logger.info("Using delineated catchments for lumped-to-distributed routing")

            # For lumped-to-distributed, SUMMA output is converted to gru/gruId format
            # by the spatial_orchestrator, so mizuRoute control file should use gru/gruId
            self.summa_uses_gru_runoff = True

            # Enable remapping: map single lumped SUMMA GRU to 25 routing HRUs with area weights
            self.needs_remap_lumped_distributed = True

            # Load the delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.config_dict.get('DOMAIN_NAME')}_catchment_delineated.shp"
            if not catchment_path.exists():
                raise FileNotFoundError(f"Delineated catchment shapefile not found: {catchment_path}")

            shp_catchments = gpd.read_file(catchment_path)
            self.logger.info(f"Loaded {len(shp_catchments)} delineated subcatchments")
            
            # Extract HRU data from delineated catchments
            hru_ids = shp_catchments['GRU_ID'].values.astype(int)
            
            # Check if we have a headwater basin (None values in river network)
            if self._check_if_headwater_basin(shp_river):
                # Create synthetic river network for headwater basin
                shp_river = self._create_synthetic_river_network(shp_river, hru_ids)
            
            # Use the delineated catchments as HRUs
            num_seg = len(shp_river)
            num_hru = len(shp_catchments)
            
            hru_to_seg_ids = shp_catchments['GRU_ID'].values.astype(int)  # Each GRU drains to segment with same ID
            
            # Convert fractional areas to actual areas (multiply by total basin area)
            total_basin_area = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_AREA')].sum()
            hru_areas = shp_catchments['avg_subbas'].values * total_basin_area
            
            # Store fractional areas for remapping
            self.subcatchment_weights = shp_catchments['avg_subbas'].values
            self.subcatchment_gru_ids = hru_ids
            
            self.logger.info(f"Created {num_hru} HRUs from delineated catchments")
            self.logger.info(f"Weight range: {self.subcatchment_weights.min():.4f} to {self.subcatchment_weights.max():.4f}")
            
        else:
            # Check if we have SUMMA attributes file with multiple HRUs per GRU
            attributes_path = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
            
            if attributes_path.exists():
                with nc4.Dataset(attributes_path, 'r') as attrs:
                    n_hrus = len(attrs.dimensions['hru'])
                    n_grus = len(attrs.dimensions['gru'])
                    
                    if n_hrus > n_grus:
                        # Multiple HRUs per GRU - SUMMA will output GRU-level runoff
                        # mizuRoute should route at GRU level
                        self.logger.info(f"Distributed SUMMA with {n_hrus} HRUs across {n_grus} GRUs")
                        self.logger.info("Creating GRU-level topology for mizuRoute (SUMMA outputs averageRoutedRunoff at GRU level)")
                        
                        # Read GRU information from SUMMA attributes file
                        gru_ids = attrs.variables['gruId'][:].astype(int)
                        
                        # For distributed SUMMA, GRU IDs should match segment IDs
                        hru_ids = gru_ids  # mizuRoute will read GRU-level data
                        hru_to_seg_ids = gru_ids  # Each GRU drains to segment with same ID
                        
                        # Calculate GRU areas by summing HRU areas within each GRU
                        hru2gru = attrs.variables['hru2gruId'][:].astype(int)
                        hru_areas_all = attrs.variables['HRUarea'][:].astype(float)
                        
                        # Sum areas for each GRU
                        gru_areas = np.zeros(n_grus)
                        for i, gru_id in enumerate(gru_ids):
                            gru_mask = hru2gru == gru_id
                            gru_areas[i] = hru_areas_all[gru_mask].sum()
                        
                        hru_areas = gru_areas
                        
                        num_seg = len(shp_river)
                        num_hru = n_grus  # mizuRoute sees GRUs as HRUs
                        
                        # Store flag for control file generation
                        self.summa_uses_gru_runoff = True
                        
                        self.logger.info(f"Created topology with {num_hru} GRUs for mizuRoute routing")
                    else:
                        # Lumped modeling: use original logic
                        self.summa_uses_gru_runoff = False
                        closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)
                        
                        if len(shp_basin) == 1:
                            shp_basin.loc[0, self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')] = closest_segment_id
                            self.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")
                        
                        num_seg = len(shp_river)
                        num_hru = len(shp_basin)
                        
                        hru_ids = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_RM_GRUID')].values.astype(int)
                        hru_to_seg_ids = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')].values.astype(int)
                        hru_areas = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_AREA')].values.astype(float)
            else:
                # No attributes file: use original logic
                self.summa_uses_gru_runoff = False
                closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)
                
                if len(shp_basin) == 1:
                    shp_basin.loc[0, self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')] = closest_segment_id
                    self.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")
                
                num_seg = len(shp_river)
                num_hru = len(shp_basin)
                
                hru_ids = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_RM_GRUID')].values.astype(int)
                hru_to_seg_ids = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')].values.astype(int)
                hru_areas = shp_basin[self.config_dict.get('RIVER_BASIN_SHP_AREA')].values.astype(float)
        
        # Ensure minimum segment length - now safe from None values
        length_col = self.config_dict.get('RIVER_NETWORK_SHP_LENGTH')
        if length_col in shp_river.columns:
            # Convert None/null values to 0 first, then set minimum
            shp_river[length_col] = shp_river[length_col].fillna(0)
            shp_river.loc[shp_river[length_col] == 0, length_col] = 1
        
        # Ensure slope column has valid values
        slope_col = self.config_dict.get('RIVER_NETWORK_SHP_SLOPE')
        if slope_col in shp_river.columns:
            shp_river[slope_col] = shp_river[slope_col].fillna(0.001)  # Default slope
            shp_river.loc[shp_river[slope_col] == 0, slope_col] = 0.001
        
        # Enforce outlets if specified
        if self.config_dict.get('SETTINGS_MIZU_MAKE_OUTLET') != 'n/a':
            river_outlet_ids = [int(id) for id in self.config_dict.get('SETTINGS_MIZU_MAKE_OUTLET').split(',')]
            seg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_SEGID')
            downseg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')
            
            for outlet_id in river_outlet_ids:
                if outlet_id in shp_river[seg_id_col].values:
                    shp_river.loc[shp_river[seg_id_col] == outlet_id, downseg_id_col] = 0
                else:
                    self.logger.warning(f"Outlet ID {outlet_id} not found in river network")
        
        # Create the netCDF file
        with nc4.Dataset(self.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_seg, num_hru)
            
            # Create segment variables (now safe from None values)
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].values.astype(int), 'Unique ID of each stream segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')].values.astype(int), 'ID of the downstream segment', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SLOPE')].values.astype(float), 'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_LENGTH')].values.astype(float), 'Segment length', 'm')
            
            # Create HRU variables (using our computed values)
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids, 'Unique hru ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids, 'ID of the stream segment to which the HRU discharges', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas, 'HRU area', 'm^2')
        
        self.logger.info(f"Network topology file created at {self.setup_dir / topology_name}")
        
    def _find_closest_segment_to_pour_point(self, shp_river):
        """
        Find the river segment closest to the pour point.
        
        Args:
            shp_river: GeoDataFrame of river network
            
        Returns:
            int: Segment ID of closest segment to pour point
        """
        from pathlib import Path
        import numpy as np
        
        # Find pour point shapefile
        pour_point_dir = self.project_dir / 'shapefiles' / 'pour_point'
        pour_point_files = list(pour_point_dir.glob('*.shp'))
        
        if not pour_point_files:
            self.logger.error(f"No pour point shapefiles found in {pour_point_dir}")
            # Fallback: use outlet segment (downSegId == 0)
            outlet_mask = shp_river[self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                # Last resort: use first segment
                fallback_seg = shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg
        
        # Load first pour point file
        pour_point_file = pour_point_files[0]
        self.logger.info(f"Loading pour point from {pour_point_file}")
        
        try:
            shp_pour_point = gpd.read_file(pour_point_file)
            
            # Ensure both are in the same CRS
            if shp_river.crs != shp_pour_point.crs:
                shp_pour_point = shp_pour_point.to_crs(shp_river.crs)
            
            # Get pour point coordinates (assume first/only point)
            pour_point_geom = shp_pour_point.geometry.iloc[0]
            
            # Calculate distances from pour point to all river segments
            shp_river_proj = shp_river.to_crs(shp_river.estimate_utm_crs())
            # Use mixin to get pour point centroid safely if needed (though it's a point)
            pour_point_centroids = self.calculate_feature_centroids(shp_pour_point.iloc[[0]])
            pour_point_proj = pour_point_centroids.to_crs(shp_river_proj.crs)
            distances = shp_river_proj.geometry.distance(pour_point_proj.iloc[0])
            
            # Find closest segment
            closest_idx = distances.idxmin()
            closest_segment_id = shp_river.loc[closest_idx, self.config_dict.get('RIVER_NETWORK_SHP_SEGID')]
            
            self.logger.info(f"Closest segment to pour point: {closest_segment_id} (distance: {distances.iloc[closest_idx]:.1f} units)")
            
            return closest_segment_id
            
        except Exception as e:
            self.logger.error(f"Error finding closest segment: {str(e)}")
            # Fallback to outlet segment
            outlet_mask = shp_river[self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                fallback_seg = shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

    def create_equal_weight_remap_file(self):
        """Create remapping file with equal weights for all segments"""
        self.logger.info("Creating equal-weight remapping file")
        
        # Load topology to get segment information
        topology_file = self.setup_dir / self.config_dict.get('SETTINGS_MIZU_TOPOLOGY')
        with xr.open_dataset(topology_file) as topo:
            seg_ids = topo['segId'].values
            hru_ids = topo['hruId'].values  # Now we have multiple HRUs
        
        n_segments = len(seg_ids)
        n_hrus = len(hru_ids)
        equal_weight = 1.0 / n_hrus  # Equal weight for each HRU
        
        remap_name = self.config_dict.get('SETTINGS_MIZU_REMAP')
        
        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Equal-weight remapping for lumped to distributed routing')
            
            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU
            
            # Create variables
            # RN_hruId: The routing HRU IDs (1, 2, 3, ..., n_hrus)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = hru_ids
            rn_hru.long_name = 'River network HRU ID'
            
            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'
            
            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'
            
            # weight: Equal weights for all HRUs
            weights = ncid.createVariable('weight', 'f8', ('data',))
            weights[:] = [equal_weight] * n_hrus
            weights.long_name = f'Equal areal weights ({equal_weight:.4f}) for all HRUs'
        
        self.logger.info(f"Equal-weight remapping file created with {n_hrus} HRUs, weight = {equal_weight:.4f}")

    def remap_summa_catchments_to_routing(self):
        self.logger.info("Remapping SUMMA catchments to routing catchments")
        if self.config_dict.get('DOMAIN_DEFINITION_METHOD') == 'lumped' and self.config_dict.get('ROUTING_DELINEATION') == 'river_network':
            self.logger.info("Area-weighted mapping for SUMMA catchments to routing catchments")
            self.create_area_weighted_remap_file()  # Changed from create_equal_weight_remap_file
            return

        hm_catchment_path = Path(self.config_dict.get('CATCHMENT_PATH'))
        hm_catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if hm_catchment_name == 'default':
            hm_catchment_name = f"{self.config_dict.get('DOMAIN_NAME')}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"

        rm_catchment_path = Path(self.config_dict.get('RIVER_BASINS_PATH'))
        rm_catchment_name = self.config_dict.get('RIVER_BASINS_NAME')
        
        intersect_path = Path(self.config_dict.get('INTERSECT_ROUTING_PATH'))
        intersect_name = self.config_dict.get('INTERSECT_ROUTING_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_routing_basins.shp'
        
        if intersect_path == 'default':
            intersect_path = self.project_dir / 'shapefiles/catchment_intersection' 
        else:
            intersect_path = Path(intersect_path)

        remap_name = self.config_dict.get('SETTINGS_MIZU_REMAP')
        
        if hm_catchment_path == 'default':
            hm_catchment_path = self.project_dir / 'shapefiles/catchment' 
        else:
            hm_catchment_path = Path(hm_catchment_path)
            
        if rm_catchment_path == 'default':
            rm_catchment_path = self.project_dir / 'shapefiles/catchment' 
        else:
            rm_catchment_path = Path(rm_catchment_path)

        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(rm_catchment_path / rm_catchment_name)
        
        # Create intersection
        esmr_obj = _create_easymore_instance()
        hm_shape = hm_shape.to_crs('EPSG:6933')
        rm_shape = rm_shape.to_crs('EPSG:6933')
        intersected_shape = esmr_obj.intersection_shp(rm_shape, hm_shape)
        intersected_shape = intersected_shape.to_crs('EPSG:4326')
        intersected_shape.to_file(intersect_path / intersect_name)
        
        # Process variables for remapping file
        self._process_remap_variables(intersected_shape)
        
        # Create remapping netCDF file
        self._create_remap_file(intersected_shape, remap_name)
        
        self.logger.info(f"Remapping file created at {self.setup_dir / remap_name}")

    def create_control_file(self):
        self.logger.debug("Creating mizuRoute control file")
        
        control_name = self.config_dict.get('SETTINGS_MIZU_CONTROL_FILE')
        
        with open(self.setup_dir / control_name, 'w') as cf:
            self._write_control_file_header(cf)
            self._write_control_file_directories(cf)
            self._write_control_file_parameters(cf)
            self._write_control_file_simulation_controls(cf)
            self._write_control_file_topology(cf)
            self._write_control_file_runoff(cf)
            self._write_control_file_remapping(cf)
            self._write_control_file_miscellaneous(cf)
        
        self.logger.debug(f"mizuRoute control file created at {self.setup_dir / control_name}")

    def _write_control_file_runoff(self, cf):
        """Write SUMMA-specific runoff file settings"""
        # Check if we should use GRU-level or HRU-level data
        uses_gru = getattr(self, 'summa_uses_gru_runoff', False)

        # Handle 'default' values - use actual defaults for mizuRoute compatibility
        routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        if routing_var in ('default', None, ''):
            routing_var = 'averageRoutedRunoff'

        routing_units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
        if routing_units in ('default', None, ''):
            routing_units = 'm/s'

        routing_dt = self.config_dict.get('SETTINGS_MIZU_ROUTING_DT', '3600')
        if routing_dt in ('default', None, ''):
            routing_dt = '3600'

        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {self.config_dict.get('EXPERIMENT_ID')}_timestep.nc    ! netCDF name for SUMMA runoff \n")
        cf.write(f"<vname_qsim>            {routing_var}    ! Variable name for SUMMA runoff \n")
        cf.write(f"<units_qsim>            {routing_units}    ! Units of input runoff \n")
        cf.write(f"<dt_qsim>               {routing_dt}    ! Time interval of input runoff in seconds \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        
        if uses_gru:
            # Distributed SUMMA outputs GRU-level runoff
            cf.write("<dname_hruid>           gru     ! Dimension name for HM_HRU ID (GRU level for distributed SUMMA) \n")
            cf.write("<vname_hruid>           gruId   ! Variable name for HM_HRU ID (GRU level for distributed SUMMA) \n")
        else:
            # Standard HRU-level runoff
            cf.write("<dname_hruid>           hru     ! Dimension name for HM_HRU ID \n")
            cf.write("<vname_hruid>           hruId   ! Variable name for HM_HRU ID \n")
            
        cf.write("<calendar>              standard    ! Calendar of the nc file \n")

    def _set_topology_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a river network .nc file for mizuRoute routing')

    def _create_topology_dimensions(self, ncid, num_seg, num_hru):
        ncid.createDimension('seg', num_seg)
        ncid.createDimension('hru', num_hru)

    def _create_grid_topology_file(self):
        """
        Create mizuRoute topology for grid-based distributed modeling.

        Each grid cell becomes both an HRU and a segment. D8 flow direction
        determines segment connectivity.
        """
        self.logger.info("Creating grid-based network topology for distributed mode")

        # Load grid shapefile with D8 topology
        grid_path = self.project_dir / 'shapefiles' / 'river_basins' / f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_distribute.shp"

        if not grid_path.exists():
            self.logger.error(f"Grid basins shapefile not found: {grid_path}")
            raise FileNotFoundError(f"Grid basins not found: {grid_path}")

        grid_gdf = gpd.read_file(grid_path)
        num_cells = len(grid_gdf)

        self.logger.info(f"Loaded {num_cells} grid cells from {grid_path}")

        topology_name = self.config_dict.get('SETTINGS_MIZU_TOPOLOGY')

        # Extract topology data from grid shapefile
        seg_ids = grid_gdf['GRU_ID'].values.astype(int)

        # Get downstream IDs from D8 topology
        # Note: shapefile truncates column names to 10 chars, so downstream_id becomes downstream
        if 'downstream_id' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream_id'].values.astype(int)
        elif 'downstream' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream'].values.astype(int)
        elif 'DSLINKNO' in grid_gdf.columns:
            down_seg_ids = grid_gdf['DSLINKNO'].values.astype(int)
        else:
            self.logger.warning("No D8 topology found, setting all cells as outlets")
            down_seg_ids = np.zeros(num_cells, dtype=int)

        # Get slopes from grid
        if 'slope' in grid_gdf.columns:
            slopes = grid_gdf['slope'].values.astype(float)
            # Ensure minimum slope
            slopes = np.maximum(slopes, 0.001)
        else:
            self.logger.warning("No slope data found, using default 0.01")
            slopes = np.full(num_cells, 0.01)

        # Get cell size for segment length
        grid_cell_size = self.config_dict.get('GRID_CELL_SIZE', 1000.0)
        lengths = np.full(num_cells, float(grid_cell_size))

        # HRU variables (each cell is also an HRU)
        hru_ids = seg_ids.copy()
        hru_to_seg_ids = seg_ids.copy()  # Each HRU drains to its own segment

        # Get HRU areas
        if 'GRU_area' in grid_gdf.columns:
            hru_areas = grid_gdf['GRU_area'].values.astype(float)
        else:
            self.logger.warning("No area data found, using cell size squared")
            hru_areas = np.full(num_cells, grid_cell_size ** 2)

        # Create the netCDF topology file
        with nc4.Dataset(self.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_cells, num_cells)

            # Create segment variables
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', seg_ids,
                                         'Unique ID of each grid cell segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', down_seg_ids,
                                         'ID of downstream grid cell (0=outlet)', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', slopes,
                                         'Grid cell slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', lengths,
                                         'Grid cell length (cell size)', 'm')

            # Create HRU variables
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids,
                                         'Unique HRU ID (=grid cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids,
                                         'Segment to which HRU drains (=cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas,
                                         'HRU area', 'm^2')

        # Count outlets for logging
        n_outlets = np.sum(down_seg_ids == 0)
        self.logger.info(f"Grid topology created: {num_cells} cells, {n_outlets} outlets")
        self.logger.info(f"Topology file: {self.setup_dir / topology_name}")

        # Set flag for control file - grid cells use GRU-level runoff
        self.summa_uses_gru_runoff = True

    def create_fuse_control_file(self):
        """Create mizuRoute control file specifically for FUSE input"""
        self.logger.debug("Creating mizuRoute control file for FUSE")
        
        control_name = self.config_dict.get('SETTINGS_MIZU_CONTROL_FILE')
        
        with open(self.setup_dir / control_name, 'w') as cf:
            self._write_control_file_header(cf)
            self._write_fuse_control_file_directories(cf)  # FUSE-specific directories
            self._write_control_file_parameters(cf)
            self._write_control_file_simulation_controls(cf)
            self._write_control_file_topology(cf)
            self._write_fuse_control_file_runoff(cf)  # FUSE-specific runoff settings
            self._write_control_file_remapping(cf)
            self._write_control_file_miscellaneous(cf)

    def _write_fuse_control_file_directories(self, cf):
        """Write FUSE-specific directory paths for mizuRoute control"""
        experiment_output_fuse = self.config_dict.get('EXPERIMENT_OUTPUT_FUSE')
        experiment_output_mizuroute = self.config_dict.get('EXPERIMENT_OUTPUT_MIZUROUTE')

        if experiment_output_fuse == 'default':
            experiment_output_fuse = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'FUSE'
        else:
            experiment_output_fuse = Path(experiment_output_fuse)

        if experiment_output_mizuroute == 'default' or not experiment_output_mizuroute:
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)
            
        # Ensure output directory exists
        experiment_output_mizuroute.mkdir(parents=True, exist_ok=True)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_fuse}/    ! Folder that contains runoff data from FUSE \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")


    def _write_fuse_control_file_runoff(self, cf):
        """Write FUSE-specific runoff file settings"""
        # Handle 'default' values - use actual defaults for mizuRoute compatibility
        routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        if routing_var in ('default', None, ''):
            routing_var = 'q_routed'

        routing_units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
        if routing_units in ('default', None, ''):
            routing_units = 'm/s'

        routing_dt = self.config_dict.get('SETTINGS_MIZU_ROUTING_DT', '3600')
        if routing_dt in ('default', None, ''):
            routing_dt = '3600'

        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {self.config_dict.get('EXPERIMENT_ID')}_timestep.nc    ! netCDF name for FUSE runoff \n")
        cf.write(f"<vname_qsim>            {routing_var}    ! Variable name for FUSE runoff \n")
        cf.write(f"<units_qsim>            {routing_units}    ! Units of input runoff \n")
        cf.write(f"<dt_qsim>               {routing_dt}    ! Time interval of input runoff in seconds \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        cf.write("<dname_hruid>           gru     ! Dimension name for HM_HRU ID \n")
        cf.write("<vname_hruid>           gruId   ! Variable name for HM_HRU ID \n")
        cf.write("<calendar>              standard    ! Calendar of the nc file \n")


    def _write_control_file_simulation_controls(self, cf):
        """Enhanced simulation control writing with proper time handling"""
        # Get simulation dates from config
        sim_start = self.config_dict.get('EXPERIMENT_TIME_START')
        sim_end = self.config_dict.get('EXPERIMENT_TIME_END')
        
        # Determine source model
        from_model = self.config_dict.get('MIZU_FROM_MODEL', '').upper()
        gr_routing = self.config_dict.get('GR_ROUTING_INTEGRATION', 'none').lower() == 'mizuroute'
        
        # Special handling for GR: force midnight alignment for daily data
        if from_model == 'GR' or gr_routing:
            if isinstance(sim_start, str):
                # Replace any time part with 00:00
                sim_start = sim_start.split(' ')[0] + " 00:00"
            if isinstance(sim_end, str):
                # Replace any time part with 00:00 (or keep as is if daily)
                sim_end = sim_end.split(' ')[0] + " 00:00"
            self.logger.debug(f"Forced GR simulation period to midnight: {sim_start} to {sim_end}")
        
        # Ensure dates are in proper format
        from datetime import datetime
        if isinstance(sim_start, str) and len(sim_start) == 10:  # YYYY-MM-DD format
            sim_start = f"{sim_start} 00:00"
        if isinstance(sim_end, str) and len(sim_end) == 10:  # YYYY-MM-DD format
            sim_end = f"{sim_end} 23:00"
        
        cf.write("!\n! --- DEFINE SIMULATION CONTROLS \n")
        cf.write(f"<case_name>             {self.config_dict.get('EXPERIMENT_ID')}    ! Simulation case name \n")
        cf.write(f"<sim_start>             {sim_start}    ! Time of simulation start \n")
        cf.write(f"<sim_end>               {sim_end}    ! Time of simulation end \n")
        cf.write(f"<route_opt>             {self.config_dict.get('SETTINGS_MIZU_OUTPUT_VARS')}    ! Option for routing schemes \n")
        cf.write(f"<newFileFrequency>      {self.config_dict.get('SETTINGS_MIZU_OUTPUT_FREQ')}    ! Frequency for new output files \n")

    def _create_topology_variables(self, ncid, shp_river, shp_basin):
        self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SEGID')].values.astype(int), 'Unique ID of each stream segment', '-')
        self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')].values.astype(int), 'ID of the downstream segment', '-')
        self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_SLOPE')].values.astype(float), 'Segment slope', '-')
        self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.config_dict.get('RIVER_NETWORK_SHP_LENGTH')].values.astype(float), 'Segment length', 'm')
        self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', shp_basin[self.config_dict.get('RIVER_BASIN_SHP_RM_GRUID')].values.astype(int), 'Unique hru ID', '-')
        self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', shp_basin[self.config_dict.get('RIVER_BASIN_SHP_HRU_TO_SEG')].values.astype(int), 'ID of the stream segment to which the HRU discharges', '-')
        self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', shp_basin[self.config_dict.get('RIVER_BASIN_SHP_AREA')].values.astype(float), 'HRU area', 'm^2')

    def _process_remap_variables(self, intersected_shape):
        int_rm_id = f"S_1_{self.config_dict.get('RIVER_BASIN_SHP_RM_HRUID')}"
        int_hm_id = f"S_2_{self.config_dict.get('CATCHMENT_SHP_GRUID')}"
        int_weight = 'AP1N'
        
        intersected_shape = intersected_shape.sort_values(by=[int_rm_id, int_hm_id])
        
        self.nc_rnhruid = intersected_shape.groupby(int_rm_id).agg({int_rm_id: pd.unique}).values.astype(int)
        self.nc_noverlaps = intersected_shape.groupby(int_rm_id).agg({int_hm_id: 'count'}).values.astype(int)
        
        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_hm_id: list}).values.tolist()
        self.nc_hmgruid = [item for sublist in multi_nested_list for item in sublist[0]]
        
        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_weight: list}).values.tolist()
        self.nc_weight = [item for sublist in multi_nested_list for item in sublist[0]]

    def _create_remap_file(self, intersected_shape, remap_name):
        num_hru = len(intersected_shape[f"S_1_{self.config_dict.get('RIVER_BASIN_SHP_RM_HRUID')}"].unique())
        num_data = len(intersected_shape)
        
        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            self._set_remap_attributes(ncid)
            self._create_remap_dimensions(ncid, num_hru, num_data)
            self._create_remap_variables(ncid)

    def _set_remap_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a remapping .nc file for mizuRoute routing')

    def _create_remap_dimensions(self, ncid, num_hru, num_data):
        ncid.createDimension('hru', num_hru)
        ncid.createDimension('data', num_data)

    def _create_remap_variables(self, ncid):
        self._create_and_fill_nc_var(ncid, 'RN_hruId', 'int', 'hru', self.nc_rnhruid, 'River network HRU ID', '-')
        self._create_and_fill_nc_var(ncid, 'nOverlaps', 'int', 'hru', self.nc_noverlaps, 'Number of overlapping HM_HRUs for each RN_HRU', '-')
        self._create_and_fill_nc_var(ncid, 'HM_hruId', 'int', 'data', self.nc_hmgruid, 'ID of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')
        self._create_and_fill_nc_var(ncid, 'weight', 'f8', 'data', self.nc_weight, 'Areal weight of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, fill_data, long_name, units):
        ncvar = ncid.createVariable(var_name, var_type, (dim,))
        ncvar[:] = fill_data
        ncvar.long_name = long_name
        ncvar.units = units

    def _write_control_file_header(self, cf):
        cf.write("! mizuRoute control file generated by SUMMA public workflow scripts \n")

    def _write_control_file_directories(self, cf):
        experiment_output_summa = self.config_dict.get('EXPERIMENT_OUTPUT_SUMMA')
        experiment_output_mizuroute = self.config_dict.get('EXPERIMENT_OUTPUT_MIZUROUTE')

        if experiment_output_summa == 'default':
            experiment_output_summa = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'SUMMA'
        else:
            experiment_output_summa = Path(experiment_output_summa)

        if experiment_output_mizuroute == 'default' or not experiment_output_mizuroute:
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.config_dict.get('EXPERIMENT_ID')}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)
            
        # Ensure output directory exists
        experiment_output_mizuroute.mkdir(parents=True, exist_ok=True)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_summa}/    ! Folder that contains runoff data from SUMMA \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")

    def _write_control_file_parameters(self, cf):
        cf.write("!\n! --- NAMELIST FILENAME \n")
        cf.write(f"<param_nml>             {self.config_dict.get('SETTINGS_MIZU_PARAMETERS')}    ! Spatially constant parameter namelist (should be stored in the ancil_dir) \n")


    def _write_control_file_topology(self, cf):
        cf.write("!\n! --- DEFINE TOPOLOGY FILE \n")
        cf.write(f"<fname_ntopOld>         {self.config_dict.get('SETTINGS_MIZU_TOPOLOGY')}    ! Name of input netCDF for River Network \n")
        cf.write("<dname_sseg>            seg    ! Dimension name for reach in river network netCDF \n")
        cf.write("<dname_nhru>            hru    ! Dimension name for RN_HRU in river network netCDF \n")
        cf.write("<seg_outlet>            -9999    ! Outlet reach ID at which to stop routing (i.e. use subset of full network). -9999 to use full network \n")
        cf.write("<varname_area>          area    ! Name of variable holding hru area \n")
        cf.write("<varname_length>        length    ! Name of variable holding segment length \n")
        cf.write("<varname_slope>         slope    ! Name of variable holding segment slope \n")
        cf.write("<varname_HRUid>         hruId    ! Name of variable holding HRU id \n")
        cf.write("<varname_hruSegId>      hruToSegId    ! Name of variable holding the stream segment below each HRU \n")
        cf.write("<varname_segId>         segId    ! Name of variable holding the ID of each stream segment \n")
        cf.write("<varname_downSegId>     downSegId    ! Name of variable holding the ID of the next downstream segment \n")

    def _write_control_file_remapping(self, cf):
        cf.write("!\n! --- DEFINE RUNOFF MAPPING FILE \n")
        # Check both config flag and lumped-to-distributed flag
        remap_flag = (
            self.config_dict.get('SETTINGS_MIZU_NEEDS_REMAP', '') == True or
            getattr(self, 'needs_remap_lumped_distributed', False)
        )
        cf.write(f"<is_remap>              {'T' if remap_flag else 'F'}    ! Logical to indicate runoff needs to be remapped to RN_HRU. T or F \n")

        if remap_flag:
            cf.write(f"<fname_remap>           {self.config_dict.get('SETTINGS_MIZU_REMAP')}    ! netCDF name of runoff remapping \n")
            cf.write("<vname_hruid_in_remap>  RN_hruId    ! Variable name for RN_HRUs \n")
            cf.write("<vname_weight>          weight    ! Variable name for areal weights of overlapping HM_HRUs \n")
            cf.write("<vname_qhruid>          HM_hruId    ! Variable name for HM_HRU ID \n")
            cf.write("<vname_num_qhru>        nOverlaps    ! Variable name for a numbers of overlapping HM_HRUs with RN_HRUs \n")
            cf.write("<dname_hru_remap>       hru    ! Dimension name for HM_HRU \n")
            cf.write("<dname_data_remap>      data    ! Dimension name for data \n")

    def _write_control_file_miscellaneous(self, cf):
        cf.write("!\n! --- MISCELLANEOUS \n")
        cf.write(f"<doesBasinRoute>        {self.config_dict.get('SETTINGS_MIZU_WITHIN_BASIN')}    ! Hillslope routing options. 0 -> no (already routed by SUMMA), 1 -> use IRF \n")

    def _get_default_time(self, time_key, default_year):
        time_value = self.config_dict.get(time_key)
        if time_value == 'default':
            raw_time = [
                    self.config_dict.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config_dict.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
            year = raw_time[0] if default_year == 'start' else raw_time[1]
            return f"{year}-{'01-01 00:00' if default_year == 'start' else '12-31 23:00'}"
        return time_value

    def _pad_string(self, string, pad_to=20):
        return f"{string:{pad_to}}"
