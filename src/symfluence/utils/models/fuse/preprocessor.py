import os
import sys
import time
import subprocess
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
from scipy import ndimage
import csv
import itertools
import re
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
from typing import Dict, List, Tuple, Any
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin, ObservationLoaderMixin, DatasetBuilderMixin
from .forcing_processor import FuseForcingProcessor
from .elevation_band_manager import FuseElevationBandManager
from .synthetic_data_generator import FuseSyntheticDataGenerator
from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.common.geospatial_utils import GeospatialUtilsMixin
from symfluence.utils.exceptions import (
    ModelExecutionError,
    FileOperationError,
    symfluence_error_handler
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from symfluence.utils.common.metrics import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_preprocessor('FUSE')
class FUSEPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, ObservationLoaderMixin, DatasetBuilderMixin):
    """
    Preprocessor for the FUSE (Framework for Understanding Structural Errors) model.
    Handles data preparation, PET calculation, and file setup for FUSE model runs.
    Inherits geospatial utilities from GeospatialUtilsMixin and observation loading from ObservationLoaderMixin.

    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        setup_dir (Path): Directory for FUSE setup files
        domain_name (str): Name of the domain being processed
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "FUSE"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # FUSE-specific paths
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'

        # Setup catchment path using base class helper
        self.catchment_path = self.get_catchment_path()
        self.catchment_name = self.catchment_path.name

        # Initialize forcing processor (use base class methods for callbacks)
        self.forcing_processor = FuseForcingProcessor(
            config=self.config,
            logger=self.logger,
            project_dir=self.project_dir,
            forcing_basin_path=self.forcing_basin_path,
            forcing_fuse_path=self.forcing_fuse_path,
            catchment_path=self.catchment_path,
            domain_name=self.domain_name,
            calculate_pet_callback=self._calculate_pet,
            calculate_catchment_centroid_callback=self.calculate_catchment_centroid,
            get_simulation_time_window_callback=self.get_simulation_time_window,
            subset_to_simulation_time_callback=self.subset_to_simulation_time
        )

        # Initialize elevation band manager
        self.elevation_band_manager = FuseElevationBandManager(
            config=self.config,
            logger=self.logger,
            project_dir=self.project_dir,
            forcing_fuse_path=self.forcing_fuse_path,
            catchment_path=self.catchment_path,
            domain_name=self.domain_name,
            calculate_catchment_centroid_callback=self.calculate_catchment_centroid
        )

        # Initialize synthetic data generator
        self.synthetic_data_generator = FuseSyntheticDataGenerator(logger=self.logger)

    def _get_fuse_file_id(self) -> str:
        """Get a short file ID for FUSE outputs and settings."""
        fuse_id = self.config.get('FUSE_FILE_ID')
        if not fuse_id:
            experiment_id = self.config.get('EXPERIMENT_ID', '')
            fuse_id = experiment_id[:6] if experiment_id else 'fuse'
            self.config['FUSE_FILE_ID'] = fuse_id
        return fuse_id

    def _get_timestep_config(self):
        """
        Get timestep configuration based on FORCING_TIME_STEP_SIZE.

        DEPRECATED: Use self.get_timestep_config() from base class instead.
        This method is kept for backward compatibility but delegates to base class.

        Returns:
            dict: Configuration with resample_freq, time_units, time_unit_factor,
                conversion_factor, and time_label
        """
        ts_config = self.get_timestep_config()
        if ts_config['time_unit'] == 'h':
            self.logger.info("FUSE forcing uses daily time units; resampling to daily for compatibility")
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,
                'time_label': 'daily',
                'timestep_seconds': 86400
            }
        return ts_config

    # NOTE: _get_simulation_time_window() and _subset_to_simulation_time() are now
    # inherited from BaseModelPreProcessor as get_simulation_time_window() and
    # subset_to_simulation_time()

    def run_preprocessing(self):
        """
        Run the complete FUSE preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting FUSE preprocessing")
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """FUSE-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data()

    def _create_model_configs(self) -> None:
        """FUSE-specific configuration file creation (template hook)."""
        self.create_elevation_bands()
        self.update_input_info()
        self.create_filemanager()

    def create_directories(self):
        """Create necessary directories for FUSE setup."""
        dirs_to_create = [
            self.setup_dir,
            self.forcing_fuse_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def copy_base_settings(self):
        """
        Copy FUSE base settings from the source directory to the project's settings directory.
        Updates the model ID in the decisions file name to match the experiment ID.

        This method performs the following steps:
        1. Determines the source directory for base settings
        2. Determines the destination directory for settings
        3. Creates the destination directory if it doesn't exist
        4. Copies all files from the source directory to the destination directory
        5. Renames the decisions file with the appropriate experiment ID

        Raises:
            FileNotFoundError: If the source directory or any source file is not found.
            PermissionError: If there are permission issues when creating directories or copying files.
        """
        self.logger.info("Copying FUSE base settings")
        
        base_settings_path = Path(self.config.get('SYMFLUENCE_CODE_DIR')) / '0_base_settings' / 'FUSE'
        settings_path = self._get_default_path('SETTINGS_FUSE_PATH', 'settings/FUSE')
        
        try:
            settings_path.mkdir(parents=True, exist_ok=True)
            
            fuse_id = self._get_fuse_file_id()
            decision_file_path = None
            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file
                
                # Handle the decisions file specially
                if 'fuse_zDecisions_' in file:
                    # Create new filename with experiment ID
                    new_filename = file.replace('902', fuse_id)
                    dest_file = settings_path / new_filename
                    decision_file_path = dest_file
                    self.logger.debug(f"Renaming decisions file from {file} to {new_filename}")
                else:
                    dest_file = settings_path / file
                
                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")
            
            if decision_file_path and decision_file_path.exists() and self.config.get('FUSE_SNOW_MODEL'):
                snow_model = self.config.get('FUSE_SNOW_MODEL')
                with open(decision_file_path, 'r') as f:
                    lines = f.readlines()
                updated_lines = []
                for line in lines:
                    if line.strip().endswith('SNOWM'):
                        parts = line.split()
                        if parts:
                            line = line.replace(parts[0], snow_model, 1)
                    updated_lines.append(line)
                with open(decision_file_path, 'w') as f:
                    f.writelines(updated_lines)
                self.logger.info(f"Updated FUSE snow model decision to {snow_model}")

            self.logger.info(f"FUSE base settings copied to {settings_path}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Source file or directory not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission error when copying files: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error copying base settings: {e}")
            raise

    def generate_synthetic_hydrograph(self, ds, area_km2, mean_temp_threshold=0.0):
        """Generate synthetic hydrograph - delegates to synthetic data generator"""
        return self.synthetic_data_generator.generate_synthetic_hydrograph(ds, area_km2, mean_temp_threshold)

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped, semi-distributed, and distributed modes.
        Supports configurable timesteps based on FORCING_TIME_STEP_SIZE config parameter.
        """
        try:
            # Get timestep configuration
            ts_config = self._get_timestep_config()
            self.logger.info(f"Using {ts_config['time_label']} timestep (resample freq: {ts_config['resample_freq']})")
            
            # Get spatial mode configuration
            spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
            
            self.logger.info(f"Preparing FUSE forcing data in {spatial_mode} mode")
            
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")
            
            variable_handler = VariableHandler(config=self.config, logger=self.logger,
                                            dataset=self.config.get('FORCING_DATASET'), model='FUSE')
            ds = xr.open_mfdataset(forcing_files, data_vars='all')
            ds = variable_handler.process_forcing_data(ds)
            ds = self.subset_to_simulation_time(ds, "Forcing")
            
            # Spatial organization based on mode BEFORE resampling
            if spatial_mode == 'lumped':
                ds = self._prepare_lumped_forcing(ds)
            elif spatial_mode == 'semi_distributed':
                ds = self._prepare_semi_distributed_forcing(ds, subcatchment_dim)
            elif spatial_mode == 'distributed':
                ds = self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown FUSE spatial mode: {spatial_mode}")
            
            # Resample to target resolution AFTER spatial organization
            self.logger.info(f"Resampling data to {ts_config['time_label']} resolution")
            with xr.set_options(use_flox=False, use_numbagg=False, use_bottleneck=False):
                ds = ds.resample(time=ts_config['resample_freq']).mean()
            
            # Process temperature and precipitation
            try:
                ds['temp'] = ds['airtemp']
                ds['pr'] = ds['pptrate']
            except:
                pass
            
            # Handle streamflow observations
            time_window = self.get_simulation_time_window()
            obs_ds = self._load_streamflow_observations(spatial_mode, ts_config, time_window)
            
            # Get PET method from config (default to 'oudin')
            pet_method = self.config.get('PET_METHOD', 'oudin').lower()
            self.logger.info(f"Using PET method: {pet_method}")
            
            # Calculate PET for the correct spatial configuration
            if spatial_mode == 'lumped':
                catchment = gpd.read_file(self.catchment_path)
                mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            else:
                # For distributed modes, calculate PET after spatial organization and resampling
                pet = self._calculate_distributed_pet(ds, spatial_mode, pet_method)
            
            # Ensure PET is also at target resolution by checking if resampling is needed
            with xr.set_options(use_flox=False, use_numbagg=False, use_bottleneck=False):
                pet_resampled = pet.resample(time=ts_config['resample_freq']).mean()
            if len(pet_resampled.time) != len(pet.time):
                self.logger.info(f"PET data resampled to {ts_config['time_label']} resolution")
                pet = pet_resampled
            else:
                self.logger.info(f"PET data is already at {ts_config['time_label']} resolution")
            
            # Create FUSE forcing dataset
            fuse_forcing = self._create_fuse_forcing_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)
            
            # Save forcing data
            output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
            encoding = self._get_encoding_dict(fuse_forcing)
            fuse_forcing.to_netcdf(output_file, unlimited_dims=['time'], 
                                encoding=encoding, format='NETCDF4')
            
            self.logger.info(f"FUSE forcing data saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise
        
            
    def _calculate_pet(self, temp_data: xr.DataArray, lat: float, method: str = 'oudin') -> xr.DataArray:
        """
        Calculate PET using the specified method.
        
        Args:
            temp_data (xr.DataArray): Temperature data
            lat (float): Latitude of the catchment centroid
            method (str): PET method ('oudin', 'hamon', or 'hargreaves')
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        method = method.lower()
        
        if method == 'oudin':
            return self.calculate_pet_oudin(temp_data, lat)
        elif method == 'hamon':
            return self.calculate_pet_hamon(temp_data, lat)
        elif method == 'hargreaves':
            return self.calculate_pet_hargreaves(temp_data, lat)
        else:
            self.logger.warning(f"Unknown PET method '{method}', defaulting to Oudin")
            return self.calculate_pet_oudin(temp_data, lat)

    def _add_forcing_variables(self, fuse_forcing, ds, pet, obs_ds, spatial_dims, n_subcatchments, ts_config=None):
        """Add forcing variables to the dataset with proper dimension handling"""
        
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"
        
        # Get the time dimension length from the coordinate system
        time_length = len(fuse_forcing.time)
        
        # Ensure all input data has the same time dimension
        self.logger.info(f"Expected time length: {time_length}")
        self.logger.info(f"ds time length: {len(ds.time)}")
        self.logger.info(f"pet time length: {len(pet.time)}")
        if obs_ds is not None:
            self.logger.info(f"obs_ds time length: {len(obs_ds.time)}")
        
        # Align all data to the common time coordinate
        common_time = fuse_forcing.time
        
        # For distributed case, make sure we're working with the right spatial dimension
        if len(spatial_dims) == 3:  # (time, spatial, 1) or (time, 1, spatial)
            if spatial_dims[1] in ['latitude', 'longitude'] and fuse_forcing.sizes[spatial_dims[1]] > 1:
                # Multiple subcatchments in this dimension
                target_shape = (time_length, n_subcatchments, 1)
            else:
                # Multiple subcatchments in the other dimension
                target_shape = (time_length, 1, n_subcatchments)
        else:
            target_shape = (time_length, n_subcatchments)
        
        # Core meteorological variables - extract only the time dimension that matches
        var_mapping = []
        
        # Handle precipitation
        if 'hru' in ds.dims and spatial_dims[1] != 'longitude':
            # Distributed data with HRU dimension
            pr_data = ds['pr'].values  # Shape: (time, hru)
            if pr_data.shape[0] != time_length:
                self.logger.warning(f"Precipitation time dimension mismatch: {pr_data.shape[0]} vs {time_length}")
                # Truncate or pad to match expected length
                if pr_data.shape[0] > time_length:
                    pr_data = pr_data[:time_length, :]
                else:
                    # Pad with the last available value
                    pad_length = time_length - pr_data.shape[0]
                    pad_values = np.repeat(pr_data[-1:, :], pad_length, axis=0)
                    pr_data = np.concatenate([pr_data, pad_values], axis=0)
        else:
            # Lumped data or single column
            pr_data = ds['pr'].values
            if pr_data.shape[0] != time_length:
                if pr_data.shape[0] > time_length:
                    pr_data = pr_data[:time_length]
                else:
                    pad_length = time_length - pr_data.shape[0]
                    pr_data = np.concatenate([pr_data, np.repeat(pr_data[-1], pad_length)])
        
        var_mapping.append(('pr', pr_data, 'precipitation', unit_str, f'Mean {time_label} precipitation'))

        
        # Handle temperature
        if 'hru' in ds.dims and spatial_dims[1] != 'longitude':
            temp_data = ds['temp'].values
            if temp_data.shape[0] != time_length:
                if temp_data.shape[0] > time_length:
                    temp_data = temp_data[:time_length, :]
                else:
                    pad_length = time_length - temp_data.shape[0]
                    pad_values = np.repeat(temp_data[-1:, :], pad_length, axis=0)
                    temp_data = np.concatenate([temp_data, pad_values], axis=0)
        else:
            temp_data = ds['temp'].values
            if temp_data.shape[0] != time_length:
                if temp_data.shape[0] > time_length:
                    temp_data = temp_data[:time_length]
                else:
                    pad_length = time_length - temp_data.shape[0]
                    temp_data = np.concatenate([temp_data, np.repeat(temp_data[-1], pad_length)])
        
        var_mapping.append(('temp', temp_data, 'temperature', 'degC', f'Mean {time_label} temperature'))    
        
        # Handle PET
        pet_data = pet.values
        if pet_data.shape[0] != time_length:
            if pet_data.shape[0] > time_length:
                pet_data = pet_data[:time_length]
            else:
                pad_length = time_length - pet_data.shape[0]
                if len(pet_data.shape) > 1:
                    pad_values = np.repeat(pet_data[-1:, :], pad_length, axis=0)
                    pet_data = np.concatenate([pet_data, pad_values], axis=0)
                else:
                    pet_data = np.concatenate([pet_data, np.repeat(pet_data[-1], pad_length)])
        
        var_mapping.append(('pet', pet_data, 'pet', unit_str, f'Mean {time_label} pet'))
        
        # Add streamflow observations
        if obs_ds is not None:
            obs_data = obs_ds['q_obs'].values
            if obs_data.shape[0] != time_length:
                if obs_data.shape[0] > time_length:
                    obs_data = obs_data[:time_length]
                else:
                    pad_length = time_length - obs_data.shape[0]
                    obs_data = np.concatenate([obs_data, np.repeat(obs_data[-1], pad_length)])
            var_mapping.append(('q_obs', obs_data, 'streamflow', unit_str, f'Mean observed {time_label} discharge'))
        else:
            # Generate synthetic hydrograph for each subcatchment
            synthetic_q = self._generate_distributed_synthetic_hydrograph(ds, n_subcatchments, time_length)
            var_mapping.append(('q_obs', synthetic_q, 'streamflow', unit_str, f'Synthetic discharge for optimization'))
        
        # Add variables to dataset
        encoding = {}
        for var_name, data, _, units, long_name in var_mapping:
            # Reshape data to match spatial structure
            if len(data.shape) == 1:  # Time series only
                # Replicate for all subcatchments
                if target_shape[1] > target_shape[2]:  # More subcatchments in second dimension
                    reshaped_data = np.tile(data[:, np.newaxis, np.newaxis], (1, target_shape[1], 1))
                else:  # More subcatchments in third dimension
                    reshaped_data = np.tile(data[:, np.newaxis, np.newaxis], (1, 1, target_shape[2]))
            elif len(data.shape) == 2 and data.shape[1] == n_subcatchments:  # (time, subcatchments)
                # Already has subcatchment data
                if target_shape[1] > target_shape[2]:
                    reshaped_data = data[:, :, np.newaxis]
                else:
                    reshaped_data = data[:, np.newaxis, :]
            else:
                # Default case: replicate along subcatchment dimension
                if target_shape[1] > target_shape[2]:
                    reshaped_data = np.tile(data.reshape(-1, 1, 1), (1, target_shape[1], 1))
                else:
                    reshaped_data = np.tile(data.reshape(-1, 1, 1), (1, 1, target_shape[2]))
            
            # Verify final shape matches expected dimensions
            expected_shape = (time_length, fuse_forcing.sizes[spatial_dims[1]], fuse_forcing.sizes[spatial_dims[2]])
            if reshaped_data.shape != expected_shape:
                self.logger.error(f"Shape mismatch for {var_name}: got {reshaped_data.shape}, expected {expected_shape}")
                raise ValueError(f"Shape mismatch for {var_name}")
            
            # Handle NaN values
            if np.any(np.isnan(reshaped_data)):
                reshaped_data = np.nan_to_num(reshaped_data, nan=-9999.0)
            
            fuse_forcing[var_name] = xr.DataArray(
                reshaped_data,
                dims=spatial_dims,
                coords=fuse_forcing.coords,
                attrs={
                    'units': units,
                    'long_name': long_name
                }
            )
            
            encoding[var_name] = {
                '_FillValue': -9999.0,
                'dtype': 'float32'
            }
        
        return encoding

    def _generate_distributed_synthetic_hydrograph(self, ds, n_subcatchments, time_length):
        """Generate distributed synthetic hydrograph - delegates to synthetic data generator"""
        return self.synthetic_data_generator.generate_distributed_synthetic_hydrograph(ds, n_subcatchments, time_length)
    def _prepare_lumped_forcing(self, ds):
        """Prepare lumped forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_lumped_forcing(ds)

    def _prepare_semi_distributed_forcing(self, ds, subcatchment_dim):
        """Prepare semi-distributed forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_semi_distributed_forcing(ds, subcatchment_dim)

    def _prepare_distributed_forcing(self, ds):
        """Prepare fully distributed forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_distributed_forcing(ds)

    def _load_subcatchment_data(self):
        """Load subcatchment information - delegates to forcing processor"""
        return self.forcing_processor._load_subcatchment_data()

    def _create_fuse_forcing_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config=None):
        """Create the final FUSE forcing dataset with proper coordinate structure"""
        
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        if spatial_mode == 'lumped':
            return self._create_lumped_dataset(ds, pet, obs_ds, ts_config)
        else:
            return self._create_distributed_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)

    def _create_distributed_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config=None):
        """Create distributed/semi-distributed FUSE forcing dataset with configurable timestep"""
        
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        # Get spatial information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Get reference coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # Create time index with correct frequency
        time_index = pd.date_range(start=ds.time.min().values, end=ds.time.max().values, freq=ts_config['resample_freq'])
        
        # Convert to numeric time values (FUSE expects days since reference)
        time_numeric = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
        
        # Create coordinate system based on subcatchment dimension choice
        if subcatchment_dim == 'latitude':
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', subcatchments.astype(float)),  # Subcatchment IDs as pseudo-lat
                'time': ('time', time_numeric)
            }
            spatial_dims = ('time', 'latitude', 'longitude')
        else:  # longitude
            coords = {
                'longitude': ('longitude', subcatchments.astype(float)),  # Subcatchment IDs as pseudo-lon
                'latitude': ('latitude', [mean_lat]),
                'time': ('time', time_numeric)
            }
            spatial_dims = ('time', 'longitude', 'latitude')
        
        # Create dataset
        fuse_forcing = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        fuse_forcing.longitude.attrs = {
            'units': 'degreesE' if subcatchment_dim != 'longitude' else 'subcatchment_id',
            'long_name': 'longitude' if subcatchment_dim != 'longitude' else 'subcatchment identifier'
        }
        fuse_forcing.latitude.attrs = {
            'units': 'degreesN' if subcatchment_dim != 'latitude' else 'subcatchment_id',
            'long_name': 'latitude' if subcatchment_dim != 'latitude' else 'subcatchment identifier'
        }
        fuse_forcing.time.attrs = {
            'units': 'days since 1970-01-01',
            'long_name': 'time'
        }
        
        # Add data variables
        self._add_forcing_variables(fuse_forcing, ds, pet, obs_ds, spatial_dims, n_subcatchments, ts_config)
        
        return fuse_forcing


    def create_filemanager(self):
        """
        Create FUSE file manager file by modifying template with project-specific settings.
        """
        self.logger.info("Creating FUSE file manager file")

        # Define source and destination paths
        template_path = self.setup_dir / 'fm_catch.txt'
        
        # Define the paths to replace
        fuse_id = self._get_fuse_file_id()
        settings = {
            'SETNGS_PATH': str(self.project_dir / 'settings' / 'FUSE') + '/',
            'INPUT_PATH': str(self.project_dir / 'forcing' / 'FUSE_input') + '/',
            'OUTPUT_PATH': str(self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FUSE') + '/',
            'METRIC': self.config.get('OPTIMIZATION_METRIC'),
            'MAXN': str(self.config.get('NUMBER_OF_ITERATIONS')),
            'FMODEL_ID': fuse_id,
            'M_DECISIONS': f"fuse_zDecisions_{fuse_id}.txt"
        }

        # Get and format dates from forcing data if available, else config
        start_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_START'), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_END'), '%Y-%m-%d %H:%M')
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        if forcing_file.exists():
            try:
                with xr.open_dataset(forcing_file) as ds:
                    time_vals = pd.to_datetime(ds.time.values)
                if len(time_vals) > 0:
                    start_time = time_vals.min().to_pydatetime()
                    end_time = time_vals.max().to_pydatetime()
            except Exception as e:
                self.logger.warning(f"Unable to read forcing time range from {forcing_file}: {e}")
        cal_start_time = datetime.strptime(self.config.get('CALIBRATION_PERIOD').split(',')[0], '%Y-%m-%d')
        cal_end_time = datetime.strptime(self.config.get('CALIBRATION_PERIOD').split(',')[1].strip(), '%Y-%m-%d')

        date_settings = {
            'date_start_sim': start_time.strftime('%Y-%m-%d'),
            'date_end_sim': end_time.strftime('%Y-%m-%d'),
            'date_start_eval': cal_start_time.strftime('%Y-%m-%d'),  # Using same dates for evaluation period
            'date_end_eval': cal_end_time.strftime('%Y-%m-%d')       # Can be modified if needed
        }

        try:
            # Read the template file
            with open(template_path, 'r') as f:
                lines = f.readlines()

            # Process each line
            modified_lines = []
            for line in lines:
                line_modified = line
                
                # Replace paths
                for path_key, new_path in settings.items():
                    if path_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_path + line[end:]
                            self.logger.debug(f"Updated {path_key} path to: {new_path}")

                # Replace dates
                for date_key, new_date in date_settings.items():
                    if date_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_date + line[end:]
                            self.logger.debug(f"Updated {date_key} to: {new_date}")

                modified_lines.append(line_modified)

            # Write the modified file
            with open(template_path, 'w') as f:
                f.writelines(modified_lines)

            self.logger.info(f"FUSE file manager created at: {template_path}")
        

        except Exception as e:
            self.logger.error(f"Error creating FUSE file manager: {str(e)}")
            raise

    def update_input_info(self):
        """Update FUSE input_info.txt based on the configured timestep."""
        input_info_path = self.setup_dir / 'input_info.txt'
        if not input_info_path.exists():
            self.logger.warning(f"FUSE input_info.txt not found at {input_info_path}")
            return

        ts_config = self._get_timestep_config()
        if ts_config['time_unit'] == 'h':
            deltim = "0.0416667"
            unit_str = "mm/h"
        else:
            deltim = "1.0"
            unit_str = "mm/d"

        unit_keys = {
            '<units_aprecip>': unit_str,
            '<units_potevap>': unit_str,
            '<units_q>': unit_str,
        }

        with open(input_info_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            line_updated = line
            if '<deltim>' in line:
                line_updated = re.sub(r"(<deltim>\\s+)(\\S+)", rf"\\1{deltim}", line)
            for key, value in unit_keys.items():
                if key in line:
                    line_updated = re.sub(rf"({re.escape(key)}\\s+)(\\S+)", rf"\\1{value}", line_updated)
            updated_lines.append(line_updated)

        with open(input_info_path, 'w') as f:
            f.writelines(updated_lines)

        self.logger.info("Updated FUSE input_info.txt for forcing timestep")

        

    # Removed: _get_catchment_centroid() - now inherited from GeospatialUtilsMixin

    def create_elevation_bands(self):
        """Create elevation bands netCDF file - delegates to elevation band manager"""
        return self.elevation_band_manager.create_elevation_bands()

    def _create_distributed_elevation_bands(self):
        """Create elevation bands for distributed mode"""
        
        # Load subcatchment information to get spatial dimensions
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Get reference coordinates (same as used in forcing file)
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # For now, create simple elevation bands for all subcatchments
        # In future, this could use subcatchment-specific elevation data
        
        # Create dataset with distributed spatial structure
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        if subcatchment_dim == 'latitude':
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', subcatchments.astype(float)),
                'elevation_band': ('elevation_band', [1])  # Simple single band for now
            }
            spatial_dims = ['elevation_band', 'latitude', 'longitude']
        else:
            coords = {
                'longitude': ('longitude', subcatchments.astype(float)),
                'latitude': ('latitude', [mean_lat]), 
                'elevation_band': ('elevation_band', [1])
            }
            spatial_dims = ['elevation_band', 'longitude', 'latitude']
        
        ds = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'} 
        ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
        
        # Create elevation band variables for all subcatchments
        target_shape = (1, n_subcatchments, 1) if subcatchment_dim == 'latitude' else (1, 1, n_subcatchments)
        
        for var_name, value, attrs in [
            ('area_frac', 1.0, {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
            ('mean_elev', 1000.0, {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
            ('prec_frac', 1.0, {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
        ]:
            
            data = np.full(target_shape, value, dtype=np.float32)
            
            ds[var_name] = xr.DataArray(
                data,
                dims=spatial_dims,
                coords=ds.coords,
                attrs=attrs
            )
        
        # Save with proper encoding
        output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
        encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
        
        ds.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
        
        self.logger.info(f"Created distributed elevation bands file: {output_file}")
        return output_file

    def _load_streamflow_observations(self, spatial_mode, ts_config=None, time_window=None):
        """
        Load streamflow observations for FUSE forcing data.

        Uses ObservationLoaderMixin for standardized observation loading.

        Args:
            spatial_mode (str): Spatial mode ('lumped', 'semi_distributed', 'distributed')
            ts_config (dict): Timestep configuration from _get_timestep_config()
            time_window (tuple): Optional (start, end) timestamp tuple for filtering

        Returns:
            xr.Dataset or None: Dataset containing observed streamflow or None if not available
        """
        # Get timestep config if not provided
        if ts_config is None:
            ts_config = self._get_timestep_config()

        # Determine target units based on timestep
        if ts_config['time_unit'] == 'h':
            target_units = 'mm_per_hour'
        elif ts_config['time_unit'] == 'D':
            target_units = 'mm_per_day'
        else:
            target_units = 'mm_per_timestep'

        # Use ObservationLoaderMixin to load observations
        obs_ds = self.load_streamflow_observations(
            output_format='xarray',
            target_units=target_units,
            resample_freq=ts_config['resample_freq'],
            time_slice=time_window,
            return_none_on_error=True
        )

        return obs_ds         

    def _create_lumped_elevation_bands(self):
        """Create elevation bands for lumped mode"""
        self.logger.info("Creating lumped elevation bands file")

        try:
            # Get catchment centroid for coordinates
            catchment = gpd.read_file(self.catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            
            # Create simple single elevation band for lumped mode
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', [mean_lat]),
                'elevation_band': ('elevation_band', [1])
            }
            
            ds = xr.Dataset(coords=coords)
            
            # Add coordinate attributes
            ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
            ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'} 
            ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
            
            # Create elevation band variables (single band covering entire catchment)
            target_shape = (1, 1, 1)  # (elevation_band, latitude, longitude)
            spatial_dims = ['elevation_band', 'latitude', 'longitude']
            
            for var_name, value, attrs in [
                ('area_frac', 1.0, {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
                ('mean_elev', 1000.0, {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
                ('prec_frac', 1.0, {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
            ]:
                
                data = np.full(target_shape, value, dtype=np.float32)
                
                ds[var_name] = xr.DataArray(
                    data,
                    dims=spatial_dims,
                    coords=ds.coords,
                    attrs=attrs
                )
            
            # Save with proper encoding
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
            
            ds.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
            
            self.logger.info(f"Created lumped elevation bands file: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error creating lumped elevation bands: {str(e)}")
            raise

    def _calculate_distributed_pet(self, ds, spatial_mode, pet_method='oudin'):
        """Calculate PET for distributed/semi-distributed modes - delegates to forcing processor"""
        return self.forcing_processor._calculate_distributed_pet(ds, spatial_mode, pet_method)

    def _get_encoding_dict(self, fuse_forcing):
        """Get encoding dictionary for netCDF output - delegates to forcing processor"""
        return self.forcing_processor.get_encoding_dict(fuse_forcing)

    def _create_lumped_dataset(self, ds, pet, obs_ds, ts_config=None):
        """
        Create lumped FUSE forcing dataset with configurable timestep.
        
        Args:
            ds: Processed forcing dataset
            pet: Calculated PET data
            obs_ds: Observed streamflow dataset (or None)
            ts_config: Timestep configuration from _get_timestep_config()
            
        Returns:
            xr.Dataset: FUSE forcing dataset for lumped mode
        """
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        # Get catchment centroid for coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # Convert all time coordinates to pandas datetime for comparison
        ds_start = pd.to_datetime(ds.time.min().values)
        ds_end = pd.to_datetime(ds.time.max().values)
        pet_start = pd.to_datetime(pet.time.min().values)
        pet_end = pd.to_datetime(pet.time.max().values)
        
        # Find overlapping time period
        start_time = max(ds_start, pet_start)
        end_time = min(ds_end, pet_end)
        
        if obs_ds is not None:
            # Convert obs_ds time to datetime based on time units
            if obs_ds.time.dtype.kind in ['i', 'u', 'f']:  # numeric types
                time_units = str(obs_ds.time.attrs.get('units', ''))
                if 'hour' in time_units:
                    obs_time_dt = pd.to_datetime('1970-01-01') + pd.to_timedelta(obs_ds.time.values, unit='h')
                else:
                    obs_time_dt = pd.to_datetime('1970-01-01') + pd.to_timedelta(obs_ds.time.values, unit='D')
                obs_start = obs_time_dt.min()
                obs_end = obs_time_dt.max()
            else:
                obs_start = pd.to_datetime(obs_ds.time.min().values)
                obs_end = pd.to_datetime(obs_ds.time.max().values)

            start_time = max(start_time, obs_start)
            end_time = min(end_time, obs_end)
        
        self.logger.info(f"Aligning all data to common time period: {start_time} to {end_time}")
        
        # Create explicit time index for the overlapping period with correct frequency
        time_index = pd.date_range(start=start_time, end=end_time, freq=ts_config['resample_freq'])
        
        # Align all datasets to the common time period
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        if obs_ds is not None:
            # Handle obs_ds reindexing based on its time format
            if obs_ds.time.dtype.kind in ['i', 'u', 'f']:  # numeric types
                time_units = str(obs_ds.time.attrs.get('units', ''))
                if 'hour' in time_units:
                    start_numeric = (pd.to_datetime(start_time) - pd.Timestamp('1970-01-01')).total_seconds() / 3600
                    end_numeric = (pd.to_datetime(end_time) - pd.Timestamp('1970-01-01')).total_seconds() / 3600
                else:
                    start_numeric = (pd.to_datetime(start_time) - pd.Timestamp('1970-01-01')).days
                    end_numeric = (pd.to_datetime(end_time) - pd.Timestamp('1970-01-01')).days

                time_numeric_index = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
                obs_ds = obs_ds.sel(time=slice(start_numeric, end_numeric))
                obs_ds = obs_ds.reindex(time=time_numeric_index)
            else:
                obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        # Convert time to numeric values since reference date for final dataset
        time_numeric = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
        
        # Create coordinates
        coords = {
            'longitude': ('longitude', [mean_lon]),
            'latitude': ('latitude', [mean_lat]),
            'time': ('time', time_numeric)
        }
        
        # Create dataset
        fuse_forcing = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        fuse_forcing.longitude.attrs = {
            'units': 'degreesE',
            'long_name': 'longitude'
        }
        fuse_forcing.latitude.attrs = {
            'units': 'degreesN',
            'long_name': 'latitude'
        }
        fuse_forcing.time.attrs = {
            'units': 'days since 1970-01-01',
            'long_name': 'time'
        }
        
        # Add forcing variables
        spatial_dims = ('time', 'latitude', 'longitude')
        
        # Determine unit string for variables
        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"
        
        # Core meteorological variables
        var_mapping = [
            ('pr', ds['pr'].values, 'precipitation', unit_str, f'Mean {time_label} precipitation'),
            ('temp', ds['temp'].values, 'temperature', 'degC', f'Mean {time_label} temperature'),
            ('pet', pet.values, 'pet', unit_str, f'Mean {time_label} pet')
        ]
        
        # Add streamflow observations
        if obs_ds is not None:
            var_mapping.append(('q_obs', obs_ds['q_obs'].values, 'streamflow', unit_str, f'Mean observed {time_label} discharge'))
        else:
            # Generate synthetic hydrograph
            synthetic_q = self.generate_synthetic_hydrograph(ds, area_km2=100.0)
            var_mapping.append(('q_obs', synthetic_q, 'streamflow', unit_str, f'Synthetic discharge for optimization'))
        
        # Add variables to dataset
        for var_name, data, _, units, long_name in var_mapping:
            # Reshape data to match spatial structure (time, lat, lon)
            if len(data.shape) == 1:  # Time series only
                reshaped_data = data[:, np.newaxis, np.newaxis]
            else:
                reshaped_data = data.reshape(-1, 1, 1)
            
            # Handle NaN values
            if np.any(np.isnan(reshaped_data)):
                reshaped_data = np.nan_to_num(reshaped_data, nan=-9999.0)
            
            # Verify dimensions match
            if reshaped_data.shape[0] != len(time_numeric):
                self.logger.error(f"Dimension mismatch for {var_name}: data has {reshaped_data.shape[0]} time points, coordinate has {len(time_numeric)}")
                raise ValueError(f"Dimension mismatch for {var_name}")
            
            fuse_forcing[var_name] = xr.DataArray(
                reshaped_data,
                dims=spatial_dims,
                coords=fuse_forcing.coords,
                attrs={
                    'units': units,
                    'long_name': long_name
                }
            )
        
        return fuse_forcing
        
    def _map_hrus_to_subcatchments(self, ds, subcatchments):
        """Map HRU data to subcatchments - delegates to forcing processor"""
        return self.forcing_processor._map_hrus_to_subcatchments(ds, subcatchments)

    def _replicate_to_subcatchments(self, ds, n_subcatchments):
        """Replicate lumped data to subcatchments - delegates to forcing processor"""
        return self.forcing_processor._replicate_to_subcatchments(ds, n_subcatchments)

    def _create_distributed_from_catchment(self, ds):
        """Create HRU-level data from catchment data - delegates to forcing processor"""
        return self.forcing_processor._create_distributed_from_catchment(ds)

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
