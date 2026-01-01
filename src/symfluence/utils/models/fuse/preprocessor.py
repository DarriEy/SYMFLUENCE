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
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
from typing import Dict, List, Tuple, Any
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin, ObservationLoaderMixin
from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.common.geospatial_utils import GeospatialUtilsMixin
from symfluence.utils.exceptions import (
    ModelExecutionError,
    FileOperationError,
    symfluence_error_handler
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from symfluence.utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_preprocessor('FUSE')
class FUSEPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, ObservationLoaderMixin):
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
        self.catchment_path = self.get_catchment_path()

    def _get_timestep_config(self):
        """
        Get timestep configuration based on FORCING_TIME_STEP_SIZE.
        
        Returns:
            dict: Configuration with resample_freq, time_units, time_unit_factor, 
                conversion_factor, and time_label
        """
        timestep_seconds = self.config.get('FORCING_TIME_STEP_SIZE', 86400)  # Default daily
        
        if timestep_seconds == 3600:  # Hourly
            return {
                'resample_freq': 'h',
                'time_units': 'hours since 1970-01-01',
                'time_unit': 'h',  # for pd.to_timedelta
                'conversion_factor': UnitConversion.MM_HOUR_TO_CMS,  # cms to mm/hour: Q(mm/hr) = Q(cms) * 3.6 / Area(km2)
                'time_label': 'hourly',
                'timestep_seconds': 3600
            }
        elif timestep_seconds == 86400:  # Daily
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,  # cms to mm/day: Q(mm/day) = Q(cms) * 86.4 / Area(km2)
                'time_label': 'daily',
                'timestep_seconds': 86400
            }
        else:
            # Generic case - calculate based on seconds
            hours = timestep_seconds / 3600
            if hours < 24:
                return {
                    'resample_freq': f'{int(hours)}h',
                    'time_units': 'hours since 1970-01-01',
                    'time_unit': 'h',
                    'conversion_factor': UnitConversion.MM_HOUR_TO_CMS * hours,  # Scale from hourly
                    'time_label': f'{int(hours)}-hourly',
                    'timestep_seconds': timestep_seconds
                }
            else:
                days = timestep_seconds / 86400
                return {
                    'resample_freq': f'{int(days)}D',
                    'time_units': 'days since 1970-01-01',
                    'time_unit': 'D',
                    'conversion_factor': UnitConversion.MM_DAY_TO_CMS * days,  # Scale from daily
                    'time_label': f'{int(days)}-daily',
                    'timestep_seconds': timestep_seconds
                }

    def run_preprocessing(self):
        """
        Run the complete FUSE preprocessing workflow.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting FUSE preprocessing")

        with symfluence_error_handler(
            "FUSE preprocessing",
            self.logger,
            error_type=ModelExecutionError
        ):
            self.create_directories()
            self.copy_base_settings()
            self.prepare_forcing_data()
            self.create_elevation_bands()
            self.create_filemanager()

            self.logger.info("FUSE preprocessing completed successfully")

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
            
            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file
                
                # Handle the decisions file specially
                if 'fuse_zDecisions_' in file:
                    # Create new filename with experiment ID
                    new_filename = file.replace('902', self.config.get('EXPERIMENT_ID'))
                    dest_file = settings_path / new_filename
                    self.logger.debug(f"Renaming decisions file from {file} to {new_filename}")
                else:
                    dest_file = settings_path / file
                
                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")
            
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
        """
        Generate a realistic synthetic hydrograph for snow optimization cases.
        
        Args:
            ds: xarray dataset with precipitation and temperature
            area_km2: catchment area in km2
            mean_temp_threshold: temperature threshold for snow/rain (°C)
        
        Returns:
            np.array: synthetic streamflow in mm/day
        """
        self.logger.info("Generating synthetic hydrograph for snow optimization")
        
        # Get precipitation and temperature data
        precip = ds['pr'].values  # mm/day
        temp = ds['temp'].values - 273.15  # Convert K to °C
        
        # Simple rainfall-runoff model parameters
        # These create a realistic but generic hydrograph
        runoff_coeff_rain = 0.3  # 30% of rain becomes runoff
        runoff_coeff_snow = 0.1  # 10% of snow becomes immediate runoff
        baseflow_recession = 0.95  # Daily baseflow recession coefficient
        
        # Initialize arrays
        n_days = len(precip)
        runoff = np.zeros(n_days)
        baseflow = np.zeros(n_days)
        snowpack = np.zeros(n_days)
        
        # Simple degree-day snowmelt parameters
        melt_factor = 3.0  # mm/°C/day
        
        # Initial baseflow (small constant)
        baseflow[0] = 0.5  # mm/day
        
        for i in range(n_days):
            # Determine if precipitation is rain or snow
            if temp[i] > mean_temp_threshold:
                # Rain
                rain = precip[i]
                snow = 0.0
            else:
                # Snow
                rain = 0.0
                snow = precip[i]
            
            # Snow accumulation and melt
            if i > 0:
                snowpack[i] = snowpack[i-1] + snow
            else:
                snowpack[i] = snow
                
            # Snowmelt (only if temperature > 0°C)
            if temp[i] > 0.0 and snowpack[i] > 0.0:
                melt = min(snowpack[i], melt_factor * temp[i])
                snowpack[i] -= melt
                rain += melt  # Add melt to effective rainfall
            
            # Calculate surface runoff
            surface_runoff = rain * runoff_coeff_rain + snow * runoff_coeff_snow
            
            # Update baseflow (simple recession + recharge)
            if i > 0:
                baseflow[i] = baseflow[i-1] * baseflow_recession + surface_runoff * 0.1
            else:
                baseflow[i] = surface_runoff * 0.1
            
            # Total runoff
            runoff[i] = surface_runoff + baseflow[i]
        
        # Add some realistic variability and ensure non-negative
        # Add small random component (±10% of mean)
        mean_runoff = np.mean(runoff)
        noise = np.random.normal(0, mean_runoff * 0.05, n_days)
        runoff = np.maximum(runoff + noise, 0.01)  # Ensure minimum flow
        
        # Apply a simple routing delay (moving average)
        from scipy import ndimage
        runoff_routed = ndimage.uniform_filter1d(runoff, size=3, mode='reflect')
        
        self.logger.info(f"Generated synthetic hydrograph: mean={np.mean(runoff_routed):.2f} mm/day, "
                        f"max={np.max(runoff_routed):.2f} mm/day")
        
        return runoff_routed

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
            obs_ds = self._load_streamflow_observations(spatial_mode, ts_config)
            
            # Get PET method from config (default to 'oudin')
            pet_method = self.config.get('PET_METHOD', 'oudin').lower()
            self.logger.info(f"Using PET method: {pet_method}")
            
            # Calculate PET for the correct spatial configuration
            if spatial_mode == 'lumped':
                catchment = gpd.read_file(self.catchment_path / self.catchment_name)
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
        """Generate synthetic hydrograph for each subcatchment with correct time dimension"""
        
        # Use only the time-matched data for generating the hydrograph
        temp_data = ds['temp'].values
        pr_data = ds['pr'].values
        
        # Ensure we're using the correct time length
        if temp_data.shape[0] > time_length:
            temp_data = temp_data[:time_length]
        if pr_data.shape[0] > time_length:
            pr_data = pr_data[:time_length]
        
        # Create a temporary dataset for hydrograph generation
        temp_ds = xr.Dataset({
            'temp': (['time'], temp_data if len(temp_data.shape) == 1 else temp_data.mean(axis=1)),
            'pr': (['time'], pr_data if len(pr_data.shape) == 1 else pr_data.mean(axis=1))
        })
        
        base_hydrograph = self.generate_synthetic_hydrograph(temp_ds, area_km2=100.0)
        
        # Ensure the base hydrograph has the correct length
        if len(base_hydrograph) != time_length:
            if len(base_hydrograph) > time_length:
                base_hydrograph = base_hydrograph[:time_length]
            else:
                pad_length = time_length - len(base_hydrograph)
                base_hydrograph = np.concatenate([base_hydrograph, np.repeat(base_hydrograph[-1], pad_length)])
        
        # Create variations for different subcatchments (simple approach)
        variations = np.random.uniform(0.8, 1.2, n_subcatchments)  # ±20% variation
        distributed_q = np.outer(base_hydrograph, variations)  # (time, subcatchments)
        
        return distributed_q
    def _prepare_lumped_forcing(self, ds):
        """Prepare lumped forcing data (current implementation)"""
        return ds.mean(dim='hru') if 'hru' in ds.dims else ds

    def _prepare_semi_distributed_forcing(self, ds, subcatchment_dim):
        """Prepare semi-distributed forcing data using subcatchment IDs"""
        self.logger.info(f"Organizing subcatchments along {subcatchment_dim} dimension")
        
        # Load subcatchment information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Reorganize data by subcatchments
        if 'hru' in ds.dims:
            if ds.sizes['hru'] == n_subcatchments:
                # Data already matches subcatchments
                ds_subcat = ds
            else:
                # Need to aggregate/map to subcatchments
                ds_subcat = self._map_hrus_to_subcatchments(ds, subcatchments)
        else:
            # Replicate lumped data to all subcatchments
            ds_subcat = self._replicate_to_subcatchments(ds, n_subcatchments)
        
        return ds_subcat

    def _prepare_distributed_forcing(self, ds):
        """Prepare fully distributed forcing data"""
        self.logger.info("Preparing distributed forcing data")
        
        # Use HRU data directly if available
        if 'hru' in ds.dims:
            return ds
        else:
            # Need to create HRU-level data from catchment data
            return self._create_distributed_from_catchment(ds)

    def _load_subcatchment_data(self):
        """Load subcatchment information for semi-distributed mode"""
        # Check if delineated catchments exist (for distributed routing)
        delineated_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
        
        if delineated_path.exists():
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

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
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # Create time index with correct frequency
        time_index = pd.date_range(start=ds.time.min().values, end=ds.time.max().values, freq=ts_config['resample_freq'])
        
        # Convert to numeric time values
        if ts_config['time_unit'] == 'h':
            time_numeric = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 3600).values
        else:
            time_numeric = (time_index - pd.Timestamp('1970-01-01')).days.values
        
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
            'units': ts_config['time_units'],
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
        settings = {
            'SETNGS_PATH': str(self.project_dir / 'settings' / 'FUSE') + '/',
            'INPUT_PATH': str(self.project_dir / 'forcing' / 'FUSE_input') + '/',
            'OUTPUT_PATH': str(self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FUSE') + '/',
            'METRIC': self.config.get('OPTIMIZATION_METRIC'),
            'MAXN': str(self.config.get('NUMBER_OF_ITERATIONS')),
            'FMODEL_ID': self.config.get('EXPERIMENT_ID'),
            'M_DECISIONS': f"fuse_zDecisions_{self.config.get('EXPERIMENT_ID')}.txt"
        }

        # Get and format dates from config
        start_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_START'), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_END'), '%Y-%m-%d %H:%M')
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

        

    # Removed: _get_catchment_centroid() - now inherited from GeospatialUtilsMixin

    def create_elevation_bands(self):
        """Create elevation bands netCDF file for FUSE input with distributed support."""
        self.logger.info("Creating elevation bands file for FUSE")

        try:
            # Check spatial mode
            spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            
            if spatial_mode == 'lumped':
                return self._create_lumped_elevation_bands()
            else:
                return self._create_distributed_elevation_bands()
                
        except Exception as e:
            self.logger.error(f"Error creating elevation bands file: {str(e)}")
            raise

    def _create_distributed_elevation_bands(self):
        """Create elevation bands for distributed mode"""
        
        # Load subcatchment information to get spatial dimensions
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Get reference coordinates (same as used in forcing file)
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
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

    def _load_streamflow_observations(self, spatial_mode, ts_config=None):
        """
        Load streamflow observations for FUSE forcing data.
        
        Args:
            spatial_mode (str): Spatial mode ('lumped', 'semi_distributed', 'distributed')
            ts_config (dict): Timestep configuration from _get_timestep_config()
            
        Returns:
            xr.Dataset or None: Dataset containing observed streamflow or None if not available
        """
        # Get timestep config if not provided
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        try:
            # Get observations file path
            obs_file_path = self.config.get('OBSERVATIONS_PATH', 'default')
            if obs_file_path == 'default':
                obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            else:
                obs_file_path = Path(obs_file_path)
            
            # Check if observations file exists
            if not obs_file_path.exists():
                self.logger.warning(f"Streamflow observations file not found: {obs_file_path}")
                self.logger.info("Will generate synthetic hydrograph for optimization")
                return None
            
            # Read observations
            self.logger.info(f"Loading streamflow observations from: {obs_file_path}")
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            
            # Resample to target timestep and get discharge
            if 'discharge_cms' in dfObs.columns:
                obs_streamflow = dfObs['discharge_cms'].resample(ts_config['resample_freq']).mean()
            elif 'discharge' in dfObs.columns: 
                obs_streamflow = dfObs['discharge'].resample(ts_config['resample_freq']).mean()
            else:
                available_cols = list(dfObs.columns)
                self.logger.warning(f"No discharge column found. Available columns: {available_cols}")
                return None
            
            # Get catchment area for unit conversion if needed
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            
            if basin_path.exists():
                basin_gdf = gpd.read_file(basin_path)
                area_km2 = basin_gdf['GRU_area'].sum() / 1e6  # Convert m2 to km2
            else:
                # Fallback area estimate
                catchment = gpd.read_file(self.catchment_path / self.catchment_name)
                catchment_proj = catchment.to_crs(catchment.estimate_utm_crs())
                area_km2 = catchment_proj.geometry.area.sum() / 1e6
                self.logger.warning(f"Using estimated catchment area: {area_km2:.2f} km2")
            
            # Convert from cms to mm/timestep for FUSE
            # Q(mm/timestep) = Q(cms) * conversion_factor / Area(km2)
            obs_streamflow_mm = obs_streamflow * ts_config['conversion_factor'] / area_km2
            
            # Create time coordinate based on timestep
            if ts_config['time_unit'] == 'h':
                # Hours since 1970-01-01
                time_values = ((obs_streamflow_mm.index - pd.Timestamp('1970-01-01')).total_seconds() / 3600).values
            else:
                # Days since 1970-01-01
                time_values = (obs_streamflow_mm.index - pd.Timestamp('1970-01-01')).days.values
            
            # Create xarray dataset
            obs_ds = xr.Dataset(
                {
                    'q_obs': xr.DataArray(
                        obs_streamflow_mm.values,
                        dims=['time'],
                        coords={'time': time_values},
                        attrs={
                            'units': f"mm/{ts_config['time_label'].replace('-', ' ')}",
                            'long_name': f"Observed {ts_config['time_label']} discharge",
                            'standard_name': 'water_volume_transport_in_river_channel'
                        }
                    )
                },
                coords={
                    'time': xr.DataArray(
                        time_values,
                        dims=['time'],
                        attrs={
                            'units': ts_config['time_units'],
                            'long_name': 'time'
                        }
                    )
                }
            )
            
            self.logger.info(f"Loaded {len(obs_streamflow_mm)} {ts_config['time_label']} streamflow observations")
            self.logger.info(f"Converted from cms to mm/{ts_config['time_label']} using area: {area_km2:.2f} km2")
            
            return obs_ds
            
        except Exception as e:
            self.logger.error(f"Error loading streamflow observations: {str(e)}")
            self.logger.info("Will generate synthetic hydrograph for optimization")
            return None         

    def _create_lumped_elevation_bands(self):
        """Create elevation bands for lumped mode"""
        self.logger.info("Creating lumped elevation bands file")

        try:
            # Get catchment centroid for coordinates
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
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
        """
        Calculate PET for distributed/semi-distributed modes.
        
        Args:
            ds: xarray dataset with temperature data
            spatial_mode (str): Spatial mode ('semi_distributed', 'distributed')
            pet_method (str): PET calculation method
            
        Returns:
            xr.DataArray: Calculated PET data
        """
        self.logger.info(f"Calculating distributed PET for {spatial_mode} mode using {pet_method}")
        
        try:
            # Get catchment for reference latitude
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            
            # For distributed modes, use the same latitude for all subcatchments/HRUs
            if 'hru' in ds.dims:
                # Use the mean temperature across all HRUs to calculate PET once
                temp_mean = ds['temp'].mean(dim='hru')
                pet_base = self._calculate_pet(temp_mean, mean_lat, pet_method)
                
                # Replicate the PET calculation for each HRU with correct dimension order
                n_hrus = ds.sizes['hru']
                
                # Create a list of the base PET for each HRU and concatenate along HRU dimension
                pet_list = []
                for i in range(n_hrus):
                    pet_list.append(pet_base)
                
                # Concatenate along new HRU dimension, ensuring time is first dimension
                pet = xr.concat(pet_list, dim='hru')
                # Transpose to ensure correct dimension order: (time, hru)
                pet = pet.transpose('time', 'hru')
                
                self.logger.info(f"Calculated distributed PET with shape: {pet.shape}")
            else:
                # Use lumped calculation as fallback
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            
            return pet
            
        except Exception as e:
            self.logger.warning(f"Error calculating distributed PET, falling back to lumped: {str(e)}")
            # Fallback to lumped calculation
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            return self._calculate_pet(ds['temp'], mean_lat, pet_method)

    def _get_encoding_dict(self, fuse_forcing):
        """
        Get encoding dictionary for netCDF output.
        
        Args:
            fuse_forcing: xarray Dataset to encode
            
        Returns:
            Dict: Encoding dictionary for netCDF
        """
        encoding = {}
        
        # Default encoding for coordinates
        for coord in fuse_forcing.coords:
            if coord == 'time':
                encoding[coord] = {
                    'dtype': 'float64'
                    # NOTE: 'units' should NOT be here - it belongs in attributes only
                }
            elif coord in ['longitude', 'latitude']:
                encoding[coord] = {
                    'dtype': 'float64'
                }
            else:
                encoding[coord] = {
                    'dtype': 'float32'
                }
        
        # Default encoding for data variables
        for var in fuse_forcing.data_vars:
            encoding[var] = {
                '_FillValue': -9999.0,
                'dtype': 'float32'
            }
        
        return encoding

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
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
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
                if 'hours' in str(obs_ds.time.attrs.get('units', '')):
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
                if ts_config['time_unit'] == 'h':
                    # Convert time index to hours since 1970-01-01 for selection
                    start_hours = (pd.to_datetime(start_time) - pd.Timestamp('1970-01-01')).total_seconds() / 3600
                    end_hours = (pd.to_datetime(end_time) - pd.Timestamp('1970-01-01')).total_seconds() / 3600
                    time_numeric_index = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 3600).values
                else:
                    # Convert time index to days since 1970-01-01 for selection
                    start_hours = (pd.to_datetime(start_time) - pd.Timestamp('1970-01-01')).days
                    end_hours = (pd.to_datetime(end_time) - pd.Timestamp('1970-01-01')).days
                    time_numeric_index = (time_index - pd.Timestamp('1970-01-01')).days.values
                
                # Select and reindex with numeric time
                obs_ds = obs_ds.sel(time=slice(start_hours, end_hours))
                obs_ds = obs_ds.reindex(time=time_numeric_index)
            else:
                obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        # Convert time to numeric values since reference date for final dataset
        if ts_config['time_unit'] == 'h':
            time_numeric = ((time_index - pd.Timestamp('1970-01-01')).total_seconds() / 3600).values
        else:
            time_numeric = (time_index - pd.Timestamp('1970-01-01')).days.values
        
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
            'units': ts_config['time_units'],
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
        """
        Map HRU data to subcatchments for semi-distributed mode.
        
        Args:
            ds: Dataset with HRU dimension
            subcatchments: Array of subcatchment IDs
            
        Returns:
            xr.Dataset: Dataset organized by subcatchments
        """
        self.logger.info("Mapping HRUs to subcatchments")
        
        # Simple approach: assume HRUs map directly to subcatchments
        # This could be enhanced with actual HRU-subcatchment mapping
        n_hrus = ds.sizes['hru']
        n_subcatchments = len(subcatchments)
        
        if n_hrus == n_subcatchments:
            # Direct mapping
            return ds.rename({'hru': 'subcatchment'})
        elif n_hrus > n_subcatchments:
            # Aggregate HRUs to subcatchments (simple averaging)
            hrus_per_subcat = n_hrus // n_subcatchments
            subcatchment_data = []
            
            for i in range(n_subcatchments):
                start_idx = i * hrus_per_subcat
                end_idx = start_idx + hrus_per_subcat
                if i == n_subcatchments - 1:  # Last subcatchment gets remaining HRUs
                    end_idx = n_hrus
                
                subcat_data = ds.isel(hru=slice(start_idx, end_idx)).mean(dim='hru')
                subcatchment_data.append(subcat_data)
            
            # Combine subcatchments
            ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
            ds_subcat['subcatchment'] = subcatchments
            
            return ds_subcat
        else:
            # Replicate HRU data to subcatchments
            return self._replicate_to_subcatchments(ds, n_subcatchments)

    def _replicate_to_subcatchments(self, ds, n_subcatchments):
        """
        Replicate lumped data to all subcatchments.
        
        Args:
            ds: Lumped dataset
            n_subcatchments: Number of subcatchments
            
        Returns:
            xr.Dataset: Dataset replicated to subcatchments
        """
        self.logger.info(f"Replicating data to {n_subcatchments} subcatchments")
        
        # Create subcatchment dimension
        subcatchment_data = []
        for i in range(n_subcatchments):
            subcatchment_data.append(ds)
        
        # Combine along new subcatchment dimension
        ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
        ds_subcat['subcatchment'] = range(1, n_subcatchments + 1)
        
        return ds_subcat

    def _create_distributed_from_catchment(self, ds):
        """
        Create HRU-level data from catchment data for distributed mode.
        
        Args:
            ds: Catchment-level dataset
            
        Returns:
            xr.Dataset: HRU-level dataset
        """
        self.logger.info("Creating distributed data from catchment data")
        
        # Load catchment shapefile to get number of HRUs
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        n_hrus = len(catchment)
        
        # Replicate catchment data to all HRUs
        hru_data = []
        for i in range(n_hrus):
            hru_data.append(ds)
        
        # Combine along HRU dimension
        ds_hru = xr.concat(hru_data, dim='hru')
        ds_hru['hru'] = range(1, n_hrus + 1)
        
        return ds_hru

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))

