"""
FUSE model preprocessor for data preparation and configuration.

Prepares forcing data, calculates PET, generates FUSE input files,
and handles synthetic data generation for the FUSE hydrological model.
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Dict, Any

import geopandas as gpd # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore

from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin, ObservationLoaderMixin, DatasetBuilderMixin
from ..registry import ModelRegistry
from .elevation_band_manager import FuseElevationBandManager
from .forcing_processor import FuseForcingProcessor
from .synthetic_data_generator import FuseSyntheticDataGenerator
from symfluence.core.constants import UnitConversion
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin
from symfluence.data.utils.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_preprocessor('FUSE')
class FUSEPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, ObservationLoaderMixin, DatasetBuilderMixin):
    """
    Preprocessor for the FUSE (Framework for Understanding Structural Errors) model.

    Handles complete preprocessing workflow for FUSE including forcing data processing,
    PET calculation, elevation band discretization, and configuration file generation.
    FUSE is a modular modeling framework that allows testing different model structures.

    Key Operations:
        - Process forcing data (precipitation, temperature, PET)
        - Calculate potential evapotranspiration using multiple methods
        - Generate elevation band discretization for distributed modeling
        - Create FUSE control and parameter files
        - Generate synthetic forcing data (for testing/calibration)
        - Calculate catchment attributes (area, elevation statistics)
        - Handle both lumped and distributed spatial configurations

    Workflow Steps:
        1. Initialize paths and load configuration
        2. Process forcing data and calculate PET
        3. Generate elevation bands (if distributed mode)
        4. Create FUSE control file (fm_control.txt)
        5. Set up parameter files with model structure selections
        6. Configure input/output file specifications
        7. Prepare observation data for evaluation

    Supported Spatial Modes:
        - Lumped: Single catchment unit
        - Distributed: Multiple elevation bands with area weighting
        - Synthetic: Generate idealized forcing for testing

    PET Calculation Methods:
        - Hamon: Temperature-based method
        - Priestley-Taylor: Radiation-based method
        - Penman-Monteith: Energy balance method (requires additional variables)

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        PETCalculatorMixin: Potential evapotranspiration calculation methods
        GeospatialUtilsMixin: Spatial operations (centroid, area calculation)
        ObservationLoaderMixin: Observation data loading capabilities
        DatasetBuilderMixin: NetCDF dataset construction utilities

    Attributes:
        config (SymfluenceConfig): Typed configuration object
        logger: Logger object for recording processing information
        project_dir (Path): Directory for the current project
        forcing_fuse_path (Path): Directory for FUSE input files
        catchment_path (Path): Path to catchment shapefile
        forcing_processor (FuseForcingProcessor): Handles forcing data transformation
        elevation_band_manager (FuseElevationBandManager): Manages elevation discretization
        synthetic_generator (FuseSyntheticDataGenerator): Generates synthetic forcing

    Example:
        >>> from symfluence.models.fuse.preprocessor import FUSEPreProcessor
        >>> preprocessor = FUSEPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates FUSE input files in: project_dir/forcing/FUSE_input/
        # Generates: input_info.txt, elev_bands.txt, fm_control.txt
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "FUSE"

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the FUSE preprocessor.

        Sets up FUSE-specific paths, forcing processors, elevation band manager,
        and synthetic data generator. The preprocessor handles complete FUSE
        input preparation including forcing data, PET calculation, and
        configuration file generation.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                FUSE model settings, paths, PET method, and spatial mode options.
            logger: Logger instance for status messages and debugging output.

        Note:
            Creates three delegate objects for modular processing:
            - forcing_processor: Handles forcing data transformation
            - elevation_band_manager: Manages elevation discretization
            - synthetic_data_generator: Creates synthetic forcing for testing
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # FUSE-specific paths
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'

        # Setup catchment path using base class helper
        self.catchment_path = self.get_catchment_path()
        self.catchment_name = self.catchment_path.name

        # Initialize forcing processor (use base class methods for callbacks)
        self.forcing_processor = FuseForcingProcessor(
            config=self.config_dict,
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
            config=self.config_dict,
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
        fuse_id = self.config_dict.get('FUSE_FILE_ID')
        if not fuse_id:
            experiment_id = self.config_dict.get('EXPERIMENT_ID', '')
            fuse_id = experiment_id[:6] if experiment_id else 'fuse'
            self.config_dict['FUSE_FILE_ID'] = fuse_id
        return fuse_id

    def _get_timestep_config(self):
        """
        Get FUSE-specific timestep configuration.

        FUSE only supports daily timesteps, so this method forces daily
        resolution regardless of the base config. This differs from
        the base class get_timestep_config() which respects the config.

        Returns:
            dict: Configuration with resample_freq, time_units, time_unit,
                conversion_factor, time_label, and timestep_seconds
        """
        ts_config = self.get_timestep_config()
        if ts_config['time_unit'] == 'h':
            self.logger.debug("FUSE uses daily timesteps; overriding hourly config")
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,
                'time_label': 'daily',
                'timestep_seconds': 86400
            }
        return ts_config

    def run_preprocessing(self):
        """
        Run the complete FUSE preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.debug("Starting FUSE preprocessing")
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
            self.logger.debug(f"Created directory: {dir_path}")

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
        self.logger.debug("Copying FUSE base settings")

        from symfluence.resources import get_base_settings_dir
        base_settings_path = get_base_settings_dir('FUSE')
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
            
            if decision_file_path and decision_file_path.exists() and self.config_dict.get('FUSE_SNOW_MODEL'):
                snow_model = self.config_dict.get('FUSE_SNOW_MODEL')
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
            t0 = time.time()
            # Get timestep configuration
            ts_config = self._get_timestep_config()
            self.logger.debug(f"Using {ts_config['time_label']} timestep (resample freq: {ts_config['resample_freq']})")
            
            # Get spatial mode configuration
            spatial_mode = self.config_dict.get('FUSE_SPATIAL_MODE', 'lumped')
            subcatchment_dim = self.config_dict.get('FUSE_SUBCATCHMENT_DIM', 'longitude')
            
            self.logger.debug(f"Preparing FUSE forcing data in {spatial_mode} mode")
            
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")

            # Basin-averaged forcing data is already in CFIF format (from model-agnostic preprocessing)
            variable_handler = VariableHandler(config=self.config_dict, logger=self.logger,
                                            dataset='CFIF', model='FUSE')
            self.logger.debug(f"FUSE forcing_files count: {len(forcing_files)}")
            
            t1 = time.time()
            # Open the forcing datasets
            if len(forcing_files) == 1:
                ds = xr.open_dataset(forcing_files[0])
            else:
                ds = xr.open_mfdataset(forcing_files, data_vars='all', combine='nested', concat_dim='time').sortby('time')
            self.logger.info(f"PERF: Opening dataset took {time.time() - t1:.2f}s")
            
            t2 = time.time()
            ds = variable_handler.process_forcing_data(ds)
            self.logger.info(f"PERF: process_forcing_data took {time.time() - t2:.2f}s")
            
            t3 = time.time()
            ds = self.subset_to_simulation_time(ds, "Forcing")
            self.logger.info(f"PERF: subset_to_simulation_time took {time.time() - t3:.2f}s")
            
            # Spatial organization based on mode BEFORE resampling
            t4 = time.time()
            if spatial_mode == 'lumped':
                ds = self._prepare_lumped_forcing(ds)
            elif spatial_mode == 'semi_distributed':
                ds = self._prepare_semi_distributed_forcing(ds, subcatchment_dim)
            elif spatial_mode == 'distributed':
                ds = self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown FUSE spatial mode: {spatial_mode}")
            self.logger.info(f"PERF: Spatial prep ({spatial_mode}) took {time.time() - t4:.2f}s")
            
            # Resample to target resolution AFTER spatial organization
            t5 = time.time()
            self.logger.debug(f"Resampling data to {ts_config['time_label']} resolution")
            ds = ds.resample(time=ts_config['resample_freq']).mean()
            self.logger.info(f"PERF: Resampling took {time.time() - t5:.2f}s")

            # Variables already renamed by variable_handler.process_forcing_data() above:
            # - airtemp -> temp
            # - pptrate -> precip

            # Handle streamflow observations
            t6 = time.time()
            time_window = self.get_simulation_time_window()
            obs_ds = self._load_streamflow_observations(spatial_mode, ts_config, time_window)
            self.logger.info(f"PERF: Loading observations took {time.time() - t6:.2f}s")
            
            # Get PET method from config (default to 'oudin')
            pet_method = self.config_dict.get('PET_METHOD', 'oudin').lower()
            self.logger.debug(f"Using PET method: {pet_method}")
            
            # Calculate PET for the correct spatial configuration
            t7 = time.time()
            if spatial_mode == 'lumped':
                catchment = gpd.read_file(self.catchment_path)
                mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            else:
                # For distributed modes, calculate PET after spatial organization and resampling
                pet = self._calculate_distributed_pet(ds, spatial_mode, pet_method)
            
            # Ensure PET is also at target resolution
            pet = pet.resample(time=ts_config['resample_freq']).mean()
            self.logger.debug(f"PET data resampled to {ts_config['time_label']} resolution")
            self.logger.info(f"PERF: PET calculation took {time.time() - t7:.2f}s")
            
            # Create FUSE forcing dataset
            t8 = time.time()
            fuse_forcing = self._create_fuse_forcing_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)
            self.logger.info(f"PERF: _create_fuse_forcing_dataset took {time.time() - t8:.2f}s")

            # Capture actual simulation dates from the final dataset
            # FUSE uses days since 1970-01-01
            try:
                t9 = time.time()
                time_vals = fuse_forcing['time'].values
                if len(time_vals) > 0:
                    dates = pd.to_datetime(time_vals, unit='D', origin='1970-01-01')
                    self.actual_start_time = dates.min().to_pydatetime()
                    self.actual_end_time = dates.max().to_pydatetime()
                    self.logger.info(f"Captured actual simulation dates: {self.actual_start_time} to {self.actual_end_time}")
                self.logger.info(f"PERF: Date capture took {time.time() - t9:.2f}s")
            except Exception as e:
                self.logger.warning(f"Could not capture actual dates from FUSE forcing: {e}")

            # Release any open handles and materialize data before writing to avoid lazy write hangs
            try:
                if hasattr(ds, "close"):
                    ds.close()
            except Exception as e:
                self.logger.debug(f"Could not close forcing dataset: {e}")

            self.logger.info("Materializing FUSE forcing dataset before write")
            fuse_forcing = fuse_forcing.load()

            # Save forcing data
            t10 = time.time()
            output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
            encoding = self._get_encoding_dict(fuse_forcing)
            fuse_forcing.to_netcdf(output_file, unlimited_dims=['time'], 
                                encoding=encoding, format='NETCDF4')
            
            self.logger.info(f"PERF: Saving NetCDF took {time.time() - t10:.2f}s")
            self.logger.info(f"PERF: Total prepare_forcing_data took {time.time() - t0:.2f}s")
            self.logger.debug(f"FUSE forcing data saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise
            
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
        """
        Add forcing variables to the FUSE dataset using efficient xarray broadcasting.

        Processes precipitation, temperature, PET, and streamflow observations
        and adds them to the FUSE forcing dataset with proper coordinate alignment,
        NaN handling, and encoding for FUSE compatibility.

        Args:
            fuse_forcing: Target xarray Dataset with coordinate structure defined.
            ds: Source forcing dataset containing 'precip' and 'temp' variables.
            pet: Potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic generation.
            spatial_dims: Tuple of dimension names, e.g., ('time', 'latitude', 'longitude').
            n_subcatchments: Number of subcatchments for distributed mode.
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            dict: Encoding dictionary for NetCDF output with compression settings.

        Note:
            FUSE interprets -9999 as actual negative values, not missing data.
            This method fills NaN values with interpolated/mean data to avoid
            FUSE misinterpreting missing data.
        """

        if ts_config is None:
            ts_config = self._get_timestep_config()

        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"

        # Determine target spatial dimension (latitude or longitude based on config)
        subcatchment_dim = self.config_dict.get('FUSE_SUBCATCHMENT_DIM', 'longitude')

        # Alignment time index (already in fuse_forcing.time but we need it as datetime for reindexing)
        # FUSE uses numeric time, but we'll use datetime for internal alignment
        time_index = pd.to_datetime('1970-01-01') + pd.to_timedelta(fuse_forcing.time.values, unit='D')

        def process_var(da, name):
            pv_start = time.time()
            # Ensure data has datetime index for alignment
            if not pd.api.types.is_datetime64_any_dtype(da.time.dtype):
                da = da.assign_coords(time=pd.to_datetime('1970-01-01') + pd.to_timedelta(da.time.values, unit='D'))

            # Align to the expected time index
            da_aligned = da.reindex(time=time_index, method='nearest', tolerance='1D').ffill(dim='time')

            # Switch back to numeric time to match fuse_forcing exactly (avoids DTypePromotionError)
            # Drop time first to ensure clean replacement of coordinate and index
            da_aligned = da_aligned.drop_vars('time').assign_coords(time=fuse_forcing.time)

            # CRITICAL FIX: Map 'hru' dimension to the target spatial dimension (latitude/longitude)
            # This ensures FUSE gets 3D (time, lat, lon) not 4D (time, lat, lon, hru)
            if 'hru' in da_aligned.dims:
                # Remove target dimension if it exists as a coordinate to avoid conflict
                if subcatchment_dim in da_aligned.coords:
                    da_aligned = da_aligned.drop_vars(subcatchment_dim)

                # Rename hru to the target spatial dimension
                da_aligned = da_aligned.rename({'hru': subcatchment_dim})
                # Assign the correct coordinate values from fuse_forcing
                da_aligned = da_aligned.assign_coords({subcatchment_dim: fuse_forcing[subcatchment_dim].values})

            # Broadcast to spatial dimensions (now should only add the singleton dimension)
            da_broadcasted = da_aligned.broadcast_like(fuse_forcing)

            # CRITICAL: Handle NaN values properly for FUSE
            # FUSE interprets -9999 as actual negative values, not as missing data
            # We need to fill NaN values with interpolated/mean data
            t_copy = time.time()
            da_values = da_broadcasted.values.copy()
            self.logger.debug(f"PERF: [{name}] da.values copy took {time.time() - t_copy:.4f}s")
            
            nan_mask = np.isnan(da_values)

            if np.any(nan_mask):
                # Find subcatchments with all-NaN data
                spatial_idx = 1 if subcatchment_dim == 'latitude' else 2  # Index in (time, lat, lon)
                all_nan_mask = np.all(nan_mask, axis=0)  # Shape: (lat, lon) or similar

                if np.any(all_nan_mask):
                    self.logger.warning(f"Variable '{name}' has {np.sum(all_nan_mask)} subcatchments with all NaN values - filling with spatial mean")

                    # Compute mean across valid subcatchments for each timestep
                    valid_data = np.where(nan_mask, np.nan, da_values)
                    spatial_mean = np.nanmean(valid_data, axis=(1, 2), keepdims=True)

                    # Fill NaN values with spatial mean
                    da_values = np.where(nan_mask, spatial_mean, da_values)

                    # If still NaN (all data missing for a timestep), use temporal mean
                    still_nan = np.isnan(da_values)
                    if np.any(still_nan):
                        temporal_mean = np.nanmean(da_values)
                        da_values = np.where(still_nan, temporal_mean, da_values)
                else:
                    # Scattered NaN values - fill with forward fill then backward fill
                    self.logger.debug(f"Variable '{name}' has {np.sum(nan_mask)} scattered NaN values - filling with interpolation")
                    da_values = np.where(nan_mask, np.nan, da_values)
                    # Use xarray for interpolation
                    da_temp = xr.DataArray(da_values, dims=da_broadcasted.dims, coords=da_broadcasted.coords)
                    da_temp = da_temp.ffill(dim='time').bfill(dim='time')
                    da_values = da_temp.values

            # Ensure non-negative for precipitation (floating-point precision can cause tiny negative values)
            if name == 'precip':
                da_values = np.maximum(da_values, 0.0)
            
            self.logger.debug(f"PERF: [{name}] process_var total took {time.time() - pv_start:.4f}s")
            return xr.DataArray(da_values, dims=da_broadcasted.dims, coords=da_broadcasted.coords).astype('float32')

        # Map variables
        var_map = {
            'precip': (ds['precip'], 'precipitation', unit_str, f'Mean {time_label} precipitation'),
            'temp': (ds['temp'], 'temperature', 'degC', f'Mean {time_label} temperature'),
            'pet': (pet, 'pet', unit_str, f'Mean {time_label} pet')
        }
        
        if obs_ds is not None:
            var_map['q_obs'] = (obs_ds['q_obs'], 'streamflow', unit_str, f'Mean observed {time_label} discharge')
        else:
            # Generate synthetic hydrograph for each subcatchment
            synthetic_q_vals = self._generate_distributed_synthetic_hydrograph(ds, n_subcatchments, len(time_index))
            # Wrap in DataArray for process_var
            # (Note: _generate_distributed_synthetic_hydrograph returns numpy array)
            synthetic_q = xr.DataArray(
                synthetic_q_vals, 
                coords={'time': ds.time}, # Use ds.time as reference
                dims=['time', 'hru'] if len(synthetic_q_vals.shape) > 1 else ['time']
            )
            var_map['q_obs'] = (synthetic_q, 'streamflow', unit_str, 'Synthetic discharge for optimization')

        # Process and add to dataset
        encoding = {}
        for var_name, (da, standard_name, units, long_name) in var_map.items():
            fuse_forcing[var_name] = process_var(da, var_name)
            fuse_forcing[var_name].attrs = {'units': units, 'long_name': long_name}
            encoding[var_name] = {'_FillValue': -9999.0, 'dtype': 'float32', 'zlib': False}
        
        # Ensure coordinates also have strict encoding for FUSE compatibility
        for coord in fuse_forcing.coords:
            encoding[coord] = {'_FillValue': None, 'dtype': 'float64'}

        return encoding

    def _generate_distributed_synthetic_hydrograph(self, ds, n_subcatchments, time_length):
        """
        Generate distributed synthetic hydrograph for testing and calibration.

        Creates synthetic streamflow data when observed data is unavailable,
        useful for model testing and structure exploration. Delegates to
        FuseSyntheticDataGenerator for the actual computation.

        Args:
            ds: Source forcing dataset with precipitation and temperature.
            n_subcatchments: Number of subcatchments to generate data for.
            time_length: Number of timesteps in the output.

        Returns:
            numpy.ndarray: Synthetic streamflow array of shape (time_length, n_subcatchments).
        """
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
        """
        Create the final FUSE forcing dataset with proper coordinate structure.

        Routes to the appropriate dataset creation method based on spatial mode.
        Lumped mode creates a single-point dataset while distributed modes
        create multi-dimensional datasets with subcatchment coordinates.

        Args:
            ds: Processed forcing dataset with 'precip' and 'temp' variables.
            pet: Calculated potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic data.
            spatial_mode: One of 'lumped', 'semi_distributed', or 'distributed'.
            subcatchment_dim: Dimension for subcatchments ('latitude' or 'longitude').
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            xr.Dataset: Complete FUSE forcing dataset ready for model input.
        """
        
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        if spatial_mode == 'lumped':
            return self._create_lumped_dataset(ds, pet, obs_ds, ts_config)
        else:
            return self._create_distributed_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)

    def _create_distributed_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config=None):
        """
        Create distributed or semi-distributed FUSE forcing dataset.

        Builds a multi-dimensional forcing dataset for distributed FUSE runs
        where subcatchments are represented along either latitude or longitude
        dimension. Uses efficient xarray operations for broadcasting.

        Args:
            ds: Processed forcing dataset with 'precip' and 'temp' variables.
            pet: Calculated potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic data.
            spatial_mode: Either 'semi_distributed' or 'distributed'.
            subcatchment_dim: Dimension for subcatchments ('latitude' or 'longitude').
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            xr.Dataset: FUSE forcing dataset with dimensions (time, latitude, longitude)
                where one spatial dimension contains subcatchment IDs and the other
                contains the catchment centroid coordinate.
        """
        
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        # Get spatial information
        subcatchments = self._load_subcatchment_data()
        
        # Get reference coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # Convert to numeric time values for FUSE
        time_numeric = ((pd.to_datetime(ds.time.values) - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
        
        # Create coordinate system
        # CRITICAL: Order must be time, lat, lon for FUSE to read correct dimensions
        if subcatchment_dim == 'latitude':
            coords = {
                'time': ('time', time_numeric),
                'latitude': ('latitude', subcatchments.astype(float)),
                'longitude': ('longitude', [float(mean_lon)])
            }
            spatial_dims = ('time', 'latitude', 'longitude')
        else:  # longitude
            coords = {
                'time': ('time', time_numeric),
                'longitude': ('longitude', subcatchments.astype(float)),
                'latitude': ('latitude', [float(mean_lat)])
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
        self._add_forcing_variables(fuse_forcing, ds, pet, obs_ds, spatial_dims, len(subcatchments), ts_config)
        
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
            'OUTPUT_PATH': str(self.project_dir / 'simulations' / self.config_dict.get('EXPERIMENT_ID') / 'FUSE') + '/',
            'METRIC': self.config_dict.get('OPTIMIZATION_METRIC'),
            'MAXN': str(self.config_dict.get('NUMBER_OF_ITERATIONS')),
            'FMODEL_ID': fuse_id,
            'M_DECISIONS': f"fuse_zDecisions_{fuse_id}.txt"
        }

        # Get and format dates from forcing data if available, else config
        start_time = datetime.strptime(self.config_dict.get('EXPERIMENT_TIME_START'), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config_dict.get('EXPERIMENT_TIME_END'), '%Y-%m-%d %H:%M')
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        
        if hasattr(self, 'actual_start_time') and hasattr(self, 'actual_end_time'):
            self.logger.info(f"Using captured dates from preprocessing: {self.actual_start_time} to {self.actual_end_time}")
            start_time = self.actual_start_time
            end_time = self.actual_end_time
        else:
            self.logger.info(f"Checking forcing file for dates: {forcing_file}")
            if forcing_file.exists():
                try:
                    self.logger.info("Forcing file exists, attempting to read time range...")
                    with xr.open_dataset(forcing_file) as ds:
                        time_vals = pd.to_datetime(ds.time.values)
                    
                    if len(time_vals) > 0:
                        start_time = time_vals.min().to_pydatetime()
                        end_time = time_vals.max().to_pydatetime()
                        self.logger.info(f"Updated simulation dates from forcing file: {start_time} to {end_time}")
                    else:
                        self.logger.warning("Forcing file has no time values!")
                except Exception as e:
                    self.logger.warning(f"Unable to read forcing time range from {forcing_file}: {e}")
            else:
                self.logger.warning(f"Forcing file not found at {forcing_file}, using config dates.")

        cal_start_time = datetime.strptime(self.config_dict.get('CALIBRATION_PERIOD').split(',')[0], '%Y-%m-%d')
        cal_end_time = datetime.strptime(self.config_dict.get('CALIBRATION_PERIOD').split(',')[1].strip(), '%Y-%m-%d')

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

    # NOTE: _create_distributed_elevation_bands() and _create_lumped_elevation_bands() have been
    # removed as they are now fully handled by FuseElevationBandManager. This eliminates ~120 lines
    # of duplicated code.

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

    def _calculate_distributed_pet(self, ds, spatial_mode, pet_method='oudin'):
        """Calculate PET for distributed/semi-distributed modes - delegates to forcing processor"""
        return self.forcing_processor._calculate_distributed_pet(ds, spatial_mode, pet_method)

    def _get_encoding_dict(self, fuse_forcing):
        """Get encoding dictionary for netCDF output - delegates to forcing processor"""
        return self.forcing_processor.get_encoding_dict(fuse_forcing)

    def _create_lumped_dataset(self, ds, pet, obs_ds, ts_config=None):
        """
        Create lumped FUSE forcing dataset with configurable timestep using xarray native alignment.
        """
        if ts_config is None:
            ts_config = self._get_timestep_config()
        
        # Get catchment centroid for coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        
        # Align all datasets to a common time period using xarray.align
        # First, ensure all have compatible time coordinates
        to_align = [ds, pet]
        if obs_ds is not None:
            # Handle obs_ds which might have numeric time
            if obs_ds.time.dtype.kind in ['i', 'u', 'f']:
                obs_ds = obs_ds.assign_coords(time=pd.to_datetime('1970-01-01') + pd.to_timedelta(obs_ds.time.values, unit='D'))
            to_align.append(obs_ds)
            
        # Join='inner' finds the overlapping period
        aligned = xr.align(*to_align, join='inner')
        ds_a, pet_a = aligned[0], aligned[1]
        obs_ds_a = aligned[2] if obs_ds is not None else None
        
        self.logger.info(f"Aligned data to overlapping period: {ds_a.time.min().values} to {ds_a.time.max().values}")
        
        # Create coordinates for FUSE
        time_numeric = ((pd.to_datetime(ds_a.time.values) - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
        # CRITICAL: Order must be time, lat, lon for FUSE to read correct dimensions
        coords = {
            'time': ('time', time_numeric),
            'latitude': ('latitude', [float(mean_lat)]),
            'longitude': ('longitude', [float(mean_lon)])
        }
        
        fuse_forcing = xr.Dataset(coords=coords)
        fuse_forcing.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        fuse_forcing.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'}
        fuse_forcing.time.attrs = {'units': 'days since 1970-01-01', 'long_name': 'time'}
        
        # Determine unit string for variables
        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"
        
        # Core meteorological variables
        var_map = {
            'precip': (ds_a['precip'], 'precipitation', unit_str, f'Mean {time_label} precipitation'),
            'temp': (ds_a['temp'], 'temperature', 'degC', f'Mean {time_label} temperature'),
            'pet': (pet_a, 'pet', unit_str, f'Mean {time_label} pet')
        }
        
        if obs_ds_a is not None:
            var_map['q_obs'] = (obs_ds_a['q_obs'], 'streamflow', unit_str, f'Mean observed {time_label} discharge')
        else:
            synthetic_q_vals = self.generate_synthetic_hydrograph(ds_a, area_km2=100.0)
            synthetic_q = xr.DataArray(synthetic_q_vals, coords={'time': ds_a.time}, dims=['time'])
            var_map['q_obs'] = (synthetic_q, 'streamflow', unit_str, 'Synthetic discharge for optimization')
            
        # Add variables with broadcasting
        for var_name, (da, _, units, long_name) in var_map.items():
            # Ensure da has same time coord type and values as fuse_forcing to avoid DTypePromotionError
            # Drop time first to ensure clean replacement of coordinate and index
            da_aligned = da.drop_vars('time').assign_coords(time=fuse_forcing.time)
            
            # Broadcast to (time, lat, lon) where lat=1, lon=1
            fuse_forcing[var_name] = da_aligned.broadcast_like(fuse_forcing).fillna(-9999.0).astype('float32')
            fuse_forcing[var_name].attrs = {'units': units, 'long_name': long_name}
            
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
        if self.config_dict.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config_dict.get(f'{file_type}'))
