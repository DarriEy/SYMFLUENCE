"""
Forcing data processing utilities for FUSE model.

This module contains the FuseForcingProcessor class which handles all forcing data
processing operations including spatial mode transformations, PET calculation,
observation loading, and NetCDF formatting for FUSE model compatibility.

Uses shared utilities from symfluence.models.utilities for common operations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.data.utilities.variable_utils import VariableHandler
from symfluence.core.constants import UnitConversion
from ..utilities import ForcingDataProcessor, DataQualityHandler, BaseForcingProcessor


class FuseForcingProcessor(BaseForcingProcessor):
    """
    Processor for FUSE forcing data with support for lumped, semi-distributed, and distributed modes.

    This class handles:
    - Spatial mode transformations (lumped/semi-distributed/distributed)
    - PET calculation for different spatial configurations
    - Forcing data resampling and alignment
    - NetCDF encoding and formatting
    - Variable mapping and dimension handling

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Root project directory
        forcing_basin_path: Path to basin-averaged forcing data
        forcing_fuse_path: Path to FUSE-specific forcing output
        catchment_path: Path to catchment shapefile
        domain_name: Name of the domain
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        project_dir: Path,
        forcing_basin_path: Path,
        forcing_fuse_path: Path,
        catchment_path: Path,
        domain_name: str,
        calculate_pet_callback,
        calculate_catchment_centroid_callback,
        get_simulation_time_window_callback,
        subset_to_simulation_time_callback
    ):
        """
        Initialize the FUSE forcing processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Root project directory
            forcing_basin_path: Path to basin-averaged forcing data
            forcing_fuse_path: Path to FUSE forcing output directory
            catchment_path: Path to catchment shapefile
            domain_name: Domain name for file naming
            calculate_pet_callback: Callback to parent's _calculate_pet method
            calculate_catchment_centroid_callback: Callback to parent's calculate_catchment_centroid method
            get_simulation_time_window_callback: Callback to parent's _get_simulation_time_window method
            subset_to_simulation_time_callback: Callback to parent's _subset_to_simulation_time method
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_basin_path,
            output_path=forcing_fuse_path,
            project_dir=project_dir,
            catchment_path=catchment_path
        )
        # Keep original attribute names for backward compatibility
        self.forcing_basin_path = self.input_path
        self.forcing_fuse_path = self.output_path
        self.domain_name = domain_name

        # Callbacks to parent methods
        self._calculate_pet = calculate_pet_callback
        self.calculate_catchment_centroid = calculate_catchment_centroid_callback
        self._get_simulation_time_window = get_simulation_time_window_callback
        self._subset_to_simulation_time = subset_to_simulation_time_callback

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "FUSE"

    def prepare_forcing_data(self, ts_config: Dict[str, Any], pet_method: str = 'oudin') -> Path:
        """
        Prepare forcing data with support for lumped, semi-distributed, and distributed modes.

        Args:
            ts_config: Timestep configuration dictionary from get_timestep_config()
            pet_method: PET calculation method ('oudin', 'hamon', 'hargreaves')

        Returns:
            Path to created forcing file

        Raises:
            FileNotFoundError: If no forcing files found
            ValueError: If unknown spatial mode specified
        """
        try:
            self.logger.debug(f"Using {ts_config['time_label']} timestep (resample freq: {ts_config['resample_freq']})")

            # Get spatial mode configuration
            spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')

            self.logger.debug(f"Preparing FUSE forcing data in {spatial_mode} mode")

            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")

            variable_handler = VariableHandler(
                config=self.config,
                logger=self.logger,
                dataset=self.config.get('FORCING_DATASET'),
                model='FUSE'
            )
            ds = xr.open_mfdataset(forcing_files, data_vars='all', combine='nested', concat_dim='time').sortby('time')
            ds = variable_handler.process_forcing_data(ds)
            ds = self._subset_to_simulation_time(ds, "Forcing")

            # Spatial organization BEFORE resampling
            if spatial_mode == 'lumped':
                ds = self._prepare_lumped_forcing(ds)
            elif spatial_mode == 'semi_distributed':
                ds = self._prepare_semi_distributed_forcing(ds, subcatchment_dim)
            elif spatial_mode == 'distributed':
                ds = self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown FUSE spatial mode: {spatial_mode}")

            # Resample to target resolution AFTER spatial organization
            self.logger.debug(f"Resampling data to {ts_config['time_label']} resolution")
            # Enable optimized backends for resampling if available
            ds = ds.resample(time=ts_config['resample_freq']).mean()

            # Process temperature and precipitation
            try:
                ds['temp'] = ds['airtemp']
                ds['pr'] = ds['pptrate']
            except:
                pass

            # Calculate PET for the correct spatial configuration
            if spatial_mode == 'lumped':
                catchment = gpd.read_file(self.catchment_path)
                mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            else:
                # For distributed modes, calculate PET after spatial organization
                pet = self._calculate_distributed_pet(ds, spatial_mode, pet_method)

            # Ensure PET is also at target resolution
            pet = pet.resample(time=ts_config['resample_freq']).mean()
            self.logger.info(f"PET data resampled to {ts_config['time_label']} resolution")

            # Save forcing data
            output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"

            self.logger.info(f"FUSE forcing data will be saved to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    def _prepare_lumped_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare lumped forcing data (spatial average)"""
        return ds.mean(dim='hru') if 'hru' in ds.dims else ds

    def _prepare_semi_distributed_forcing(self, ds: xr.Dataset, subcatchment_dim: str) -> xr.Dataset:
        """Prepare semi-distributed forcing data using subcatchment IDs"""
        self.logger.info(f"Organizing subcatchments along {subcatchment_dim} dimension")

        # Load subcatchment information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)

        # Reorganize data by subcatchments
        if 'hru' in ds.dims:
            if ds.sizes['hru'] == n_subcatchments:
                ds_subcat = ds
            else:
                ds_subcat = self._map_hrus_to_subcatchments(ds, subcatchments)
        else:
            ds_subcat = self._replicate_to_subcatchments(ds, n_subcatchments)

        return ds_subcat

    def _prepare_distributed_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare fully distributed forcing data"""
        self.logger.info("Preparing distributed forcing data")

        # Use HRU data directly if available
        if 'hru' in ds.dims:
            return ds
        else:
            return self._create_distributed_from_catchment(ds)

    def _calculate_distributed_pet(
        self,
        ds: xr.Dataset,
        spatial_mode: str,
        pet_method: str = 'oudin'
    ) -> xr.DataArray:
        """
        Calculate PET for distributed/semi-distributed modes.

        Args:
            ds: xarray dataset with temperature data
            spatial_mode: Spatial mode ('semi_distributed', 'distributed')
            pet_method: PET calculation method

        Returns:
            xr.DataArray: Calculated PET data
        """
        self.logger.info(f"Calculating distributed PET for {spatial_mode} mode using {pet_method}")

        try:
            # Get catchment for reference latitude
            catchment = gpd.read_file(self.catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)

            # For distributed modes, use the same latitude for all subcatchments/HRUs
            if 'hru' in ds.dims:
                # Use the mean temperature across all HRUs to calculate PET once
                temp_mean = ds['temp'].mean(dim='hru')
                pet_base = self._calculate_pet(temp_mean, mean_lat, pet_method)

                # Broadcast the PET calculation to all HRUs (more efficient than concat)
                pet = pet_base.broadcast_like(ds['temp'])

                self.logger.info(f"Calculated distributed PET with shape: {pet.shape}")
            else:
                # Use lumped calculation as fallback
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)

            return pet

        except Exception as e:
            self.logger.warning(f"Error calculating distributed PET, falling back to lumped: {str(e)}")
            catchment = gpd.read_file(self.catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            return self._calculate_pet(ds['temp'], mean_lat, pet_method)

    def _load_subcatchment_data(self) -> np.ndarray:
        """Load subcatchment information for semi-distributed mode"""
        # Check if delineated catchments exist (for distributed routing)
        delineated_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"

        if delineated_path.exists():
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs
            catchment = gpd.read_file(self.catchment_path)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def _map_hrus_to_subcatchments(self, ds: xr.Dataset, subcatchments: np.ndarray) -> xr.Dataset:
        """Map HRU data to subcatchments for semi-distributed mode"""
        self.logger.info("Mapping HRUs to subcatchments")

        n_hrus = ds.sizes['hru']
        n_subcatchments = len(subcatchments)

        if n_hrus == n_subcatchments:
            return ds.rename({'hru': 'subcatchment'})
        elif n_hrus > n_subcatchments:
            # Aggregate HRUs to subcatchments
            hrus_per_subcat = n_hrus // n_subcatchments
            subcatchment_data = []

            for i in range(n_subcatchments):
                start_idx = i * hrus_per_subcat
                end_idx = start_idx + hrus_per_subcat if i < n_subcatchments - 1 else n_hrus
                subcat_data = ds.isel(hru=slice(start_idx, end_idx)).mean(dim='hru')
                subcatchment_data.append(subcat_data)

            ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
            ds_subcat['subcatchment'] = subcatchments
            return ds_subcat
        else:
            return self._replicate_to_subcatchments(ds, n_subcatchments)

    def _replicate_to_subcatchments(self, ds: xr.Dataset, n_subcatchments: int) -> xr.Dataset:
        """Replicate lumped data to all subcatchments using broadcasting"""
        self.logger.info(f"Replicating data to {n_subcatchments} subcatchments")

        sub_ids = xr.DataArray(range(1, n_subcatchments + 1), dims='subcatchment', name='subcatchment')
        return ds.broadcast_like(sub_ids).assign_coords(subcatchment=sub_ids)

    def _create_distributed_from_catchment(self, ds: xr.Dataset) -> xr.Dataset:
        """Create HRU-level data from catchment data for distributed mode using broadcasting"""
        self.logger.info("Creating distributed data from catchment data")

        catchment = gpd.read_file(self.catchment_path)
        n_hrus = len(catchment)

        hru_ids = xr.DataArray(range(1, n_hrus + 1), dims='hru', name='hru')
        return ds.broadcast_like(hru_ids).assign_coords(hru=hru_ids)

    def get_encoding_dict(self, fuse_forcing: xr.Dataset) -> Dict[str, Dict]:
        """
        Get encoding dictionary for netCDF output.

        Uses shared DataQualityHandler for fill values and ForcingDataProcessor
        for consistent encoding patterns.

        Args:
            fuse_forcing: xarray Dataset to encode

        Returns:
            Dict: Encoding dictionary for netCDF
        """
        # Use shared utility for data variable encoding
        dqh = DataQualityHandler()
        fill_value = dqh.get_fill_value('float32')

        fdp = ForcingDataProcessor(self.config)
        encoding = fdp.create_encoding_dict(
            fuse_forcing,
            fill_value=fill_value,
            dtype='float32',
            compression=False
        )

        # Add coordinate-specific encoding (FUSE requires float64 for coords)
        for coord in fuse_forcing.coords:
            if coord == 'time':
                encoding[coord] = {'dtype': 'float64'}
            elif coord in ['longitude', 'latitude', 'lon', 'lat']:
                encoding[coord] = {'dtype': 'float64'}
            else:
                encoding[coord] = {'dtype': 'float32'}

        return encoding
