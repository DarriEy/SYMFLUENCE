"""
SAC-SMA Model Preprocessor.

Prepares forcing data (P, T, PET) for SAC-SMA + Snow-17 model execution.
Follows the same pattern as HBV preprocessor (lumped mode).
"""

from typing import Dict, Any, Optional, Union
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.mixins import SpatialModeDetectionMixin
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.constants import UnitDetectionThresholds


@ModelRegistry.register_preprocessor('SACSMA')
class SacSmaPreProcessor(BaseModelPreProcessor, SpatialModeDetectionMixin):  # type: ignore[misc]
    """Preprocessor for SAC-SMA + Snow-17 model.

    Prepares forcing data:
    - Precipitation (mm/day)
    - Temperature (°C)
    - Potential evapotranspiration (mm/day)
    """


    MODEL_NAME = "SACSMA"
    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        params: Optional[Dict[str, float]] = None
    ):
        super().__init__(config, logger)

        self.params = params or {}
        self.sacsma_forcing_dir = self.forcing_dir
        self.spatial_mode = self.detect_spatial_mode('SACSMA')

        self.pet_method = self._get_config_value(
            lambda: self.config.model.sacsma.pet_method
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            'input'
        )

        self.latitude = self._get_config_value(
            lambda: self.config.model.sacsma.latitude
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            None
        )

    def run_preprocessing(self) -> bool:
        """Run SAC-SMA preprocessing workflow."""
        self.logger.info("Starting SAC-SMA preprocessing (lumped mode)")
        self.create_directories()
        self._prepare_lumped_forcing()
        self.logger.info("SAC-SMA preprocessing completed successfully")
        return True

    def _prepare_lumped_forcing(self) -> bool:
        """Prepare forcing data for lumped SAC-SMA simulation."""
        self.logger.info("Preparing lumped forcing data for SAC-SMA")

        from symfluence.models.utilities import ForcingDataProcessor

        # Load basin-averaged forcing
        fdp = ForcingDataProcessor(self.config, self.logger)

        ds = None
        if hasattr(self, 'forcing_basin_path') and self.forcing_basin_path.exists():
            ds = fdp.load_forcing_data(self.forcing_basin_path)
            if ds is not None:
                ds = self.subset_to_simulation_time(ds, "Forcing")

        if ds is None:
            merged_file = self.merged_forcing_path / f"{self.domain_name}_merged_forcing.nc"
            if merged_file.exists():
                ds = xr.open_dataset(merged_file)
                ds = self.subset_to_simulation_time(ds, "Forcing")

        if ds is None:
            raise FileNotFoundError(
                f"No forcing data found for domain '{self.domain_name}'."
            )

        time = pd.to_datetime(ds.time.values)

        # Extract precipitation
        precip = None
        for var in ['pr', 'precip', 'pptrate', 'prcp', 'precipitation']:
            if var in ds:
                precip = ds[var].values
                precip_units = ds[var].attrs.get('units', '')
                self.logger.info(f"Using precipitation variable: {var}")
                break
        if precip is None:
            raise ValueError("Precipitation variable not found in forcing data.")

        # Convert flux units to mm/day if needed
        units_norm = precip_units.strip().lower().replace(" ", "")
        if 'mm/s' in units_norm or 'kgm-2s-1' in units_norm or ('kg' in units_norm and 's-1' in units_norm):
            precip = precip * 86400
            self.logger.info(f"Converted precipitation from {precip_units} to mm/day")

        # Extract temperature
        temp = None
        for var in ['temp', 'tas', 'airtemp', 'tair', 'temperature', 'tmean']:
            if var in ds:
                temp = ds[var].values
                break
        if temp is None:
            raise ValueError("Temperature variable not found in forcing data.")

        # K to C
        if np.nanmean(temp) > UnitDetectionThresholds.TEMP_KELVIN_VS_CELSIUS:
            temp = temp - 273.15
            self.logger.info("Converted temperature from K to °C")

        # Average across spatial dims for lumped
        if precip.ndim > 1:
            precip = np.nanmean(precip, axis=1)
        if temp.ndim > 1:
            temp = np.nanmean(temp, axis=1)

        # PET
        pet = self._get_pet(ds, temp, time)

        # Flatten
        precip = precip.flatten()
        temp = temp.flatten()
        pet = pet.flatten()

        # Build DataFrame for aggregation
        forcing_df = pd.DataFrame({
            'time': time,
            'pr': precip,
            'temp': temp,
            'pet': pet,
        })
        forcing_df['time'] = pd.to_datetime(forcing_df['time'])

        # Aggregate to daily if sub-daily (SAC-SMA runs daily timestep)
        time_diff = forcing_df['time'].diff().median()
        if time_diff < pd.Timedelta(days=1):
            self.logger.info(
                f"Aggregating sub-daily forcing ({time_diff}) to daily for SAC-SMA"
            )
            forcing_df = forcing_df.set_index('time').resample('D').agg({
                'pr': 'mean',    # mean of mm/day rates = daily total (mm/day)
                'temp': 'mean',  # daily mean temperature (°C)
                'pet': 'mean',   # mean of mm/day rates = daily total (mm/day)
            }).reset_index()
            forcing_df = forcing_df.dropna()
            self.logger.info(f"Aggregated to {len(forcing_df)} daily timesteps")

        # Subset to simulation window
        time_window = self.get_simulation_time_window()
        if time_window:
            start_time, end_time = time_window
            forcing_df = forcing_df[
                (forcing_df['time'] >= start_time) &
                (forcing_df['time'] <= end_time)
            ]

        # Save NetCDF
        ds_out = xr.Dataset(
            data_vars={
                'pr': (['time'], forcing_df['pr'].values.astype(np.float32)),
                'temp': (['time'], forcing_df['temp'].values.astype(np.float32)),
                'pet': (['time'], forcing_df['pet'].values.astype(np.float32)),
            },
            coords={
                'time': pd.to_datetime(forcing_df['time']),
            },
            attrs={
                'model': 'SAC-SMA + Snow-17',
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'units_pr': 'mm/day',
                'units_temp': 'degC',
                'units_pet': 'mm/day',
            }
        )

        nc_file = self.sacsma_forcing_dir / f"{self.domain_name}_sacsma_forcing.nc"
        encoding = create_netcdf_encoding(ds_out, compression=True)
        ds_out.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved SAC-SMA forcing: {nc_file} ({len(forcing_df)} timesteps)")

        # Save CSV
        csv_file = self.sacsma_forcing_dir / f"{self.domain_name}_sacsma_forcing.csv"
        forcing_df.to_csv(csv_file, index=False)

        # Copy observations
        self._prepare_observations()

        return True

    def _get_pet(self, ds: xr.Dataset, temp: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
        """Get or calculate PET."""
        for var in ['pet', 'pET', 'potEvap', 'evap', 'evspsbl']:
            if var in ds:
                self.logger.info(f"Using PET from forcing data (variable: {var})")
                pet = ds[var].values
                pet_units = ds[var].attrs.get('units', '')
                units_norm = pet_units.strip().lower()
                if 'mm/s' in units_norm or 'kg' in units_norm:
                    pet = pet * 86400
                if pet.ndim > 1:
                    pet = np.nanmean(pet, axis=1)
                return pet

        # Calculate PET using Hamon method
        return self._calculate_hamon_pet(temp.flatten(), time)

    def _calculate_hamon_pet(self, temp: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
        """Calculate PET using Hamon method."""
        from symfluence.models.mixins.pet_calculator import PETCalculatorMixin

        self.logger.info("Calculating PET using Hamon method")
        lat = self.latitude
        if lat is None:
            try:
                import geopandas as gpd
                catchment = gpd.read_file(self.get_catchment_path())
                centroid = catchment.to_crs(epsg=4326).unary_union.centroid
                lat = centroid.y
            except Exception:
                lat = 45.0
                self.logger.warning(f"Using default latitude {lat}°")

        doy = np.asarray(time.dayofyear) if hasattr(time, 'dayofyear') else np.asarray(pd.to_datetime(time).dayofyear)
        return PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat)

    def _prepare_observations(self) -> None:
        """Copy observation data for validation."""
        obs_dir = self.project_observations_dir / 'streamflow' / 'preprocessed'
        obs_file = obs_dir / f"{self.domain_name}_streamflow_processed.csv"

        if obs_file.exists():
            self.logger.info(f"Observations available at: {obs_file}")
            obs_df = pd.read_csv(obs_file)
            output_obs = self.sacsma_forcing_dir / f"{self.domain_name}_observations.csv"
            obs_df.to_csv(output_obs, index=False)
