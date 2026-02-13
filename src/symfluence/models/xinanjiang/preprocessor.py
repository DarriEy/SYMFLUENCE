"""
Xinanjiang Model Preprocessor.

Prepares forcing data (P, PET) for Xinanjiang model execution.
Follows the same pattern as SAC-SMA preprocessor (lumped mode).
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


@ModelRegistry.register_preprocessor('XINANJIANG')
class XinanjiangPreProcessor(BaseModelPreProcessor, SpatialModeDetectionMixin):  # type: ignore[misc]
    """Preprocessor for Xinanjiang model.

    Prepares forcing data:
    - Precipitation (mm/day)
    - Potential evapotranspiration (mm/day)
    - Temperature (C) - for PET calculation if needed
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        params: Optional[Dict[str, float]] = None
    ):
        super().__init__(config, logger)

        self.params = params or {}
        self.xinanjiang_forcing_dir = self.forcing_dir
        self.spatial_mode = self.detect_spatial_mode('XINANJIANG')

        self.pet_method = self._get_config_value(
            lambda: self.config.model.xinanjiang.pet_method
            if self.config.model and hasattr(self.config.model, 'xinanjiang') and self.config.model.xinanjiang
            else None,
            'input'
        )

        self.latitude = self._get_config_value(
            lambda: self.config.model.xinanjiang.latitude
            if self.config.model and hasattr(self.config.model, 'xinanjiang') and self.config.model.xinanjiang
            else None,
            None
        )

    def _get_model_name(self) -> str:
        return "XINANJIANG"

    def run_preprocessing(self) -> bool:
        """Run Xinanjiang preprocessing workflow."""
        self.logger.info("Starting Xinanjiang preprocessing (lumped mode)")
        self.create_directories()
        self._prepare_lumped_forcing()
        self.logger.info("Xinanjiang preprocessing completed successfully")
        return True

    def _prepare_lumped_forcing(self) -> bool:
        """Prepare forcing data for lumped Xinanjiang simulation."""
        self.logger.info("Preparing lumped forcing data for Xinanjiang")

        from symfluence.models.utilities import ForcingDataProcessor

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
        for var in ['pr', 'precip', 'pptrate', 'prcp', 'precipitation', 'precipitation_flux']:
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
        for var in ['temp', 'tas', 'airtemp', 'tair', 'temperature', 'tmean', 'air_temperature']:
            if var in ds:
                temp = ds[var].values
                self.logger.info(f"Using temperature variable: {var}")
                break

        if temp is not None:
            # K to C
            if np.nanmean(temp) > UnitDetectionThresholds.TEMP_KELVIN_VS_CELSIUS:
                temp = temp - 273.15
                self.logger.info("Converted temperature from K to C")

        # Average across spatial dims for lumped
        if precip.ndim > 1:
            precip = np.nanmean(precip, axis=1)
        if temp is not None and temp.ndim > 1:
            temp = np.nanmean(temp, axis=1)

        # Get PET
        pet = self._get_pet(ds, temp, time)

        # Flatten
        precip = precip.flatten()
        if temp is not None:
            temp = temp.flatten()
        pet = pet.flatten()

        # Build DataFrame for aggregation
        forcing_data = {
            'time': time,
            'pr': precip,
            'pet': pet,
        }
        if temp is not None:
            forcing_data['temp'] = temp
        forcing_df = pd.DataFrame(forcing_data)
        forcing_df['time'] = pd.to_datetime(forcing_df['time'])

        # Aggregate to daily if sub-daily (Xinanjiang runs daily timestep)
        time_diff = forcing_df['time'].diff().median()
        if time_diff < pd.Timedelta(days=1):
            self.logger.info(
                f"Aggregating sub-daily forcing ({time_diff}) to daily for Xinanjiang"
            )
            agg_dict = {
                'pr': 'mean',
                'pet': 'mean',
            }
            if 'temp' in forcing_df.columns:
                agg_dict['temp'] = 'mean'
            forcing_df = forcing_df.set_index('time').resample('D').agg(agg_dict).reset_index()
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
        out_vars = {
            'pr': (['time'], forcing_df['pr'].values.astype(np.float32)),
            'pet': (['time'], forcing_df['pet'].values.astype(np.float32)),
        }
        out_attrs = {
            'model': 'Xinanjiang',
            'spatial_mode': 'lumped',
            'domain': self.domain_name,
            'units_pr': 'mm/day',
            'units_pet': 'mm/day',
        }
        if 'temp' in forcing_df.columns:
            out_vars['temp'] = (['time'], forcing_df['temp'].values.astype(np.float32))
            out_attrs['units_temp'] = 'degC'

        ds_out = xr.Dataset(
            data_vars=out_vars,
            coords={'time': pd.to_datetime(forcing_df['time'])},
            attrs=out_attrs,
        )
        ds_out['pr'].attrs = {'units': 'mm/day', 'long_name': 'Precipitation'}
        ds_out['pet'].attrs = {'units': 'mm/day', 'long_name': 'Potential evapotranspiration'}
        if 'temp' in ds_out:
            ds_out['temp'].attrs = {'units': 'degC', 'long_name': 'Air temperature'}

        self.xinanjiang_forcing_dir.mkdir(parents=True, exist_ok=True)
        nc_file = self.xinanjiang_forcing_dir / f"{self.domain_name}_xinanjiang_forcing.nc"
        encoding = create_netcdf_encoding(ds_out, compression=True)
        ds_out.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved Xinanjiang forcing: {nc_file} ({len(forcing_df)} timesteps)")

        # Save CSV
        csv_file = self.xinanjiang_forcing_dir / f"{self.domain_name}_xinanjiang_forcing.csv"
        forcing_df.to_csv(csv_file, index=False)

        # Copy observations
        self._prepare_observations()

        ds.close()
        return True

    def _get_pet(self, ds: xr.Dataset, temp, time: pd.DatetimeIndex) -> np.ndarray:
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
        if temp is not None:
            return self._calculate_hamon_pet(temp.flatten(), time)

        # Fallback: simple fraction of precip
        self.logger.warning("No PET data or temperature available; using simple fraction of precip")
        precip = None
        for var in ['pr', 'precip', 'pptrate', 'prcp']:
            if var in ds:
                precip = ds[var].values
                break
        if precip is not None:
            if precip.ndim > 1:
                precip = np.nanmean(precip, axis=1)
            return precip.flatten() * 0.6
        return np.ones(len(time)) * 2.0  # ~2 mm/day constant fallback

    def _calculate_hamon_pet(self, temp: np.ndarray, time_values) -> np.ndarray:
        """Calculate PET using simplified Hamon method (mm/day).

        Uses the same formulation as SAC-SMA and HBV preprocessors:
        PET = 0.55 * (dayl/12)^2 * esat, where esat is saturated vapor
        pressure (kPa) from the Magnus formula. PET is zero when temp <= 0.
        """
        self.logger.info("Calculating PET using Hamon method")
        dates = pd.to_datetime(time_values)
        doy = dates.dayofyear.values
        lat_rad = np.radians(self.latitude if self.latitude else 45.0)

        # Solar declination
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)
        # Sunset hour angle
        omega = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
        # Day length (hours)
        dayl = 24.0 / np.pi * omega

        # Saturated vapour pressure (kPa) — Magnus formula
        esat = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))

        # Hamon PET (mm/day)
        pet = 0.55 * (dayl / 12.0) ** 2 * esat
        pet = np.where(temp > 0.0, pet, 0.0)
        pet = np.maximum(pet, 0.0)

        return pet

    def _prepare_observations(self) -> None:
        """Copy observation files for validation."""
        obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        if obs_dir.exists():
            import shutil
            dest_dir = self.xinanjiang_forcing_dir
            for obs_file in obs_dir.glob(f"{self.domain_name}*streamflow*.csv"):
                dest = dest_dir / obs_file.name
                if not dest.exists():
                    shutil.copy2(obs_file, dest)
                    self.logger.debug(f"Copied observation file: {obs_file.name}")
