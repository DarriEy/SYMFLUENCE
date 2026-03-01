# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HBV Model Preprocessor.

Prepares forcing data (precipitation, temperature, PET) for HBV-96 model execution.
Supports both lumped and distributed modes.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion, UnitDetectionThresholds
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.mixins import SpatialModeDetectionMixin
from symfluence.models.registry import ModelRegistry
from symfluence.models.spatial_modes import SpatialMode
from symfluence.models.utilities import ForcingDataProcessor


@ModelRegistry.register_preprocessor('HBV')
class HBVPreProcessor(BaseModelPreProcessor, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Preprocessor for HBV-96 model.

    Prepares forcing data including:
    - Precipitation (mm/day)
    - Temperature (°C)
    - Potential evapotranspiration (mm/day)

    Supports lumped mode (single time series) and distributed mode
    (per-HRU time series for mizuRoute integration).
    """


    MODEL_NAME = "HBV"
    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize HBV preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance
            params: Optional parameter overrides
        """
        super().__init__(config, logger)

        self.params = params or {}

        # HBV-specific paths
        self.hbv_setup_dir = self.setup_dir
        self.hbv_forcing_dir = self.forcing_dir
        self.hbv_results_dir = self.project_dir / 'simulations' / self.experiment_id / 'HBV'

        # Determine spatial mode using mixin
        self.spatial_mode = self.detect_spatial_mode('HBV')

        # PET method configuration
        self.pet_method = self._get_config_value(
            lambda: self.config.model.hbv.pet_method if self.config.model and self.config.model.hbv else None,
            'input'
        )

        self.latitude = self._get_config_value(
            lambda: self.config.model.hbv.latitude if self.config.model and self.config.model.hbv else None,
            None
        )

        # Unit handling
        self.allow_unit_heuristics = self._get_config_value(
            lambda: self.config.model.hbv.allow_unit_heuristics if self.config.model and self.config.model.hbv else None,
            False
        )

        # Timestep configuration (1=hourly, 24=daily)
        self.timestep_hours = self._get_config_value(
            lambda: self.config.model.hbv.timestep_hours if self.config.model and self.config.model.hbv else None,
            24
        )

    def run_preprocessing(self) -> bool:
        """
        Run HBV preprocessing workflow.

        Creates forcing data files in the appropriate format for HBV model execution.

        Returns:
            True if preprocessing completed successfully.

        Raises:
            FileNotFoundError: If required forcing data files are not found.
            RuntimeError: If preprocessing fails due to data errors.
        """
        self.logger.info(f"Starting HBV preprocessing in {self.spatial_mode} mode")

        # Create directories
        self.create_directories()

        # Prepare forcing data (will raise exceptions on failure)
        if self.spatial_mode == SpatialMode.LUMPED:
            self._prepare_lumped_forcing()
        else:
            self._prepare_distributed_forcing()

        self.logger.info("HBV preprocessing completed successfully")
        return True

    def _prepare_lumped_forcing(self) -> bool:
        """
        Prepare forcing data for lumped HBV simulation.

        Creates BOTH hourly and daily forcing files from the same source data
        to ensure consistency between different timestep configurations.

        Returns:
            True if successful.
        """
        self.logger.info("Preparing lumped forcing data for HBV")

        try:
            # Load basin-averaged forcing
            forcing_ds = self._load_basin_averaged_forcing()
            if forcing_ds is None:
                raise FileNotFoundError(
                    f"No forcing data found for domain '{self.domain_name}'. "
                    f"Checked: {self.forcing_basin_path}, {self.merged_forcing_path}"
                )

            # Get timestep configuration from source data
            timestep_config = self.get_timestep_config()

            # Extract variables
            time = pd.to_datetime(forcing_ds.time.values)

            # Precipitation (check various naming conventions)
            precip_vars = ['pr', 'precip', 'pptrate', 'prcp', 'precipitation']
            precip = None
            precip_var_name = None
            for var in precip_vars:
                if var in forcing_ds:
                    precip = forcing_ds[var].values
                    precip_var_name = var
                    self.logger.info(f"Using precipitation variable: {var}")
                    break
            if precip is None:
                raise ValueError(
                    f"Precipitation variable not found in forcing data. "
                    f"Tried: {precip_vars}. Available: {list(forcing_ds.data_vars)}"
                )

            # Convert precipitation units if needed
            precip_units = forcing_ds[precip_var_name].attrs.get('units', '')
            precip = self._convert_flux_units(
                precip,
                precip_units,
                timestep_config['time_label'],
                "Precipitation"
            )

            # Temperature (check various naming conventions)
            temp_vars = ['temp', 'tas', 'airtemp', 'tair', 'temperature', 'tmean']
            temp = None
            for var in temp_vars:
                if var in forcing_ds:
                    temp = forcing_ds[var].values
                    self.logger.info(f"Using temperature variable: {var}")
                    break
            if temp is None:
                raise ValueError(
                    f"Temperature variable not found in forcing data. "
                    f"Tried: {temp_vars}. Available: {list(forcing_ds.data_vars)}"
                )

            # Convert temperature from K to C if needed (heuristic detection)
            if np.nanmean(temp) > UnitDetectionThresholds.TEMP_KELVIN_VS_CELSIUS:
                temp = temp - 273.15
                self.logger.info("Converted temperature from K to °C")

            # For lumped mode, average across HRUs if data has spatial dimension
            # This handles basin-averaged forcing that still has HRU dimension
            if precip.ndim > 1:
                self.logger.info(f"Averaging precipitation across {precip.shape[1]} HRUs for lumped mode")
                precip = np.nanmean(precip, axis=1)
            if temp.ndim > 1:
                self.logger.info(f"Averaging temperature across {temp.shape[1]} HRUs for lumped mode")
                temp = np.nanmean(temp, axis=1)

            # Potential evapotranspiration (calculated at daily resolution, will be distributed)
            pet_result = self._get_pet(forcing_ds, temp, time, timestep_config['time_label'])
            if pet_result is None:
                raise ValueError(
                    "Could not calculate or extract PET. "
                    "Ensure forcing data contains PET variable or temperature for calculation."
                )
            pet_values, pet_is_daily = pet_result

            # Create hourly DataFrame (source resolution)
            if timestep_config['time_label'] == 'hourly':
                self.logger.info("Source data is hourly - creating both hourly and daily forcing files")

                # Build hourly forcing DataFrame
                # PET from Hamon is daily - distribute to hourly (mm/day -> mm/hour)
                pet_hourly = pet_values / 24.0 if pet_is_daily else pet_values

                forcing_hourly = pd.DataFrame({
                    'time': time,
                    'pr': precip.flatten(),    # mm/hour
                    'temp': temp.flatten(),    # °C
                    'pet': pet_hourly.flatten() # mm/hour
                }).set_index('time')

                # Subset to simulation time window
                time_window = self.get_simulation_time_window()
                if time_window:
                    start_time, end_time = time_window
                    forcing_hourly = forcing_hourly[
                        (forcing_hourly.index >= start_time) &
                        (forcing_hourly.index <= end_time)
                    ]

                # Save hourly forcing (1h)
                self._save_forcing_file(forcing_hourly, timestep_hours=1)

                # Create daily forcing by resampling
                forcing_daily = forcing_hourly.resample('D').agg({
                    'pr': 'sum',      # mm/hour * 24 = mm/day
                    'temp': 'mean',   # daily mean temperature
                    'pet': 'sum'      # mm/hour * 24 = mm/day
                })

                # Save daily forcing (24h)
                self._save_forcing_file(forcing_daily.reset_index(), timestep_hours=24)

                self.logger.info(f"Created hourly forcing: {len(forcing_hourly)} timesteps")
                self.logger.info(f"Created daily forcing: {len(forcing_daily)} timesteps")

            else:
                # Daily source data - only create daily file
                self.logger.info("Source data is daily - creating daily forcing file only")

                forcing_daily = pd.DataFrame({
                    'time': time,
                    'pr': precip.flatten(),
                    'temp': temp.flatten(),
                    'pet': pet_values.flatten()
                })

                # Subset to simulation time window
                time_window = self.get_simulation_time_window()
                if time_window:
                    start_time, end_time = time_window
                    forcing_daily['time'] = pd.to_datetime(forcing_daily['time'])
                    forcing_daily = forcing_daily[
                        (forcing_daily['time'] >= start_time) &
                        (forcing_daily['time'] <= end_time)
                    ]

                # Save daily forcing
                self._save_forcing_file(forcing_daily, timestep_hours=24)

            # Load and save observations if available
            self._prepare_observations()

            return True

        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error preparing lumped forcing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise RuntimeError(f"HBV lumped forcing preparation failed: {e}") from e

    def _save_forcing_file(self, forcing_df: pd.DataFrame, timestep_hours: int) -> None:
        """Save forcing DataFrame to both CSV and NetCDF files.

        Args:
            forcing_df: DataFrame with 'pr', 'temp', 'pet' columns (and 'time' column or index)
            timestep_hours: Timestep in hours (1 for hourly, 24 for daily)
        """
        # Ensure time is a column
        if 'time' not in forcing_df.columns:
            forcing_df = forcing_df.reset_index()

        # Determine units
        if timestep_hours == 24:
            pr_units = 'mm/day'
            pet_units = 'mm/day'
            timestep_label = 'daily'
        else:
            pr_units = f'mm/{timestep_hours}h'
            pet_units = f'mm/{timestep_hours}h'
            timestep_label = f'{timestep_hours}-hourly'

        # Save CSV
        csv_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_{timestep_hours}h.csv"
        forcing_df.to_csv(csv_file, index=False)
        self.logger.debug(f"Saved CSV: {csv_file}")

        # Save NetCDF
        ds = xr.Dataset(
            data_vars={
                'pr': (['time'], forcing_df['pr'].values.astype(np.float32)),
                'temp': (['time'], forcing_df['temp'].values.astype(np.float32)),
                'pet': (['time'], forcing_df['pet'].values.astype(np.float32)),
            },
            coords={
                'time': pd.to_datetime(forcing_df['time']),
            },
            attrs={
                'model': 'HBV-96',
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'timestep_hours': timestep_hours,
                'timestep_label': timestep_label,
                'units_pr': pr_units,
                'units_temp': 'degC',
                'units_pet': pet_units,
            }
        )

        nc_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_{timestep_hours}h.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(nc_file, encoding=encoding)
        ds.close()
        self.logger.info(f"Saved forcing: {nc_file} ({timestep_label}, {len(forcing_df)} timesteps)")

    def _prepare_distributed_forcing(self) -> bool:
        """
        Prepare forcing data for distributed HBV simulation.

        Creates per-HRU time series for mizuRoute integration.

        Returns:
            True if successful.
        """
        self.logger.info("Preparing distributed forcing data for HBV")

        try:
            # Load gridded forcing
            forcing_ds = self._load_merged_forcing()
            if forcing_ds is None:
                self.logger.warning("Merged forcing not found, falling back to basin-averaged")
                return self._prepare_lumped_forcing()

            # Get HRU information
            catchment_path = self.get_catchment_path()
            if not catchment_path.exists():
                raise FileNotFoundError(
                    f"Catchment shapefile not found: {catchment_path}. "
                    "Required for distributed HBV preprocessing."
                )

            import geopandas as gpd
            catchment = gpd.read_file(catchment_path)
            n_hrus = len(catchment)
            self.logger.info(f"Processing forcing for {n_hrus} HRUs")

            # Get timestep configuration
            timestep_config = self.get_timestep_config()

            # Extract time coordinate
            time = pd.to_datetime(forcing_ds.time.values)

            # Initialize arrays for HRU data
            hru_ids = catchment[self.hru_id_col].values if self.hru_id_col in catchment.columns else np.arange(n_hrus) + 1

            # For distributed mode, we need to extract forcing for each HRU
            # This depends on how the forcing data is structured
            # Typically: (time, hru) or (time, lat, lon)

            if 'hru' in forcing_ds.dims:
                # Forcing already per-HRU
                precip_var_name = 'pr' if 'pr' in forcing_ds else 'precip'
                precip = forcing_ds[precip_var_name].values
                temp = self._get_temperature_variable(forcing_ds)
                pet = self._get_pet_distributed(forcing_ds, temp, time, timestep_config['time_label'])

                # Convert precipitation units if needed
                precip_units = forcing_ds[precip_var_name].attrs.get('units', '')
            else:
                # Need to spatially average forcing to HRUs
                self.logger.info("Spatially averaging gridded forcing to HRUs")
                precip, temp, pet = self._spatially_average_to_hrus(forcing_ds, catchment)

                # Try to get units from original forcing dataset
                precip_var_candidates = ['pr', 'precip', 'pptrate', 'prcp', 'precipitation']
                precip_units = ''
                for var in precip_var_candidates:
                    if var in forcing_ds:
                        precip_units = forcing_ds[var].attrs.get('units', '')
                        break

            # Convert precipitation from rate units to depth units if needed
            precip = self._convert_flux_units(
                precip,
                precip_units,
                timestep_config['time_label'],
                "Distributed precipitation"
            )

            # Convert temperature from K to C if needed
            if np.nanmean(temp) > 100:
                temp = temp - 273.15

            # Handle temporal resolution based on timestep_hours config
            pet_is_daily = False
            if timestep_config['time_label'] == 'hourly' and pet.shape[0] >= 24:
                if pet.ndim == 2:
                    pet_is_daily = np.allclose(pet[::24, :], pet[1:24, :], equal_nan=True)
                else:
                    pet_is_daily = np.allclose(pet[::24], pet[1:24], equal_nan=True)

            if timestep_config['time_label'] == 'hourly' and self.timestep_hours == 24:
                # Default behavior: resample hourly to daily for daily HBV
                self.logger.info("Resampling hourly data to daily for HBV (HBV_TIMESTEP_HOURS=24)")
                precip, temp, pet, time = self._resample_to_daily(
                    precip, temp, pet, time, pet_is_daily=pet_is_daily
                )
            elif timestep_config['time_label'] == 'hourly' and self.timestep_hours < 24:
                # Sub-daily mode: preserve hourly resolution
                self.logger.info(f"Preserving hourly resolution for distributed HBV (HBV_TIMESTEP_HOURS={self.timestep_hours})")
                # PET from Hamon is daily, distribute to hourly
                if pet_is_daily:
                    self.logger.info("Distributing daily PET across hourly timesteps")
                    pet = pet / 24.0

            # Determine units based on timestep
            if self.timestep_hours == 24:
                pr_units = 'mm/day'
                pet_units = 'mm/day'
                timestep_label = 'daily'
            else:
                pr_units = f'mm/{self.timestep_hours}h'
                pet_units = f'mm/{self.timestep_hours}h'
                timestep_label = f'{self.timestep_hours}-hourly'

            # Create xarray Dataset for distributed forcing
            ds = xr.Dataset(
                data_vars={
                    'pr': (['time', 'hru'], precip),
                    'temp': (['time', 'hru'], temp),
                    'pet': (['time', 'hru'], pet),
                    'hru_id': (['hru'], hru_ids.astype(np.int32)),
                },
                coords={
                    'time': time,
                    'hru': np.arange(n_hrus),
                },
                attrs={
                    'model': 'HBV-96',
                    'spatial_mode': 'distributed',
                    'domain': self.domain_name,
                    'n_hrus': n_hrus,
                    'timestep_hours': self.timestep_hours,
                    'timestep_label': timestep_label,
                    'units_pr': pr_units,
                    'units_temp': 'degC',
                    'units_pet': pet_units,
                }
            )

            # Subset to simulation time window
            time_window = self.get_simulation_time_window()
            if time_window:
                start_time, end_time = time_window
                ds = ds.sel(time=slice(start_time, end_time))

            # Save distributed forcing
            output_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_distributed_{self.timestep_hours}h.nc"
            encoding = create_netcdf_encoding(ds, compression=True)
            ds.to_netcdf(output_file, encoding=encoding)
            self.logger.info(f"Saved distributed forcing to: {output_file}")

            # Also save observations
            self._prepare_observations()

            return True

        except FileNotFoundError:
            # Re-raise file not found errors without wrapping
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error preparing distributed forcing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise RuntimeError(f"HBV distributed forcing preparation failed: {e}") from e

    def _load_basin_averaged_forcing(self) -> Optional[xr.Dataset]:
        """
        Load basin-averaged forcing data using ForcingDataProcessor.

        Returns:
            xr.Dataset with forcing data, or None if no data found
            (allows caller to implement fallback logic).

        Raises:
            RuntimeError: If forcing data loading fails due to errors
            (not just missing files).
        """
        # Use shared ForcingDataProcessor for loading (handles multi-file data)
        fdp = ForcingDataProcessor(self.config, self.logger)

        # Try forcing_basin_path first (from base class)
        if hasattr(self, 'forcing_basin_path') and self.forcing_basin_path.exists():
            self.logger.info(f"Loading basin-averaged forcing from: {self.forcing_basin_path}")
            try:
                ds = fdp.load_forcing_data(self.forcing_basin_path)
                if ds is not None:
                    # Subset to simulation time window
                    ds = self.subset_to_simulation_time(ds, "Forcing")
                    return ds
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                raise RuntimeError(
                    f"Error loading forcing data from {self.forcing_basin_path}: {e}"
                ) from e

        # Fall back to merged forcing
        return self._load_merged_forcing()

    def _load_merged_forcing(self) -> Optional[xr.Dataset]:
        """
        Load merged forcing data.

        Returns:
            xr.Dataset with forcing data, or None if file not found
            (allows caller to implement fallback logic).
        """
        merged_file = self.merged_forcing_path / f"{self.domain_name}_merged_forcing.nc"
        if merged_file.exists():
            self.logger.info(f"Loading merged forcing from: {merged_file}")
            ds = xr.open_dataset(merged_file)
            ds = self.subset_to_simulation_time(ds, "Forcing")
            return ds

        self.logger.warning(f"Merged forcing file not found at: {merged_file}")
        return None

    def _get_temperature_variable(self, ds: xr.Dataset) -> np.ndarray:
        """Extract temperature variable from dataset."""
        for var in ['temp', 'tas', 'airtemp', 'tair', 'temperature']:
            if var in ds:
                return ds[var].values
        raise ValueError("Temperature variable not found in forcing dataset")

    def _normalize_units(self, units: str) -> str:
        """Normalize unit strings for robust comparisons."""
        return units.strip().lower().replace(" ", "")

    def _apply_flux_unit_heuristic(
        self,
        values: np.ndarray,
        timestep_label: str,
        context: str
    ) -> np.ndarray:
        """Apply magnitude-based unit heuristics (opt-in)."""
        if np.nanmean(np.abs(values)) < UnitDetectionThresholds.FLUX_RATE_MM_S_VS_MM_DAY:
            if timestep_label == 'hourly':
                self.logger.warning(f"{context} values appear to be in mm/s; converting to mm/hour (×3600)")
                return values * UnitConversion.SECONDS_PER_HOUR
            self.logger.warning(f"{context} values appear to be in mm/s; converting to mm/day (×86400)")
            return values * UnitConversion.SECONDS_PER_DAY
        return values

    def _convert_flux_units(
        self,
        values: np.ndarray,
        units: str,
        timestep_label: str,
        context: str
    ) -> np.ndarray:
        """
        Convert flux units to mm per timestep.

        Supported units:
        - mm/day, mm/d
        - mm/hour, mm/hr
        - mm/s
        - kg m-2 s-1 (or similar mass flux)
        """
        units_raw = units or ""
        units_norm = self._normalize_units(units_raw)

        if not units_norm:
            if self.allow_unit_heuristics:
                return self._apply_flux_unit_heuristic(values, timestep_label, context)
            raise ValueError(
                f"{context} units missing. Provide explicit units in forcing data or set "
                "HBV_ALLOW_UNIT_HEURISTICS=true to enable magnitude-based detection."
            )

        is_rate = (
            "mm/s" in units_norm or
            "mms-1" in units_norm or
            ("kg" in units_norm and ("m-2" in units_norm or "m^-2" in units_norm or "m2" in units_norm)
             and ("s-1" in units_norm or "/s" in units_norm))
        )

        if is_rate:
            if timestep_label == 'hourly':
                self.logger.info(f"Converted {context} from {units_raw} to mm/hour (×3600)")
                return values * UnitConversion.SECONDS_PER_HOUR
            self.logger.info(f"Converted {context} from {units_raw} to mm/day (×86400)")
            return values * UnitConversion.SECONDS_PER_DAY

        if "mm/day" in units_norm or "mm/d" in units_norm:
            if timestep_label == 'hourly':
                raise ValueError(
                    f"{context} units are {units_raw} but forcing is hourly. "
                    "Convert to mm/hour or supply correct units."
                )
            return values

        if "mm/hour" in units_norm or "mm/hr" in units_norm or "mmh-1" in units_norm:
            if timestep_label != 'hourly':
                raise ValueError(
                    f"{context} units are {units_raw} but forcing is daily. "
                    "Convert to mm/day or supply correct units."
                )
            return values

        if self.allow_unit_heuristics:
            self.logger.warning(f"Unrecognized {context} units '{units_raw}'. Falling back to heuristic detection.")
            return self._apply_flux_unit_heuristic(values, timestep_label, context)

        raise ValueError(
            f"Unrecognized {context} units '{units_raw}'. Provide explicit units or set "
            "HBV_ALLOW_UNIT_HEURISTICS=true."
        )

    def _get_pet(
        self,
        ds: xr.Dataset,
        temp: np.ndarray,
        time: pd.DatetimeIndex,
        timestep_label: str
    ) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Get or calculate PET (Potential Evapotranspiration).

        Args:
            ds: Forcing dataset
            temp: Temperature array (°C)
            time: Time index

        Returns:
            PET array (mm/day) or None if calculation fails.
        """
        # Check for PET in forcing data
        for var in ['pet', 'pET', 'potEvap', 'evap', 'evspsbl']:
            if var in ds:
                self.logger.info(f"Using PET from forcing data (variable: {var})")
                pet = ds[var].values
                pet_units = ds[var].attrs.get('units', '')
                pet = self._convert_flux_units(pet, pet_units, timestep_label, "PET")
                pet_values = pet.flatten() if pet.ndim > 1 else pet
                return pet_values, (timestep_label != 'hourly')

        # Calculate PET if not in forcing
        if self.pet_method == 'hamon':
            return self._calculate_hamon_pet(temp.flatten(), time), True
        elif self.pet_method == 'thornthwaite':
            return self._calculate_thornthwaite_pet(temp.flatten(), time), True
        else:
            self.logger.warning("No PET found in forcing and pet_method='input'. Using Hamon method.")
            return self._calculate_hamon_pet(temp.flatten(), time), True

    def _get_pet_distributed(
        self,
        ds: xr.Dataset,
        temp: np.ndarray,
        time: pd.DatetimeIndex,
        timestep_label: str
    ) -> np.ndarray:
        """Get or calculate PET for distributed mode."""
        for var in ['pet', 'pET', 'potEvap', 'evap', 'evspsbl']:
            if var in ds:
                pet = ds[var].values
                pet_units = ds[var].attrs.get('units', '')
                return self._convert_flux_units(pet, pet_units, timestep_label, "PET")

        # Calculate PET for each HRU using catchment-mean temperature
        mean_temp = np.nanmean(temp, axis=1)  # Average across HRUs for PET calc
        pet_1d = self._calculate_hamon_pet(mean_temp, time)

        # Broadcast to all HRUs
        return np.broadcast_to(pet_1d[:, np.newaxis], temp.shape)

    def _calculate_hamon_pet(
        self,
        temp: np.ndarray,
        time: pd.DatetimeIndex
    ) -> np.ndarray:
        """Calculate PET using Hamon method."""
        from symfluence.models.mixins.pet_calculator import PETCalculatorMixin

        self.logger.info("Calculating PET using Hamon method")

        if self.latitude is None:
            try:
                import geopandas as gpd
                catchment = gpd.read_file(self.get_catchment_path())
                centroid = catchment.to_crs(epsg=4326).unary_union.centroid
                lat = centroid.y
            except (FileNotFoundError, KeyError, IndexError, ValueError):
                lat = 45.0
                self.logger.warning(f"Using default latitude {lat}° for PET calculation")
        else:
            lat = self.latitude

        doy = np.asarray(time.dayofyear)
        return PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat)

    def _calculate_thornthwaite_pet(
        self,
        temp: np.ndarray,
        time: pd.DatetimeIndex
    ) -> np.ndarray:
        """Calculate PET using Thornthwaite method."""
        from symfluence.models.mixins.pet_calculator import PETCalculatorMixin

        self.logger.info("Calculating PET using Thornthwaite method")
        lat = self.latitude if self.latitude else 45.0
        return PETCalculatorMixin.thornthwaite_pet_numpy(temp, time, lat)

    def _spatially_average_to_hrus(
        self,
        ds: xr.Dataset,
        catchment
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spatially average gridded forcing to HRUs.

        Args:
            ds: Gridded forcing dataset
            catchment: GeoDataFrame with HRU geometries

        Returns:
            Tuple of (precip, temp, pet) arrays with shape (time, n_hrus)
        """
        import rasterio
        from rasterio.features import geometry_mask

        n_hrus = len(catchment)
        n_times = len(ds.time)

        # Get spatial coordinates
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lats = ds.lat.values
            lons = ds.lon.values
        elif 'y' in ds.coords and 'x' in ds.coords:
            lats = ds.y.values
            lons = ds.x.values
        else:
            raise ValueError("Cannot find spatial coordinates in forcing dataset")

        # Create affine transform
        res_lat = np.abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
        res_lon = np.abs(lons[1] - lons[0]) if len(lons) > 1 else 0.1
        transform = rasterio.transform.from_origin(
            lons.min() - res_lon/2,
            lats.max() + res_lat/2,
            res_lon,
            res_lat
        )

        # Extract variables
        precip_var = 'pr' if 'pr' in ds else 'precip'
        temp_var = self._find_var(ds, ['temp', 'tas', 'airtemp', 'tair'])

        precip_grid = ds[precip_var].values
        temp_grid = ds[temp_var].values

        # Initialize output arrays
        precip_hru = np.zeros((n_times, n_hrus))
        temp_hru = np.zeros((n_times, n_hrus))

        # Ensure CRS match
        catchment_reproj = catchment.to_crs(epsg=4326)

        # Spatial average for each HRU
        for i, geom in enumerate(catchment_reproj.geometry):
            try:
                mask = geometry_mask(
                    [geom],
                    out_shape=(len(lats), len(lons)),
                    transform=transform,
                    invert=True
                )

                for t in range(n_times):
                    precip_masked = np.ma.masked_array(precip_grid[t], ~mask)
                    temp_masked = np.ma.masked_array(temp_grid[t], ~mask)

                    precip_hru[t, i] = np.ma.mean(precip_masked)
                    temp_hru[t, i] = np.ma.mean(temp_masked)

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Error averaging HRU {i}: {e}. Using catchment mean.")
                precip_hru[:, i] = np.nanmean(precip_grid, axis=(1, 2))
                temp_hru[:, i] = np.nanmean(temp_grid, axis=(1, 2))

        # Calculate PET (broadcast across HRUs)
        mean_temp = np.nanmean(temp_hru, axis=1)
        time_idx = pd.to_datetime(ds.time.values)
        pet_1d = self._calculate_hamon_pet(mean_temp, time_idx)
        pet_hru = np.broadcast_to(pet_1d[:, np.newaxis], (n_times, n_hrus))

        return precip_hru, temp_hru, pet_hru.copy()

    def _find_var(self, ds: xr.Dataset, candidates: list) -> str:
        """Find first matching variable name."""
        for var in candidates:
            if var in ds:
                return var
        raise ValueError(f"None of {candidates} found in dataset")

    def _resample_to_daily(
        self,
        precip: np.ndarray,
        temp: np.ndarray,
        pet: np.ndarray,
        time: pd.DatetimeIndex,
        pet_is_daily: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Resample hourly data to daily."""
        pet_agg = 'mean' if pet_is_daily else 'sum'
        # Create DataFrames for resampling
        n_hrus = precip.shape[1] if precip.ndim > 1 else 1

        if n_hrus == 1:
            df = pd.DataFrame({
                'pr': precip.flatten(),
                'temp': temp.flatten(),
                'pet': pet.flatten()
            }, index=time)

            df_daily = df.resample('D').agg({
                'pr': 'sum',
                'temp': 'mean',
                'pet': pet_agg
            })

            return (  # type: ignore[return-value]
                np.asarray(df_daily['pr'].values)[:, np.newaxis],
                np.asarray(df_daily['temp'].values)[:, np.newaxis],
                np.asarray(df_daily['pet'].values)[:, np.newaxis],
                df_daily.index
            )
        else:
            # Multi-HRU case
            precip_daily = []
            temp_daily = []
            pet_daily = []

            for i in range(n_hrus):
                df = pd.DataFrame({
                    'pr': precip[:, i],
                    'temp': temp[:, i],
                    'pet': pet[:, i]
                }, index=time)

                df_daily = df.resample('D').agg({
                    'pr': 'sum',
                    'temp': 'mean',
                    'pet': pet_agg
                })

                precip_daily.append(df_daily['pr'].values)
                temp_daily.append(df_daily['temp'].values)
                pet_daily.append(df_daily['pet'].values)

            return (  # type: ignore[return-value]
                np.column_stack(precip_daily),
                np.column_stack(temp_daily),
                np.column_stack(pet_daily),
                df_daily.index
            )

    def _prepare_observations(self) -> None:
        """Prepare observation data for validation/calibration."""
        obs_dir = self.project_observations_dir / 'streamflow' / 'preprocessed'
        obs_file = obs_dir / f"{self.domain_name}_streamflow_processed.csv"

        if obs_file.exists():
            self.logger.info(f"Observations available at: {obs_file}")

            # Copy to HBV output dir for easy access
            obs_df = pd.read_csv(obs_file)
            output_obs = self.hbv_forcing_dir / f"{self.domain_name}_observations.csv"
            obs_df.to_csv(output_obs, index=False)
        else:
            self.logger.warning(f"No observation file found at: {obs_file}")

    def load_forcing_and_obs(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Load prepared forcing and observation data.

        Returns:
            Tuple of (forcing_dict, observations)
            forcing_dict contains 'precip', 'temp', 'pet' arrays
        """
        if self.spatial_mode == SpatialMode.DISTRIBUTED:
            forcing_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_distributed_{self.timestep_hours}h.nc"
        else:
            forcing_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_{self.timestep_hours}h.nc"

        if not forcing_file.exists():
            raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

        ds = xr.open_dataset(forcing_file)

        forcing = {
            'precip': ds['pr'].values,
            'temp': ds['temp'].values,
            'pet': ds['pet'].values,
            'time': pd.to_datetime(ds.time.values),
        }

        # Load observations if available
        obs_file = self.hbv_forcing_dir / f"{self.domain_name}_observations.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            observations = obs_df.iloc[:, 0].values
        else:
            observations = None

        return forcing, observations  # type: ignore[return-value]
