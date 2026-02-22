"""
TOPMODEL Preprocessor.

Prepares forcing data (precipitation, temperature, PET) for TOPMODEL execution.
Supports lumped mode with ERA5 or other gridded forcing datasets.
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
from symfluence.core.constants import UnitConversion


@ModelRegistry.register_preprocessor('TOPMODEL')
class TopmodelPreProcessor(BaseModelPreProcessor, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Preprocessor for TOPMODEL.

    Prepares forcing data including:
    - Precipitation (mm/day)
    - Temperature (deg C)
    - Potential evapotranspiration (mm/day)
    """


    MODEL_NAME = "TOPMODEL"
    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        params: Optional[Dict[str, float]] = None
    ):
        """Initialize TOPMODEL preprocessor."""
        super().__init__(config, logger)

        self.params = params or {}

        # TOPMODEL-specific paths
        self.topmodel_setup_dir = self.setup_dir
        self.topmodel_forcing_dir = self.forcing_dir
        self.topmodel_results_dir = self.project_dir / 'simulations' / self.experiment_id / 'TOPMODEL'

        # Determine spatial mode
        self.spatial_mode = self.detect_spatial_mode('TOPMODEL')

        # PET method configuration
        self.pet_method = self._get_config_value(
            lambda: self.config.model.topmodel.pet_method if self.config.model and self.config.model.topmodel else None,
            'input'
        )

        self.latitude = self._get_config_value(
            lambda: self.config.model.topmodel.latitude if self.config.model and self.config.model.topmodel else None,
            None
        )

    def run_preprocessing(self) -> bool:
        """
        Run TOPMODEL preprocessing workflow.

        Returns:
            True if preprocessing completed successfully.
        """
        self.logger.info("Starting TOPMODEL preprocessing")

        try:
            # Create output directories
            self.topmodel_forcing_dir.mkdir(parents=True, exist_ok=True)

            # Load and process forcing data
            forcing_data = self._load_forcing_data()

            if forcing_data is None:
                self.logger.error("Failed to load forcing data")
                return False

            # Calculate PET if needed
            if self.pet_method != 'input':
                forcing_data = self._calculate_pet(forcing_data)

            # Save forcing data
            self._save_forcing(forcing_data)

            self.logger.info("TOPMODEL preprocessing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"TOPMODEL preprocessing failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing_data(self) -> Optional[Dict[str, Any]]:
        """Load forcing data using shared ForcingDataProcessor.

        Tries model-ready store first, falls back to basin_averaged_data.
        Handles multi-file (monthly chunked) and single-file forcing data.
        """
        from symfluence.models.utilities import ForcingDataProcessor

        fdp = ForcingDataProcessor(self.config, self.logger)

        ds = None
        if hasattr(self, 'forcing_basin_path') and self.forcing_basin_path.exists():
            ds = fdp.load_forcing_data(self.forcing_basin_path)
            if ds is not None:
                ds = self.subset_to_simulation_time(ds, "Forcing")

        if ds is None:
            self.logger.error(
                f"No forcing data found. "
                f"Checked: {getattr(self, 'forcing_basin_path', 'N/A')}"
            )
            return None

        self.logger.info(f"Loaded forcing dataset with {len(ds.time)} timesteps")

        try:
            # Extract precipitation
            precip = self._extract_variable(ds, ['pptrate', 'pr', 'precipitation', 'PREC', 'tp'])
            if precip is None:
                self.logger.error("Precipitation variable not found")
                return None

            # Extract temperature
            temp = self._extract_variable(ds, ['airtemp', 'temp', 'temperature', 'T2', 't2m', 'tas'])
            if temp is None:
                self.logger.error("Temperature variable not found")
                return None

            # Convert units
            precip_values = precip.values.flatten()
            temp_values = temp.values.flatten()

            # Get time coordinate and detect timestep
            time_index = pd.to_datetime(ds.time.values)
            timestep_seconds = 3600.0  # default hourly
            if len(time_index) > 1:
                timestep_seconds = (time_index[1] - time_index[0]).total_seconds()

            # Precipitation: convert rate (kg/m2/s) to depth per timestep (mm/timestep)
            precip_units = precip.attrs.get('units', '')
            if np.nanmean(precip_values[precip_values > 0]) < 0.01:
                precip_values = precip_values * timestep_seconds
                self.logger.info(
                    f"Converted precipitation from {precip_units or 'kg/m2/s'} "
                    f"to mm/timestep (×{timestep_seconds})"
                )

            # Temperature: convert from K to C if needed
            if np.nanmean(temp_values) > 100:
                temp_values = temp_values - 273.15
                self.logger.info("Converted temperature from K to deg C")

            # Extract PET if available
            pet_values = None
            pet = self._extract_variable(ds, ['pet', 'evspsblpot', 'PET'])
            if pet is not None:
                pet_values = pet.values.flatten()
                if np.nanmean(pet_values[pet_values > 0]) < 0.01:
                    pet_values = pet_values * timestep_seconds

            # Resample to daily if sub-daily
            if timestep_seconds < 86400:
                self.logger.info(f"Resampling from {timestep_seconds}s to daily")
                df = pd.DataFrame({
                    'precip': precip_values[:len(time_index)],
                    'temp': temp_values[:len(time_index)],
                }, index=time_index)
                if pet_values is not None:
                    df['pet'] = pet_values[:len(time_index)]

                daily = pd.DataFrame()
                daily['precip'] = df['precip'].resample('D').sum()
                daily['temp'] = df['temp'].resample('D').mean()
                if 'pet' in df.columns:
                    daily['pet'] = df['pet'].resample('D').sum()

                daily = daily.dropna()
                precip_values = daily['precip'].values
                temp_values = daily['temp'].values
                pet_values = daily['pet'].values if 'pet' in daily.columns else pet_values
                time_index = daily.index

            ds.close()

            result = {
                'precip': precip_values,
                'temp': temp_values,
                'time': time_index,
            }
            if pet_values is not None:
                result['pet'] = pet_values

            return result

        except Exception as e:
            self.logger.error(f"Error loading forcing: {e}")
            return None

    def _extract_variable(self, ds: xr.Dataset, names: list) -> Optional[xr.DataArray]:
        """Try to extract a variable by multiple possible names."""
        for name in names:
            if name in ds.data_vars:
                return ds[name]
        return None

    def _calculate_pet(self, forcing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate PET using the Oudin method (or Hamon fallback)."""
        from symfluence.models.mixins.pet_calculator import PETCalculatorMixin

        temp = forcing_data['temp']
        time_index = forcing_data['time']
        lat = self.latitude if self.latitude else 45.0
        doy = np.array([t.timetuple().tm_yday for t in time_index])

        if self.pet_method == 'oudin':
            pet = PETCalculatorMixin.oudin_pet_numpy(temp, doy, lat)
        else:
            pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat, coefficient=0.1651)

        forcing_data['pet'] = pet
        self.logger.info(f"Calculated PET using {self.pet_method} method, mean={np.mean(pet):.2f} mm/day")
        return forcing_data

    def _save_forcing(self, forcing_data: Dict[str, Any]) -> None:
        """Save forcing data in CSV and NetCDF formats."""
        domain_name = self.domain_name
        time_index = forcing_data['time']

        precip = forcing_data['precip']
        temp = forcing_data['temp']
        pet = forcing_data.get('pet', np.zeros_like(precip))

        n = min(len(precip), len(temp), len(pet), len(time_index))
        precip, temp, pet = precip[:n], temp[:n], pet[:n]
        time_index = time_index[:n]

        # Save CSV
        df = pd.DataFrame({
            'datetime': time_index,
            'pr': precip,
            'temp': temp,
            'pet': pet,
        })
        csv_path = self.topmodel_forcing_dir / f"{domain_name}_topmodel_forcing.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV forcing: {csv_path}")

        # Save NetCDF
        ds = xr.Dataset(
            {
                'pr': (['time'], precip.astype(np.float64)),
                'temp': (['time'], temp.astype(np.float64)),
                'pet': (['time'], pet.astype(np.float64)),
            },
            coords={'time': time_index},
            attrs={
                'model': 'TOPMODEL',
                'description': 'Forcing data for TOPMODEL native Python implementation',
                'units_pr': 'mm/day',
                'units_temp': 'degC',
                'units_pet': 'mm/day',
            }
        )
        nc_path = self.topmodel_forcing_dir / f"{domain_name}_topmodel_forcing.nc"
        encoding = create_netcdf_encoding(ds)
        ds.to_netcdf(nc_path, encoding=encoding)
        self.logger.info(f"Saved NetCDF forcing: {nc_path}")
