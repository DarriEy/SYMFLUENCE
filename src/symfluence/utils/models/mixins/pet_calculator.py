"""
Potential Evapotranspiration (PET) Calculator Mixin.

Provides three PET calculation methods for use in hydrological model preprocessors:
- Oudin's formula (simple, temperature-based)
- Hamon's method (temperature and daylight-based)
- Hargreaves method (simplified version)

All methods automatically detect and handle temperature units (Kelvin or Celsius).
"""

import numpy as np
import pandas as pd
import xarray as xr


class PETCalculatorMixin:
    """
    Mixin class providing PET calculation methods.

    This mixin provides three common PET calculation methods that can be
    used by model preprocessors. All methods:
    - Auto-detect temperature units (Kelvin vs Celsius)
    - Handle both lumped and distributed (multi-HRU) configurations
    - Return xarray DataArrays with proper metadata

    Usage:
        class MyModelPreProcessor(BaseModelPreProcessor, PETCalculatorMixin):
            def prepare_forcing(self):
                pet = self.calculate_pet_oudin(temp_data, latitude)

    Note:
        Requires self.logger to be available (provided by BaseModelPreProcessor)
    """

    def calculate_pet_oudin(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate potential evapotranspiration using Oudin's formula.

        Oudin's formula is a simple temperature-based method:
        PET = Ra * (T + 5) / 100 when T > -5°C, else 0

        Reference:
            Oudin et al. (2005). "Which potential evapotranspiration input
            for a lumped rainfall-runoff model?"

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata

        Raises:
            ValueError: If temperature data has unexpected range
        """
        self.logger.info("Calculating PET using Oudin's formula")

        # Load data if using dask
        if hasattr(temp_data.data, 'compute'):
            self.logger.debug("Loading temperature data from dask array...")
            temp_data = temp_data.load()

        # Auto-detect temperature units
        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            temp_mean = float(temp_data.mean())
            temp_min = float(temp_data.min())
            temp_max = float(temp_data.max())

        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}, Min={temp_min:.2f}, Max={temp_max:.2f}")

        # Convert to Celsius if needed
        if temp_mean > 100:  # Likely Kelvin
            self.logger.info("Temperature appears to be in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Likely Celsius
            self.logger.info("Temperature appears to be in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")

        # Verify converted temperature is reasonable
        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            temp_mean_C = float(temp_C.mean())
            self.logger.debug(
                f"Temperature in Celsius: Mean={temp_mean_C:.2f}°C, "
                f"Min={float(temp_C.min()):.2f}°C, Max={float(temp_C.max()):.2f}°C"
            )

        if temp_mean_C < -60 or temp_mean_C > 60:
            self.logger.error(f"Temperature is unrealistic: {temp_mean_C:.2f}°C")
            raise ValueError(f"Unrealistic temperature after conversion: {temp_mean_C:.2f}°C")

        # Get day of year
        time_values = pd.to_datetime(temp_data.time.values)
        doy = xr.DataArray(time_values.dayofyear, coords={'time': temp_data.time}, dims=['time'])

        # Calculate solar radiation components
        lat_rad = np.deg2rad(lat)

        # Solar declination (radians)
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle with numerical stability
        cos_arg = -np.tan(lat_rad) * np.tan(solar_decl)
        cos_arg = cos_arg.clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            self.logger.debug(f"Solar radiation Ra: Mean={float(Ra.mean()):.2f} MJ/m²/day")

        # Broadcast Ra if needed for multi-HRU configurations
        if 'hru' in temp_C.dims:
            Ra = Ra.broadcast_like(temp_C)

        # Oudin's formula: PET = Ra * (T + 5) / 100 when T + 5 > 0
        pet = xr.where(temp_C + 5.0 > 0.0, Ra * (temp_C + 5.0) / 100.0, 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Oudin et al. (2005)',
            'latitude': lat
        }

        # Log results
        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            pet_mean = float(pet.mean())
            self.logger.info(
                f"PET calculation complete: Mean={pet_mean:.3f} mm/day, "
                f"Min={float(pet.min()):.3f} mm/day, Max={float(pet.max()):.3f} mm/day"
            )

        # Warn if PET is suspiciously low
        if pet_mean < 0.1:
            with xr.set_options(use_numbagg=False, use_bottleneck=False):
                n_valid = int((temp_C + 5.0 > 0.0).sum())
            self.logger.warning(f"Very low PET! Days with T>-5°C: {n_valid}/{len(temp_C.time)}")

        return pet

    def calculate_pet_hamon(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hamon's method.

        Hamon's method uses temperature and daylight hours to estimate PET.

        Reference:
            Hamon (1961). "Estimating Potential Evapotranspiration"

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata

        Raises:
            ValueError: If temperature data has unexpected range
        """
        self.logger.info("Calculating PET using Hamon's method")

        # Load data if needed
        if hasattr(temp_data.data, 'compute'):
            temp_data = temp_data.load()

        # Check and convert temperature
        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            temp_mean = float(temp_data.mean())

        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}")

        # Auto-detect units
        if temp_mean > 100:  # Kelvin
            self.logger.info("Temperature in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Celsius
            self.logger.info("Temperature in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature has unexpected range: mean={temp_mean:.2f}")

        # Get values for computation
        temp_C_vals = temp_C.values

        # Verify reasonable range
        temp_mean_C = np.nanmean(temp_C_vals)
        self.logger.debug(f"Temperature (°C): Mean={temp_mean_C:.2f}, "
                         f"Min={np.nanmin(temp_C_vals):.2f}, Max={np.nanmax(temp_C_vals):.2f}")

        if temp_mean_C < -60 or temp_mean_C > 60:
            raise ValueError(f"Temperature unrealistic: {temp_mean_C:.2f}°C")

        # Day of year
        dates = pd.to_datetime(temp_data.time.values)
        doy = dates.dayofyear.values

        # Calculate daylight hours
        lat_rad = np.deg2rad(lat)
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)
        cos_arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        daylight_hours = 24.0 * sunset_angle / np.pi

        # Saturated vapor pressure (kPa)
        e_sat = 0.6108 * np.exp(17.27 * temp_C_vals / (temp_C_vals + 237.3))

        # Hamon PET (mm/day)
        # Handle multi-dimensional arrays (e.g., time x hru)
        if len(temp_C_vals.shape) > 1:
            daylight_hours = daylight_hours.reshape(-1, 1)

        pet_values = 0.1651 * daylight_hours * e_sat * 2.54
        pet_values = np.maximum(pet_values, 0.0)

        # Create DataArray with proper dimensions
        if 'hru' in temp_data.dims:
            pet = xr.DataArray(pet_values, coords=temp_data.coords, dims=temp_data.dims)
        else:
            pet = xr.DataArray(pet_values, coords={'time': temp_data.time}, dims=['time'])

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'method': 'Hamon (1961)',
            'latitude': lat
        }

        self.logger.info(f"PET calculation complete: Mean={np.nanmean(pet_values):.3f} mm/day, "
                        f"Max={np.nanmax(pet_values):.3f} mm/day")

        return pet

    def calculate_pet_hargreaves(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hargreaves method (simplified version).

        This is a simplified Hargreaves method that uses only mean temperature.
        The full method requires Tmin and Tmax; here we assume a typical
        diurnal temperature range of 10°C.

        Reference:
            Hargreaves & Samani (1985). "Reference Crop Evapotranspiration
            from Temperature"

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata

        Raises:
            ValueError: If temperature data has unexpected range

        Note:
            This simplified version assumes a diurnal temperature range of 10°C
            when min/max temperatures are not available.
        """
        self.logger.info("Calculating PET using Hargreaves method (simplified)")

        # Load data if needed
        if hasattr(temp_data.data, 'compute'):
            temp_data = temp_data.load()

        # Check and convert temperature
        temp_mean = float(temp_data.mean())
        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}")

        # Auto-detect units
        if temp_mean > 100:  # Kelvin
            self.logger.info("Temperature in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Celsius
            self.logger.info("Temperature in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature has unexpected range: mean={temp_mean:.2f}")

        # Verify reasonable range
        temp_mean_C = float(temp_C.mean())
        self.logger.debug(f"Temperature (°C): Mean={temp_mean_C:.2f}")

        if temp_mean_C < -60 or temp_mean_C > 60:
            raise ValueError(f"Temperature unrealistic: {temp_mean_C:.2f}°C")

        # Get time information
        time_values = pd.to_datetime(temp_data.time.values)
        doy = xr.DataArray(time_values.dayofyear, coords={'time': temp_data.time}, dims=['time'])

        # Calculate extraterrestrial radiation (Ra)
        lat_rad = np.deg2rad(lat)

        # Solar declination
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle
        cos_arg = -np.tan(lat_rad) * np.tan(solar_decl)
        cos_arg = cos_arg.clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

        self.logger.debug(f"Solar radiation Ra: Mean={float(Ra.mean()):.2f} MJ/m²/day")

        # Broadcast Ra if needed for multi-HRU
        if 'hru' in temp_C.dims:
            Ra = Ra.broadcast_like(temp_C)

        # Hargreaves formula (simplified without Tmin/Tmax)
        # PET = 0.0023 * Ra * (Tmean + 17.8) * TD^0.5
        # Without Tmin/Tmax, we use a typical diurnal range of 10°C
        # Convert Ra from MJ/m²/day to mm/day: multiply by 0.408

        TD = 10.0  # Assumed temperature range (°C) when min/max not available
        pet = 0.0023 * (Ra * 0.408) * (temp_C + 17.8) * np.sqrt(TD)

        # Ensure non-negative
        pet = xr.where(pet > 0, pet, 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Hargreaves (simplified)',
            'latitude': lat,
            'note': 'Simplified version using assumed diurnal temperature range of 10°C'
        }

        with xr.set_options(use_numbagg=False, use_bottleneck=False):
            pet_mean = float(pet.mean())
            self.logger.info(
                f"PET calculation complete: Mean={pet_mean:.3f} mm/day, "
                f"Min={float(pet.min()):.3f} mm/day, Max={float(pet.max()):.3f} mm/day"
            )

        return pet
