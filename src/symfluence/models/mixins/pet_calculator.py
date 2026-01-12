"""
Potential Evapotranspiration (PET) Calculator Mixin.

Provides three PET calculation methods for use in hydrological model preprocessors:
- Oudin's formula (simple, temperature-based)
- Hamon's method (temperature and daylight-based)
- Hargreaves method (simplified version)

All methods automatically detect and handle temperature units (Kelvin or Celsius).
"""

import numpy as np
import xarray as xr
from symfluence.core.constants import PhysicalConstants


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

    def _get_robust_temp_c(self, temp_data: xr.DataArray) -> xr.DataArray:
        """
        Helper to get temperature in Celsius with robust unit detection.
        
        Args:
            temp_data: Input temperature DataArray
            
        Returns:
            Temperature in Celsius
        """
        # Check metadata units attribute if available
        meta_units = str(temp_data.attrs.get('units', '')).lower()
        
        # Convert to Celsius if needed based on metadata
        if 'k' in meta_units and 'c' not in meta_units:
            self.logger.info(f"Metadata indicates temperature is in Kelvin ({meta_units}), converting to Celsius")
            temp_C = temp_data - PhysicalConstants.KELVIN_OFFSET
            temp_C.attrs['units'] = 'degC'
            return temp_C
        elif 'c' in meta_units:
            self.logger.info(f"Metadata indicates temperature is in Celsius ({meta_units}), using as-is")
            return temp_data

        # If metadata is missing or ambiguous, we need to check the data range.
        # We only compute the mean to minimize overhead if it's a dask array.
        self.logger.debug("Metadata units missing or ambiguous, checking data range for unit detection")
        temp_mean = float(temp_data.mean().compute()) if hasattr(temp_data.data, 'compute') else float(temp_data.mean())

        self.logger.debug(f"Input temperature mean: {temp_mean:.2f}")

        if temp_mean > 100:  # Likely Kelvin based on range
            self.logger.info("Temperature appears to be in Kelvin based on range, converting to Celsius")
            temp_C = temp_data - PhysicalConstants.KELVIN_OFFSET
        elif -60 < temp_mean < 60:  # Likely Celsius based on range
            self.logger.info("Temperature appears to be in Celsius based on range, using as-is")
            temp_C = temp_data
        elif temp_mean <= -100:
            # Case where data might be Celsius but was incorrectly treated as Kelvin before
            if -60 < (temp_mean + PhysicalConstants.KELVIN_OFFSET) < 60:
                self.logger.info("Temperature appears to have been double-converted. Reversing.")
                temp_C = temp_data + PhysicalConstants.KELVIN_OFFSET
            else:
                raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")
        else:
            raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")

        # Verification (only in debug)
        if self.logger.isEnabledFor(10):  # DEBUG level
            temp_mean_C = float(temp_C.mean().compute()) if hasattr(temp_C.data, 'compute') else float(temp_C.mean())
            self.logger.debug(f"Temperature in Celsius: Mean={temp_mean_C:.2f}°C")
            if temp_mean_C < -60 or temp_mean_C > 60:
                raise ValueError(f"Unrealistic temperature after conversion: {temp_mean_C:.2f}°C")
            
        return temp_C

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
        """
        self.logger.info("Calculating PET using Oudin's formula")

        temp_C = self._get_robust_temp_c(temp_data)

        # Get day of year using xarray accessor (dask-friendly)
        doy = temp_data.time.dt.dayofyear

        # Calculate solar radiation components
        lat_rad = np.deg2rad(lat)

        # Solar declination (radians)
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle with numerical stability
        cos_arg = (-np.tan(lat_rad) * np.tan(solar_decl)).clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

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

        # Log results (only summary)
        self.logger.debug("PET calculation complete (Oudin)")

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
        """
        self.logger.debug("Calculating PET using Hamon's method")

        temp_C = self._get_robust_temp_c(temp_data)

        # Day of year using xarray accessor
        doy = temp_data.time.dt.dayofyear

        # Calculate daylight hours
        lat_rad = np.deg2rad(lat)
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)
        cos_arg = (-np.tan(lat_rad) * np.tan(decl)).clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        daylight_hours = 24.0 * sunset_angle / np.pi

        # Saturated vapor pressure (kPa)
        # Use xarray-wrapped exp for dask support
        e_sat = 0.6108 * xr.apply_ufunc(np.exp, 17.27 * temp_C / (temp_C + 237.3), dask="parallelized")

        # Hamon PET (mm/day)
        pet = 0.1651 * daylight_hours * e_sat * 2.54
        pet = pet.where(pet > 0, 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'method': 'Hamon (1961)',
            'latitude': lat
        }

        self.logger.debug("PET calculation complete (Hamon)")

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
        """
        self.logger.info("Calculating PET using Hargreaves method (simplified)")

        temp_C = self._get_robust_temp_c(temp_data)

        # Get day of year
        doy = temp_data.time.dt.dayofyear

        # Calculate extraterrestrial radiation (Ra)
        lat_rad = np.deg2rad(lat)

        # Solar declination
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle
        cos_arg = (-np.tan(lat_rad) * np.tan(solar_decl)).clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

        # Hargreaves formula (simplified without Tmin/Tmax)
        # PET = 0.0023 * Ra * (Tmean + 17.8) * TD^0.5
        # TD = 10.0  # Assumed temperature range (°C) when min/max not available
        pet = 0.0023 * (Ra * 0.408) * (temp_C + 17.8) * np.sqrt(10.0)

        # Ensure non-negative
        pet = pet.where(pet > 0, 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Hargreaves (simplified)',
            'latitude': lat,
            'note': 'Simplified version using assumed diurnal temperature range of 10°C'
        }

        self.logger.debug("PET calculation complete (Hargreaves)")

        return pet