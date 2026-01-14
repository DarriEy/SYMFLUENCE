"""
RHESSys Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to RHESSys format.
RHESSys (Regional Hydro-Ecologic Simulation System) requires meteorological
forcing for ecohydrological simulations.
"""

from typing import Dict, List, Callable
from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry


@ForcingAdapterRegistry.register_adapter('RHESSYS')
class RHESSysForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for RHESSys model.

    RHESSys variable naming conventions:
    - tmax: Maximum daily air temperature (°C)
    - tmin: Minimum daily air temperature (°C)
    - rain: Precipitation (mm/day)
    - vpd: Vapor pressure deficit (Pa)
    - dayl: Day length (s)
    - srad: Solar radiation (W m-2)
    - tavg: Average air temperature (°C) - optional

    Note:
        RHESSys uses daily aggregated data, often requiring temporal
        aggregation from sub-daily CFIF data. Some unit conversions
        are needed (K to °C, flux to mm/day).
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to RHESSys names.

        Returns:
            Dict mapping CFIF names to RHESSys variable names
        """
        return {
            'air_temperature': 'tavg',  # Will need min/max derivation
            'air_temperature_max': 'tmax',
            'air_temperature_min': 'tmin',
            'precipitation_flux': 'rain',
            'surface_downwelling_shortwave_flux': 'srad',
            'vapor_pressure_deficit': 'vpd',
            'day_length': 'dayl',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by RHESSys.

        Returns:
            List of required CFIF variable names
        """
        return [
            'air_temperature',  # Or air_temperature_max and air_temperature_min
            'precipitation_flux',
            'surface_downwelling_shortwave_flux',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for RHESSys.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'air_temperature_max',
            'air_temperature_min',
            'vapor_pressure_deficit',
            'day_length',
            'relative_humidity',  # For VPD calculation
            'specific_humidity',  # For VPD calculation
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversion functions for RHESSys.

        RHESSys uses different units than CFIF:
        - Temperature: °C instead of K
        - Precipitation: mm/day instead of kg m-2 s-1

        Returns:
            Dict mapping CFIF variable names to conversion functions
        """
        return {
            'air_temperature': lambda x: x - 273.15,  # K to °C
            'air_temperature_max': lambda x: x - 273.15,  # K to °C
            'air_temperature_min': lambda x: x - 273.15,  # K to °C
            'precipitation_flux': lambda x: x * 86400,  # kg m-2 s-1 to mm/day
        }

    def add_metadata(self, ds):
        """Add RHESSys-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'RHESSys'
        ds.attrs['temporal_resolution'] = 'daily'
        ds.attrs['note'] = 'RHESSys typically uses daily aggregated meteorological data'
        return ds
