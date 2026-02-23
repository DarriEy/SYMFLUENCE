"""
SUMMA Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to SUMMA format.
Since CFIF was designed with SUMMA-compatible conventions, this adapter
primarily handles variable renaming.
"""

from typing import Dict, List

import xarray as xr

from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry


@ForcingAdapterRegistry.register_adapter('SUMMA')
class SUMMAForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for SUMMA model.

    SUMMA variable naming conventions:
        - airtemp: Air temperature (K)
        - pptrate: Precipitation rate (kg m-2 s-1)
        - SWRadAtm: Downward shortwave radiation (W m-2)
        - LWRadAtm: Downward longwave radiation (W m-2)
        - spechum: Specific humidity (kg kg-1)
        - windspd: Wind speed (m s-1)
        - airpres: Surface air pressure (Pa)

    Note:
        CFIF uses the same standard units as SUMMA, so no unit
        conversions are needed.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to SUMMA names.

        Returns:
            Dict mapping CFIF names to SUMMA variable names
        """
        return {
            'air_temperature': 'airtemp',
            'precipitation_flux': 'pptrate',
            'surface_downwelling_shortwave_flux': 'SWRadAtm',
            'surface_downwelling_longwave_flux': 'LWRadAtm',
            'specific_humidity': 'spechum',
            'relative_humidity': 'relhum',
            'wind_speed': 'windspd',
            'eastward_wind': 'windspd_u',
            'northward_wind': 'windspd_v',
            'surface_air_pressure': 'airpres',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by SUMMA.

        Returns:
            List of required CFIF variable names
        """
        return [
            'air_temperature',
            'precipitation_flux',
            'surface_downwelling_shortwave_flux',
            'surface_downwelling_longwave_flux',
            'specific_humidity',
            'wind_speed',
            'surface_air_pressure',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for SUMMA.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'relative_humidity',
            'eastward_wind',
            'northward_wind',
        ]

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add SUMMA-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['Conventions'] = 'CF-1.6'
        ds.attrs['model_format'] = 'SUMMA'
        return ds
