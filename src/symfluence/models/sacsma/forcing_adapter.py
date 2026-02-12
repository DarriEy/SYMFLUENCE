"""
SAC-SMA Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to SAC-SMA format.
"""

from typing import Dict, List, Callable
import xarray as xr

from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry


@ForcingAdapterRegistry.register_adapter('SACSMA')
class SacSmaForcingAdapter(ForcingAdapter):
    """Forcing adapter for SAC-SMA + Snow-17 model.

    SAC-SMA variable naming:
        - pr: Precipitation (mm/day)
        - temp: Temperature (°C)
        - pet: Potential evapotranspiration (mm/day)
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        return {
            'precipitation_flux': 'pr',
            'air_temperature': 'temp',
            'potential_evapotranspiration': 'pet',
        }

    def get_required_variables(self) -> List[str]:
        return [
            'precipitation_flux',
            'air_temperature',
        ]

    def get_optional_variables(self) -> List[str]:
        return [
            'potential_evapotranspiration',
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        return {
            'air_temperature': lambda x: x - 273.15,          # K to °C
            'precipitation_flux': lambda x: x * 86400,         # kg/m²/s to mm/day
            'potential_evapotranspiration': lambda x: x * 86400,
        }

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'SACSMA'
        ds.attrs['temporal_resolution'] = 'daily'
        return ds
