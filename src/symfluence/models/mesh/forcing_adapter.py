"""
MESH Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to MESH format.
MESH uses CLASS (Canadian Land Surface Scheme) which requires specific
meteorological variables.
"""

from typing import Dict, List
from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry


@ForcingAdapterRegistry.register_adapter('MESH')
class MESHForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for MESH model.

    MESH variable naming conventions (CLASS-based):
    - FSIN: Incoming shortwave radiation (W m-2)
    - FLIN: Incoming longwave radiation (W m-2)
    - TA: Air temperature (K)
    - QA: Specific humidity (kg kg-1)
    - UV: Wind speed (m s-1)
    - PRES: Surface air pressure (Pa)
    - PRE: Precipitation rate (kg m-2 s-1)

    Note:
        MESH uses the same standard units as CFIF, so minimal unit
        conversions are needed.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to MESH names.

        Returns:
            Dict mapping CFIF names to MESH variable names
        """
        return {
            'surface_downwelling_shortwave_flux': 'FSIN',
            'surface_downwelling_longwave_flux': 'FLIN',
            'air_temperature': 'TA',
            'specific_humidity': 'QA',
            'wind_speed': 'UV',
            'surface_air_pressure': 'PRES',
            'precipitation_flux': 'PRE',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by MESH.

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
        Get optional variables for MESH.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'relative_humidity',  # Can be derived from specific humidity
        ]

    def add_metadata(self, ds):
        """Add MESH-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'MESH'
        ds.attrs['land_surface_scheme'] = 'CLASS'
        ds.attrs['Conventions'] = 'CF-1.6'
        return ds
