# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to GR model format.
GR models (GR4J, GR5J, etc.) use simple daily forcing variables.
"""

from typing import Callable, Dict, List

import xarray as xr

from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry


@ForcingAdapterRegistry.register_adapter('GR')
class GRForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for GR family models (GR4J, GR5J, etc.).

    GR variable naming conventions:
        - P: Precipitation (mm/day)
        - E: Potential evapotranspiration (mm/day)
        - T: Temperature (°C) - used with CemaNeige snow module

    Note:
        GR models are lumped daily models that require simple
        forcing variables: precipitation and evapotranspiration.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to GR names.

        Returns:
            Dict mapping CFIF names to GR variable names
        """
        return {
            'precipitation_flux': 'P',
            'potential_evapotranspiration': 'E',
            'air_temperature': 'T',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by GR.

        Returns:
            List of required CFIF variable names
        """
        return [
            'precipitation_flux',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for GR.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'potential_evapotranspiration',
            'air_temperature',  # For CemaNeige snow module
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversions for GR.

        GR uses:
            - Precipitation in mm/day
            - Evapotranspiration in mm/day
            - Temperature in °C

        Returns:
            Dict of conversion functions
        """
        # GR uses daily data, so convert flux to mm/day
        # kg m-2 s-1 * 86400 s/day = mm/day
        seconds_per_day = 86400

        return {
            'air_temperature': lambda x: x - 273.15,  # K to °C
            'precipitation_flux': lambda x: x * seconds_per_day,
            'potential_evapotranspiration': lambda x: x * seconds_per_day,
        }

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add GR-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'GR'
        ds.attrs['temporal_resolution'] = 'daily'
        return ds
