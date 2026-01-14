"""
Unit conversion utilities for CFIF.

This module provides functions for converting between different unit systems
commonly used in meteorological and hydrological datasets.

Standard CFIF units (SI-based):
    - Temperature: K (Kelvin)
    - Precipitation: kg m-2 s-1 (mass flux)
    - Pressure: Pa (Pascals)
    - Radiation: W m-2
    - Specific humidity: kg kg-1
    - Wind speed: m s-1
"""

from typing import Dict, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import xarray as xr

# Type alias for array-like data
ArrayLike = Union[np.ndarray, float, 'xr.DataArray']


class UnitConverter:
    """
    Central unit conversion utility.

    Provides methods for converting common meteorological variables
    to CFIF standard units.

    Example:
        >>> converter = UnitConverter()
        >>> temp_k = converter.convert('temperature', temp_c, 'degC', 'K')
        >>> precip_flux = converter.convert('precipitation', precip_mm_hr, 'mm/hr', 'kg m-2 s-1')
    """

    # Temperature conversions
    TEMPERATURE_CONVERSIONS = {
        ('degC', 'K'): lambda x: x + 273.15,
        ('K', 'degC'): lambda x: x - 273.15,
        ('degF', 'K'): lambda x: (x - 32) * 5/9 + 273.15,
        ('K', 'degF'): lambda x: (x - 273.15) * 9/5 + 32,
        ('degC', 'degF'): lambda x: x * 9/5 + 32,
        ('degF', 'degC'): lambda x: (x - 32) * 5/9,
    }

    # Precipitation conversions (to kg m-2 s-1)
    # Note: mm = kg m-2 for water (density = 1000 kg/m3)
    PRECIPITATION_CONVERSIONS = {
        ('mm/s', 'kg m-2 s-1'): lambda x: x,  # 1 mm/s = 1 kg m-2 s-1
        ('mm/hr', 'kg m-2 s-1'): lambda x: x / 3600,
        ('mm/day', 'kg m-2 s-1'): lambda x: x / 86400,
        ('m/s', 'kg m-2 s-1'): lambda x: x * 1000,
        ('kg m-2 s-1', 'mm/hr'): lambda x: x * 3600,
        ('kg m-2 s-1', 'mm/day'): lambda x: x * 86400,
    }

    # Pressure conversions (to Pa)
    PRESSURE_CONVERSIONS = {
        ('hPa', 'Pa'): lambda x: x * 100,
        ('Pa', 'hPa'): lambda x: x / 100,
        ('kPa', 'Pa'): lambda x: x * 1000,
        ('Pa', 'kPa'): lambda x: x / 1000,
        ('mbar', 'Pa'): lambda x: x * 100,  # mbar = hPa
        ('Pa', 'mbar'): lambda x: x / 100,
    }

    # Wind speed conversions (to m s-1)
    WIND_CONVERSIONS = {
        ('km/hr', 'm s-1'): lambda x: x / 3.6,
        ('m s-1', 'km/hr'): lambda x: x * 3.6,
        ('mph', 'm s-1'): lambda x: x * 0.44704,
        ('m s-1', 'mph'): lambda x: x / 0.44704,
        ('knots', 'm s-1'): lambda x: x * 0.514444,
        ('m s-1', 'knots'): lambda x: x / 0.514444,
    }

    # Humidity conversions
    HUMIDITY_CONVERSIONS = {
        ('g/kg', 'kg kg-1'): lambda x: x / 1000,
        ('kg kg-1', 'g/kg'): lambda x: x * 1000,
        ('fraction', '%'): lambda x: x * 100,
        ('%', 'fraction'): lambda x: x / 100,
    }

    def __init__(self):
        """Initialize the converter with all conversion tables."""
        self._converters: Dict[str, Dict] = {
            'temperature': self.TEMPERATURE_CONVERSIONS,
            'precipitation': self.PRECIPITATION_CONVERSIONS,
            'pressure': self.PRESSURE_CONVERSIONS,
            'wind': self.WIND_CONVERSIONS,
            'humidity': self.HUMIDITY_CONVERSIONS,
        }

    def convert(
        self,
        var_type: str,
        data: ArrayLike,
        from_units: str,
        to_units: str
    ) -> ArrayLike:
        """
        Convert data between units.

        Args:
            var_type: Variable type ('temperature', 'precipitation', etc.)
            data: Input data array
            from_units: Source units
            to_units: Target units

        Returns:
            Converted data array

        Raises:
            ValueError: If conversion not supported
        """
        if from_units == to_units:
            return data

        conversions = self._converters.get(var_type)
        if conversions is None:
            raise ValueError(f"Unknown variable type: {var_type}")

        converter = conversions.get((from_units, to_units))
        if converter is None:
            raise ValueError(
                f"No conversion from '{from_units}' to '{to_units}' "
                f"for variable type '{var_type}'"
            )

        return converter(data)

    def get_supported_units(self, var_type: str) -> set:
        """Get all supported units for a variable type."""
        conversions = self._converters.get(var_type, {})
        units = set()
        for from_u, to_u in conversions.keys():
            units.add(from_u)
            units.add(to_u)
        return units


# Convenience functions for common conversions

def convert_temperature(
    data: ArrayLike,
    from_units: str = 'degC',
    to_units: str = 'K'
) -> ArrayLike:
    """
    Convert temperature data.

    Args:
        data: Temperature values
        from_units: Source units ('degC', 'K', 'degF')
        to_units: Target units ('degC', 'K', 'degF')

    Returns:
        Converted temperature values
    """
    converter = UnitConverter()
    return converter.convert('temperature', data, from_units, to_units)


def convert_precipitation(
    data: ArrayLike,
    from_units: str = 'mm/hr',
    to_units: str = 'kg m-2 s-1'
) -> ArrayLike:
    """
    Convert precipitation data.

    Args:
        data: Precipitation values
        from_units: Source units ('mm/s', 'mm/hr', 'mm/day', 'm/s')
        to_units: Target units ('kg m-2 s-1', 'mm/hr', 'mm/day')

    Returns:
        Converted precipitation values
    """
    converter = UnitConverter()
    return converter.convert('precipitation', data, from_units, to_units)


def convert_pressure(
    data: ArrayLike,
    from_units: str = 'hPa',
    to_units: str = 'Pa'
) -> ArrayLike:
    """
    Convert pressure data.

    Args:
        data: Pressure values
        from_units: Source units ('hPa', 'kPa', 'mbar', 'Pa')
        to_units: Target units ('Pa', 'hPa', 'kPa', 'mbar')

    Returns:
        Converted pressure values
    """
    converter = UnitConverter()
    return converter.convert('pressure', data, from_units, to_units)


def convert_radiation(data: ArrayLike, from_units: str, to_units: str) -> ArrayLike:
    """
    Convert radiation data.

    Currently a passthrough since standard radiation units (W m-2) are
    used across most datasets. Included for API consistency.

    Args:
        data: Radiation values
        from_units: Source units (typically 'W m-2')
        to_units: Target units (typically 'W m-2')

    Returns:
        Radiation values (unchanged if units match)
    """
    if from_units == to_units:
        return data
    # Add conversions here if needed (e.g., MJ m-2 day-1 to W m-2)
    raise ValueError(f"Radiation conversion from '{from_units}' to '{to_units}' not supported")


def convert_humidity(
    data: ArrayLike,
    from_units: str = 'g/kg',
    to_units: str = 'kg kg-1'
) -> ArrayLike:
    """
    Convert humidity data.

    Args:
        data: Humidity values
        from_units: Source units ('g/kg', 'kg kg-1', '%', 'fraction')
        to_units: Target units ('g/kg', 'kg kg-1', '%', 'fraction')

    Returns:
        Converted humidity values
    """
    converter = UnitConverter()
    return converter.convert('humidity', data, from_units, to_units)


def convert_wind(
    data: ArrayLike,
    from_units: str = 'km/hr',
    to_units: str = 'm s-1'
) -> ArrayLike:
    """
    Convert wind speed data.

    Args:
        data: Wind speed values
        from_units: Source units ('km/hr', 'm s-1', 'mph', 'knots')
        to_units: Target units ('km/hr', 'm s-1', 'mph', 'knots')

    Returns:
        Converted wind speed values
    """
    converter = UnitConverter()
    return converter.convert('wind', data, from_units, to_units)
