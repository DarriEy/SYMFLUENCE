"""
Physical constants and unit conversion factors for SYMFLUENCE.

Centralizes all hardcoded constants to eliminate duplication and
improve maintainability across the codebase.
"""

from typing import Dict


class UnitConversion:
    """
    Unit conversion factors for hydrological calculations.

    All factors are scientifically derived and documented to provide
    a single source of truth for unit conversions throughout SYMFLUENCE.
    """

    # Time constants
    SECONDS_PER_HOUR = 3600
    """Seconds in one hour."""

    SECONDS_PER_DAY = 86400
    """Seconds in one day (24 hours × 3600 seconds)."""

    HOURS_PER_DAY = 24
    """Hours in one day."""

    DAYS_PER_YEAR = 365.25
    """Average days per year accounting for leap years."""

    # Streamflow conversions
    MM_DAY_TO_CMS = SECONDS_PER_DAY / 1000.0
    """
    Convert mm/day to m³/s (cms) per km² of catchment area.

    Formula: Q(cms) = Q(mm/day) * Area(km²) / MM_DAY_TO_CMS

    Derivation:
        1 mm/day over 1 km² =
        (0.001 m) × (1,000,000 m²) / (86,400 s) =
        1000 m³ / 86,400 s =
        0.01157 m³/s

    Therefore: 86.4 = 86,400 / 1000
    """

    MM_HOUR_TO_CMS = SECONDS_PER_HOUR / 1000.0
    """
    Convert mm/hour to m³/s per km² of catchment area.

    Formula: Q(cms) = Q(mm/hour) * Area(km²) / MM_HOUR_TO_CMS

    Derivation:
        1 mm/hour over 1 km² =
        (0.001 m) × (1,000,000 m²) / (3,600 s) =
        1000 m³ / 3,600 s =
        0.278 m³/s

    Therefore: 3.6 = 3,600 / 1000
    """

    CFS_TO_CMS = 0.028316846592
    """
    Convert cubic feet per second to cubic meters per second.

    1 cubic foot = 0.028316846592 cubic meters (exact)
    """

    # Area conversions
    M2_TO_KM2 = 1e-6
    """Convert square meters to square kilometers (m² / 1,000,000)."""

    KM2_TO_M2 = 1e6
    """Convert square kilometers to square meters (km² × 1,000,000)."""

    HA_TO_KM2 = 0.01
    """Convert hectares to square kilometers (1 ha = 0.01 km²)."""

    # Pressure conversions
    PA_TO_KPA = 0.001
    """Convert Pascals to kiloPascals."""

    # Length conversions
    FEET_TO_METERS = 0.3048
    """
    Convert feet to meters.

    1 foot = 0.3048 meters (exact, international foot)
    """

    METERS_TO_FEET = 3.28084
    """
    Convert meters to feet.

    1 meter = 3.28084 feet (approximate)
    """

    @classmethod
    def mm_per_timestep_to_cms_factor(
        cls,
        timestep_seconds: int
    ) -> float:
        """
        Get conversion factor for mm/timestep to cms.

        This is the divisor to convert mm per timestep to m³/s per km².

        Args:
            timestep_seconds: Model timestep in seconds

        Returns:
            Conversion factor (timestep_seconds / 1000)
        """
        return timestep_seconds / 1000.0


class PhysicalConstants:
    """
    Physical constants for hydrological and meteorological calculations.

    Values are from standard reference sources and widely accepted
    approximations used in hydrological modeling.
    """

    # Temperature conversions
    KELVIN_OFFSET = 273.15
    """Offset to convert Celsius to Kelvin (T_K = T_C + 273.15)."""

    # Water properties
    WATER_DENSITY = 1000.0
    """Water density in kg/m³ at 4°C (maximum density)."""

    LATENT_HEAT_VAPORIZATION = 2.45e6
    """
    Latent heat of vaporization of water in J/kg at 20°C.

    Used in evapotranspiration calculations. Value varies with temperature:
    - 0°C: 2.501 × 10⁶ J/kg
    - 20°C: 2.453 × 10⁶ J/kg
    - 100°C: 2.257 × 10⁶ J/kg
    """

    SPECIFIC_HEAT_WATER = 4186.0
    """Specific heat capacity of water in J/(kg·K) at 15°C."""

    # Atmospheric constants
    STEFAN_BOLTZMANN = 5.67e-8
    """Stefan-Boltzmann constant in W/(m²·K⁴) for radiation calculations."""

    GAS_CONSTANT_DRY_AIR = 287.05
    """Specific gas constant for dry air in J/(kg·K)."""

    # Earth constants
    GRAVITY = 9.80665
    """
    Standard acceleration due to gravity in m/s².

    This is the internationally adopted standard value. Actual gravity
    varies slightly with latitude and elevation.
    """

    EARTH_RADIUS_KM = 6371.0
    """Mean radius of Earth in kilometers."""


class ModelDefaults:
    """Default configuration values used across models."""

    # Timesteps
    DEFAULT_TIMESTEP_HOURLY = 3600
    """Default hourly timestep in seconds."""

    DEFAULT_TIMESTEP_DAILY = 86400
    """Default daily timestep in seconds."""

    # Spatial configuration
    DEFAULT_DISCRETIZATION = 'lumped'
    """Default spatial discretization method."""

    # Temporal configuration
    DEFAULT_SPINUP_DAYS = 365
    """Default spin-up period in days for model initialization."""

    # Numerical precision
    DEFAULT_TOLERANCE = 1e-6
    """Default numerical tolerance for convergence checks."""


# Export convenience dictionary for backward compatibility
UNIT_CONVERSIONS: Dict[str, float] = {
    'mm_day_to_cms': UnitConversion.MM_DAY_TO_CMS,
    'mm_hour_to_cms': UnitConversion.MM_HOUR_TO_CMS,
    'cfs_to_cms': UnitConversion.CFS_TO_CMS,
    'm2_to_km2': UnitConversion.M2_TO_KM2,
    'km2_to_m2': UnitConversion.KM2_TO_M2,
    'seconds_per_day': UnitConversion.SECONDS_PER_DAY,
}
"""
Convenience dictionary for accessing unit conversion factors.

Provided for backward compatibility and convenience. Prefer using
UnitConversion class directly for better IDE support and documentation.
"""