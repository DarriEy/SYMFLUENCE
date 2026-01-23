"""
Physical constants and unit conversion factors for SYMFLUENCE.

Centralizes all hardcoded constants to eliminate duplication and
improve maintainability across the codebase.
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import pandas as pd


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

    # Optimization/Calibration
    PENALTY_SCORE = -9999.0
    """
    Standard penalty score for failed model evaluations during optimization.

    Used when a model run fails, produces invalid output, or encounters errors.
    This value ensures failed runs are strongly penalized in optimization but
    still distinguishable from uninitialized values.
    """


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


class UnitConverter:
    """Centralized unit conversion utilities for hydrological data.

    Provides static methods for common unit conversions with optional
    auto-detection of input units based on data magnitude heuristics.

    This class consolidates scattered conversion logic from evaluators
    to ensure consistent unit handling across the evaluation suite.

    Thresholds and Constants:
        MASS_FLUX_THRESHOLD: 1e-6 m/s - values above indicate mass flux units
        SWE_UNIT_THRESHOLD: 250 - SWE values above indicate mm units
        SECONDS_PER_DAY: 86400 - conversion factor for daily rates
        INCHES_TO_MM: 25.4 - standard conversion factor
    """

    MASS_FLUX_THRESHOLD = 1e-6  # m/s threshold for detecting mass flux
    SWE_UNIT_THRESHOLD = 250    # values below likely in inches
    SECONDS_PER_DAY = 86400
    INCHES_TO_MM = 25.4

    @classmethod
    def et_mass_flux_to_mm_day(
        cls,
        data: 'pd.Series',
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert ET from kg m⁻² s⁻¹ to mm/day.

        SUMMA outputs evapotranspiration in mass flux units (kg m⁻² s⁻¹).
        This converts to the more common mm/day representation.

        Physical basis:
            - 1 kg m⁻² s⁻¹ = 1 mm/s (assuming water density = 1000 kg/m³)
            - 1 mm/s × 86400 s/day = 86400 mm/day

        Args:
            data: ET time series in kg m⁻² s⁻¹
            logger: Optional logger for debug messages

        Returns:
            ET time series in mm/day
        """
        if logger:
            logger.debug("Converting ET from kg m⁻² s⁻¹ to mm/day")
        return data * cls.SECONDS_PER_DAY

    @classmethod
    def swe_inches_to_mm(
        cls,
        data: 'pd.Series',
        auto_detect: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert SWE from inches to mm with optional auto-detection.

        SNOTEL and some other data sources report SWE in inches.
        This method can auto-detect based on data magnitude.

        Auto-detection logic:
            If max(data) < SWE_UNIT_THRESHOLD (250), assume inches
            Rationale: 250 inches = 6350 mm is an extremely high SWE value

        Args:
            data: SWE time series (potentially in inches)
            auto_detect: If True, detect units from data magnitude
            logger: Optional logger for debug messages

        Returns:
            SWE time series in mm
        """
        import numpy as np

        if auto_detect:
            max_val = np.nanmax(data)
            if max_val < cls.SWE_UNIT_THRESHOLD:
                if logger:
                    logger.debug(
                        f"SWE max={max_val:.1f} < {cls.SWE_UNIT_THRESHOLD}: "
                        f"assuming inches, converting to mm"
                    )
                return data * cls.INCHES_TO_MM
            else:
                if logger:
                    logger.debug(
                        f"SWE max={max_val:.1f} >= {cls.SWE_UNIT_THRESHOLD}: "
                        f"assuming already in mm"
                    )
                return data
        else:
            if logger:
                logger.debug("Converting SWE from inches to mm (forced)")
            return data * cls.INCHES_TO_MM

    @classmethod
    def detect_and_convert_mass_flux(
        cls,
        data: 'pd.Series',
        logger: Optional[logging.Logger] = None
    ) -> Tuple['pd.Series', bool]:
        """Detect mass flux and convert to volume flux if needed.

        SUMMA may output runoff in mass flux (kg m⁻² s⁻¹) incorrectly
        labeled as volume flux (m s⁻¹). This detects and corrects.

        Detection logic:
            If mean(data) > MASS_FLUX_THRESHOLD (1e-6 m/s), likely mass flux
            Rationale: 1e-6 m/s = 86.4 mm/day runoff (extremely high)

        Args:
            data: Runoff time series (possibly mislabeled units)
            logger: Optional logger for debug messages

        Returns:
            Tuple of (converted_data, was_converted)
        """
        import numpy as np

        mean_val = np.nanmean(data)
        if mean_val > cls.MASS_FLUX_THRESHOLD:
            if logger:
                logger.debug(
                    f"Mean runoff {mean_val:.2e} > {cls.MASS_FLUX_THRESHOLD:.0e} m/s: "
                    f"detected mass flux, dividing by 1000"
                )
            return data / 1000.0, True
        return data, False

    @classmethod
    def streamflow_mm_day_to_cms(
        cls,
        data: 'pd.Series',
        catchment_area_m2: float,
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert mm/day to m³/s given catchment area.

        Converts per-unit-area runoff depth to volumetric discharge.

        Formula:
            Q(m³/s) = depth(mm/day) × area(m²) / (1000 mm/m × 86400 s/day)

        Args:
            data: Runoff time series in mm/day
            catchment_area_m2: Catchment area in square meters
            logger: Optional logger for debug messages

        Returns:
            Discharge time series in m³/s
        """
        if logger:
            logger.debug(
                f"Converting mm/day to m³/s with area={catchment_area_m2:.0f} m²"
            )
        return data * catchment_area_m2 / (1000.0 * cls.SECONDS_PER_DAY)


class UnitDetectionThresholds:
    """
    Heuristic thresholds for automatic unit detection.

    These thresholds are used to infer input data units based on typical
    value ranges. While convenient, explicit unit specification is always
    preferred when available.

    Warning:
        These heuristics may fail for unusual data ranges. Always verify
        unit conversions are applied correctly for your specific data.
    """

    # Area unit detection
    AREA_KM2_VS_M2 = 1000.0
    """
    Threshold for detecting km² vs m² area units.

    If mean(area) < 1000, values are assumed to be in km² and converted to m².
    Rationale: A catchment of 1000 m² (0.001 km²) is unrealistically small,
    while 1000 km² is a reasonable catchment size.
    """

    # Temperature unit detection
    TEMP_KELVIN_VS_CELSIUS = 100.0
    """
    Threshold for detecting Kelvin vs Celsius temperature units.

    If mean(temp) > 100, values are assumed to be in Kelvin and converted to °C.
    Rationale: Mean temperatures on Earth never exceed 100°C, but 100K (-173°C)
    is well below any terrestrial temperature, so values > 100 must be Kelvin.
    """

    # Flux rate detection
    FLUX_RATE_MM_S_VS_MM_DAY = 0.01
    """
    Threshold for detecting mm/s vs mm/day flux rates (PET, evaporation).

    If mean(|flux|) < 0.01, values are assumed to be in mm/s (or similar small
    rate units) and converted to mm/day.
    Rationale: Typical daily PET is 1-10 mm/day. A mean of 0.01 mm/day is
    unrealistically low, suggesting the data is in mm/s (× 86400 for mm/day).
    """


class SupportedModels:
    """
    Registry of supported model names for dynamic imports.

    Centralizes the whitelist of valid model names to prevent arbitrary
    code execution via dynamic imports. All model names must be listed
    here before they can be imported dynamically.
    """

    # Core hydrological models
    ALL: Tuple[str, ...] = (
        'summa', 'fuse', 'hype', 'ngen', 'mesh', 'gr', 'rhessys',
        'lstm', 'gnn', 'mizuroute', 'hbv'
    )
    """All models supported by SYMFLUENCE."""

    # Models with forcing adapters
    WITH_FORCING_ADAPTER: Tuple[str, ...] = (
        'summa', 'fuse', 'hype', 'ngen', 'mesh', 'gr', 'rhessys'
    )
    """Models that have forcing adapter modules."""

    # Models with visualization plotters
    WITH_PLOTTERS: Tuple[str, ...] = (
        'summa', 'fuse', 'hype', 'ngen', 'lstm'
    )
    """Models with registered visualization plotters."""

    # Models with initialization presets
    WITH_PRESETS: Tuple[str, ...] = (
        'summa', 'fuse', 'hype', 'gr', 'ngen'
    )
    """Models that have initialization preset modules."""

    @classmethod
    def is_valid(cls, model_name: str) -> bool:
        """Check if a model name is in the supported whitelist."""
        return model_name.lower() in cls.ALL
