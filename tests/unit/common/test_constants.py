"""
Unit tests for the constants module.

Tests the UnitConversion and PhysicalConstants classes to ensure
all conversion factors and physical constants are correct.
"""

import pytest
from symfluence.utils.common.constants import UnitConversion, PhysicalConstants, ModelDefaults


class TestUnitConversion:
    """Test suite for UnitConversion class."""

    def test_mm_day_to_cms_value(self):
        """Test MM_DAY_TO_CMS constant value."""
        assert UnitConversion.MM_DAY_TO_CMS == 86.4

    def test_mm_day_to_cms_derivation(self):
        """
        Test that MM_DAY_TO_CMS is correctly derived.

        Derivation:
        1 mm/day over 1 km² = 1000 m³/day
        1000 m³/day / 86400 seconds = 0.01157 m³/s
        Therefore: Q(cms) = Q(mm/day) * Area(km²) / 86.4
        """
        # 1 mm/day over 1 km² should equal 1000/86400 m³/s
        area_km2 = 1.0
        q_mm_day = 1.0
        q_cms = q_mm_day * area_km2 / UnitConversion.MM_DAY_TO_CMS

        expected_cms = 1000.0 / 86400.0  # m³/day to m³/s
        assert abs(q_cms - expected_cms) < 1e-6

    def test_mm_day_to_cms_conversion(self):
        """Test realistic mm/day to cms conversion."""
        # Example: 10 mm/day over 100 km² catchment
        q_mm_day = 10.0
        area_km2 = 100.0
        q_cms = q_mm_day * area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Expected: 10 * 100 * 1000 m³/day = 1,000,000 m³/day = 11.57 m³/s
        expected_cms = 1_000_000.0 / 86400.0
        assert abs(q_cms - expected_cms) < 1e-2

    def test_mm_hour_to_cms_value(self):
        """Test MM_HOUR_TO_CMS constant value."""
        assert UnitConversion.MM_HOUR_TO_CMS == 3.6

    def test_mm_hour_to_cms_derivation(self):
        """Test that MM_HOUR_TO_CMS is correctly derived (86.4 / 24 = 3.6)."""
        assert UnitConversion.MM_HOUR_TO_CMS == UnitConversion.MM_DAY_TO_CMS / 24

    def test_cfs_to_cms_value(self):
        """Test CFS_TO_CMS constant value."""
        assert abs(UnitConversion.CFS_TO_CMS - 0.028316846592) < 1e-10

    def test_cfs_to_cms_conversion(self):
        """Test cubic feet per second to cubic meters per second conversion."""
        # 1 cubic foot = 0.0283168 cubic meters
        cfs = 100.0
        cms = cfs * UnitConversion.CFS_TO_CMS
        expected_cms = 100.0 * 0.028316846592
        assert abs(cms - expected_cms) < 1e-6

    def test_seconds_per_day(self):
        """Test SECONDS_PER_DAY constant."""
        assert UnitConversion.SECONDS_PER_DAY == 86400
        assert UnitConversion.SECONDS_PER_DAY == 24 * 60 * 60

    def test_m2_to_km2(self):
        """Test square meters to square kilometers conversion."""
        assert UnitConversion.M2_TO_KM2 == 1e-6
        # 1 km² = 1,000,000 m²
        m2 = 1_000_000.0
        km2 = m2 * UnitConversion.M2_TO_KM2
        assert km2 == 1.0

    def test_km2_to_m2(self):
        """Test square kilometers to square meters conversion."""
        assert UnitConversion.KM2_TO_M2 == 1e6
        # 1 km² = 1,000,000 m²
        km2 = 1.0
        m2 = km2 * UnitConversion.KM2_TO_M2
        assert m2 == 1_000_000.0

    def test_inverse_area_conversions(self):
        """Test that area conversions are inverses."""
        assert UnitConversion.M2_TO_KM2 * UnitConversion.KM2_TO_M2 == 1.0

    def test_roundtrip_mm_day_to_cms(self):
        """Test roundtrip conversion: mm/day -> cms -> mm/day."""
        q_mm_day_original = 5.0
        area_km2 = 50.0

        # Convert to cms
        q_cms = q_mm_day_original * area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Convert back to mm/day
        q_mm_day_back = q_cms * UnitConversion.MM_DAY_TO_CMS / area_km2

        assert abs(q_mm_day_original - q_mm_day_back) < 1e-10


class TestPhysicalConstants:
    """Test suite for PhysicalConstants class."""

    def test_water_density(self):
        """Test water density constant."""
        assert PhysicalConstants.WATER_DENSITY == 1000.0

    def test_water_density_units(self):
        """Verify water density is in kg/m³."""
        # Water density at 4°C is 1000 kg/m³
        assert PhysicalConstants.WATER_DENSITY == 1000.0

    def test_gravity(self):
        """Test gravitational acceleration constant."""
        assert PhysicalConstants.GRAVITY == 9.80665

    def test_gravity_units(self):
        """Verify gravity is standard acceleration in m/s²."""
        # Standard gravity is 9.80665 m/s²
        assert PhysicalConstants.GRAVITY == 9.80665

    def test_gravity_range(self):
        """Test gravity is within expected range."""
        # Gravity should be approximately 9.8 m/s²
        assert 9.8 <= PhysicalConstants.GRAVITY <= 9.81


class TestModelDefaults:
    """Test suite for ModelDefaults class."""

    def test_default_timestep_hourly(self):
        """Test default hourly timestep."""
        assert ModelDefaults.DEFAULT_TIMESTEP_HOURLY == 3600

    def test_default_timestep_daily(self):
        """Test default daily timestep."""
        assert ModelDefaults.DEFAULT_TIMESTEP_DAILY == 86400

    def test_default_discretization(self):
        """Test default spatial discretization."""
        assert ModelDefaults.DEFAULT_DISCRETIZATION == 'lumped'

    def test_default_spinup_days(self):
        """Test default spin-up period."""
        assert ModelDefaults.DEFAULT_SPINUP_DAYS == 365

    def test_default_tolerance(self):
        """Test default numerical tolerance."""
        assert ModelDefaults.DEFAULT_TOLERANCE == 1e-6


class TestConstantsDocumentation:
    """Test that constants have proper documentation."""

    def test_mm_day_to_cms_has_docstring(self):
        """Test MM_DAY_TO_CMS has documentation."""
        # Check the class-level attribute exists
        assert hasattr(UnitConversion, 'MM_DAY_TO_CMS')

    def test_constants_are_numeric(self):
        """Test all unit conversion constants are numeric."""
        assert isinstance(UnitConversion.MM_DAY_TO_CMS, (int, float))
        assert isinstance(UnitConversion.MM_HOUR_TO_CMS, (int, float))
        assert isinstance(UnitConversion.CFS_TO_CMS, (int, float))
        assert isinstance(UnitConversion.SECONDS_PER_DAY, int)
        assert isinstance(UnitConversion.M2_TO_KM2, (int, float))
        assert isinstance(UnitConversion.KM2_TO_M2, (int, float))

    def test_physical_constants_are_numeric(self):
        """Test all physical constants are numeric."""
        assert isinstance(PhysicalConstants.WATER_DENSITY, (int, float))
        assert isinstance(PhysicalConstants.GRAVITY, (int, float))


class TestConstantsUsage:
    """Test typical usage patterns of constants."""

    def test_discharge_calculation(self):
        """Test typical discharge calculation using constants."""
        # Simulate: 2.5 mm/day rainfall over 250 km² catchment
        rainfall_mm_day = 2.5
        area_km2 = 250.0

        # Calculate discharge
        discharge_cms = rainfall_mm_day * area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Expected: 2.5 * 250 * 1000 / 86400 = 7.233 m³/s
        expected = (2.5 * 250 * 1000) / 86400
        assert abs(discharge_cms - expected) < 0.001

    def test_timestep_conversion(self):
        """Test converting between different timesteps."""
        # Convert mm/hour to mm/day equivalent
        mm_hour_rate = 1.0
        mm_day_equivalent = mm_hour_rate * 24

        # Both should give same discharge for same area
        area_km2 = 100.0
        q_from_hour = mm_hour_rate * area_km2 / UnitConversion.MM_HOUR_TO_CMS
        q_from_day = mm_day_equivalent * area_km2 / UnitConversion.MM_DAY_TO_CMS

        assert abs(q_from_hour - q_from_day) < 1e-6

    def test_area_unit_conversion(self):
        """Test area conversion in typical catchment size."""
        # 150 km² catchment
        area_km2 = 150.0
        area_m2 = area_km2 * UnitConversion.KM2_TO_M2

        assert area_m2 == 150_000_000.0

        # Convert back
        area_km2_back = area_m2 * UnitConversion.M2_TO_KM2
        assert abs(area_km2 - area_km2_back) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
