"""
Tests for constants module, including UnitConverter.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from symfluence.core.constants import PhysicalConstants, UnitConversion, UnitConverter


class TestUnitConverter:
    """Test UnitConverter class methods."""

    def test_et_mass_flux_to_mm_day(self):
        """Test ET conversion from kg m⁻² s⁻¹ to mm/day."""
        # Input: 1e-5 kg m⁻² s⁻¹ (typical ET rate)
        data = pd.Series([1e-5, 2e-5, 0.0])
        result = UnitConverter.et_mass_flux_to_mm_day(data)

        # Expected: multiply by 86400
        expected = pd.Series([0.864, 1.728, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_et_mass_flux_with_logger(self):
        """Test ET conversion logs debug message when logger provided."""
        logger = MagicMock()
        data = pd.Series([1e-5])
        UnitConverter.et_mass_flux_to_mm_day(data, logger=logger)

        logger.debug.assert_called()

    def test_swe_inches_to_mm_auto_detect_inches(self):
        """Test SWE conversion auto-detects inches when max < 250."""
        # SNOTEL-like data in inches (max 50 inches)
        data = pd.Series([10.0, 30.0, 50.0])
        result = UnitConverter.swe_inches_to_mm(data, auto_detect=True)

        # Should convert: 10 * 25.4 = 254, 30 * 25.4 = 762, 50 * 25.4 = 1270
        expected = pd.Series([254.0, 762.0, 1270.0])
        pd.testing.assert_series_equal(result, expected)

    def test_swe_inches_to_mm_auto_detect_already_mm(self):
        """Test SWE conversion skips when data already in mm."""
        # Data already in mm (max 500 mm > threshold 250)
        data = pd.Series([100.0, 300.0, 500.0])
        result = UnitConverter.swe_inches_to_mm(data, auto_detect=True)

        # Should not convert
        pd.testing.assert_series_equal(result, data)

    def test_swe_inches_to_mm_forced_conversion(self):
        """Test SWE conversion when auto_detect=False."""
        # Force conversion even if values are high
        data = pd.Series([500.0])
        result = UnitConverter.swe_inches_to_mm(data, auto_detect=False)

        # Should convert: 500 * 25.4 = 12700
        expected = pd.Series([12700.0])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_and_convert_mass_flux_detects_mass_flux(self):
        """Test mass flux detection when mean exceeds threshold."""
        # Mean of 1e-5 > threshold of 1e-6
        data = pd.Series([1e-5, 2e-5, 1.5e-5])
        result, was_converted = UnitConverter.detect_and_convert_mass_flux(data)

        assert was_converted is True
        # Should divide by 1000
        expected = pd.Series([1e-8, 2e-8, 1.5e-8])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_and_convert_mass_flux_no_conversion_needed(self):
        """Test mass flux detection when mean is below threshold."""
        # Mean of 1e-8 < threshold of 1e-6
        data = pd.Series([1e-8, 2e-8, 1.5e-8])
        result, was_converted = UnitConverter.detect_and_convert_mass_flux(data)

        assert was_converted is False
        pd.testing.assert_series_equal(result, data)

    def test_streamflow_mm_day_to_cms(self):
        """Test streamflow conversion from mm/day to m³/s."""
        data = pd.Series([1.0, 2.0, 3.0])  # mm/day
        catchment_area = 1e6  # 1 km² in m²

        result = UnitConverter.streamflow_mm_day_to_cms(data, catchment_area)

        # Formula: mm/day * m² / (1000 * 86400)
        # 1 mm/day * 1e6 m² / (1000 * 86400) = 1e6 / 8.64e7 = 0.01157 m³/s
        expected = data * 1e6 / (1000.0 * 86400)
        pd.testing.assert_series_equal(result, expected)

    def test_streamflow_conversion_with_logger(self):
        """Test streamflow conversion logs debug message."""
        logger = MagicMock()
        data = pd.Series([1.0])
        UnitConverter.streamflow_mm_day_to_cms(data, 1e6, logger=logger)

        logger.debug.assert_called()


class TestUnitConverterConstants:
    """Test UnitConverter class constants."""

    def test_mass_flux_threshold(self):
        """Test mass flux threshold value."""
        assert UnitConverter.MASS_FLUX_THRESHOLD == 1e-6

    def test_swe_unit_threshold(self):
        """Test SWE unit threshold value."""
        assert UnitConverter.SWE_UNIT_THRESHOLD == 250

    def test_seconds_per_day(self):
        """Test seconds per day constant."""
        assert UnitConverter.SECONDS_PER_DAY == 86400

    def test_inches_to_mm(self):
        """Test inches to mm conversion factor."""
        assert UnitConverter.INCHES_TO_MM == 25.4


class TestUnitConversion:
    """Test UnitConversion class constants."""

    def test_mm_day_to_cms_factor(self):
        """Test mm/day to cms conversion factor."""
        assert UnitConversion.MM_DAY_TO_CMS == pytest.approx(86.4)

    def test_cfs_to_cms_factor(self):
        """Test cfs to cms conversion factor."""
        assert UnitConversion.CFS_TO_CMS == pytest.approx(0.028316846592)

    def test_area_conversions(self):
        """Test area conversion factors."""
        assert UnitConversion.M2_TO_KM2 == 1e-6
        assert UnitConversion.KM2_TO_M2 == 1e6

    def test_mm_per_timestep_to_cms_factor(self):
        """Test dynamic timestep conversion factor calculation."""
        # Hourly timestep
        assert UnitConversion.mm_per_timestep_to_cms_factor(3600) == 3.6
        # Daily timestep
        assert UnitConversion.mm_per_timestep_to_cms_factor(86400) == 86.4


class TestPhysicalConstants:
    """Test PhysicalConstants class values."""

    def test_water_density(self):
        """Test water density value."""
        assert PhysicalConstants.WATER_DENSITY == 1000.0

    def test_latent_heat_vaporization(self):
        """Test latent heat of vaporization value."""
        assert PhysicalConstants.LATENT_HEAT_VAPORIZATION == pytest.approx(2.45e6)

    def test_gravity(self):
        """Test standard gravity value."""
        assert PhysicalConstants.GRAVITY == pytest.approx(9.80665)

    def test_kelvin_offset(self):
        """Test Kelvin to Celsius offset."""
        assert PhysicalConstants.KELVIN_OFFSET == 273.15
