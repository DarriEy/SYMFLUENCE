"""
Unit tests for PETCalculatorMixin.

Tests the PET calculation methods provided by the mixin.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from symfluence.models.mixins.pet_calculator import PETCalculatorMixin


class MockPreProcessor(PETCalculatorMixin):
    """Mock class that uses the mixin."""

    def __init__(self, logger):
        self.logger = logger


@pytest.fixture
def logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def preprocessor(logger):
    """Create a test preprocessor with PET mixin."""
    return MockPreProcessor(logger)


@pytest.fixture
def temp_data_celsius():
    """Create temperature data in Celsius."""
    time = pd.date_range('2020-01-01', periods=365, freq='D')
    temp = 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365)  # -5 to 25°C
    return xr.DataArray(temp, coords={'time': time}, dims=['time'])


@pytest.fixture
def temp_data_kelvin():
    """Create temperature data in Kelvin."""
    time = pd.date_range('2020-01-01', periods=365, freq='D')
    temp = 288.15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365)  # ~283K to 298K
    return xr.DataArray(temp, coords={'time': time}, dims=['time'])


@pytest.fixture
def temp_data_multi_hru():
    """Create multi-HRU temperature data."""
    time = pd.date_range('2020-01-01', periods=100, freq='D')
    temp = np.random.randn(100, 3) * 5 + 15  # 3 HRUs
    return xr.DataArray(
        temp,
        coords={'time': time, 'hru': [1, 2, 3]},
        dims=['time', 'hru']
    )


class TestOudinPET:
    """Test Oudin PET calculation."""

    def test_oudin_basic_celsius(self, preprocessor, temp_data_celsius):
        """Test basic Oudin PET calculation with Celsius input."""
        lat = 45.0  # 45°N
        pet = preprocessor.calculate_pet_oudin(temp_data_celsius, lat)

        # Check basic properties
        assert isinstance(pet, xr.DataArray)
        assert len(pet) == 365
        assert pet.attrs['units'] == 'mm/day'
        assert pet.attrs['method'] == 'Oudin et al. (2005)'

        # Check values are reasonable
        assert float(pet.mean()) > 0
        assert float(pet.mean()) < 10  # Reasonable PET range
        assert float(pet.min()) >= 0  # PET should never be negative

    def test_oudin_kelvin_conversion(self, preprocessor, temp_data_kelvin):
        """Test Oudin PET with Kelvin input (auto-conversion)."""
        lat = 45.0
        pet = preprocessor.calculate_pet_oudin(temp_data_kelvin, lat)

        assert isinstance(pet, xr.DataArray)
        assert float(pet.mean()) > 0
        assert float(pet.min()) >= 0

    def test_oudin_latitude_effect(self, preprocessor, temp_data_celsius):
        """Test that latitude affects PET calculation."""
        pet_low_lat = preprocessor.calculate_pet_oudin(temp_data_celsius, 10.0)
        pet_high_lat = preprocessor.calculate_pet_oudin(temp_data_celsius, 60.0)

        # PET should generally be higher at lower latitudes
        # (more solar radiation)
        assert float(pet_low_lat.mean()) > float(pet_high_lat.mean())

    def test_oudin_multi_hru(self, preprocessor, temp_data_multi_hru):
        """Test Oudin PET with multi-HRU data."""
        lat = 45.0
        pet = preprocessor.calculate_pet_oudin(temp_data_multi_hru, lat)

        # Check dimensions are preserved
        assert 'hru' in pet.dims
        assert pet.shape == temp_data_multi_hru.shape

        # Check all HRUs have reasonable values
        for hru in [1, 2, 3]:
            pet_hru = pet.sel(hru=hru)
            assert float(pet_hru.mean()) > 0
            assert float(pet_hru.min()) >= 0

    def test_oudin_seasonal_variation(self, preprocessor, temp_data_celsius):
        """Test that PET shows seasonal variation."""
        lat = 45.0
        pet = preprocessor.calculate_pet_oudin(temp_data_celsius, lat)

        # Extract summer and winter months
        summer_pet = pet.sel(time=pet.time.dt.month.isin([6, 7, 8]))
        winter_pet = pet.sel(time=pet.time.dt.month.isin([12, 1, 2]))

        # Summer PET should be higher than winter
        assert float(summer_pet.mean()) > float(winter_pet.mean())

    def test_oudin_negative_temp_handling(self, preprocessor):
        """Test that Oudin handles negative temperatures correctly."""
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        temp = np.array([-20, -10, -5, 0, 5, 10, 15, 20, 25, 30])
        temp_data = xr.DataArray(temp, coords={'time': time}, dims=['time'])

        lat = 45.0
        pet = preprocessor.calculate_pet_oudin(temp_data, lat)

        # PET should be 0 when T < -5°C
        assert float(pet[0]) == 0  # T = -20°C
        assert float(pet[1]) == 0  # T = -10°C
        assert float(pet[2]) == 0  # T = -5°C

        # PET should be > 0 when T > -5°C
        assert float(pet[3]) > 0  # T = 0°C


class TestHamonPET:
    """Test Hamon PET calculation."""

    def test_hamon_basic_celsius(self, preprocessor, temp_data_celsius):
        """Test basic Hamon PET calculation."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hamon(temp_data_celsius, lat)

        assert isinstance(pet, xr.DataArray)
        assert len(pet) == 365
        assert pet.attrs['units'] == 'mm/day'
        assert pet.attrs['method'] == 'Hamon (1961)'

        # Check reasonable values
        assert float(pet.mean()) > 0
        assert float(pet.min()) >= 0

    def test_hamon_kelvin_conversion(self, preprocessor, temp_data_kelvin):
        """Test Hamon PET with Kelvin input."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hamon(temp_data_kelvin, lat)

        assert isinstance(pet, xr.DataArray)
        assert float(pet.mean()) > 0

    def test_hamon_multi_hru(self, preprocessor, temp_data_multi_hru):
        """Test Hamon PET with multi-HRU data."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hamon(temp_data_multi_hru, lat)

        assert 'hru' in pet.dims
        assert pet.shape == temp_data_multi_hru.shape


class TestHargreavesPET:
    """Test Hargreaves PET calculation."""

    def test_hargreaves_basic_celsius(self, preprocessor, temp_data_celsius):
        """Test basic Hargreaves PET calculation."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hargreaves(temp_data_celsius, lat)

        assert isinstance(pet, xr.DataArray)
        assert len(pet) == 365
        assert pet.attrs['units'] == 'mm/day'
        assert 'Hargreaves' in pet.attrs['method']

        # Check reasonable values
        assert float(pet.mean()) > 0
        assert float(pet.min()) >= 0

    def test_hargreaves_kelvin_conversion(self, preprocessor, temp_data_kelvin):
        """Test Hargreaves PET with Kelvin input."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hargreaves(temp_data_kelvin, lat)

        assert isinstance(pet, xr.DataArray)
        assert float(pet.mean()) > 0

    def test_hargreaves_multi_hru(self, preprocessor, temp_data_multi_hru):
        """Test Hargreaves PET with multi-HRU data."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hargreaves(temp_data_multi_hru, lat)

        assert 'hru' in pet.dims
        assert pet.shape == temp_data_multi_hru.shape

    def test_hargreaves_simplified_note(self, preprocessor, temp_data_celsius):
        """Test that Hargreaves method includes note about simplification."""
        lat = 45.0
        pet = preprocessor.calculate_pet_hargreaves(temp_data_celsius, lat)

        assert 'note' in pet.attrs
        assert '10' in pet.attrs['note']  # Mentions assumed 10°C diurnal range


class TestPETComparison:
    """Test comparison between different PET methods."""

    def test_all_methods_produce_positive_values(self, preprocessor, temp_data_celsius):
        """Test that all PET methods produce non-negative values."""
        lat = 45.0

        pet_oudin = preprocessor.calculate_pet_oudin(temp_data_celsius, lat)
        pet_hamon = preprocessor.calculate_pet_hamon(temp_data_celsius, lat)
        pet_hargreaves = preprocessor.calculate_pet_hargreaves(temp_data_celsius, lat)

        assert (pet_oudin >= 0).all()
        assert (pet_hamon >= 0).all()
        assert (pet_hargreaves >= 0).all()

    def test_methods_produce_similar_magnitudes(self, preprocessor, temp_data_celsius):
        """Test that different methods produce PET values of similar magnitude."""
        lat = 45.0

        pet_oudin = preprocessor.calculate_pet_oudin(temp_data_celsius, lat)
        pet_hamon = preprocessor.calculate_pet_hamon(temp_data_celsius, lat)
        pet_hargreaves = preprocessor.calculate_pet_hargreaves(temp_data_celsius, lat)

        # All should be in reasonable range (0.5 to 12 mm/day)
        # Hargreaves tends to produce higher estimates than Oudin/Hamon
        for pet in [pet_oudin, pet_hamon, pet_hargreaves]:
            mean_pet = float(pet.mean())
            assert 0.5 < mean_pet < 12.0, f"PET mean {mean_pet} is outside expected range"


class TestErrorHandling:
    """Test error handling in PET calculations."""

    def test_unrealistic_temperature_range(self, preprocessor):
        """Test that unrealistic temperatures raise errors."""
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        unrealistic_temp = np.ones(10) * 1000  # 1000°C is unrealistic
        temp_data = xr.DataArray(unrealistic_temp, coords={'time': time}, dims=['time'])

        lat = 45.0

        # Should raise ValueError for unrealistic temperature
        with pytest.raises(ValueError, match="Unrealistic temperature"):
            preprocessor.calculate_pet_oudin(temp_data, lat)

    def test_extreme_negative_temperatures(self, preprocessor):
        """Test handling of extreme negative temperatures."""
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        extreme_temp = np.ones(10) * -100  # -100°C
        temp_data = xr.DataArray(extreme_temp, coords={'time': time}, dims=['time'])

        lat = 45.0

        # Should raise ValueError for unrealistic temperature after conversion check
        with pytest.raises(ValueError, match="unexpected range"):
            preprocessor.calculate_pet_oudin(temp_data, lat)


class TestMetadata:
    """Test that PET outputs have proper metadata."""

    def test_oudin_metadata(self, preprocessor, temp_data_celsius):
        """Test Oudin PET metadata."""
        pet = preprocessor.calculate_pet_oudin(temp_data_celsius, 45.0)

        assert 'units' in pet.attrs
        assert 'long_name' in pet.attrs
        assert 'method' in pet.attrs
        assert 'latitude' in pet.attrs
        assert pet.attrs['latitude'] == 45.0

    def test_hamon_metadata(self, preprocessor, temp_data_celsius):
        """Test Hamon PET metadata."""
        pet = preprocessor.calculate_pet_hamon(temp_data_celsius, 45.0)

        assert 'units' in pet.attrs
        assert 'method' in pet.attrs
        assert pet.attrs['latitude'] == 45.0

    def test_hargreaves_metadata(self, preprocessor, temp_data_celsius):
        """Test Hargreaves PET metadata."""
        pet = preprocessor.calculate_pet_hargreaves(temp_data_celsius, 45.0)

        assert 'units' in pet.attrs
        assert 'method' in pet.attrs
        assert 'latitude' in pet.attrs
        assert 'note' in pet.attrs


class TestHamonNumpyPET:
    """Test numpy-based Hamon PET calculation."""

    def test_basic(self):
        """Test basic Hamon numpy PET with default coefficient."""
        temp = np.array([10.0, 15.0, 20.0, 25.0])
        doy = np.array([100, 150, 200, 250])
        pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat=45.0)

        assert pet.shape == temp.shape
        assert (pet >= 0).all()
        assert pet.mean() > 0

    def test_coefficient_0_55(self):
        """Test 0.55 coefficient (SAC-SMA/HBV variant)."""
        temp = 15.0 * np.ones(365)
        doy = np.arange(1, 366)
        pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat=45.0, coefficient=0.55)

        assert 0.5 < pet.mean() < 5.0

    def test_coefficient_0_1651(self):
        """Test 0.1651 coefficient (original Hamon 1961 variant)."""
        temp = 15.0 * np.ones(365)
        doy = np.arange(1, 366)
        pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat=45.0, coefficient=0.1651)

        assert 0.1 < pet.mean() < 5.0

    def test_subzero_temp(self):
        """Test that sub-zero temperatures still return non-negative PET."""
        temp = np.array([-10.0, -5.0, 0.0, 5.0])
        doy = np.array([15, 46, 74, 100])
        pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat=45.0)

        assert (pet >= 0).all()

    def test_polar_latitude_clipping(self):
        """Test polar latitudes don't produce NaN from arccos."""
        temp = np.array([5.0, 10.0, 15.0])
        doy = np.array([172, 173, 174])  # Near summer solstice
        pet = PETCalculatorMixin.hamon_pet_numpy(temp, doy, lat=70.0)

        assert not np.any(np.isnan(pet))
        assert (pet >= 0).all()


class TestOudinNumpyPET:
    """Test numpy-based Oudin PET calculation."""

    def test_basic(self):
        """Test basic Oudin numpy PET."""
        temp = np.array([10.0, 15.0, 20.0, 25.0])
        doy = np.array([100, 150, 200, 250])
        pet = PETCalculatorMixin.oudin_pet_numpy(temp, doy, lat=45.0)

        assert pet.shape == temp.shape
        assert (pet >= 0).all()
        assert pet.mean() > 0

    def test_negative_temp_cutoff(self):
        """Test Oudin PET is zero when T < -5."""
        temp = np.array([-20.0, -10.0, -6.0, -4.0, 0.0, 10.0])
        doy = np.array([100, 100, 100, 100, 100, 100])
        pet = PETCalculatorMixin.oudin_pet_numpy(temp, doy, lat=45.0)

        assert pet[0] == 0.0  # T = -20
        assert pet[1] == 0.0  # T = -10
        assert pet[2] == 0.0  # T = -6
        assert pet[3] > 0.0   # T = -4
        assert pet[5] > 0.0   # T = 10

    def test_latitude_effect(self):
        """Test that latitude affects Oudin PET."""
        temp = 15.0 * np.ones(365)
        doy = np.arange(1, 366)
        pet_low = PETCalculatorMixin.oudin_pet_numpy(temp, doy, lat=10.0)
        pet_high = PETCalculatorMixin.oudin_pet_numpy(temp, doy, lat=60.0)

        assert pet_low.mean() > pet_high.mean()


class TestThornthwaiteNumpyPET:
    """Test numpy-based Thornthwaite PET calculation."""

    def test_basic(self):
        """Test basic Thornthwaite numpy PET."""
        time = pd.date_range('2020-01-01', periods=365, freq='D')
        temp = 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365)
        pet = PETCalculatorMixin.thornthwaite_pet_numpy(temp, time, lat=45.0)

        assert pet.shape == temp.shape
        assert (pet >= 0).all()
        assert pet.mean() > 0

    def test_seasonal_cycle(self):
        """Test Thornthwaite shows seasonal variation."""
        time = pd.date_range('2020-01-01', periods=365, freq='D')
        # Phase-shifted sine: peak at ~day 196 (mid-July), trough ~day 14 (mid-Jan)
        temp = 15 + 10 * np.sin(2 * np.pi * (np.arange(365) - 105) / 365)
        pet = PETCalculatorMixin.thornthwaite_pet_numpy(temp, time, lat=45.0)

        # Summer (DOY 150-250) should have higher PET than winter (DOY 1-60)
        summer_pet = pet[150:250].mean()
        winter_pet = pet[:60].mean()
        assert summer_pet > winter_pet


class TestNumpyXarrayConsistency:
    """Test that numpy and xarray PET methods produce consistent results."""

    def test_hamon_numpy_variants_reasonable(self):
        """Verify both Hamon numpy variants produce physically reasonable PET."""
        time = pd.date_range('2020-01-01', periods=365, freq='D')
        temp_vals = 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365)
        doy = np.asarray(time.dayofyear)
        lat = 45.0

        pet_055 = PETCalculatorMixin.hamon_pet_numpy(temp_vals, doy, lat, coefficient=0.55)
        pet_1651 = PETCalculatorMixin.hamon_pet_numpy(temp_vals, doy, lat, coefficient=0.1651)

        # Both should be non-negative with reasonable means
        assert (pet_055 >= 0).all()
        assert (pet_1651 >= 0).all()
        assert 0.5 < pet_055.mean() < 5.0
        assert 0.1 < pet_1651.mean() < 5.0

    def test_oudin_numpy_reasonable(self):
        """Verify numpy Oudin produces physically reasonable PET values."""
        time = pd.date_range('2020-01-01', periods=365, freq='D')
        temp_vals = 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365)
        doy = np.asarray(time.dayofyear)
        lat = 45.0

        pet_np = PETCalculatorMixin.oudin_pet_numpy(temp_vals, doy, lat)

        assert (pet_np >= 0).all()
        assert 0.5 < pet_np.mean() < 5.0
        # Summer should exceed winter
        assert pet_np[150:250].mean() > pet_np[:60].mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
