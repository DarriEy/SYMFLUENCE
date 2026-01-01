"""
Unit tests for PETCalculatorMixin.

Tests the PET calculation methods provided by the mixin.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import Mock
from src.symfluence.utils.models.mixins import PETCalculatorMixin


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
