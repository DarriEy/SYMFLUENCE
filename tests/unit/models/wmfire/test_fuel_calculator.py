"""Unit tests for WMFire FuelCalculator and FuelMoistureModel classes."""
import numpy as np
import pytest

from symfluence.models.wmfire.fuel_calculator import (
    FuelCalculator,
    FuelMoistureModel,
    FuelStats,
    estimate_initial_moisture,
)


class TestFuelCalculator:
    """Tests for FuelCalculator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        calc = FuelCalculator()
        assert calc.carbon_to_fuel_ratio == 2.0
        assert 'litr1c' in calc.pool_weights
        assert 'litr4c' in calc.pool_weights

    def test_init_custom_ratio(self):
        """Test custom carbon to fuel ratio."""
        calc = FuelCalculator(carbon_to_fuel_ratio=2.5)
        assert calc.carbon_to_fuel_ratio == 2.5

    def test_init_custom_weights(self):
        """Test custom pool weights."""
        weights = {'litr1c': 0.5, 'litr2c': 0.5}
        calc = FuelCalculator(pool_weights=weights)
        assert calc.pool_weights == weights

    def test_calculate_fuel_load_scalar(self):
        """Test fuel load calculation with scalar values."""
        calc = FuelCalculator(carbon_to_fuel_ratio=2.0)

        # 1 kg C/m² in each pool
        pools = {
            'litr1c': 1.0,
            'litr2c': 1.0,
            'litr3c': 1.0,
            'litr4c': 1.0,
        }

        fuel = calc.calculate_fuel_load(pools)

        # Expected: sum of (weight * carbon * ratio) for each pool
        # With default weights: 0.35 + 0.30 + 0.25 + 0.10 = 1.0
        # fuel = 1.0 * 2.0 = 2.0 kg/m²
        assert fuel == pytest.approx(2.0, rel=0.01)

    def test_calculate_fuel_load_varying_carbon(self):
        """Test fuel load with varying carbon amounts."""
        calc = FuelCalculator(carbon_to_fuel_ratio=2.0)

        pools = {
            'litr1c': 0.5,   # 0.35 * 0.5 * 2 = 0.35
            'litr2c': 1.0,   # 0.30 * 1.0 * 2 = 0.60
            'litr3c': 2.0,   # 0.25 * 2.0 * 2 = 1.00
            'litr4c': 0.0,   # 0.10 * 0.0 * 2 = 0.00
        }

        fuel = calc.calculate_fuel_load(pools)
        expected = 0.35 + 0.60 + 1.00 + 0.00
        assert fuel == pytest.approx(expected, rel=0.01)

    def test_calculate_fuel_load_with_area(self):
        """Test fuel load calculation with cell area."""
        calc = FuelCalculator(carbon_to_fuel_ratio=2.0)

        pools = {'litr1c': 1.0, 'litr2c': 1.0, 'litr3c': 1.0, 'litr4c': 1.0}
        cell_area = 900.0  # 30m x 30m cell

        fuel = calc.calculate_fuel_load(pools, cell_area_m2=cell_area)

        # 2.0 kg/m² * 900 m² = 1800 kg
        assert fuel == pytest.approx(1800.0, rel=0.01)

    def test_calculate_fuel_load_array(self):
        """Test fuel load with numpy arrays."""
        calc = FuelCalculator(carbon_to_fuel_ratio=2.0)

        pools = {
            'litr1c': np.array([1.0, 2.0, 3.0]),
            'litr2c': np.array([1.0, 2.0, 3.0]),
            'litr3c': np.array([1.0, 2.0, 3.0]),
            'litr4c': np.array([1.0, 2.0, 3.0]),
        }

        fuel = calc.calculate_fuel_load(pools)

        assert len(fuel) == 3
        assert fuel[0] == pytest.approx(2.0, rel=0.01)  # Sum of weights * 1 * 2
        assert fuel[1] == pytest.approx(4.0, rel=0.01)  # Sum of weights * 2 * 2
        assert fuel[2] == pytest.approx(6.0, rel=0.01)  # Sum of weights * 3 * 2

    def test_calculate_fuel_load_grid(self):
        """Test grid-based fuel load calculation."""
        calc = FuelCalculator()

        # 2x2 grids
        grids = {
            'litr1c': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'litr2c': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'litr3c': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'litr4c': np.array([[1.0, 2.0], [3.0, 4.0]]),
        }

        fuel_grid = calc.calculate_fuel_load_grid(grids)

        assert fuel_grid.shape == (2, 2)
        # Each cell has total carbon, multiplied by ratio
        # Top-left: 1 * 2 * (sum of weights) = 2.0
        assert fuel_grid[0, 0] == pytest.approx(2.0, rel=0.01)
        assert fuel_grid[1, 1] == pytest.approx(8.0, rel=0.01)

    def test_calculate_load_coefficients_uniform(self):
        """Test load coefficient calculation with uniform fuel."""
        calc = FuelCalculator()

        fuel = np.ones((10, 10)) * 2.0  # Uniform 2 kg/m²
        k1, k2 = calc.calculate_load_coefficients(fuel)

        # With mean=2, load_factor = 2/2 = 1.0, so k1 = 3.9
        assert k1 == pytest.approx(3.9, rel=0.1)
        # With zero std (uniform), cv=0, so k2 = default
        assert k2 == pytest.approx(0.07, rel=0.1)

    def test_calculate_load_coefficients_variable(self):
        """Test load coefficient calculation with variable fuel."""
        calc = FuelCalculator()

        # Create variable fuel load
        fuel = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k1, k2 = calc.calculate_load_coefficients(fuel)

        # Mean = 3, so load_factor = 3/2 = 1.5, k1 = 3.9 * 1.5 = 5.85
        mean = np.mean(fuel)
        std = np.std(fuel)
        expected_k1 = 3.9 * np.clip(mean / 2.0, 0.5, 2.0)
        expected_k2 = 0.07 * (1.0 + std / mean)

        assert k1 == pytest.approx(expected_k1, rel=0.1)
        assert k2 == pytest.approx(expected_k2, rel=0.1)

    def test_get_fuel_stats(self):
        """Test fuel statistics calculation."""
        calc = FuelCalculator()

        fuel = np.array([[1.0, 2.0], [3.0, np.nan]])
        stats = calc.get_fuel_stats(fuel)

        assert isinstance(stats, FuelStats)
        assert stats.mean == pytest.approx(2.0, rel=0.01)
        assert stats.min == pytest.approx(1.0, rel=0.01)
        assert stats.max == pytest.approx(3.0, rel=0.01)
        assert stats.total == pytest.approx(6.0, rel=0.01)

    def test_fuel_stats_to_dict(self):
        """Test FuelStats to dict conversion."""
        stats = FuelStats(mean=2.0, std=0.5, min=1.0, max=3.0, total=10.0)
        d = stats.to_dict()

        assert d['mean'] == 2.0
        assert d['std'] == 0.5
        assert d['min'] == 1.0
        assert d['max'] == 3.0
        assert d['total'] == 10.0


class TestFuelMoistureModel:
    """Tests for FuelMoistureModel class."""

    def test_init_defaults(self):
        """Test default initialization."""
        model = FuelMoistureModel()
        assert model.fuel_class == '10hr'
        assert model.timelag == 10.0

    def test_init_custom_class(self):
        """Test initialization with different fuel classes."""
        model_1hr = FuelMoistureModel(fuel_class='1hr')
        assert model_1hr.timelag == 1.0

        model_100hr = FuelMoistureModel(fuel_class='100hr')
        assert model_100hr.timelag == 100.0

    def test_equilibrium_moisture_dry(self):
        """Test EMC at low relative humidity."""
        model = FuelMoistureModel()

        # Dry conditions: 20% RH, 30°C
        emc = model.equilibrium_moisture(0.2, 30.0)

        # Should be low (around 5-8%)
        assert 0.03 < emc < 0.10

    def test_equilibrium_moisture_humid(self):
        """Test EMC at high relative humidity."""
        model = FuelMoistureModel()

        # Humid conditions: 80% RH, 20°C
        emc = model.equilibrium_moisture(0.8, 20.0)

        # Should be higher (around 15-25%)
        assert 0.15 < emc < 0.30

    def test_equilibrium_moisture_rh_percentage(self):
        """Test EMC handles RH as percentage (0-100)."""
        model = FuelMoistureModel()

        # Pass RH as percentage instead of fraction
        emc_pct = model.equilibrium_moisture(80, 20.0)  # 80%
        emc_frac = model.equilibrium_moisture(0.8, 20.0)  # 0.8

        assert emc_pct == pytest.approx(emc_frac, rel=0.01)

    def test_equilibrium_moisture_array(self):
        """Test EMC with array inputs."""
        model = FuelMoistureModel()

        rh = np.array([0.2, 0.5, 0.8])
        temp = np.array([30.0, 25.0, 20.0])

        emc = model.equilibrium_moisture(rh, temp)

        assert len(emc) == 3
        # EMC should increase with RH
        assert emc[0] < emc[1] < emc[2]

    def test_equilibrium_moisture_temperature_effect(self):
        """Test temperature effect on EMC."""
        model = FuelMoistureModel()

        # Same RH, different temperatures
        emc_cold = model.equilibrium_moisture(0.5, 10.0)
        emc_warm = model.equilibrium_moisture(0.5, 30.0)

        # Warmer temperatures should give lower EMC
        assert emc_warm < emc_cold

    def test_update_moisture_drying(self):
        """Test moisture update when drying."""
        model = FuelMoistureModel(fuel_class='10hr')

        current_mc = 0.20  # 20% moisture
        emc = 0.10  # Equilibrium is 10%

        # After 10 hours (one time constant), should be ~63% toward EMC
        new_mc = model.update_moisture(current_mc, emc, timestep_hours=10.0)

        # Should be between current and EMC, closer to EMC
        assert emc < new_mc < current_mc
        # After one time constant: mc = emc + (mc0 - emc) * e^-1
        expected = emc + (current_mc - emc) * np.exp(-1)
        assert new_mc == pytest.approx(expected, rel=0.01)

    def test_update_moisture_wetting(self):
        """Test moisture update when wetting."""
        model = FuelMoistureModel(fuel_class='10hr')

        current_mc = 0.10  # 10% moisture
        emc = 0.25  # Equilibrium is 25%

        new_mc = model.update_moisture(current_mc, emc, timestep_hours=10.0)

        # Should be between current and EMC, moving toward EMC
        assert current_mc < new_mc < emc

    def test_update_moisture_short_timestep(self):
        """Test moisture update with short timestep."""
        model = FuelMoistureModel(fuel_class='10hr')

        current_mc = 0.20
        emc = 0.10

        # Short timestep should change less
        new_mc_short = model.update_moisture(current_mc, emc, timestep_hours=1.0)
        new_mc_long = model.update_moisture(current_mc, emc, timestep_hours=10.0)

        # Short timestep should be closer to current
        assert abs(new_mc_short - current_mc) < abs(new_mc_long - current_mc)

    def test_calculate_moisture_coefficients(self):
        """Test moisture coefficient calculation."""
        model = FuelMoistureModel()

        mc = np.array([0.10, 0.15, 0.20])
        k1, k2 = model.calculate_moisture_coefficients(mc)

        # Default k1 = 3.8, scaled by 0.15/mean_mc
        # mean_mc = 0.15, so factor = 1.0, k1 = 3.8
        assert k1 == pytest.approx(3.8, rel=0.1)

    def test_critical_moisture(self):
        """Test critical moisture thresholds."""
        model = FuelMoistureModel()

        assert model.critical_moisture('grass') == 0.15
        assert model.critical_moisture('shrub') == 0.20
        assert model.critical_moisture('forest') == 0.25
        assert model.critical_moisture('unknown') == 0.25  # Default


class TestEstimateInitialMoisture:
    """Tests for estimate_initial_moisture function."""

    def test_seasonal_pattern(self):
        """Test seasonal moisture pattern."""
        # Northern hemisphere, temperate
        jan_mc = estimate_initial_moisture(1, 45.0, 'temperate')
        aug_mc = estimate_initial_moisture(8, 45.0, 'temperate')

        # August should be drier than January
        assert aug_mc < jan_mc

    def test_southern_hemisphere(self):
        """Test Southern Hemisphere pattern is shifted."""
        # Northern hemisphere January (winter = wet)
        nh_jan = estimate_initial_moisture(1, 45.0, 'temperate')
        # Southern hemisphere January (summer = dry)
        sh_jan = estimate_initial_moisture(1, -45.0, 'temperate')

        # Southern hemisphere January should be drier
        assert sh_jan < nh_jan

    def test_climate_adjustment(self):
        """Test climate type adjustments."""
        month = 8  # August
        lat = 35.0

        mc_arid = estimate_initial_moisture(month, lat, 'arid')
        mc_temperate = estimate_initial_moisture(month, lat, 'temperate')
        mc_boreal = estimate_initial_moisture(month, lat, 'boreal')

        # Arid should be driest, boreal wettest
        assert mc_arid < mc_temperate < mc_boreal

    def test_bounds(self):
        """Test moisture is bounded."""
        # Extreme cases
        for month in range(1, 13):
            for climate in ['arid', 'temperate', 'boreal', 'tropical']:
                mc = estimate_initial_moisture(month, 45.0, climate)
                assert 0.05 <= mc <= 0.35
