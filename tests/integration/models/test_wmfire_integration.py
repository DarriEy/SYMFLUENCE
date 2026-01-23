"""Integration tests for WMFire module."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Skip all tests if dependencies not available
pytest.importorskip('geopandas')


class TestWMFireIntegration:
    """Integration tests for WMFire fire modeling components."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration with WMFire settings."""
        config = MagicMock()

        # WMFire configuration
        wmfire = MagicMock()
        wmfire.grid_resolution = 30
        wmfire.timestep_hours = 24
        wmfire.ndays_average = 30.0
        wmfire.fuel_source = 'static'
        wmfire.moisture_source = 'static'
        wmfire.carbon_to_fuel_ratio = 2.0
        wmfire.write_geotiff = True
        wmfire.load_k1 = None
        wmfire.load_k2 = None
        wmfire.moisture_k1 = None
        wmfire.moisture_k2 = None

        # RHESSys configuration
        rhessys = MagicMock()
        rhessys.wmfire = wmfire
        rhessys.use_wmfire = True

        # Model configuration
        model = MagicMock()
        model.rhessys = rhessys

        config.model = model
        return config

    @pytest.fixture
    def sample_catchment(self, tmp_path):
        """Create sample catchment GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import box

        # Create 4 HRU polygons in a 2x2 arrangement
        # Each HRU is 100m x 100m
        polygons = [
            box(500000, 4500000, 500100, 4500100),  # HRU 1 (SW)
            box(500100, 4500000, 500200, 4500100),  # HRU 2 (SE)
            box(500000, 4500100, 500100, 4500200),  # HRU 3 (NW)
            box(500100, 4500100, 500200, 4500200),  # HRU 4 (NE)
        ]

        gdf = gpd.GeoDataFrame({
            'HRU_ID': [1, 2, 3, 4],
            'elev_mean': [1000, 1100, 1200, 1300],
            'slope_mean': [10, 15, 20, 25],
            'aspect_mean': [180, 90, 270, 0],
            'geometry': polygons
        }, crs='EPSG:32610')

        return gdf

    def test_fire_grid_creation(self, mock_config, sample_catchment):
        """Test creating fire grids from catchment."""
        from symfluence.models.wmfire import FireGridManager

        manager = FireGridManager(mock_config)
        patch_grid, dem_grid = manager.create_fire_grid(sample_catchment)

        # Verify grid properties
        assert patch_grid.nrows > 0
        assert patch_grid.ncols > 0
        assert patch_grid.crs == 'EPSG:32610'
        assert patch_grid.resolution == 30

        # Verify DEM matches patch grid dimensions
        assert dem_grid.nrows == patch_grid.nrows
        assert dem_grid.ncols == patch_grid.ncols

        # Verify patch IDs are in grid
        unique_patches = np.unique(patch_grid.data)
        # Should have patches 1-4 plus 0 for background
        assert 1 in unique_patches or 2 in unique_patches

    def test_fire_grid_text_export(self, mock_config, sample_catchment, tmp_path):
        """Test exporting fire grids to text format."""
        from symfluence.models.wmfire import FireGridManager

        manager = FireGridManager(mock_config)
        patch_grid, dem_grid = manager.create_fire_grid(sample_catchment)

        # Write text files
        patch_path = tmp_path / 'patch_grid.txt'
        dem_path = tmp_path / 'dem_grid.txt'

        patch_path.write_text(patch_grid.to_text())
        dem_path.write_text(dem_grid.to_text())

        # Verify files exist and have content
        assert patch_path.exists()
        assert dem_path.exists()

        patch_content = patch_path.read_text()
        dem_content = dem_path.read_text()

        # Check content is non-empty and has expected structure
        patch_lines = patch_content.strip().split('\n')
        dem_lines = dem_content.strip().split('\n')

        assert len(patch_lines) == patch_grid.nrows
        assert len(dem_lines) == dem_grid.nrows

    def test_fire_def_generation(self, mock_config, sample_catchment, tmp_path):
        """Test generating fire.def with matching dimensions."""
        from symfluence.models.wmfire import FireGridManager, FireDefGenerator

        # Create grids
        manager = FireGridManager(mock_config)
        patch_grid, _ = manager.create_fire_grid(sample_catchment)

        # Generate fire.def
        gen = FireDefGenerator(mock_config)
        fire_def_path = tmp_path / 'fire.def'
        gen.write_fire_def(fire_def_path, patch_grid)

        # Verify file exists
        assert fire_def_path.exists()

        # Parse and verify dimensions match
        content = fire_def_path.read_text()
        assert f'{patch_grid.nrows}    n_rows' in content
        assert f'{patch_grid.ncols}    n_cols' in content

    def test_fuel_calculation_workflow(self, mock_config):
        """Test fuel load calculation from litter pools."""
        from symfluence.models.wmfire import FuelCalculator

        # Initialize calculator with config ratio
        calc = FuelCalculator(
            carbon_to_fuel_ratio=mock_config.model.rhessys.wmfire.carbon_to_fuel_ratio
        )

        # Simulate RHESSys litter output (kg C/m²)
        litter_pools = {
            'litr1c': 0.5,  # Labile
            'litr2c': 1.0,  # Cellulose
            'litr3c': 1.5,  # Lignin
            'litr4c': 0.5,  # Recalcitrant
        }

        fuel_load = calc.calculate_fuel_load(litter_pools)

        # Total carbon = 3.5 kg/m²
        # Weighted sum = 0.5*0.35 + 1.0*0.30 + 1.5*0.25 + 0.5*0.10
        #             = 0.175 + 0.30 + 0.375 + 0.05 = 0.90
        # Fuel = 0.90 * 2.0 = 1.80 kg/m²
        expected = 0.5*0.35 + 1.0*0.30 + 1.5*0.25 + 0.5*0.10
        expected *= 2.0  # Carbon to fuel ratio
        assert fuel_load == pytest.approx(expected, rel=0.01)

    def test_fuel_moisture_dynamics(self, mock_config):
        """Test fuel moisture model dynamics."""
        from symfluence.models.wmfire import FuelMoistureModel

        model = FuelMoistureModel(fuel_class='10hr')

        # Start at 20% moisture
        current_mc = 0.20

        # Simulate drying conditions: 30% RH, 30°C
        emc = model.equilibrium_moisture(0.30, 30.0)
        assert emc < current_mc  # EMC should be lower

        # Update moisture over 24 hours
        new_mc = model.update_moisture(current_mc, emc, timestep_hours=24.0)

        # Should have dried toward EMC
        assert current_mc > new_mc > emc

    def test_coefficient_adjustment(self, mock_config):
        """Test coefficient calculation from fuel/moisture data."""
        from symfluence.models.wmfire import FuelCalculator, FuelMoistureModel

        # Calculate fuel load coefficients
        fuel_calc = FuelCalculator()
        fuel_loads = np.array([1.0, 2.0, 3.0, 2.5, 1.5])  # Variable fuel
        load_k1, load_k2 = fuel_calc.calculate_load_coefficients(fuel_loads)

        assert load_k1 > 0
        assert load_k2 > 0

        # Calculate moisture coefficients
        moisture_model = FuelMoistureModel()
        moisture = np.array([0.10, 0.15, 0.12, 0.08, 0.11])
        moist_k1, moist_k2 = moisture_model.calculate_moisture_coefficients(moisture)

        assert moist_k1 > 0
        assert moist_k2 > 0

    def test_full_fire_setup_workflow(self, mock_config, sample_catchment, tmp_path):
        """Test complete fire setup workflow."""
        from symfluence.models.wmfire import (
            FireGridManager,
            FireDefGenerator,
            FuelCalculator,
        )

        # Step 1: Create fire grids
        manager = FireGridManager(mock_config)
        patch_grid, dem_grid = manager.create_fire_grid(sample_catchment)

        # Step 2: Calculate fuel statistics
        calc = FuelCalculator()
        # Simulate spatially variable fuel
        fuel_loads = np.random.uniform(1.0, 3.0, size=patch_grid.data.shape)
        fuel_stats = calc.get_fuel_stats(fuel_loads)
        load_k1, load_k2 = calc.calculate_load_coefficients(fuel_loads)

        # Step 3: Generate fire.def with adjusted coefficients
        gen = FireDefGenerator(mock_config)
        fire_def_path = tmp_path / 'fire.def'
        gen.write_fire_def(
            fire_def_path,
            patch_grid,
            fuel_stats={'load_k1': load_k1, 'load_k2': load_k2}
        )

        # Step 4: Write grid files
        fire_dir = tmp_path / 'fire'
        fire_dir.mkdir()
        (fire_dir / 'patch_grid.txt').write_text(patch_grid.to_text())
        (fire_dir / 'dem_grid.txt').write_text(dem_grid.to_text())

        # Verify all files exist
        assert fire_def_path.exists()
        assert (fire_dir / 'patch_grid.txt').exists()
        assert (fire_dir / 'dem_grid.txt').exists()

        # Verify dimensions are consistent
        content = fire_def_path.read_text()
        assert f'{patch_grid.nrows}    n_rows' in content
        assert f'{patch_grid.ncols}    n_cols' in content

    def test_different_resolutions(self, sample_catchment, tmp_path):
        """Test fire grid creation at different resolutions."""
        from symfluence.models.wmfire import FireGridManager

        for resolution in [30, 60, 90]:
            config = MagicMock()
            config.model.rhessys.wmfire.grid_resolution = resolution

            manager = FireGridManager(config)
            patch_grid, _ = manager.create_fire_grid(sample_catchment)

            assert patch_grid.resolution == resolution

            # Higher resolution = more cells
            if resolution == 30:
                base_cells = patch_grid.nrows * patch_grid.ncols
            else:
                current_cells = patch_grid.nrows * patch_grid.ncols
                # Approximately 4x fewer cells at 60m, 9x fewer at 90m
                expected_factor = (30 / resolution) ** 2
                assert current_cells <= base_cells


class TestWMFireConfigIntegration:
    """Tests for WMFire configuration integration."""

    def test_config_propagation(self):
        """Test that config values propagate correctly."""
        from symfluence.models.wmfire import FireGridManager, FireDefGenerator
        import geopandas as gpd
        from shapely.geometry import box

        # Create config with custom values
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.grid_resolution = 60
        wmfire.ndays_average = 45.0
        wmfire.load_k1 = 4.5
        wmfire.load_k2 = 0.10
        wmfire.moisture_k1 = 4.0
        wmfire.moisture_k2 = 0.30
        wmfire.write_geotiff = False
        config.model.rhessys.wmfire = wmfire

        # Create simple catchment
        gdf = gpd.GeoDataFrame({
            'HRU_ID': [1],
            'elev_mean': [1500],
            'geometry': [box(0, 0, 100, 100)]
        }, crs='EPSG:32610')

        # Grid manager should use config resolution
        manager = FireGridManager(config)
        assert manager.resolution == 60

        # Fire def generator should use config coefficients
        patch_grid, _ = manager.create_fire_grid(gdf)
        gen = FireDefGenerator(config)
        content = gen.generate_fire_def(patch_grid)

        assert '45.0    ndays_average' in content
        assert '4.50    load_k1' in content
        assert '0.10    load_k2' in content
