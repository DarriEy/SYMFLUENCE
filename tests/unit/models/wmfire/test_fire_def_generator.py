"""Unit tests for WMFire FireDefGenerator class."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from symfluence.models.wmfire.fire_def_generator import (
    FireDefGenerator,
    FireDefParameters,
    validate_fire_def,
)
from symfluence.models.wmfire.fire_grid import FireGrid


class TestFireDefParameters:
    """Tests for FireDefParameters dataclass."""

    def test_defaults(self):
        """Test default parameter values."""
        params = FireDefParameters()

        assert params.n_rows == 3
        assert params.n_cols == 3
        assert params.ndays_average == 30.0
        assert params.load_k1 == 3.9
        assert params.load_k2 == 0.07
        assert params.moisture_k1 == 3.8
        assert params.moisture_k2 == 0.27
        assert params.spread_calc_type == 9

    def test_custom_values(self):
        """Test custom parameter values."""
        params = FireDefParameters(
            n_rows=10,
            n_cols=20,
            ndays_average=60.0,
            load_k1=4.5
        )

        assert params.n_rows == 10
        assert params.n_cols == 20
        assert params.ndays_average == 60.0
        assert params.load_k1 == 4.5
        # Others should still be defaults
        assert params.load_k2 == 0.07


class TestFireDefGenerator:
    """Tests for FireDefGenerator class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.model.rhessys.wmfire = None
        return config

    @pytest.fixture
    def mock_config_with_wmfire(self):
        """Create mock configuration with WMFire settings."""
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.ndays_average = 45.0
        wmfire.load_k1 = 4.0
        wmfire.load_k2 = 0.08
        wmfire.moisture_k1 = 4.0
        wmfire.moisture_k2 = 0.30
        config.model.rhessys.wmfire = wmfire
        return config

    @pytest.fixture
    def sample_grid(self):
        """Create sample FireGrid for testing."""
        data = np.zeros((5, 10), dtype='int32')
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 150.0)
        return FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

    def test_init(self, mock_config):
        """Test generator initialization."""
        gen = FireDefGenerator(mock_config)
        assert gen.config == mock_config
        assert gen._wmfire_config is None

    def test_init_with_wmfire_config(self, mock_config_with_wmfire):
        """Test generator with WMFire config."""
        gen = FireDefGenerator(mock_config_with_wmfire)
        assert gen._wmfire_config is not None

    def test_generate_fire_def_basic(self, mock_config, sample_grid):
        """Test basic fire.def generation."""
        gen = FireDefGenerator(mock_config)
        content = gen.generate_fire_def(sample_grid)

        # Check key parameters are present
        assert 'fire_parm_ID' in content
        assert 'n_rows' in content
        assert 'n_cols' in content
        assert '5    n_rows' in content
        assert '10    n_cols' in content

    def test_generate_fire_def_with_config(self, mock_config_with_wmfire, sample_grid):
        """Test fire.def generation with WMFire config."""
        gen = FireDefGenerator(mock_config_with_wmfire)
        content = gen.generate_fire_def(sample_grid)

        # Check config values are applied
        assert '45.0    ndays_average' in content
        assert '4.00    load_k1' in content
        assert '0.08    load_k2' in content

    def test_generate_fire_def_with_fuel_stats(self, mock_config, sample_grid):
        """Test fire.def generation with fuel statistics."""
        gen = FireDefGenerator(mock_config)

        fuel_stats = {
            'load_k1': 5.0,
            'load_k2': 0.10,
        }

        content = gen.generate_fire_def(sample_grid, fuel_stats=fuel_stats)

        assert '5.00    load_k1' in content
        assert '0.10    load_k2' in content

    def test_generate_fire_def_with_moisture_stats(self, mock_config, sample_grid):
        """Test fire.def generation with moisture statistics."""
        gen = FireDefGenerator(mock_config)

        moisture_stats = {
            'moisture_k1': 4.5,
            'moisture_k2': 0.35,
        }

        content = gen.generate_fire_def(sample_grid, moisture_stats=moisture_stats)

        assert '4.50    moisture_k1' in content
        assert '0.35    moisture_k2' in content
        # Should also set ignition moisture
        assert '4.50    moisture_ign_k1' in content

    def test_generate_fire_def_with_overrides(self, mock_config, sample_grid):
        """Test fire.def generation with direct overrides."""
        gen = FireDefGenerator(mock_config)

        content = gen.generate_fire_def(
            sample_grid,
            fire_verbose=1,
            fire_write=1,
            ran_seed=42
        )

        assert '1    fire_verbose' in content
        assert '1    fire_write' in content
        assert '42    ran_seed' in content

    def test_write_fire_def(self, mock_config, sample_grid, tmp_path):
        """Test writing fire.def to file."""
        gen = FireDefGenerator(mock_config)

        output_path = tmp_path / 'defs' / 'fire.def'
        result = gen.write_fire_def(output_path, sample_grid)

        assert result == output_path
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert '5    n_rows' in content
        assert '10    n_cols' in content

    def test_generate_default_fire_def(self, mock_config, tmp_path):
        """Test generating default fire.def with dimensions only."""
        gen = FireDefGenerator(mock_config)

        content = gen.generate_default_fire_def(
            n_rows=8,
            n_cols=12
        )

        assert '8    n_rows' in content
        assert '12    n_cols' in content

    def test_generate_default_fire_def_to_file(self, mock_config, tmp_path):
        """Test generating default fire.def and writing to file."""
        gen = FireDefGenerator(mock_config)

        output_path = tmp_path / 'fire.def'
        content = gen.generate_default_fire_def(
            n_rows=8,
            n_cols=12,
            output_path=output_path
        )

        assert output_path.exists()
        file_content = output_path.read_text()
        assert content == file_content

    def test_format_fire_def_structure(self, mock_config, sample_grid):
        """Test fire.def file structure."""
        gen = FireDefGenerator(mock_config)
        content = gen.generate_fire_def(sample_grid)

        lines = content.strip().split('\n')

        # Should have all expected parameters
        param_names = [
            'fire_parm_ID', 'ndays_average', 'load_k1', 'load_k2',
            'slope_k1', 'slope_k2', 'moisture_k1', 'moisture_k2',
            'winddir_k1', 'winddir_k2', 'moisture_ign_k1', 'moisture_ign_k2',
            'windmax', 'ignition_col', 'ignition_row', 'ignition_tmin',
            'fire_verbose', 'fire_write', 'fire_in_buffer',
            'n_rows', 'n_cols', 'spread_calc_type',
            'mean_log_wind', 'sd_log_wind',
            'mean1_rvm', 'mean2_rvm', 'kappa1_rvm', 'kappa2_rvm', 'p_rvm',
            'ign_def_mod', 'veg_k1', 'veg_k2', 'mean_ign',
            'ran_seed', 'calc_fire_effects', 'include_wui',
            'fire_size_name', 'wind_shift'
        ]

        for param in param_names:
            found = any(param in line for line in lines)
            assert found, f"Parameter '{param}' not found in fire.def"


class TestValidateFireDef:
    """Tests for validate_fire_def function."""

    @pytest.fixture
    def valid_fire_def(self, tmp_path):
        """Create a valid fire.def file."""
        content = """1    fire_parm_ID
30.0    ndays_average
3.9    load_k1
0.07    load_k2
5    n_rows
10    n_cols
"""
        path = tmp_path / 'fire.def'
        path.write_text(content)
        return path

    def test_validate_valid_file(self, valid_fire_def):
        """Test validation of valid file."""
        params = validate_fire_def(valid_fire_def)

        assert params['fire_parm_ID'] == 1
        assert params['ndays_average'] == 30.0
        assert params['n_rows'] == 5
        assert params['n_cols'] == 10

    def test_validate_missing_file(self, tmp_path):
        """Test validation of missing file."""
        with pytest.raises(ValueError, match="not found"):
            validate_fire_def(tmp_path / 'missing.def')

    def test_validate_missing_required(self, tmp_path):
        """Test validation with missing required parameters."""
        content = """1    fire_parm_ID
30.0    ndays_average
"""
        path = tmp_path / 'incomplete.def'
        path.write_text(content)

        with pytest.raises(ValueError, match="Missing required"):
            validate_fire_def(path)

    def test_validate_parses_floats(self, tmp_path):
        """Test that floats are parsed correctly."""
        content = """1    fire_parm_ID
30.5    ndays_average
3.9    load_k1
5    n_rows
10    n_cols
"""
        path = tmp_path / 'fire.def'
        path.write_text(content)

        params = validate_fire_def(path)
        assert isinstance(params['ndays_average'], float)
        assert params['ndays_average'] == 30.5
        assert isinstance(params['n_rows'], int)

    def test_validate_ignores_comments(self, tmp_path):
        """Test that comments are ignored."""
        content = """# This is a comment
1    fire_parm_ID
# Another comment
5    n_rows
10    n_cols
"""
        path = tmp_path / 'fire.def'
        path.write_text(content)

        params = validate_fire_def(path)
        assert params['fire_parm_ID'] == 1


class TestFireDefGeneratorIntegration:
    """Integration tests for FireDefGenerator."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from config to file."""
        # Create config
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.ndays_average = 60.0
        wmfire.load_k1 = None  # Use default
        wmfire.load_k2 = None
        wmfire.moisture_k1 = None
        wmfire.moisture_k2 = None
        config.model.rhessys.wmfire = wmfire

        # Create grid
        data = np.ones((10, 15), dtype='int32')
        transform = (60.0, 0.0, 0.0, 0.0, -60.0, 600.0)
        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=60.0
        )

        # Generate fire.def
        gen = FireDefGenerator(config)
        output_path = tmp_path / 'fire.def'
        gen.write_fire_def(output_path, grid)

        # Validate output
        params = validate_fire_def(output_path)
        assert params['n_rows'] == 10
        assert params['n_cols'] == 15
        assert params['ndays_average'] == 60.0

    def test_coefficient_adjustment_workflow(self, tmp_path):
        """Test workflow with coefficient adjustment from stats."""
        config = MagicMock()
        config.model.rhessys.wmfire = None

        data = np.ones((5, 5), dtype='int32')
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 150.0)
        grid = FireGrid(
            data=data,
            transform=transform,
            crs='EPSG:32610',
            resolution=30.0
        )

        # Simulated fuel and moisture statistics
        fuel_stats = {'load_k1': 4.2, 'load_k2': 0.09}
        moisture_stats = {'moisture_k1': 3.5, 'moisture_k2': 0.25}

        gen = FireDefGenerator(config)
        output_path = tmp_path / 'fire.def'
        gen.write_fire_def(
            output_path,
            grid,
            fuel_stats=fuel_stats,
            moisture_stats=moisture_stats
        )

        params = validate_fire_def(output_path)
        assert params['load_k1'] == pytest.approx(4.2, rel=0.01)
        assert params['load_k2'] == pytest.approx(0.09, rel=0.01)
        assert params['moisture_k1'] == pytest.approx(3.5, rel=0.01)
        assert params['moisture_k2'] == pytest.approx(0.25, rel=0.01)
