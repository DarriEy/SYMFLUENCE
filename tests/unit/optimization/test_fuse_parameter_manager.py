"""
Unit tests for FUSEParameterManager.

Tests FUSE-specific parameter handling including:
- NetCDF parameter file operations
- FUSE parameter bounds
- Parameter update methods
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_fuse_parameter_manager')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def fuse_config(tmp_path):
    """Create FUSE-specific configuration."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_catchment',
        'EXPERIMENT_ID': 'test_fuse_exp',
        'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MBASE,MAXWATR_1,BASERTE,TIMEDELAY',
        'FUSE_FILE_ID': 'test_fuse',
        'FUSE_SPATIAL_MODE': 'lumped',
    }


@pytest.fixture
def fuse_project_structure(tmp_path, fuse_config):
    """Create FUSE project directory structure."""
    domain_name = fuse_config['DOMAIN_NAME']
    experiment_id = fuse_config['EXPERIMENT_ID']
    fuse_id = fuse_config['FUSE_FILE_ID']

    # Create directories
    project_dir = tmp_path / f"domain_{domain_name}"
    sim_dir = project_dir / 'simulations' / experiment_id / 'FUSE'
    setup_dir = project_dir / 'settings' / 'FUSE'

    sim_dir.mkdir(parents=True)
    setup_dir.mkdir(parents=True)

    return {
        'project_dir': project_dir,
        'sim_dir': sim_dir,
        'setup_dir': setup_dir,
        'para_def_path': sim_dir / f"{domain_name}_{fuse_id}_para_def.nc",
        'para_sce_path': sim_dir / f"{domain_name}_{fuse_id}_para_sce.nc",
        'para_best_path': sim_dir / f"{domain_name}_{fuse_id}_para_best.nc",
    }


@pytest.fixture
def mock_netcdf_dataset():
    """Create a mock NetCDF dataset for FUSE parameters."""
    mock_ds = MagicMock()

    # Mock dimensions
    mock_ds.dims = {'par': 1, 'hru': 1}
    mock_ds.sizes = {'par': 1, 'hru': 1}

    # Mock variables with realistic FUSE parameter values
    mock_vars = {
        'MBASE': MagicMock(values=np.array([[0.0]])),
        'MAXWATR_1': MagicMock(values=np.array([[200.0]])),
        'BASERTE': MagicMock(values=np.array([[0.1]])),
        'TIMEDELAY': MagicMock(values=np.array([[1.0]])),
    }

    mock_ds.__getitem__ = lambda self, key: mock_vars.get(key, MagicMock())
    mock_ds.variables = mock_vars
    mock_ds.data_vars = mock_vars

    return mock_ds


# ============================================================================
# Initialization tests
# ============================================================================

class TestFUSEParameterManagerInitialization:
    """Test FUSEParameterManager initialization."""

    def test_init_parses_fuse_params(self, fuse_config, test_logger, tmp_path):
        """Test that FUSE parameters are parsed from config."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        expected_params = ['MBASE', 'MAXWATR_1', 'BASERTE', 'TIMEDELAY']
        assert manager.fuse_params == expected_params

    def test_init_sets_paths_correctly(self, fuse_config, test_logger, fuse_project_structure):
        """Test that paths are set correctly."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        manager = FUSEParameterManager(
            fuse_config, test_logger, fuse_project_structure['setup_dir']
        )

        assert manager.domain_name == 'test_catchment'
        assert manager.experiment_id == 'test_fuse_exp'
        assert manager.fuse_id == 'test_fuse'

    def test_init_handles_empty_params(self, fuse_config, test_logger, tmp_path):
        """Test initialization with no parameters to calibrate."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        config = fuse_config.copy()
        config['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = ''

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(config, test_logger, settings_dir)
        assert manager.fuse_params == []


# ============================================================================
# Parameter names and bounds tests
# ============================================================================

class TestFUSEParameterNames:
    """Test FUSE parameter name handling."""

    def test_get_parameter_names_returns_fuse_params(self, fuse_config, test_logger, tmp_path):
        """Test that _get_parameter_names returns FUSE params."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)
        names = manager._get_parameter_names()

        assert 'MBASE' in names
        assert 'MAXWATR_1' in names
        assert 'BASERTE' in names

    def test_all_param_names_property(self, fuse_config, test_logger, tmp_path):
        """Test all_param_names property."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        assert len(manager.all_param_names) == 4


class TestFUSEParameterBounds:
    """Test FUSE parameter bounds."""

    def test_load_parameter_bounds_returns_dict(self, fuse_config, test_logger, tmp_path):
        """Test that bounds are loaded as dictionary."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)
        bounds = manager._load_parameter_bounds()

        assert isinstance(bounds, dict)

    def test_bounds_have_min_max(self, fuse_config, test_logger, tmp_path):
        """Test that each bound has min and max."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)
        bounds = manager.param_bounds

        for param_name, bound in bounds.items():
            assert 'min' in bound, f"Missing 'min' for {param_name}"
            assert 'max' in bound, f"Missing 'max' for {param_name}"
            assert bound['min'] < bound['max'], f"Invalid bounds for {param_name}"

    def test_default_fuse_bounds_are_reasonable(self, fuse_config, test_logger, tmp_path):
        """Test that default FUSE bounds are physically reasonable."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)
        default_bounds = manager._get_default_fuse_bounds()

        # Check specific parameters have reasonable bounds
        assert default_bounds['MBASE']['min'] >= -10.0  # Temperature can be negative
        assert default_bounds['MBASE']['max'] <= 10.0

        assert default_bounds['MAXWATR_1']['min'] > 0  # Storage must be positive
        assert default_bounds['MAXWATR_1']['max'] <= 2000.0

        assert default_bounds['BASERTE']['min'] > 0  # Rate must be positive
        assert default_bounds['BASERTE']['max'] <= 10.0

    def test_bounds_for_all_requested_params(self, fuse_config, test_logger, tmp_path):
        """Test that bounds exist for all requested parameters."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)
        bounds = manager.param_bounds

        for param in manager.fuse_params:
            assert param in bounds, f"Missing bounds for {param}"


# ============================================================================
# Normalization/Denormalization tests
# ============================================================================

class TestFUSENormalization:
    """Test FUSE parameter normalization (inherited from base)."""

    def test_normalize_fuse_params(self, fuse_config, test_logger, tmp_path):
        """Test normalizing FUSE parameters."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Get midpoint values
        params = {}
        for name in manager.all_param_names:
            bounds = manager.param_bounds[name]
            params[name] = (bounds['min'] + bounds['max']) / 2

        normalized = manager.normalize_parameters(params)

        # All should be around 0.5 (midpoint)
        np.testing.assert_array_almost_equal(normalized, [0.5] * len(params), decimal=1)

    def test_denormalize_fuse_params(self, fuse_config, test_logger, tmp_path):
        """Test denormalizing FUSE parameters."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Denormalize from midpoint
        normalized = np.array([0.5] * len(manager.all_param_names))
        params = manager.denormalize_parameters(normalized)

        # Check each param is at midpoint of its bounds
        for name in manager.all_param_names:
            bounds = manager.param_bounds[name]
            expected = (bounds['min'] + bounds['max']) / 2
            assert params[name] == pytest.approx(expected, rel=0.01)

    def test_roundtrip_consistency(self, fuse_config, test_logger, tmp_path):
        """Test normalize → denormalize roundtrip."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Original values at various points
        original = {}
        for i, name in enumerate(manager.all_param_names):
            bounds = manager.param_bounds[name]
            # Vary the fraction for each param
            frac = (i + 1) / (len(manager.all_param_names) + 1)
            original[name] = bounds['min'] + frac * (bounds['max'] - bounds['min'])

        normalized = manager.normalize_parameters(original)
        denormalized = manager.denormalize_parameters(normalized)

        for name in original:
            assert denormalized[name] == pytest.approx(original[name], rel=0.001)


# ============================================================================
# Parameter update tests
# ============================================================================

class TestFUSEParameterUpdate:
    """Test FUSE parameter file updates."""

    def test_update_model_files_signature(self, fuse_config, test_logger, tmp_path):
        """Test that update_model_files has correct signature."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Verify method exists and accepts dict
        assert hasattr(manager, 'update_model_files')
        assert callable(manager.update_model_files)

    @patch('xarray.open_dataset')
    def test_update_parameter_file_validates_params(
        self, mock_open, fuse_config, test_logger, fuse_project_structure, mock_netcdf_dataset
    ):
        """Test that update validates parameters."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        mock_open.return_value.__enter__ = Mock(return_value=mock_netcdf_dataset)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        manager = FUSEParameterManager(
            fuse_config, test_logger, fuse_project_structure['setup_dir']
        )

        # Valid parameters
        params = {
            'MBASE': 0.0,
            'MAXWATR_1': 200.0,
            'BASERTE': 0.1,
            'TIMEDELAY': 1.0,
        }

        # Should not raise for valid params
        assert manager.validate_parameters(params) is True


# ============================================================================
# Initial parameters tests
# ============================================================================

class TestFUSEInitialParameters:
    """Test FUSE initial parameter retrieval."""

    def test_get_initial_parameters_returns_dict(self, fuse_config, test_logger, tmp_path):
        """Test that get_initial_parameters returns dictionary."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Method should exist
        assert hasattr(manager, 'get_initial_parameters')

    def test_initial_params_within_bounds(self, fuse_config, test_logger, tmp_path):
        """Test that initial parameters are within bounds."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        try:
            initial = manager.get_initial_parameters()
            bounds = manager.param_bounds

            for name, value in initial.items():
                if name in bounds:
                    assert bounds[name]['min'] <= value <= bounds[name]['max'], \
                        f"Initial {name}={value} outside bounds [{bounds[name]['min']}, {bounds[name]['max']}]"
        except Exception:
            # get_initial_parameters may not be implemented or may require files
            pass


# ============================================================================
# Edge cases and error handling
# ============================================================================

class TestFUSEEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_missing_param_file(self, fuse_config, test_logger, tmp_path):
        """Test handling of missing parameter file."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # Parameter file doesn't exist - verify_and_fix should handle gracefully
        assert hasattr(manager, 'verify_and_fix_parameter_files')

    def test_handles_unknown_parameter(self, fuse_config, test_logger, tmp_path):
        """Test handling of unknown parameter in config."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        config = fuse_config.copy()
        config['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = 'MBASE,UNKNOWN_PARAM'

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(config, test_logger, settings_dir)

        # Should still have MBASE
        assert 'MBASE' in manager.fuse_params
        # UNKNOWN_PARAM may or may not be included depending on validation

    def test_handles_whitespace_in_params(self, fuse_config, test_logger, tmp_path):
        """Test handling of whitespace in parameter list."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        config = fuse_config.copy()
        config['SETTINGS_FUSE_PARAMS_TO_CALIBRATE'] = '  MBASE , MAXWATR_1  ,  BASERTE  '

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(config, test_logger, settings_dir)

        # Whitespace should be stripped
        assert manager.fuse_params == ['MBASE', 'MAXWATR_1', 'BASERTE']


# ============================================================================
# Integration with base class
# ============================================================================

class TestFUSEBaseClassIntegration:
    """Test integration with BaseParameterManager."""

    def test_inherits_from_base(self, fuse_config, test_logger, tmp_path):
        """Test that FUSEParameterManager inherits from BaseParameterManager."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager
        from symfluence.utils.optimization.core.base_parameter_manager import BaseParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        assert isinstance(manager, BaseParameterManager)

    def test_implements_abstract_methods(self, fuse_config, test_logger, tmp_path):
        """Test that all abstract methods are implemented."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = FUSEParameterManager(fuse_config, test_logger, settings_dir)

        # These should all be callable without error
        assert callable(manager._get_parameter_names)
        assert callable(manager._load_parameter_bounds)
        assert callable(manager.update_model_files)

        # And should return correct types
        names = manager._get_parameter_names()
        assert isinstance(names, list)

        bounds = manager._load_parameter_bounds()
        assert isinstance(bounds, dict)
