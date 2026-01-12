"""
Unit tests for MESH Parameter Manager

Tests that the MESH parameter manager can:
1. Load parameter bounds from registry
2. Normalize and denormalize parameters correctly
3. Handle parameter initialization
"""

import pytest
from pathlib import Path
import logging
import tempfile

from symfluence.optimization.parameter_managers import MESHParameterManager


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_logger')


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mesh_config(temp_dir):
    """Create a basic MESH configuration."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'MESH_PARAMS_TO_CALIBRATE': 'ZSNL,MANN,RCHARG',
    }


class TestMESHParameterManager:
    """Tests for MESH Parameter Manager."""

    def test_can_instantiate(self, mesh_config, logger, temp_dir):
        """Test that MESHParameterManager can be instantiated."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        assert manager is not None
        assert manager.domain_name == 'test_domain'
        assert manager.experiment_id == 'test_exp'

    def test_parameter_names(self, mesh_config, logger, temp_dir):
        """Test that parameter names are parsed correctly."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        param_names = manager._get_parameter_names()
        assert param_names == ['ZSNL', 'MANN', 'RCHARG']

    def test_load_bounds(self, mesh_config, logger, temp_dir):
        """Test that parameter bounds can be loaded from registry."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()

        assert 'ZSNL' in bounds
        assert 'MANN' in bounds
        assert 'RCHARG' in bounds

        # Check specific bounds
        assert bounds['ZSNL']['min'] == 0.001
        assert bounds['ZSNL']['max'] == 0.1
        assert bounds['MANN']['min'] == 0.01
        assert bounds['MANN']['max'] == 0.3

    def test_normalize_denormalize(self, mesh_config, logger, temp_dir):
        """Test parameter normalization and denormalization roundtrip."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)

        # Test parameters
        params = {'ZSNL': 0.05, 'MANN': 0.15, 'RCHARG': 0.5}

        # Normalize
        normalized = manager.normalize_parameters(params)
        assert len(normalized) == 3

        # All normalized values should be between 0 and 1
        for val in normalized:
            assert 0 <= val <= 1

        # Denormalize
        denorm = manager.denormalize_parameters(normalized)

        # Check roundtrip accuracy
        for param_name in params:
            assert abs(denorm[param_name] - params[param_name]) < 1e-6

    def test_get_default_initial_values(self, mesh_config, logger, temp_dir):
        """Test that default initial values are midpoint of bounds."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        defaults = manager._get_default_initial_values()

        # ZSNL bounds: 0.001 to 0.1, midpoint should be ~0.0505
        assert abs(defaults['ZSNL'] - 0.0505) < 1e-6

        # MANN bounds: 0.01 to 0.3, midpoint should be 0.155
        assert abs(defaults['MANN'] - 0.155) < 1e-6

        # RCHARG bounds: 0.0 to 1.0, midpoint should be 0.5
        assert abs(defaults['RCHARG'] - 0.5) < 1e-6

    def test_param_file_mapping(self, mesh_config, logger, temp_dir):
        """Test that parameters are mapped to correct files."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)

        assert manager.param_file_map['ZSNL'] == 'CLASS'
        assert manager.param_file_map['MANN'] == 'CLASS'
        assert manager.param_file_map['RCHARG'] == 'hydrology'
        assert manager.param_file_map['DTMINUSR'] == 'routing'
