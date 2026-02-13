"""Tests for IGNACIO parameter manager."""

import pytest
import logging
import tempfile
from pathlib import Path


@pytest.fixture
def logger():
    return logging.getLogger('test_ignacio_pm')


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def ignacio_config(temp_dir):
    return {
        'DOMAIN_NAME': 'test_fire',
        'EXPERIMENT_ID': 'fire_test',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'IGNACIO_PARAMS_TO_CALIBRATE': 'ffmc,dmc,dc,fmc,curing,initial_radius',
    }


class TestIGNACIOParameterManagerRegistration:
    """Tests for parameter manager registration."""

    def test_parameter_manager_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'IGNACIO' in OptimizerRegistry._parameter_managers

    def test_parameter_manager_is_correct_class(self):
        from symfluence.optimization.registry import OptimizerRegistry
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        assert OptimizerRegistry._parameter_managers.get('IGNACIO') == IGNACIOParameterManager


class TestIGNACIOParameterBounds:
    """Tests for IGNACIO FBP parameter bounds."""

    def test_ignacio_bounds_available(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert len(bounds) == 6

    def test_ffmc_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'ffmc' in bounds
        assert bounds['ffmc']['min'] == 0.0
        assert bounds['ffmc']['max'] == 101.0

    def test_dmc_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'dmc' in bounds
        assert bounds['dmc']['min'] == 0.0
        assert bounds['dmc']['max'] == 200.0

    def test_dc_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'dc' in bounds
        assert bounds['dc']['min'] == 0.0
        assert bounds['dc']['max'] == 800.0

    def test_fmc_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'fmc' in bounds
        assert bounds['fmc']['min'] == 50.0
        assert bounds['fmc']['max'] == 150.0

    def test_curing_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'curing' in bounds
        assert bounds['curing']['min'] == 0.0
        assert bounds['curing']['max'] == 100.0

    def test_initial_radius_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
        bounds = get_ignacio_bounds()
        assert 'initial_radius' in bounds
        assert bounds['initial_radius']['min'] == 1.0
        assert bounds['initial_radius']['max'] == 100.0


class TestIGNACIOParameterManagerInstance:
    """Tests for IGNACIO parameter manager instances."""

    def test_can_instantiate(self, ignacio_config, logger, temp_dir):
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        manager = IGNACIOParameterManager(ignacio_config, logger, temp_dir)
        assert manager is not None

    def test_parameter_names(self, ignacio_config, logger, temp_dir):
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        manager = IGNACIOParameterManager(ignacio_config, logger, temp_dir)
        names = manager._get_parameter_names()
        assert 'ffmc' in names
        assert 'dmc' in names
        assert 'initial_radius' in names
        assert len(names) == 6

    def test_load_bounds(self, ignacio_config, logger, temp_dir):
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        manager = IGNACIOParameterManager(ignacio_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()
        assert len(bounds) == 6
        for param in ['ffmc', 'dmc', 'dc', 'fmc', 'curing', 'initial_radius']:
            assert param in bounds

    def test_normalize_denormalize_roundtrip(self, ignacio_config, logger, temp_dir):
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        manager = IGNACIOParameterManager(ignacio_config, logger, temp_dir)

        params = {'ffmc': 88.0, 'dmc': 30.0, 'dc': 150.0, 'fmc': 100.0, 'curing': 85.0, 'initial_radius': 10.0}
        normalized = manager.normalize_parameters(params)
        denormalized = manager.denormalize_parameters(normalized)

        for key in params:
            assert abs(denormalized[key] - params[key]) < 0.1, \
                f"Roundtrip failed for {key}: {params[key]} -> {denormalized[key]}"

    def test_get_initial_parameters(self, ignacio_config, logger, temp_dir):
        from symfluence.models.ignacio.calibration.parameter_manager import IGNACIOParameterManager
        manager = IGNACIOParameterManager(ignacio_config, logger, temp_dir)
        initial = manager.get_initial_parameters()
        assert initial is not None
        assert len(initial) == 6
