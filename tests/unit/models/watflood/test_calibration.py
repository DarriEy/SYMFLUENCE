"""Tests for WATFLOOD calibration components."""

import pytest


class TestWATFLOODCalibrationRegistration:
    """Tests for WATFLOOD calibration component registration."""

    def test_optimizer_registered(self):
        from symfluence.models.watflood.calibration import WATFLOODModelOptimizer  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'WATFLOOD' in OptimizerRegistry._optimizers

    def test_worker_registered(self):
        from symfluence.models.watflood.calibration import WATFLOODWorker  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'WATFLOOD' in OptimizerRegistry._workers

    def test_parameter_manager_registered(self):
        from symfluence.models.watflood.calibration import WATFLOODParameterManager  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'WATFLOOD' in OptimizerRegistry._parameter_managers


class TestWATFLOODParameterBounds:
    """Tests for WATFLOOD parameter bounds in registry."""

    def test_watflood_bounds_count(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_watflood_bounds
        bounds = get_watflood_bounds()
        assert len(bounds) == 16

    def test_log_transforms(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_watflood_bounds
        bounds = get_watflood_bounds()
        for param in ['FLZCOEF', 'AK2', 'AK2FS']:
            assert bounds[param]['transform'] == 'log', f"{param} should have log transform"

    def test_linear_transforms(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_watflood_bounds
        bounds = get_watflood_bounds()
        linear_params = ['PWR', 'R2N', 'AK', 'AKF', 'REESSION', 'RETN',
                         'R3', 'DS', 'FPET', 'FTALL', 'FM', 'BASE', 'SUBLIM_FACTOR']
        for param in linear_params:
            assert bounds[param]['transform'] == 'linear', f"{param} should have linear transform"

    def test_key_param_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_watflood_bounds
        bounds = get_watflood_bounds()
        assert bounds['FLZCOEF']['min'] == pytest.approx(1e-6)
        assert bounds['FLZCOEF']['max'] == pytest.approx(0.01)
        assert bounds['BASE']['min'] == pytest.approx(-3.0)
        assert bounds['BASE']['max'] == pytest.approx(2.0)
        assert bounds['DS']['max'] == pytest.approx(20.0)
