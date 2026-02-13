"""Tests for VIC calibration components."""

import pytest
import logging
import tempfile
from pathlib import Path


class TestVICCalibrationRegistration:
    """Tests for VIC calibration component registration."""

    def test_optimizer_registered(self):
        # VIC calibration requires explicit import to trigger decorator registration
        from symfluence.models.vic.calibration import VICModelOptimizer  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'VIC' in OptimizerRegistry._optimizers

    def test_worker_registered(self):
        from symfluence.models.vic.calibration import VICWorker  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'VIC' in OptimizerRegistry._workers

    def test_parameter_manager_registered(self):
        from symfluence.models.vic.calibration import VICParameterManager  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'VIC' in OptimizerRegistry._parameter_managers


class TestVICParameterBounds:
    """Tests for VIC parameter bounds in registry."""

    def test_vic_bounds_available(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_vic_bounds
        bounds = get_vic_bounds()
        assert len(bounds) > 0

    def test_infilt_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_vic_bounds
        bounds = get_vic_bounds()
        assert 'infilt' in bounds
        assert bounds['infilt']['min'] == 0.001
        assert bounds['infilt']['max'] == 0.9

    def test_soil_depth_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_vic_bounds
        bounds = get_vic_bounds()
        assert 'depth1' in bounds
        assert 'depth2' in bounds
        assert 'depth3' in bounds

    def test_baseflow_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_vic_bounds
        bounds = get_vic_bounds()
        assert 'Ds' in bounds
        assert 'Dsmax' in bounds
        assert 'Ws' in bounds
