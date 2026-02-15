"""Tests for IGNACIO preprocessor."""

import pytest


class TestIGNACIOPreProcessorImport:
    """Tests for IGNACIO preprocessor import."""

    def test_preprocessor_can_be_imported(self):
        from symfluence.models.ignacio.preprocessor import IGNACIOPreProcessor
        assert IGNACIOPreProcessor is not None


class TestIGNACIOConfigImport:
    """Tests for IGNACIO config import."""

    def test_config_can_be_imported(self):
        from symfluence.models.ignacio.config import IGNACIOConfig
        assert IGNACIOConfig is not None

    def test_config_defaults(self):
        from symfluence.models.ignacio.config import IGNACIOConfig
        config = IGNACIOConfig()
        assert config.project_name == 'ignacio_run'
        assert config.default_ffmc == 88.0
        assert config.default_dmc == 30.0
        assert config.default_dc == 150.0
        assert config.fmc == 100.0
        assert config.curing == 85.0
        assert config.initial_radius == 10.0

    def test_config_dt_range(self):
        from symfluence.models.ignacio.config import IGNACIOConfig
        config = IGNACIOConfig()
        assert 0.1 <= config.dt <= 60.0


class TestIGNACIOOptimizerRegistration:
    """Tests for IGNACIO optimizer registration."""

    def test_optimizer_registered(self):
        from symfluence.models.ignacio.calibration.optimizer import IGNACIOModelOptimizer
        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry.is_registered('IGNACIO')
        assert OptimizerRegistry.get_optimizer('IGNACIO') is IGNACIOModelOptimizer
