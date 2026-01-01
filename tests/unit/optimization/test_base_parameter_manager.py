"""
Unit tests for BaseParameterManager.

Tests the shared normalization, denormalization, and validation logic
that will be inherited by all model-specific parameter managers.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

from symfluence.utils.optimization.core.base_parameter_manager import BaseParameterManager


class ConcreteParameterManager(BaseParameterManager):
    """Concrete implementation for testing BaseParameterManager."""

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path,
                 param_names: List[str] = None, param_bounds: Dict[str, Dict[str, float]] = None):
        super().__init__(config, logger, settings_dir)
        self._test_param_names = param_names or ['param1', 'param2', 'param3']
        self._test_param_bounds = param_bounds or {
            'param1': {'min': 0.0, 'max': 10.0},
            'param2': {'min': -5.0, 'max': 5.0},
            'param3': {'min': 100.0, 'max': 500.0}
        }
        self.update_files_called = False
        self.initial_params = {
            'param1': 5.0,
            'param2': 0.0,
            'param3': 300.0
        }

    def _get_parameter_names(self) -> List[str]:
        return self._test_param_names

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        return self._test_param_bounds

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        self.update_files_called = True
        return True

    def get_initial_parameters(self) -> Dict[str, Any]:
        return self.initial_params.copy()


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_base_parameter_manager')


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp'
    }


@pytest.fixture
def settings_dir(tmp_path):
    """Create a temporary settings directory."""
    return tmp_path / 'settings'


@pytest.fixture
def param_manager(config, logger, settings_dir):
    """Create a test parameter manager."""
    return ConcreteParameterManager(config, logger, settings_dir)


class TestBaseParameterManagerInitialization:
    """Test initialization and property access."""

    def test_init_stores_config(self, param_manager, config):
        """Test that config is stored correctly."""
        assert param_manager.config == config

    def test_init_stores_logger(self, param_manager, logger):
        """Test that logger is stored correctly."""
        assert param_manager.logger == logger

    def test_init_stores_settings_dir(self, param_manager, settings_dir):
        """Test that settings_dir is stored correctly."""
        assert param_manager.settings_dir == settings_dir

    def test_all_param_names_property(self, param_manager):
        """Test that all_param_names property returns correct names."""
        expected = ['param1', 'param2', 'param3']
        assert param_manager.all_param_names == expected

    def test_param_bounds_property(self, param_manager):
        """Test that param_bounds property returns correct bounds."""
        bounds = param_manager.param_bounds
        assert 'param1' in bounds
        assert bounds['param1']['min'] == 0.0
        assert bounds['param1']['max'] == 10.0

    def test_lazy_initialization_of_param_names(self, config, logger, settings_dir):
        """Test that parameter names are loaded lazily."""
        manager = ConcreteParameterManager(config, logger, settings_dir)
        # _param_names should be empty initially
        assert manager._param_names == []
        # Accessing property should trigger loading
        names = manager.all_param_names
        assert len(names) == 3
        assert manager._param_names == names

    def test_lazy_initialization_of_param_bounds(self, config, logger, settings_dir):
        """Test that parameter bounds are loaded lazily."""
        manager = ConcreteParameterManager(config, logger, settings_dir)
        # _param_bounds should be empty initially
        assert manager._param_bounds == {}
        # Accessing property should trigger loading
        bounds = manager.param_bounds
        assert len(bounds) == 3
        assert manager._param_bounds == bounds


class TestNormalization:
    """Test parameter normalization to [0, 1] range."""

    def test_normalize_at_min_bound(self, param_manager):
        """Test normalization when parameters are at minimum bounds."""
        params = {
            'param1': 0.0,   # min = 0.0, max = 10.0  → normalized = 0.0
            'param2': -5.0,  # min = -5.0, max = 5.0 → normalized = 0.0
            'param3': 100.0  # min = 100.0, max = 500.0 → normalized = 0.0
        }
        normalized = param_manager.normalize_parameters(params)

        np.testing.assert_array_almost_equal(normalized, [0.0, 0.0, 0.0])

    def test_normalize_at_max_bound(self, param_manager):
        """Test normalization when parameters are at maximum bounds."""
        params = {
            'param1': 10.0,   # max = 10.0 → normalized = 1.0
            'param2': 5.0,    # max = 5.0 → normalized = 1.0
            'param3': 500.0   # max = 500.0 → normalized = 1.0
        }
        normalized = param_manager.normalize_parameters(params)

        np.testing.assert_array_almost_equal(normalized, [1.0, 1.0, 1.0])

    def test_normalize_at_midpoint(self, param_manager):
        """Test normalization when parameters are at midpoint."""
        params = {
            'param1': 5.0,    # (5-0)/(10-0) = 0.5
            'param2': 0.0,    # (0-(-5))/(5-(-5)) = 0.5
            'param3': 300.0   # (300-100)/(500-100) = 0.5
        }
        normalized = param_manager.normalize_parameters(params)

        np.testing.assert_array_almost_equal(normalized, [0.5, 0.5, 0.5])

    def test_normalize_arbitrary_values(self, param_manager):
        """Test normalization with arbitrary values."""
        params = {
            'param1': 2.5,    # (2.5-0)/(10-0) = 0.25
            'param2': 2.5,    # (2.5-(-5))/(5-(-5)) = 0.75
            'param3': 200.0   # (200-100)/(500-100) = 0.25
        }
        normalized = param_manager.normalize_parameters(params)

        np.testing.assert_array_almost_equal(normalized, [0.25, 0.75, 0.25], decimal=6)

    def test_normalize_clips_below_zero(self, param_manager):
        """Test that normalization clips values below 0."""
        params = {
            'param1': -5.0,   # Below min, should clip to 0.0
            'param2': 0.0,
            'param3': 300.0
        }
        normalized = param_manager.normalize_parameters(params)

        assert normalized[0] == 0.0
        assert normalized[0] >= 0.0

    def test_normalize_clips_above_one(self, param_manager):
        """Test that normalization clips values above 1."""
        params = {
            'param1': 15.0,   # Above max, should clip to 1.0
            'param2': 0.0,
            'param3': 300.0
        }
        normalized = param_manager.normalize_parameters(params)

        assert normalized[0] == 1.0
        assert normalized[0] <= 1.0

    def test_normalize_returns_numpy_array(self, param_manager):
        """Test that normalization returns a numpy array."""
        params = {'param1': 5.0, 'param2': 0.0, 'param3': 300.0}
        normalized = param_manager.normalize_parameters(params)

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == (3,)

    def test_normalize_missing_parameter_uses_default(self, param_manager, caplog):
        """Test that missing parameters default to 0.5 with warning."""
        params = {'param1': 5.0, 'param3': 300.0}  # param2 missing

        with caplog.at_level(logging.WARNING):
            normalized = param_manager.normalize_parameters(params)

        assert normalized[1] == 0.5  # Missing param should be 0.5
        assert any('param2' in record.message and 'missing' in record.message.lower()
                   for record in caplog.records)

    def test_normalize_preserves_parameter_order(self, param_manager):
        """Test that normalization preserves parameter order."""
        params = {'param1': 10.0, 'param2': -5.0, 'param3': 500.0}
        normalized = param_manager.normalize_parameters(params)

        # Order should match all_param_names
        assert len(normalized) == 3
        assert normalized[0] == 1.0  # param1
        assert normalized[1] == 0.0  # param2
        assert normalized[2] == 1.0  # param3


class TestDenormalization:
    """Test parameter denormalization from [0, 1] to actual values."""

    def test_denormalize_from_zero(self, param_manager):
        """Test denormalization from 0.0 (minimum bounds)."""
        normalized = np.array([0.0, 0.0, 0.0])
        params = param_manager.denormalize_parameters(normalized)

        assert params['param1'] == 0.0
        assert params['param2'] == -5.0
        assert params['param3'] == 100.0

    def test_denormalize_from_one(self, param_manager):
        """Test denormalization from 1.0 (maximum bounds)."""
        normalized = np.array([1.0, 1.0, 1.0])
        params = param_manager.denormalize_parameters(normalized)

        assert params['param1'] == 10.0
        assert params['param2'] == 5.0
        assert params['param3'] == 500.0

    def test_denormalize_from_midpoint(self, param_manager):
        """Test denormalization from 0.5 (midpoint)."""
        normalized = np.array([0.5, 0.5, 0.5])
        params = param_manager.denormalize_parameters(normalized)

        assert params['param1'] == 5.0
        assert params['param2'] == 0.0
        assert params['param3'] == 300.0

    def test_denormalize_arbitrary_values(self, param_manager):
        """Test denormalization with arbitrary normalized values."""
        normalized = np.array([0.25, 0.75, 0.125])
        params = param_manager.denormalize_parameters(normalized)

        # param1: 0 + 0.25 * (10-0) = 2.5
        # param2: -5 + 0.75 * (5-(-5)) = -5 + 7.5 = 2.5
        # param3: 100 + 0.125 * (500-100) = 100 + 50 = 150
        assert params['param1'] == pytest.approx(2.5)
        assert params['param2'] == pytest.approx(2.5)
        assert params['param3'] == pytest.approx(150.0)

    def test_denormalize_returns_dict(self, param_manager):
        """Test that denormalization returns a dictionary."""
        normalized = np.array([0.5, 0.5, 0.5])
        params = param_manager.denormalize_parameters(normalized)

        assert isinstance(params, dict)
        assert len(params) == 3
        assert all(name in params for name in ['param1', 'param2', 'param3'])

    def test_denormalize_clips_to_bounds(self, param_manager):
        """Test that denormalization clips to bounds for safety."""
        # Slightly above 1.0 and below 0.0 (might happen due to numerical errors)
        normalized = np.array([1.001, -0.001, 0.5])
        params = param_manager.denormalize_parameters(normalized)

        # Should clip to max and min
        assert params['param1'] <= 10.0
        assert params['param2'] >= -5.0

    def test_denormalize_calls_format_hook(self, config, logger, settings_dir):
        """Test that denormalization calls _format_parameter_value hook."""
        class CustomManager(ConcreteParameterManager):
            def _format_parameter_value(self, param_name: str, value: float) -> Any:
                # Custom formatting: return value as string
                return f"{value:.2f}"

        manager = CustomManager(config, logger, settings_dir)
        normalized = np.array([0.5, 0.5, 0.5])
        params = manager.denormalize_parameters(normalized)

        # Should be formatted as strings
        assert isinstance(params['param1'], str)
        assert params['param1'] == "5.00"


class TestRoundTripConsistency:
    """Test that normalize → denormalize → normalize is consistent."""

    def test_roundtrip_at_bounds(self, param_manager):
        """Test roundtrip at parameter bounds."""
        original = {'param1': 0.0, 'param2': 5.0, 'param3': 500.0}

        normalized = param_manager.normalize_parameters(original)
        denormalized = param_manager.denormalize_parameters(normalized)
        renormalized = param_manager.normalize_parameters(denormalized)

        np.testing.assert_array_almost_equal(normalized, renormalized)
        for key in original:
            assert denormalized[key] == pytest.approx(original[key])

    def test_roundtrip_at_midpoints(self, param_manager):
        """Test roundtrip at midpoints."""
        original = {'param1': 5.0, 'param2': 0.0, 'param3': 300.0}

        normalized = param_manager.normalize_parameters(original)
        denormalized = param_manager.denormalize_parameters(normalized)
        renormalized = param_manager.normalize_parameters(denormalized)

        np.testing.assert_array_almost_equal(normalized, renormalized)
        for key in original:
            assert denormalized[key] == pytest.approx(original[key])

    def test_roundtrip_arbitrary_values(self, param_manager):
        """Test roundtrip with arbitrary values."""
        original = {'param1': 3.7, 'param2': -2.3, 'param3': 427.8}

        normalized = param_manager.normalize_parameters(original)
        denormalized = param_manager.denormalize_parameters(normalized)
        renormalized = param_manager.normalize_parameters(denormalized)

        np.testing.assert_array_almost_equal(normalized, renormalized)
        for key in original:
            assert denormalized[key] == pytest.approx(original[key])


class TestValidation:
    """Test parameter validation."""

    def test_validate_parameters_within_bounds(self, param_manager):
        """Test validation passes for parameters within bounds."""
        params = {'param1': 5.0, 'param2': 0.0, 'param3': 300.0}

        assert param_manager.validate_parameters(params) is True

    def test_validate_parameters_at_bounds(self, param_manager):
        """Test validation passes for parameters at bounds."""
        params_min = {'param1': 0.0, 'param2': -5.0, 'param3': 100.0}
        params_max = {'param1': 10.0, 'param2': 5.0, 'param3': 500.0}

        assert param_manager.validate_parameters(params_min) is True
        assert param_manager.validate_parameters(params_max) is True

    def test_validate_parameters_below_bounds(self, param_manager, caplog):
        """Test validation fails for parameters below bounds."""
        params = {'param1': -1.0, 'param2': 0.0, 'param3': 300.0}

        with caplog.at_level(logging.WARNING):
            result = param_manager.validate_parameters(params)

        assert result is False
        assert any('param1' in record.message and 'outside bounds' in record.message
                   for record in caplog.records)

    def test_validate_parameters_above_bounds(self, param_manager, caplog):
        """Test validation fails for parameters above bounds."""
        params = {'param1': 5.0, 'param2': 0.0, 'param3': 600.0}

        with caplog.at_level(logging.WARNING):
            result = param_manager.validate_parameters(params)

        assert result is False
        assert any('param3' in record.message and 'outside bounds' in record.message
                   for record in caplog.records)

    def test_validate_ignores_unknown_parameters(self, param_manager):
        """Test that validation ignores parameters not in bounds."""
        params = {'param1': 5.0, 'param2': 0.0, 'param3': 300.0, 'unknown_param': 999.9}

        # Should pass - unknown param is ignored
        assert param_manager.validate_parameters(params) is True

    def test_validate_empty_params(self, param_manager):
        """Test validation with empty parameter dict."""
        params = {}

        # Should pass - no params to validate
        assert param_manager.validate_parameters(params) is True


class TestHelperMethods:
    """Test helper and hook methods."""

    def test_extract_scalar_from_float(self, param_manager):
        """Test extracting scalar from float."""
        assert param_manager._extract_scalar_value(5.5) == 5.5

    def test_extract_scalar_from_int(self, param_manager):
        """Test extracting scalar from int."""
        assert param_manager._extract_scalar_value(5) == 5.0

    def test_extract_scalar_from_numpy_scalar(self, param_manager):
        """Test extracting scalar from numpy scalar."""
        value = np.float64(7.5)
        assert param_manager._extract_scalar_value(value) == 7.5

    def test_extract_scalar_from_single_element_array(self, param_manager):
        """Test extracting scalar from single-element array."""
        value = np.array([8.5])
        assert param_manager._extract_scalar_value(value) == 8.5

    def test_extract_scalar_from_multi_element_array(self, param_manager):
        """Test extracting scalar from multi-element array (uses mean)."""
        value = np.array([1.0, 2.0, 3.0])
        assert param_manager._extract_scalar_value(value) == 2.0

    def test_format_parameter_value_default(self, param_manager):
        """Test default parameter formatting returns float."""
        formatted = param_manager._format_parameter_value('param1', 5.5)

        assert isinstance(formatted, float)
        assert formatted == 5.5

    def test_get_parameter_bounds_returns_copy(self, param_manager):
        """Test that get_parameter_bounds returns a copy."""
        bounds1 = param_manager.get_parameter_bounds()
        bounds2 = param_manager.get_parameter_bounds()

        # Should be equal but not the same object
        assert bounds1 == bounds2
        assert bounds1 is not bounds2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_parameter(self, config, logger, settings_dir):
        """Test with single parameter."""
        manager = ConcreteParameterManager(
            config, logger, settings_dir,
            param_names=['x'],
            param_bounds={'x': {'min': 0.0, 'max': 1.0}}
        )

        params = {'x': 0.5}
        normalized = manager.normalize_parameters(params)

        assert len(normalized) == 1
        assert normalized[0] == 0.5

    def test_many_parameters(self, config, logger, settings_dir):
        """Test with many parameters."""
        param_names = [f'p{i}' for i in range(20)]
        param_bounds = {name: {'min': 0.0, 'max': 100.0} for name in param_names}

        manager = ConcreteParameterManager(
            config, logger, settings_dir,
            param_names=param_names,
            param_bounds=param_bounds
        )

        params = {name: 50.0 for name in param_names}
        normalized = manager.normalize_parameters(params)

        assert len(normalized) == 20
        np.testing.assert_array_almost_equal(normalized, [0.5] * 20)

    def test_zero_range_parameter(self, config, logger, settings_dir, caplog):
        """Test with zero-range parameter (min == max)."""
        manager = ConcreteParameterManager(
            config, logger, settings_dir,
            param_names=['constant'],
            param_bounds={'constant': {'min': 5.0, 'max': 5.0}}
        )

        params = {'constant': 5.0}

        # Should handle division by zero gracefully with a warning
        with caplog.at_level(logging.WARNING):
            normalized = manager.normalize_parameters(params)

        # Should log warning about zero range
        assert any('zero range' in record.message.lower() for record in caplog.records)
        # Result should default to 0.5
        assert normalized[0] == 0.5

    def test_negative_range_parameters(self, config, logger, settings_dir):
        """Test with negative value ranges."""
        manager = ConcreteParameterManager(
            config, logger, settings_dir,
            param_names=['neg'],
            param_bounds={'neg': {'min': -100.0, 'max': -10.0}}
        )

        params = {'neg': -55.0}  # Midpoint
        normalized = manager.normalize_parameters(params)

        assert normalized[0] == 0.5

        denormalized = manager.denormalize_parameters(normalized)
        assert denormalized['neg'] == pytest.approx(-55.0)
