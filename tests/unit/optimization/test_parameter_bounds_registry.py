"""
Unit tests for Parameter Bounds Registry

Tests that parameter bounds are correctly registered for all models
and that config overrides preserve transform metadata.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from symfluence.optimization.core.parameter_bounds_registry import get_hype_bounds, get_mesh_bounds
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager


class TestParameterBoundsRegistry:
    """Test that bounds are correctly registered."""

    def test_hype_bounds_exist(self):
        """Test that all HYPE parameters have bounds."""
        bounds = get_hype_bounds()

        expected_params = [
            'ttmp', 'cmlt', 'ttpi', 'cmrefr',  # Snow
            'cevp', 'lp', 'epotdist',  # ET
            'rrcs1', 'rrcs2', 'rrcs3', 'wcwp', 'wcfc', 'wcep', 'srrcs',  # Soil
            'rivvel', 'damp', 'qmean',  # Routing
            'ilratk', 'ilratp',  # Lakes
        ]

        for param in expected_params:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert bounds[param]['min'] < bounds[param]['max']

    def test_mesh_bounds_exist(self):
        """Test that all MESH parameters have bounds."""
        bounds = get_mesh_bounds()

        expected_params = [
            'ZSNL', 'ZPLG', 'ZPLS', 'FRZTH', 'MANN',  # CLASS
            'RCHARG', 'DRAINFRAC', 'BASEFLW',  # Hydrology
            'DTMINUSR',  # Routing
        ]

        for param in expected_params:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert bounds[param]['min'] < bounds[param]['max']


# ---------------------------------------------------------------------------
# Tests for _apply_config_bounds_override
# ---------------------------------------------------------------------------


class _ConcreteParameterManager(BaseParameterManager):
    """Minimal concrete subclass for testing the base class helper."""

    def _get_parameter_names(self):
        return []

    def _load_parameter_bounds(self):
        return {}

    def update_model_files(self, params):
        return True

    def get_initial_parameters(self):
        return {}


@pytest.fixture
def manager():
    """Create a concrete BaseParameterManager for testing."""
    config = {'DOMAIN_NAME': 'test'}
    logger = logging.getLogger('test_bounds')
    return _ConcreteParameterManager(config, logger, Path('/tmp'))


class TestApplyConfigBoundsOverride:
    """Tests for BaseParameterManager._apply_config_bounds_override."""

    def test_list_override_preserves_transform(self, manager):
        """[min, max] list override must preserve registry transform."""
        bounds = {
            'KSAT': {'min': 1.0, 'max': 500.0, 'transform': 'log'},
            'DRN': {'min': 0.0, 'max': 5.0},
        }
        config_bounds = {
            'KSAT': [10.0, 200.0],
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['KSAT']['min'] == 10.0
        assert result['KSAT']['max'] == 200.0
        assert result['KSAT']['transform'] == 'log'
        # DRN unchanged
        assert result['DRN'] == {'min': 0.0, 'max': 5.0}

    def test_tuple_override_preserves_transform(self, manager):
        """(min, max) tuple override must preserve registry transform."""
        bounds = {
            'satdk': {'min': 1e-7, 'max': 1e-2, 'transform': 'log'},
        }
        config_bounds = {
            'satdk': (1e-6, 1e-3),
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['satdk']['min'] == 1e-6
        assert result['satdk']['max'] == 1e-3
        assert result['satdk']['transform'] == 'log'

    def test_dict_override_preserves_transform(self, manager):
        """{'min': ..., 'max': ...} dict override must preserve registry transform."""
        bounds = {
            'FLZ': {'min': 0.0001, 'max': 0.1, 'transform': 'log'},
        }
        config_bounds = {
            'FLZ': {'min': 0.001, 'max': 0.05},
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['FLZ']['min'] == 0.001
        assert result['FLZ']['max'] == 0.05
        assert result['FLZ']['transform'] == 'log'

    def test_dict_override_allows_explicit_transform(self, manager):
        """Dict override with explicit transform must use the explicit value."""
        bounds = {
            'param_a': {'min': 1.0, 'max': 100.0},
        }
        config_bounds = {
            'param_a': {'min': 5.0, 'max': 50.0, 'transform': 'log'},
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['param_a']['min'] == 5.0
        assert result['param_a']['max'] == 50.0
        assert result['param_a']['transform'] == 'log'

    def test_dict_override_can_change_transform(self, manager):
        """Dict override can change transform from log to linear."""
        bounds = {
            'param_a': {'min': 1.0, 'max': 100.0, 'transform': 'log'},
        }
        config_bounds = {
            'param_a': {'min': 5.0, 'max': 50.0, 'transform': 'linear'},
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['param_a']['transform'] == 'linear'

    def test_list_override_no_transform_in_registry(self, manager):
        """List override for param without transform should not add one."""
        bounds = {
            'simple_param': {'min': 0.0, 'max': 10.0},
        }
        config_bounds = {
            'simple_param': [1.0, 5.0],
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['simple_param']['min'] == 1.0
        assert result['simple_param']['max'] == 5.0
        assert 'transform' not in result['simple_param']

    def test_new_param_added(self, manager):
        """Config override can add a new parameter not in registry."""
        bounds = {
            'existing': {'min': 0.0, 'max': 1.0},
        }
        config_bounds = {
            'new_param': [0.5, 2.0],
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert 'new_param' in result
        assert result['new_param']['min'] == 0.5
        assert result['new_param']['max'] == 2.0

    def test_invalid_format_logged_and_skipped(self, manager):
        """Invalid config bounds format should be skipped with a warning."""
        bounds = {
            'param_a': {'min': 0.0, 'max': 10.0, 'transform': 'log'},
        }
        config_bounds = {
            'param_a': 'invalid',
            'param_b': [1],  # Only one element
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        # param_a should be unchanged (invalid format skipped)
        assert result['param_a']['min'] == 0.0
        assert result['param_a']['max'] == 10.0
        assert result['param_a']['transform'] == 'log'

    def test_none_config_bounds_returns_unchanged(self, manager):
        """None config_bounds should return bounds unchanged."""
        bounds = {
            'param_a': {'min': 0.0, 'max': 10.0, 'transform': 'log'},
        }

        result = manager._apply_config_bounds_override(bounds, None)

        assert result is bounds
        assert result['param_a']['transform'] == 'log'

    def test_empty_config_bounds_returns_unchanged(self, manager):
        """Empty config_bounds dict should return bounds unchanged."""
        bounds = {
            'param_a': {'min': 0.0, 'max': 10.0, 'transform': 'log'},
        }

        result = manager._apply_config_bounds_override(bounds, {})

        assert result is bounds

    def test_multiple_overrides_preserve_all_transforms(self, manager):
        """Multiple simultaneous overrides must preserve all transforms."""
        bounds = {
            'KSAT': {'min': 1.0, 'max': 500.0, 'transform': 'log'},
            'FLZ': {'min': 0.0001, 'max': 0.1, 'transform': 'log'},
            'DRN': {'min': 0.0, 'max': 5.0},
            'SDEP': {'min': 0.5, 'max': 10.0},
        }
        config_bounds = {
            'KSAT': [10.0, 200.0],
            'FLZ': {'min': 0.001, 'max': 0.05},
            'DRN': [0.5, 3.0],
        }

        result = manager._apply_config_bounds_override(bounds, config_bounds)

        assert result['KSAT']['transform'] == 'log'
        assert result['FLZ']['transform'] == 'log'
        assert 'transform' not in result['DRN']
        # SDEP untouched
        assert result['SDEP'] == {'min': 0.5, 'max': 10.0}
