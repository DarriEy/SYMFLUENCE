"""Tests for WATFLOOD parameter definitions."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestParameterDefinitions:
    """Tests for WATFLOOD parameter constants."""

    def test_param_bounds_count(self):
        from symfluence.models.watflood.parameters import PARAM_BOUNDS
        assert len(PARAM_BOUNDS) == 16

    def test_all_bounds_valid(self):
        from symfluence.models.watflood.parameters import PARAM_BOUNDS
        for name, bounds in PARAM_BOUNDS.items():
            assert bounds['min'] < bounds['max'], f"{name}: min ({bounds['min']}) >= max ({bounds['max']})"

    def test_log_params_positive_min(self):
        from symfluence.models.watflood.parameters import PARAM_BOUNDS
        log_params = [k for k, v in PARAM_BOUNDS.items() if v.get('transform') == 'log']
        assert len(log_params) == 3
        for name in log_params:
            assert PARAM_BOUNDS[name]['min'] > 0, f"Log param {name} must have min > 0"

    def test_all_bounds_in_keyword_map(self):
        from symfluence.models.watflood.parameters import PAR_KEYWORD_MAP, PARAM_BOUNDS
        for param in PARAM_BOUNDS:
            assert param in PAR_KEYWORD_MAP, f"{param} missing from PAR_KEYWORD_MAP"


class TestParameterApplication:
    """Tests for parameter substitution in .par files."""

    def _make_manager(self, settings_dir):
        """Create a WATFLOODParameterManager with mock config."""
        from symfluence.models.watflood.calibration.parameter_manager import WATFLOODParameterManager

        mock_config = MagicMock()
        mock_config.get = MagicMock(return_value=None)

        manager = WATFLOODParameterManager.__new__(WATFLOODParameterManager)
        manager.config = mock_config
        manager.logger = logging.getLogger('test')
        manager.settings_dir = settings_dir
        manager.watflood_params = ['FLZCOEF', 'PWR', 'R2N', 'AK']
        manager.par_file = 'test.par'
        return manager

    def test_update_par_value_flz(self, tmp_path):
        manager = self._make_manager(tmp_path)
        content = ":flz, 0.005, some comment\n:pwr, 2.0, power\n"
        result = manager._update_par_value(content, 'FLZCOEF', 1.5e-4)
        assert '1.500E-04' in result
        assert ':pwr, 2.0' in result  # unchanged

    def test_update_par_value_multiple_occurrences(self, tmp_path):
        manager = self._make_manager(tmp_path)
        content = ":ak, 10.0,\n:akfs, 5.0,\n:ak, 20.0,\n"
        result = manager._update_par_value(content, 'AK', 50.0)
        assert result.count('5.000E+01') == 2  # both :ak lines updated
        assert ':akfs, 5.0' in result  # akfs unchanged
