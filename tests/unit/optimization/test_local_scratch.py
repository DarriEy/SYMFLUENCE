# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for USE_LOCAL_SCRATCH / SLURM_TMPDIR handling.

Covers GitHub issue #90: USE_LOCAL_SCRATCH silently ignored due to
Pydantic validation fallback and SLURM_TMPDIR timing.
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ---------------------------------------------------------------------------
# Required fields for a valid SymfluenceConfig
# ---------------------------------------------------------------------------

_REQUIRED_BASE = {
    'DOMAIN_DEFINITION_METHOD': 'subset',
    'SUB_GRID_DISCRETIZATION': 'GRU',
    'FORCING_DATASET': 'ERA5',
    'SYMFLUENCE_CODE_DIR': '/tmp/code',
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_config(tmp_path):
    """SymfluenceConfig with USE_LOCAL_SCRATCH enabled."""
    from symfluence.core.config.models import SymfluenceConfig

    return SymfluenceConfig(**{
        **_REQUIRED_BASE,
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test',
        'EXPERIMENT_ID': 'exp1',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'OPTIMIZATION_METHODS': ['iteration'],
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'OPTIMIZATION_METRIC': 'KGE',
        'NUMBER_OF_ITERATIONS': 3,
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-31 23:00',
        'CALIBRATION_PERIOD': '2020-01-01, 2020-01-31',
        'PARALLEL_WORKERS': 1,
        'USE_LOCAL_SCRATCH': True,
    })


@pytest.fixture
def test_logger():
    logger = logging.getLogger('test_local_scratch')
    logger.setLevel(logging.DEBUG)
    return logger


def _make_scratch_manager(tmp_path, config, logger):
    """Create a LocalScratchManager for testing."""
    from symfluence.optimization.mixins.parallel.local_scratch_manager import (
        LocalScratchManager,
    )

    project_dir = tmp_path / 'domain_test'
    project_dir.mkdir(parents=True, exist_ok=True)
    return LocalScratchManager(
        config=config,
        logger=logger,
        project_dir=project_dir,
        algorithm_name='dds',
    )


# ---------------------------------------------------------------------------
# Tests for LocalScratchManager SLURM_TMPDIR handling
# ---------------------------------------------------------------------------

class TestLocalScratchSetup:

    def test_creates_slurm_tmpdir_when_not_yet_existing(
        self, tmp_path, scratch_config, test_logger
    ):
        """Core fix for issue #90: SLURM_TMPDIR is set but the directory
        doesn't exist yet (common when the SLURM script creates it after
        the Python process starts).  The manager should mkdir it."""
        slurm_dir = str(tmp_path / 'slurm_scratch')
        assert not Path(slurm_dir).exists()

        with patch.dict(os.environ, {'SLURM_TMPDIR': slurm_dir}):
            mgr = _make_scratch_manager(tmp_path, scratch_config, test_logger)

        assert Path(slurm_dir).exists()
        assert mgr.use_scratch is True

    def test_uses_existing_slurm_tmpdir(
        self, tmp_path, scratch_config, test_logger
    ):
        """When SLURM_TMPDIR already exists, scratch should work as before."""
        slurm_dir = tmp_path / 'slurm_scratch'
        slurm_dir.mkdir()

        with patch.dict(os.environ, {'SLURM_TMPDIR': str(slurm_dir)}):
            mgr = _make_scratch_manager(tmp_path, scratch_config, test_logger)

        assert mgr.use_scratch is True

    def test_falls_back_when_slurm_tmpdir_not_set(
        self, tmp_path, scratch_config, test_logger
    ):
        """USE_LOCAL_SCRATCH=True but no SLURM_TMPDIR env var -> warn + disabled."""
        mock_logger = MagicMock()

        env = os.environ.copy()
        env.pop('SLURM_TMPDIR', None)
        with patch.dict(os.environ, env, clear=True):
            mgr = _make_scratch_manager(tmp_path, scratch_config, mock_logger)

        assert mgr.use_scratch is False
        warn_msgs = [str(c) for c in mock_logger.warning.call_args_list]
        assert any('SLURM_TMPDIR' in m for m in warn_msgs)

    def test_falls_back_when_scratch_dir_not_creatable(
        self, tmp_path, scratch_config, test_logger
    ):
        """SLURM_TMPDIR points somewhere we can't mkdir -> warn + disabled."""
        mock_logger = MagicMock()

        fake_dir = str(tmp_path / 'uncreatable_scratch')
        _real_mkdir = Path.mkdir

        def _mkdir_that_blocks_slurm(self_path, *args, **kwargs):
            if str(self_path) == fake_dir:
                raise OSError("permission denied")
            return _real_mkdir(self_path, *args, **kwargs)

        with patch.dict(os.environ, {'SLURM_TMPDIR': fake_dir}), \
             patch('pathlib.Path.mkdir', _mkdir_that_blocks_slurm):
            mgr = _make_scratch_manager(tmp_path, scratch_config, mock_logger)

        assert mgr.use_scratch is False
        warn_msgs = [str(c) for c in mock_logger.warning.call_args_list]
        assert any('could not be created' in m for m in warn_msgs)

    def test_no_scratch_when_disabled(
        self, tmp_path, test_logger
    ):
        """USE_LOCAL_SCRATCH=False (default) -> always disabled."""
        from symfluence.core.config.models import SymfluenceConfig

        config = SymfluenceConfig(**{
            **_REQUIRED_BASE,
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp1',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'OPTIMIZATION_METRIC': 'KGE',
            'NUMBER_OF_ITERATIONS': 3,
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-01-31 23:00',
            'CALIBRATION_PERIOD': '2020-01-01, 2020-01-31',
            'PARALLEL_WORKERS': 1,
            'USE_LOCAL_SCRATCH': False,
        })

        slurm_dir = tmp_path / 'slurm_scratch'
        slurm_dir.mkdir()
        with patch.dict(os.environ, {'SLURM_TMPDIR': str(slurm_dir)}):
            mgr = _make_scratch_manager(tmp_path, config, test_logger)

        assert mgr.use_scratch is False


# ---------------------------------------------------------------------------
# Tests for coercion warning on dict fallback
# ---------------------------------------------------------------------------

class TestCoercionWarning:

    def test_dict_fallback_emits_warning(self, tmp_path, test_logger):
        """When Pydantic validation fails and config stays a dict, the
        optimizer should log a visible warning (issue #90 root cause)."""
        incomplete_config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'USE_LOCAL_SCRATCH': True,
        }

        from symfluence.core.config.coercion import coerce_config

        result = coerce_config(incomplete_config, warn=True)
        assert isinstance(result, dict), "incomplete config should fall back to dict"

    def test_dict_config_still_reads_use_local_scratch(
        self, tmp_path, test_logger
    ):
        """Even when config is a plain dict (coercion failed), the
        _get_config_value dict_key fallback should still find
        USE_LOCAL_SCRATCH=True so it's not silently defaulted to False."""
        from symfluence.optimization.optimizers.base_model_optimizer import (
            BaseModelOptimizer,
        )

        class _Stub(BaseModelOptimizer):
            def _get_model_name(self):
                return 'TEST'

            def _run_model_for_final_evaluation(self, output_dir):
                return True

            def _get_final_file_manager_path(self):
                return tmp_path

        dict_config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp1',
            'USE_LOCAL_SCRATCH': True,
        }

        stub = object.__new__(_Stub)
        stub.__dict__['_config'] = dict_config

        value = stub._get_config_value(
            lambda: stub.config.system.use_local_scratch,
            default=False,
            dict_key='USE_LOCAL_SCRATCH',
        )
        assert value is True
