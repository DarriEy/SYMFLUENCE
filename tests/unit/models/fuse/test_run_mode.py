# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE run-mode disentanglement.

Calibration always uses run_pre (trial parameters are read from
para_def.nc); run_def has exactly one role — the runner's initial default
run, which generates para_def.nc before calibration starts. A run_def
request reaching the calibration path would silently score the default
parameter set on every trial, so it is rejected with a warning.
"""
from __future__ import annotations

import logging

from symfluence.models.fuse.calibration.model_execution import detect_fuse_run_mode


class TestDetectFuseRunMode:
    def test_default_is_run_pre(self):
        assert detect_fuse_run_mode({}, {}) == 'run_pre'

    def test_explicit_run_pre_is_honored(self):
        assert detect_fuse_run_mode({'FUSE_RUN_MODE': 'run_pre'}, {}) == 'run_pre'

    def test_config_run_def_is_rejected_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            mode = detect_fuse_run_mode({'FUSE_RUN_MODE': 'run_def'}, {})
        assert mode == 'run_pre'
        assert any('run_def' in r.message and 'ignored' in r.message
                   for r in caplog.records)

    def test_kwargs_run_def_is_rejected_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            mode = detect_fuse_run_mode({}, {'mode': 'run_def'})
        assert mode == 'run_pre'
        assert any('run_def' in r.message for r in caplog.records)

    def test_regionalization_methods_all_use_run_pre(self):
        for method in ('lumped', 'semi_distributed', 'transfer_function'):
            config = {'PARAMETER_REGIONALIZATION': method}
            assert detect_fuse_run_mode(config, {}) == 'run_pre'
