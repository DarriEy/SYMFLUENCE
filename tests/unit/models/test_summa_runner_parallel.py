# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for SUMMA domain parallel execution."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.summa.runner import SummaRunner


def make_summa_runner(tmp_path: Path, local_workers: int = 0) -> SummaRunner:
    """Create a SUMMA runner with local parallel execution enabled."""
    data_dir = tmp_path / 'data'
    code_dir = tmp_path / 'code'
    install_dir = tmp_path / 'summa_bin'
    settings_dir = data_dir / 'domain_test_domain' / 'settings' / 'SUMMA'

    install_dir.mkdir(parents=True)
    settings_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)

    summa_exe = install_dir / 'summa_sundials.exe'
    summa_exe.write_text('#!/bin/sh\n', encoding='utf-8')
    summa_exe.chmod(0o755)
    (settings_dir / 'fileManager.txt').write_text('', encoding='utf-8')

    config = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(data_dir),
        SYMFLUENCE_CODE_DIR=str(code_dir),
        DOMAIN_NAME='test_domain',
        EXPERIMENT_ID='test_run',
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-01-02 00:00',
        DOMAIN_DEFINITION_METHOD='semidistributed',
        SUB_GRID_DISCRETIZATION='GRUs',
        HYDROLOGICAL_MODEL='SUMMA',
        ROUTING_MODEL='none',
        FORCING_DATASET='ERA5',
        SUMMA_INSTALL_PATH=str(install_dir),
        SETTINGS_SUMMA_PATH=str(settings_dir),
        SETTINGS_SUMMA_USE_PARALLEL_SUMMA=True,
        SETTINGS_SUMMA_PARALLEL_BACKEND='local',
        SETTINGS_SUMMA_LOCAL_WORKERS=local_workers,
    )

    return SummaRunner(config, logging.getLogger(__name__))


def prepare_local_parallel_runner(runner: SummaRunner, tmp_path: Path) -> None:
    """Patch expensive runner steps for focused local backend tests."""
    runner.output_dir = tmp_path / 'output'
    runner.output_dir.mkdir()
    runner._count_grus = MagicMock(return_value=3)
    runner.estimate_optimal_grus_per_job = MagicMock(return_value=2)
    runner._pre_execution = MagicMock(return_value=True)
    runner._merge_parallel_outputs = MagicMock(return_value=runner.output_dir)


def test_local_parallel_summa_runs_one_command_per_gru(monkeypatch, tmp_path):
    """Test local parallel SUMMA runs each GRU subset and merges outputs."""
    runner = make_summa_runner(tmp_path, local_workers=2)
    prepare_local_parallel_runner(runner, tmp_path)
    calls = []

    def fake_run(command, stdout, stderr, cwd, env, check, timeout):
        calls.append({
            'command': command,
            'cwd': cwd,
            'env': env,
            'timeout': timeout,
        })
        stdout.write('ok\n')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr('symfluence.models.summa.runner.subprocess.run', fake_run)

    result = runner.run_parallel_summa()

    assert result == runner.output_dir
    runner._merge_parallel_outputs.assert_called_once()

    commands = sorted(calls, key=lambda call: int(call['command'][2]))
    assert len(commands) == 3
    for index, call in enumerate(commands, start=1):
        command = call['command']
        assert command[1:5] == ['-g', str(index), '1', '-m']
        assert command[5] == str(runner.file_manager)
        assert call['cwd'] == runner.output_dir
        assert call['env']['OMP_NUM_THREADS'] == '1'
        assert call['env']['MKL_NUM_THREADS'] == '1'
        assert call['env']['OPENBLAS_NUM_THREADS'] == '1'


def test_local_parallel_summa_skips_merge_after_failure(monkeypatch, tmp_path):
    """Test local parallel SUMMA skips merge if any GRU run fails."""
    runner = make_summa_runner(tmp_path, local_workers=2)
    prepare_local_parallel_runner(runner, tmp_path)

    def fake_run(command, stdout, stderr, cwd, env, check, timeout):
        return_code = 1 if command[2] == '2' else 0
        return subprocess.CompletedProcess(command, return_code)

    monkeypatch.setattr('symfluence.models.summa.runner.subprocess.run', fake_run)

    result = runner.run_parallel_summa()

    assert result is None
    runner._merge_parallel_outputs.assert_not_called()
