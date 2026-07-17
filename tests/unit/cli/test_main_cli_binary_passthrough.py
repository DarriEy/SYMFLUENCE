# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the binary pass-through fast path in main_cli."""
from __future__ import annotations

from unittest.mock import patch

from symfluence import main_cli


def _run(argv, monkeypatch):
    monkeypatch.setattr('sys.argv', ['symfluence', *argv])
    with patch(
        'symfluence.cli.commands.binary_commands.BinaryCommands.exec_binary',
        return_value=0,
    ) as mock_exec:
        code = main_cli.main()
    return code, mock_exec


def test_dry_run_previews_instead_of_executing(monkeypatch, capsys):
    """--dry-run before `binary <tool>` must never execute the real binary."""
    code, mock_exec = _run(['--dry-run', 'binary', 'summa', 'run'], monkeypatch)
    assert code == 0
    mock_exec.assert_not_called()
    out = capsys.readouterr().out
    assert 'would execute' in out
    assert 'summa' in out


def test_plain_passthrough_still_executes(monkeypatch):
    code, mock_exec = _run(['binary', 'summa', '--version'], monkeypatch)
    assert code == 0
    mock_exec.assert_called_once_with('summa', ['--version'])


def test_debug_prefix_still_dispatches_passthrough(monkeypatch):
    code, mock_exec = _run(['--debug', 'binary', 'summa', '--version'], monkeypatch)
    assert code == 0
    mock_exec.assert_called_once_with('summa', ['--version'])
