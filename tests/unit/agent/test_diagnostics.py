# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the shared agent-setup diagnostics."""
from __future__ import annotations

from symfluence.agent import diagnostics


def test_diagnostics_cover_all_subsystems(tmp_path):
    checks = diagnostics.run_diagnostics(tmp_path)
    labels = {c.label for c in checks}
    assert {'agent CLI', 'skills', 'subagents', 'cache dir',
            'MCP server', 'project context'} <= labels


def test_diagnostics_statuses_are_valid(tmp_path):
    for check in diagnostics.run_diagnostics(tmp_path):
        assert check.status in (diagnostics.OK, diagnostics.WARN, diagnostics.FAIL)
        assert check.detail


def test_no_cli_reports_failure(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda binary: None)
    checks = diagnostics.run_diagnostics(tmp_path)
    cli_check = next(c for c in checks if c.label == 'agent CLI')
    assert cli_check.status == diagnostics.FAIL
