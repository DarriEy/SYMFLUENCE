# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the Agent Command Center screen and its service."""
from __future__ import annotations

import asyncio

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")

from symfluence.agent.handoff import AgentHandoff  # noqa: E402
from symfluence.tui.app import SymfluenceTUI  # noqa: E402
from symfluence.tui.services.agent_service import AgentService  # noqa: E402


# ---------------------------------------------------------------- service

def test_snapshot_gathers_everything(tmp_path):
    snapshot = AgentService().snapshot(tmp_path)
    assert snapshot.workdir == tmp_path
    assert [r.name for r in snapshot.runtimes[:3]] == ['claude', 'codex', 'gemini']
    assert len(snapshot.skills) >= 6
    assert len(snapshot.subagents) >= 2
    assert len(snapshot.mcp_tools) >= 4
    assert snapshot.checks


def test_snapshot_detects_project_context(tmp_path):
    (tmp_path / 'config.yaml').write_text(
        'DOMAIN_NAME: bow\nHYDROLOGICAL_MODEL: SUMMA\n', encoding='utf-8'
    )
    (tmp_path / 'domain_bow').mkdir()
    snapshot = AgentService().snapshot(tmp_path)
    assert snapshot.configs and snapshot.configs[0][0] == 'config.yaml'
    assert snapshot.domains == ['domain_bow']


# ----------------------------------------------------------------- screen

def test_launch_exits_with_handoff():
    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('l')
        return app.return_value

    result = asyncio.run(_test())
    assert isinstance(result, AgentHandoff)
    assert result.prompt is None
    assert result.no_skills is False


def test_toggle_priming_carries_into_handoff():
    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('k')
            await pilot.press('l')
        return app.return_value

    assert asyncio.run(_test()).no_skills is True


def test_agent_defaults_preselect_runtime_and_priming():
    async def _test():
        app = SymfluenceTUI(
            initial_mode='agent',
            agent_defaults={'cli': 'gemini', 'no_skills': True},
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('l')
        return app.return_value

    result = asyncio.run(_test())
    assert result.cli == 'gemini'
    assert result.no_skills is True


def test_agent_mode_registered_in_app():
    async def _test():
        app = SymfluenceTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('7')
            await pilot.pause()
            return app.current_mode

    assert asyncio.run(_test()) == 'agent'
