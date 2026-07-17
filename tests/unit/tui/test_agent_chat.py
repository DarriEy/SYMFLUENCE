# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the modelling chat screen (fake claude binary, no network)."""
from __future__ import annotations

import asyncio
import json
import stat

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")

from textual.widgets import Input  # noqa: E402

from symfluence.agent import registry  # noqa: E402
from symfluence.tui.app import SymfluenceTUI  # noqa: E402
from symfluence.tui.screens.agent_chat import (  # noqa: E402
    AgentChatScreen,
    ChatMessage,
    ToolCallCard,
)
from symfluence.tui.services.run_monitor import RunMonitor  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')
    monkeypatch.delenv('SYMFLUENCE_AGENT_CLI', raising=False)


def _fake_claude(tmp_path, events) -> registry.AgentLauncher:
    body = "import json\n" + "\n".join(
        f"print(json.dumps({event!r}))" for event in events
    )
    script = tmp_path / 'fake-claude'
    script.write_text(f"#!/usr/bin/env python3\n{body}", encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return registry.AgentLauncher(
        name='claude', binary=str(script), env_keys=(),
        skills_mode='claude_native', oneshot=(str(script), '-p', '{prompt}'),
        system_prompt_args=('--append-system-prompt', '{prompt}'),
        supports_headless=True,
    )


_TURN = [
    {'type': 'system', 'subtype': 'init', 'session_id': 'sid-chat',
     'tools': [], 'mcp_servers': []},
    {'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 't1',
         'name': 'mcp__symfluence__validate_config',
         'input': {'config_path': 'config.yaml'}}]}},
    {'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 't1', 'is_error': False,
         'content': [{'type': 'text', 'text': '{"valid": true}'}]}]}},
    {'type': 'assistant', 'message': {'content': [
        {'type': 'text', 'text': 'Your config is valid.'}]}},
    {'type': 'result', 'is_error': False, 'result': 'Your config is valid.',
     'session_id': 'sid-chat', 'duration_ms': 900},
]


def test_chat_turn_renders_messages_and_tool_cards(tmp_path):
    launcher = _fake_claude(tmp_path, _TURN)

    async def _test():
        app = SymfluenceTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = AgentChatScreen(launcher, workdir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()

            chat_input = screen.query_one("#chat-input", Input)
            chat_input.value = "validate my config"
            await pilot.press('enter')

            for _ in range(100):  # wait for the turn worker to finish
                await pilot.pause(0.05)
                if not screen._session.busy and not chat_input.disabled:
                    break

            messages = [m.renderable for m in screen.query(ChatMessage)]
            cards = list(screen.query(ToolCallCard))
            return str(messages), cards, screen._session.session_id

    messages, cards, session_id = asyncio.run(_test())
    assert 'validate my config' in messages         # user message
    assert 'Your config is valid.' in messages      # assistant message
    assert len(cards) == 1
    assert cards[0].tool_name == 'validate_config'
    assert cards[0].done and not cards[0].failed
    assert '"valid"' in cards[0].output
    assert session_id == 'sid-chat'


def test_chat_escape_backs_out_when_idle(tmp_path):
    launcher = _fake_claude(tmp_path, _TURN)

    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(AgentChatScreen(launcher, workdir=tmp_path))
            await pilot.pause()
            opened = type(app.screen).__name__
            await pilot.press('escape')
            await pilot.pause()
            return opened, type(app.screen).__name__

    opened, closed = asyncio.run(_test())
    assert opened == 'AgentChatScreen'
    assert closed == 'AgentHomeScreen'


def test_run_monitor_handles_missing_everything(tmp_path):
    status = RunMonitor(None).poll()
    assert status.config_name is None
    assert status.calibration is None
    assert status.jobs == []

    config = tmp_path / 'c.yaml'
    config.write_text(
        f"DOMAIN_NAME: X\nSYMFLUENCE_DATA_DIR: {tmp_path}\n", encoding='utf-8')
    status = RunMonitor(config).poll()
    assert status.config_name == 'c.yaml'
    assert status.domain == 'X'
    assert status.calibration is None  # no optimization dir yet — degrades quietly
    assert status.last_log_line is None
