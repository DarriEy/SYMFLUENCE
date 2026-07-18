# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the headless Claude driver (stream-JSON parsing, no network)."""
from __future__ import annotations

import asyncio
import json

import pytest

from symfluence.agent import headless, registry
from symfluence.agent.modes import AgentMode

CLAUDE = registry.get('claude')


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')  # keep argv small in tests
    monkeypatch.delenv('SYMFLUENCE_AGENT_CLI', raising=False)


# ---------------------------------------------------------------- parsing

def test_parse_init_event():
    line = json.dumps({
        'type': 'system', 'subtype': 'init', 'session_id': 'sid-1',
        'tools': ['Read', 'mcp__symfluence__validate_config'],
        'mcp_servers': [{'name': 'symfluence', 'status': 'connected'}],
    })
    [event] = headless.parse_stream_event(line)
    assert isinstance(event, headless.SessionInit)
    assert event.session_id == 'sid-1'
    assert 'Read' in event.tools
    assert event.mcp_servers == ('symfluence',)


def test_parse_text_delta_and_assistant_blocks():
    delta = json.dumps({
        'type': 'stream_event',
        'event': {'type': 'content_block_delta',
                  'delta': {'type': 'text_delta', 'text': 'Hel'}},
    })
    [event] = headless.parse_stream_event(delta)
    assert event == headless.TextDelta('Hel')

    assistant = json.dumps({
        'type': 'assistant',
        'message': {'content': [
            {'type': 'text', 'text': 'Validating now.'},
            {'type': 'tool_use', 'id': 't1',
             'name': 'mcp__symfluence__validate_config',
             'input': {'config_path': 'c.yaml'}},
        ]},
    })
    text, tool = headless.parse_stream_event(assistant)
    assert text == headless.AssistantText('Validating now.')
    assert isinstance(tool, headless.ToolCallStarted)
    assert tool.name.endswith('validate_config')
    assert tool.arguments == {'config_path': 'c.yaml'}


def test_parse_tool_result_and_turn_result():
    user = json.dumps({
        'type': 'user',
        'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 't1', 'is_error': False,
             'content': [{'type': 'text', 'text': '{"valid": true}'}]},
        ]},
    })
    [finished] = headless.parse_stream_event(user)
    assert isinstance(finished, headless.ToolCallFinished)
    assert finished.tool_id == 't1'
    assert '"valid"' in finished.output

    result = json.dumps({
        'type': 'result', 'is_error': False, 'result': 'All good.',
        'session_id': 'sid-1', 'duration_ms': 1500, 'total_cost_usd': 0.01,
    })
    [turn] = headless.parse_stream_event(result)
    assert isinstance(turn, headless.TurnResult)
    assert turn.text == 'All good.'
    assert turn.duration_s == pytest.approx(1.5)


def test_parser_tolerates_noise():
    assert headless.parse_stream_event('') == []
    assert headless.parse_stream_event('{"type": "brand_new_event"}') == []
    assert headless.parse_stream_event('[1, 2]') == []
    [err] = headless.parse_stream_event('not json at all')
    assert isinstance(err, headless.DriverError)


# ------------------------------------------------------------- session ids

def test_session_persistence_roundtrip(tmp_path):
    workdir = tmp_path / 'proj'
    assert headless.load_session_id(workdir, AgentMode.MODELLING) is None
    headless.save_session_id(workdir, AgentMode.MODELLING, 'sid-42')
    assert headless.load_session_id(workdir, AgentMode.MODELLING) == 'sid-42'
    assert headless.load_session_id(workdir, AgentMode.CODING) is None


# ----------------------------------------------------------------- driver

def test_driver_rejects_non_headless_launcher(tmp_path):
    with pytest.raises(ValueError, match='headlessly'):
        headless.HeadlessClaudeDriver(registry.get('codex'), tmp_path)


def test_build_turn_argv_carries_streaming_and_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    driver = headless.HeadlessClaudeDriver(CLAUDE, tmp_path)
    driver.session_id = 'sid-9'
    argv = driver.build_turn_argv('validate my config')

    assert argv[0] == 'claude'
    assert 'validate my config' in argv
    for flag in ('--output-format', '--include-partial-messages', '--verbose',
                 '--resume', '--allowedTools', '--disallowedTools'):
        assert flag in argv, f'missing {flag}'
    assert argv[argv.index('--resume') + 1] == 'sid-9'
    allowed = argv[argv.index('--allowedTools') + 1]
    assert 'mcp__symfluence__*' in allowed
    disallowed = argv[argv.index('--disallowedTools') + 1]
    assert 'Write' in disallowed
    # Modelling priming rode along.
    assert '--append-system-prompt' in argv
    assert '--mcp-config' in argv


def test_interactive_approvals_swap_denials_for_prompt_tool(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    prompting = headless.HeadlessClaudeDriver(
        CLAUDE, tmp_path, interactive_approvals=True)
    argv = prompting.build_turn_argv('hi')
    assert '--permission-prompt-tool' in argv
    assert argv[argv.index('--permission-prompt-tool') + 1] == \
        'mcp__symfluence__approve_action'
    assert '--disallowedTools' not in argv     # denials become prompts

    denying = headless.HeadlessClaudeDriver(CLAUDE, tmp_path)
    argv = denying.build_turn_argv('hi')
    assert '--permission-prompt-tool' not in argv
    assert '--disallowedTools' in argv         # no UI watching: hard denials


def _fake_claude(tmp_path, body: str) -> registry.AgentLauncher:
    """A stand-in 'claude' binary that prints canned stream-JSON."""
    import sys

    # Run the script through the interpreter rather than a shebang: Windows
    # cannot exec shebang scripts, and POSIX shebang lines cannot carry
    # interpreter paths with spaces. The autouse SYMFLUENCE_NO_SKILLS=1 keeps
    # priming argv empty, so nothing lands between sys.executable and the
    # script path.
    script = tmp_path / 'fake-claude.py'
    script.write_text(body, encoding='utf-8')
    launcher = registry.AgentLauncher(
        name='claude', binary=str(script), env_keys=(),
        skills_mode='claude_native',
        oneshot=(sys.executable, str(script), '-p', '{prompt}'),
        system_prompt_args=('--append-system-prompt', '{prompt}'),
        supports_headless=True,
    )
    return launcher


def test_run_turn_streams_events_and_saves_session(tmp_path):
    events_ndjson = [
        {'type': 'system', 'subtype': 'init', 'session_id': 'sid-real',
         'tools': [], 'mcp_servers': []},
        {'type': 'assistant',
         'message': {'content': [{'type': 'text', 'text': 'Done.'}]}},
        {'type': 'result', 'is_error': False, 'result': 'Done.',
         'session_id': 'sid-real', 'duration_ms': 10},
    ]
    body = "import json\n" + "\n".join(
        f"print(json.dumps({event!r}))" for event in events_ndjson
    )
    launcher = _fake_claude(tmp_path, body)
    driver = headless.HeadlessClaudeDriver(launcher, tmp_path)

    async def _collect():
        return [event async for event in driver.run_turn('hi')]

    events = asyncio.run(_collect())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ['SessionInit', 'AssistantText', 'TurnResult']
    assert driver.session_id == 'sid-real'
    # Persisted for the next TUI start.
    assert headless.load_session_id(tmp_path, AgentMode.MODELLING) == 'sid-real'


def test_run_turn_reports_bad_exit_as_driver_error(tmp_path):
    launcher = _fake_claude(
        tmp_path, "import sys; sys.stderr.write('boom'); sys.exit(2)")
    driver = headless.HeadlessClaudeDriver(launcher, tmp_path)

    async def _collect():
        return [event async for event in driver.run_turn('hi')]

    events = asyncio.run(_collect())
    assert len(events) == 1
    assert isinstance(events[0], headless.DriverError)
    assert 'boom' in events[0].message


def test_interrupt_kills_running_turn(tmp_path):
    launcher = _fake_claude(tmp_path, "import time\ntime.sleep(60)")
    driver = headless.HeadlessClaudeDriver(launcher, tmp_path)

    async def _run():
        gen = driver.run_turn('hi')
        task = asyncio.ensure_future(anext(gen, None))
        await asyncio.sleep(0.3)  # let the subprocess spawn
        assert driver.turn_running
        assert driver.interrupt() is True
        await asyncio.wait_for(task, timeout=10)
        # Drain remaining events; generator must finish quickly after the kill.
        async for _ in gen:
            pass

    asyncio.run(_run())
    assert driver.turn_running is False
