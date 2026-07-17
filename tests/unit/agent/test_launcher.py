# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the `symfluence agent` launcher (CLI detection + handoff).

No real agent binary is launched: ``shutil.which`` is faked and ``os.execvp``
is captured.
"""
from __future__ import annotations

import pytest

from symfluence.agent import launcher
from symfluence.cli.exit_codes import ExitCode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate from the host environment: no override, no keys, no skill writes."""
    monkeypatch.delenv('SYMFLUENCE_AGENT_CLI', raising=False)
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')  # don't touch the filesystem
    for key in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        monkeypatch.delenv(key, raising=False)


def _fake_which(present):
    """Return a ``shutil.which`` stub where only *present* binaries resolve."""
    present = set(present)
    return lambda binary: f'/usr/bin/{binary}' if binary in present else None


def _capture_execvp(monkeypatch):
    """Replace ``os.execvp`` with a recorder so the test process is not replaced."""
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(launcher.os, 'execvp', lambda file, args: calls.append((file, list(args))))
    return calls


def test_no_cli_returns_dependency_error(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which([]))
    calls = _capture_execvp(monkeypatch)
    assert launcher.launch_agent() == ExitCode.DEPENDENCY_ERROR
    assert calls == []


def test_priority_prefers_claude(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude', 'codex', 'gemini']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent()
    assert calls[0][1][0] == 'claude'


def test_priority_falls_through_to_codex(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['codex', 'gemini']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent()
    assert calls[0][1][0] == 'codex'


def test_override_wins_over_priority(monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_AGENT_CLI', 'gemini')
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude', 'gemini']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent()
    assert calls[0][1][0] == 'gemini'


def test_override_not_on_path_errors(monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_AGENT_CLI', 'codex')
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    assert launcher.launch_agent() == ExitCode.DEPENDENCY_ERROR
    assert calls == []


def test_oneshot_argv_claude(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(prompt='do the thing')
    file, argv = calls[0]
    assert file == 'claude'
    assert argv[0] == 'claude'
    assert argv[1] == '-p'  # NO_SKILLS so nothing injected before the flags
    assert argv[-1] == 'do the thing'


def test_oneshot_argv_codex(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['codex']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(prompt='hello')
    _, argv = calls[0]
    assert argv[:2] == ['codex', 'exec']
    assert argv[-1] == 'hello'


def test_interactive_has_no_prompt(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent()
    _, argv = calls[0]
    assert argv == ['claude']


def test_missing_key_warns_but_still_launches(monkeypatch):
    # No ANTHROPIC_API_KEY set (cleaned by fixture); launch should still proceed.
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent()
    assert calls and calls[0][1][0] == 'claude'


def test_extra_args_forwarded(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(extra_args=['--model', 'sonnet'])
    _, argv = calls[0]
    assert argv[-2:] == ['--model', 'sonnet']


def test_cli_param_overrides_detection(monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude', 'gemini']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(cli='gemini')
    assert calls[0][1][0] == 'gemini'


def test_cli_param_wins_over_env_override(monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_AGENT_CLI', 'claude')
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude', 'codex']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(cli='codex')
    assert calls[0][1][0] == 'codex'


def test_no_skills_param_launches_bare_cli(monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)
    launcher.launch_agent(no_skills=True)
    assert calls[0][1] == ['claude']


def test_primed_claude_launch_wires_full_context(monkeypatch, tmp_path):
    """With priming enabled, the exec argv carries all four context layers."""
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.shutil, 'which', _fake_which(['claude']))
    calls = _capture_execvp(monkeypatch)

    launcher.launch_agent()

    _, argv = calls[0]
    for flag in ('--add-dir', '--append-system-prompt', '--mcp-config', '--agents'):
        assert flag in argv, f"missing {flag} in primed argv"
