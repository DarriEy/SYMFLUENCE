# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the agent verb routing (TUI vs direct, per-mode)."""
from __future__ import annotations

from argparse import Namespace

import pytest

from symfluence.agent.handoff import AgentHandoff
from symfluence.agent.modes import AgentMode
from symfluence.cli.commands.agent_commands import AgentCommands


@pytest.fixture
def spies(monkeypatch):
    """Spy on both destinations: direct launch_agent and the TUI."""
    record = {'direct': None, 'tui': None}

    def fake_launch_agent(prompt=None, extra_args=None, cli=None,
                          no_skills=False, mode=AgentMode.CODING):
        record['direct'] = {
            'prompt': prompt, 'extra_args': extra_args,
            'cli': cli, 'no_skills': no_skills, 'mode': mode,
        }
        return 0

    def fake_launch_tui(**kwargs):
        record['tui'] = kwargs
        return AgentHandoff(cli='claude')

    import symfluence.agent as agent_pkg
    import symfluence.agent.launcher as launcher
    monkeypatch.setattr(agent_pkg, 'launch_agent', fake_launch_agent)
    monkeypatch.setattr(launcher, 'launch_agent', fake_launch_agent)

    import symfluence.tui as tui_pkg
    monkeypatch.setattr(tui_pkg, 'launch_tui', fake_launch_tui)

    return record


def _args(**kwargs) -> Namespace:
    defaults = {
        'prompt': None, 'cli': None, 'no_skills': False,
        'direct': False, 'extra': None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _fake_tty(monkeypatch, value: bool) -> None:
    import sys
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: value, raising=False)
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: value, raising=False)


def test_code_direct_flag_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.code(_args(direct=True))
    assert spies['direct']['mode'] is AgentMode.CODING
    assert spies['tui'] is None


def test_code_oneshot_prompt_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.code(_args(prompt='do the thing'))
    assert spies['direct']['prompt'] == 'do the thing'
    assert spies['tui'] is None


def test_code_no_tty_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, False)
    AgentCommands.code(_args())
    assert spies['direct'] is not None
    assert spies['tui'] is None


def _fake_textual_available(monkeypatch, available: bool) -> None:
    """Pin the textual-availability probe so routing tests are CI-independent."""
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'textual':
            return object() if available else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr('importlib.util.find_spec', fake_find_spec)


def test_code_missing_textual_falls_back_to_direct(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    _fake_textual_available(monkeypatch, False)
    AgentCommands.code(_args())
    assert spies['direct'] is not None
    assert spies['tui'] is None


def test_code_interactive_tty_opens_agent_screen(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    _fake_textual_available(monkeypatch, True)
    AgentCommands.code(_args(cli='codex', no_skills=True, extra=['--resume']))

    # TUI opened on the agent screen with the presets forwarded ...
    assert spies['tui']['initial_mode'] == 'agent'
    assert spies['tui']['agent_defaults'] == {
        'cli': 'codex', 'no_skills': True, 'extra_args': ['--resume'],
    }
    # ... and its handoff result was completed via launch_agent.
    assert spies['direct']['cli'] == 'claude'


def test_model_launches_directly_with_modelling_mode(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.model(_args(prompt='validate my config'))
    assert spies['direct']['mode'] is AgentMode.MODELLING
    assert spies['direct']['prompt'] == 'validate my config'
    assert spies['tui'] is None


def test_model_interactive_launches_directly(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.model(_args())
    assert spies['direct']['mode'] is AgentMode.MODELLING
    assert spies['tui'] is None


def test_launch_is_deprecated_alias_for_code(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.launch(_args(direct=True))
    assert spies['direct']['mode'] is AgentMode.CODING


def test_home_opens_tui_on_tty(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    _fake_textual_available(monkeypatch, True)
    AgentCommands.home(_args())
    assert spies['tui']['initial_mode'] == 'agent'


def test_home_without_tty_prints_guidance(monkeypatch, spies):
    _fake_tty(monkeypatch, False)
    assert AgentCommands.home(_args()) == 0
    assert spies['tui'] is None
    assert spies['direct'] is None  # never auto-launches a session
