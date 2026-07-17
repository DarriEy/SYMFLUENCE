# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the `agent launch` TUI-vs-direct routing decision."""
from __future__ import annotations

from argparse import Namespace

import pytest

from symfluence.agent.handoff import AgentHandoff
from symfluence.cli.commands.agent_commands import AgentCommands


@pytest.fixture
def spies(monkeypatch):
    """Spy on both destinations: direct launch_agent and the TUI."""
    record = {'direct': None, 'tui': None}

    def fake_launch_agent(prompt=None, extra_args=None, cli=None, no_skills=False):
        record['direct'] = {
            'prompt': prompt, 'extra_args': extra_args,
            'cli': cli, 'no_skills': no_skills,
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


def test_direct_flag_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.launch(_args(direct=True))
    assert spies['direct'] is not None
    assert spies['tui'] is None


def test_oneshot_prompt_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.launch(_args(prompt='do the thing'))
    assert spies['direct']['prompt'] == 'do the thing'
    assert spies['tui'] is None


def test_no_tty_skips_tui(monkeypatch, spies):
    _fake_tty(monkeypatch, False)
    AgentCommands.launch(_args())
    assert spies['direct'] is not None
    assert spies['tui'] is None


def test_interactive_tty_opens_command_center(monkeypatch, spies):
    _fake_tty(monkeypatch, True)
    AgentCommands.launch(_args(cli='codex', no_skills=True))

    # TUI opened on the agent screen with the presets forwarded ...
    assert spies['tui']['initial_mode'] == 'agent'
    assert spies['tui']['agent_defaults'] == {'cli': 'codex', 'no_skills': True}
    # ... and its handoff result was completed via launch_agent.
    assert spies['direct']['cli'] == 'claude'
