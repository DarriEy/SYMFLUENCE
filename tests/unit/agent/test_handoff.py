# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the TUI → exec handoff contract."""
from __future__ import annotations

from symfluence.agent import handoff


def test_non_handoff_results_are_ignored():
    assert handoff.complete_handoff(None) is None
    assert handoff.complete_handoff('quit') is None
    assert handoff.complete_handoff(42) is None


def test_handoff_maps_to_launch_agent(monkeypatch):
    from symfluence.agent.modes import AgentMode

    calls = {}

    def fake_launch(prompt=None, extra_args=None, cli=None, no_skills=False,
                    mode=AgentMode.CODING):
        calls.update(prompt=prompt, extra_args=extra_args, cli=cli,
                     no_skills=no_skills, mode=mode)
        return 0

    import symfluence.agent.launcher as launcher
    monkeypatch.setattr(launcher, 'launch_agent', fake_launch)

    request = handoff.AgentHandoff(
        cli='codex', prompt='do it', no_skills=True, mode=AgentMode.MODELLING,
    )
    assert handoff.complete_handoff(request) == 0
    assert calls == {
        'prompt': 'do it', 'extra_args': [], 'cli': 'codex', 'no_skills': True,
        'mode': AgentMode.MODELLING,
    }


def test_handoff_mode_defaults_to_coding():
    from symfluence.agent.modes import AgentMode

    assert handoff.AgentHandoff().mode is AgentMode.CODING
