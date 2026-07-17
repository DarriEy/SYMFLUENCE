# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the agent mode profiles."""
from __future__ import annotations

import pytest

from symfluence.agent.modes import PROFILES, AgentMode, get_profile


def test_both_modes_have_profiles():
    assert set(PROFILES) == {AgentMode.MODELLING, AgentMode.CODING}


def test_get_profile_accepts_enum_and_string():
    assert get_profile(AgentMode.MODELLING) is PROFILES[AgentMode.MODELLING]
    assert get_profile('model') is PROFILES[AgentMode.MODELLING]
    assert get_profile('code') is PROFILES[AgentMode.CODING]


def test_get_profile_rejects_unknown_mode():
    with pytest.raises(ValueError):
        get_profile('debug')


def test_modelling_profile_is_operational_subset():
    profile = get_profile(AgentMode.MODELLING)
    assert profile.skills == (
        'explore-platform', 'run-workflow-locally', 'debug-calibration',
    )
    assert profile.subagents == ('calibration-debugger', 'platform-scout')
    assert profile.prefers_native_chat is True
    # Headless sessions must not write files directly.
    assert 'Write' in profile.disallowed_tools
    assert 'Edit' in profile.disallowed_tools
    assert any(t.startswith('mcp__symfluence__') for t in profile.allowed_tools)


def test_coding_profile_is_unrestricted():
    profile = get_profile(AgentMode.CODING)
    assert profile.skills is None       # all packaged skills
    assert profile.subagents is None    # all packaged subagents
    assert profile.mcp_tools is None    # all MCP tools
    assert profile.prefers_native_chat is False
    assert profile.disallowed_tools == ()


def test_modelling_skills_exist_in_package():
    """Profile skill names must match packaged skill directories."""
    from symfluence.resources import get_skills_dir

    packaged = {p.name for p in get_skills_dir().iterdir()
                if (p / 'SKILL.md').is_file()}
    profile = get_profile(AgentMode.MODELLING)
    assert set(profile.skills) <= packaged


def test_modelling_subagents_exist_in_package():
    from symfluence.resources import get_agents_dir

    packaged = {p.stem for p in get_agents_dir().glob('*.md')}
    profile = get_profile(AgentMode.MODELLING)
    assert set(profile.subagents) <= packaged
