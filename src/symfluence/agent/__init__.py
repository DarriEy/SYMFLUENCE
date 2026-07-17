# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SYMFLUENCE agent interface.

``symfluence agent model`` / ``symfluence agent code`` hand off to an installed
coding-agent CLI (Claude Code, Codex, Gemini, ...), primed as *the SYMFLUENCE
agent* under that mode's profile: packaged skills, an identity block with live
project context, the SYMFLUENCE MCP server (``symfluence agent mcp``), and
specialist subagents. See :mod:`symfluence.agent.modes`,
:mod:`symfluence.agent.launcher`, :mod:`symfluence.agent.priming`,
:mod:`symfluence.agent.context`, :mod:`symfluence.agent.mcp_server`, and
:mod:`symfluence.agent.registry`.
"""
from __future__ import annotations

from .launcher import build_launch_argv, launch_agent, resolve_active
from .modes import AgentMode, ModeProfile, get_profile
from .registry import AgentLauncher, all_launchers, register

__all__ = [
    'AgentLauncher',
    'AgentMode',
    'ModeProfile',
    'all_launchers',
    'build_launch_argv',
    'get_profile',
    'launch_agent',
    'register',
    'resolve_active',
]
