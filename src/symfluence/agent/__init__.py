# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SYMFLUENCE agent interface.

``symfluence agent launch`` is a thin wrapper that hands off to an installed
coding-agent CLI (Claude Code, Codex, Gemini, ...), primed with the SYMFLUENCE
skills. See :mod:`symfluence.agent.launcher` and :mod:`symfluence.agent.registry`.
"""
from __future__ import annotations

from .launcher import launch_agent
from .registry import AgentLauncher, all_launchers, register

__all__ = [
    'launch_agent',
    'AgentLauncher',
    'all_launchers',
    'register',
]
