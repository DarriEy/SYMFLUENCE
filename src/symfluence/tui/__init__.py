# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE TUI Package.

Textual-based interactive terminal application for hydrological modeling workflows.
Provides domain browsing, run history, workflow execution, calibration monitoring,
and results comparison — all from an SSH terminal.

Install dependencies: pip install "symfluence[tui]"
Launch: symfluence tui launch
"""
from __future__ import annotations


def launch_tui(config_path=None, demo=None, initial_mode=None, agent_defaults=None):
    """
    Build and launch the SYMFLUENCE TUI as an interactive terminal application.

    Thin wrapper that imports the actual app module lazily so that
    ``from symfluence.tui import launch_tui`` succeeds even when Textual
    is not installed (the ImportError is raised only when called).

    Args:
        config_path: Optional path to a YAML config file to preload.
        demo: Optional demo name (e.g. 'bow') to load a built-in config.
        initial_mode: Optional mode to open on (e.g. 'agent'); default dashboard.
        agent_defaults: Optional presets for the Agent Command Center
            (``{'cli': ..., 'no_skills': ...}``).

    Returns:
        The app's exit result. When the Agent Command Center requests a launch
        this is an :class:`symfluence.agent.handoff.AgentHandoff` for the
        caller to complete; otherwise None.
    """
    try:
        import textual  # noqa: F401
    except ImportError:
        raise ImportError(
            "Textual is required for the SYMFLUENCE TUI.\n"
            'Install with:  pip install "symfluence[tui]"'
        ) from None

    from .app import SymfluenceTUI
    app = SymfluenceTUI(
        config_path=config_path,
        demo=demo,
        initial_mode=initial_mode,
        agent_defaults=agent_defaults,
    )
    return app.run()


__all__ = ['launch_tui']
