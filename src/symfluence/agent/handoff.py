# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The contract between the TUI Agent Command Center and the process handoff.

The TUI never execs into a host CLI itself — Textual owns the terminal while it
runs. Instead the Agent screen exits the app with an :class:`AgentHandoff`
result; after Textual has torn down and restored the terminal, the CLI command
layer passes that result to :func:`complete_handoff`, which performs the actual
``launch_agent`` exec. Keeping this a typed, one-way contract also leaves room
to embed the agent inside the TUI later (a pty-backed screen would simply stop
returning handoffs) without touching the screens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .modes import AgentMode


@dataclass(frozen=True)
class AgentHandoff:
    """A launch request produced by the TUI for the command layer to execute.

    Attributes:
        cli: Launcher name to use (None = auto-detect).
        prompt: One-shot prompt (None = interactive session).
        no_skills: Launch the bare CLI without SYMFLUENCE priming.
        extra_args: Extra argv forwarded verbatim to the host CLI.
        mode: Which agent mode's priming profile to launch with.
    """

    cli: str | None = None
    prompt: str | None = None
    no_skills: bool = False
    extra_args: list[str] = field(default_factory=list)
    mode: AgentMode = AgentMode.CODING


def complete_handoff(result: object) -> int | None:
    """Execute ``result`` if it is an :class:`AgentHandoff`.

    Returns the launch exit code (only reached on failure — success execs and
    never returns), or None when ``result`` is not a handoff request.
    """
    if not isinstance(result, AgentHandoff):
        return None

    from .launcher import launch_agent

    return launch_agent(
        prompt=result.prompt,
        extra_args=result.extra_args,
        cli=result.cli,
        no_skills=result.no_skills,
        mode=result.mode,
    )
