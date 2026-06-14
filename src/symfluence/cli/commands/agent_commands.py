# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""AI agent command handlers for SYMFLUENCE CLI.

`symfluence agent launch` hands off to an installed coding-agent CLI (Claude Code,
Codex, Gemini, ...), primed with the SYMFLUENCE skills. The legacy `start`/`run`
verbs are deprecated aliases for `launch`.
"""
from __future__ import annotations

from argparse import Namespace

from .base import BaseCommand, cli_exception_handler


class AgentCommands(BaseCommand):
    """Handlers for the `symfluence agent` commands."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence agent launch [PROMPT]

        Launch an installed coding-agent CLI in the current project, primed with
        the SYMFLUENCE skills. With no prompt, an interactive session starts; with
        a prompt, the agent runs it once and exits.
        """
        from symfluence.agent import launch_agent

        return launch_agent(
            prompt=BaseCommand.get_arg(args, 'prompt', None),
            extra_args=BaseCommand.get_arg(args, 'extra', None),
        )

    @staticmethod
    @cli_exception_handler
    def start(args: Namespace) -> int:
        """Deprecated alias for `agent launch` (interactive)."""
        BaseCommand._console.warning(
            "`symfluence agent start` is deprecated; use `symfluence agent launch`."
        )
        from symfluence.agent import launch_agent

        return launch_agent(prompt=None, extra_args=BaseCommand.get_arg(args, 'extra', None))

    @staticmethod
    @cli_exception_handler
    def run(args: Namespace) -> int:
        """Deprecated alias for `agent launch PROMPT` (one-shot)."""
        BaseCommand._console.warning(
            '`symfluence agent run` is deprecated; use `symfluence agent launch "..."`.'
        )
        from symfluence.agent import launch_agent

        return launch_agent(
            prompt=args.prompt,
            extra_args=BaseCommand.get_arg(args, 'extra', None),
        )
