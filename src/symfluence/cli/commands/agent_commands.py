# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""AI agent command handlers for SYMFLUENCE CLI.

`symfluence agent launch` hands off to an installed coding-agent CLI (Claude
Code, Codex, Gemini, ...), primed as the SYMFLUENCE agent. The other verbs
inspect and support that interface: `list` (detected CLIs), `skills` (packaged
domain guides), `doctor` (diagnose the setup), and `mcp` (serve the SYMFLUENCE
MCP server on stdio for the host CLI). The legacy `start`/`run` verbs are
deprecated aliases for `launch`.
"""
from __future__ import annotations

import os
import shutil
from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


def _skill_frontmatter(skill_md: Path) -> dict:
    """Parse a SKILL.md YAML frontmatter block, returning {} on any problem."""
    from symfluence.resources import parse_frontmatter

    parsed = parse_frontmatter(skill_md)
    return parsed[0] if parsed else {}


class AgentCommands(BaseCommand):
    """Handlers for the `symfluence agent` commands."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence agent launch [PROMPT] [--cli NAME] [--no-skills] [--direct]

        By default this opens the Agent Command Center — a dedicated screen in
        the SYMFLUENCE TUI for reviewing runtimes, project context, and
        preflight checks before handing off to the agent CLI. The handoff
        itself always happens after the TUI exits.

        Direct handoff (today's immediate exec) happens when any of these hold:
        a one-shot PROMPT is given, ``--direct`` is passed, the session has no
        TTY, or the TUI extra (textual) is not installed.
        """
        import sys

        from symfluence.agent import launch_agent

        prompt = BaseCommand.get_arg(args, 'prompt', None)
        cli = BaseCommand.get_arg(args, 'cli', None)
        no_skills = BaseCommand.get_arg(args, 'no_skills', False)
        extra = BaseCommand.get_arg(args, 'extra', None)

        def direct() -> int:
            return launch_agent(
                prompt=prompt, extra_args=extra, cli=cli, no_skills=no_skills,
            )

        if prompt or BaseCommand.get_arg(args, 'direct', False):
            return direct()
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return direct()

        # Only the availability probe may fall back to direct: an ImportError
        # raised later, while the TUI session is running, is a real error and
        # must surface — not silently exec the agent.
        import importlib.util
        if importlib.util.find_spec('textual') is None:
            BaseCommand._console.debug(
                'TUI extra not installed (pip install "symfluence[tui]"); '
                'handing off directly.'
            )
            return direct()

        from symfluence.tui import launch_tui

        result = launch_tui(
            initial_mode='agent',
            agent_defaults={
                'cli': cli, 'no_skills': no_skills, 'extra_args': list(extra or []),
            },
        )

        from symfluence.agent.handoff import complete_handoff

        code = complete_handoff(result)
        return code if code is not None else ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def list_clis(args: Namespace) -> int:
        """
        Execute: symfluence agent list

        Show the registered coding-agent CLIs, which are installed, and which
        one `agent launch` would pick.
        """
        from symfluence.agent import all_launchers
        from symfluence.agent.launcher import resolve_active

        console = BaseCommand._console
        active = resolve_active()
        override = os.environ.get('SYMFLUENCE_AGENT_CLI')

        console.info("Registered agent CLIs (detection-priority order):")
        for spec in all_launchers():
            path = shutil.which(spec.binary)
            marker = '  ← would launch' if active and spec.name == active.name else ''
            installed = path if path else 'not installed'
            keys = ', '.join(spec.env_keys) if spec.env_keys else '-'
            key_set = any(os.environ.get(k) for k in spec.env_keys)
            key_note = 'set' if key_set else 'not set'
            console.indent(f"{spec.name:<8} {installed}{marker}")
            console.indent(f"keys: {keys} ({key_note})", level=2)

        if override:
            console.info(f"SYMFLUENCE_AGENT_CLI={override} overrides detection.")
        if active is None:
            console.warning("No agent CLI found on PATH; `agent launch` would fail.")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def skills(args: Namespace) -> int:
        """
        Execute: symfluence agent skills

        List the packaged SYMFLUENCE skills exposed to the agent at launch.
        """
        from symfluence.resources import get_skills_dir

        console = BaseCommand._console
        try:
            skills_dir = get_skills_dir()
        except FileNotFoundError as e:
            console.error(str(e))
            return ExitCode.DEPENDENCY_ERROR

        console.info("Packaged SYMFLUENCE skills:")
        count = 0
        for skill in sorted(skills_dir.iterdir()):
            skill_md = skill / 'SKILL.md'
            if not skill_md.is_file():
                continue
            meta = _skill_frontmatter(skill_md)
            description = ' '.join(str(meta.get('description', '')).split())
            if len(description) > 100:
                description = description[:97] + '...'
            console.indent(f"{skill.name:<22} {description}")
            count += 1
        console.info(f"{count} skill(s). Materialized for the agent at launch.")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence agent doctor

        Diagnose the agent setup: CLI detection, API keys, packaged skills and
        subagents, cache directory, MCP server, and project context.
        """
        from symfluence.agent.diagnostics import FAIL, OK, run_diagnostics

        console = BaseCommand._console
        symbols = {OK: '✓', FAIL: '✗'}

        console.info("SYMFLUENCE agent doctor")
        console.rule()

        checks = run_diagnostics(Path.cwd())
        for check in checks:
            console.indent(f"{symbols.get(check.status, '!')} {check.label}: {check.detail}")

        console.rule()
        failures = sum(1 for c in checks if c.status == FAIL)
        if failures:
            console.error(f"{failures} check(s) failed.")
            return ExitCode.DEPENDENCY_ERROR
        console.success("Agent setup looks healthy.")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def mcp(args: Namespace) -> int:
        """
        Execute: symfluence agent mcp

        Serve the SYMFLUENCE MCP server on stdio (blocks until EOF). Host CLIs
        launched by `agent launch` connect to this automatically; it can also be
        registered manually in any MCP-capable tool.
        """
        from symfluence.agent.mcp_server import serve

        return serve()

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
