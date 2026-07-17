# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""AI agent command handlers for SYMFLUENCE CLI.

The agent surface has two first-class modes:

- `symfluence agent model` — a modelling session: drive experiments (configs,
  runs, calibrations, results) conversationally, primed with the operational
  skills and the MCP tools.
- `symfluence agent code` — a coding session: extend the platform in a host
  coding-agent CLI (Claude Code, Codex, Gemini, ...) primed with every packaged
  skill and subagent.

Bare `symfluence agent` opens the Agent screen in the TUI to pick a mode.
`doctor` diagnoses the setup (runtimes, keys, per-mode priming, MCP server),
and `mcp` serves the SYMFLUENCE MCP server on stdio for host CLIs. The legacy
`launch` verb is a deprecated alias for `code`.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from symfluence.agent.modes import AgentMode, get_profile

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


def _direct_launch(args: Namespace, mode: AgentMode) -> int:
    """Hand off to the host CLI immediately with ``mode``'s priming."""
    from symfluence.agent import launch_agent

    return launch_agent(
        prompt=BaseCommand.get_arg(args, 'prompt', None),
        extra_args=BaseCommand.get_arg(args, 'extra', None),
        cli=BaseCommand.get_arg(args, 'cli', None),
        no_skills=BaseCommand.get_arg(args, 'no_skills', False),
        mode=mode,
    )


def _tui_available() -> bool:
    """Whether an interactive TUI session is possible right now.

    Only the availability probe may steer routing: an ImportError raised later,
    while the TUI session is running, is a real error and must surface — not
    silently fall back to a direct exec.
    """
    import importlib.util
    import sys

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if importlib.util.find_spec('textual') is None:
        BaseCommand._console.debug(
            'TUI extra not installed (pip install "symfluence[tui]"); '
            'handing off directly.'
        )
        return False
    return True


def _run_agent_tui(args: Namespace) -> int:
    """Open the TUI on the agent screen and complete any resulting handoff."""
    from symfluence.tui import launch_tui

    result = launch_tui(
        initial_mode='agent',
        agent_defaults={
            'cli': BaseCommand.get_arg(args, 'cli', None),
            'no_skills': BaseCommand.get_arg(args, 'no_skills', False),
            'extra_args': list(BaseCommand.get_arg(args, 'extra', None) or []),
        },
    )

    from symfluence.agent.handoff import complete_handoff

    code = complete_handoff(result)
    return code if code is not None else ExitCode.SUCCESS


class AgentCommands(BaseCommand):
    """Handlers for the `symfluence agent` commands."""

    @staticmethod
    @cli_exception_handler
    def home(args: Namespace) -> int:
        """
        Execute: symfluence agent

        Open the Agent screen in the TUI to pick a modelling or coding session.
        Without a TTY or the TUI extra, print how to start each mode directly.
        """
        if _tui_available():
            return _run_agent_tui(args)

        console = BaseCommand._console
        console.info("SYMFLUENCE agent — two session modes:")
        for mode in AgentMode:
            profile = get_profile(mode)
            console.indent(f"symfluence agent {mode.value:<6} {profile.tagline}")
        console.info("Run `symfluence agent doctor` to check the setup.")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def model(args: Namespace) -> int:
        """
        Execute: symfluence agent model [PROMPT] [--cli NAME] [--no-skills]

        Start a modelling session: drive experiments (configs, runs,
        calibrations, results) conversationally, with modelling priming and
        the MCP tools. With a PROMPT the session runs once and exits.
        """
        return _direct_launch(args, AgentMode.MODELLING)

    @staticmethod
    @cli_exception_handler
    def code(args: Namespace) -> int:
        """
        Execute: symfluence agent code [PROMPT] [--cli NAME] [--no-skills] [--direct]

        Start a coding session in the host coding-agent CLI, primed with every
        packaged SYMFLUENCE skill and subagent. Interactive sessions open the
        TUI agent screen first; a PROMPT, ``--direct``, a missing TTY, or a
        missing TUI extra hands off immediately.
        """
        prompt = BaseCommand.get_arg(args, 'prompt', None)
        if prompt or BaseCommand.get_arg(args, 'direct', False):
            return _direct_launch(args, AgentMode.CODING)
        if not _tui_available():
            return _direct_launch(args, AgentMode.CODING)
        return _run_agent_tui(args)

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """Deprecated alias for `agent code`."""
        BaseCommand._console.warning(
            "`symfluence agent launch` is deprecated; use `symfluence agent code` "
            "(or `symfluence agent model` for a modelling session)."
        )
        return AgentCommands.code(args)

    @staticmethod
    @cli_exception_handler
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence agent doctor [--json]

        Diagnose the agent setup: detected runtimes, API keys, per-mode
        priming, cache directory, MCP server, and project context.
        """
        import json as json_mod
        import os
        import shutil

        from symfluence.agent import all_launchers
        from symfluence.agent.diagnostics import FAIL, OK, run_diagnostics
        from symfluence.agent.launcher import resolve_active

        console = BaseCommand._console
        checks = run_diagnostics(Path.cwd())
        active = resolve_active()

        runtimes = []
        for spec in all_launchers():
            path = shutil.which(spec.binary)
            runtimes.append({
                'name': spec.name,
                'installed': bool(path),
                'path': path,
                'key_set': any(os.environ.get(k) for k in spec.env_keys),
                'headless': spec.supports_headless,
                'active': bool(active and spec.name == active.name),
            })

        failed = any(c.status == FAIL for c in checks)

        if BaseCommand.get_arg(args, 'json', False):
            payload = {
                'checks': [
                    {'status': c.status, 'label': c.label, 'detail': c.detail}
                    for c in checks
                ],
                'runtimes': runtimes,
                'modes': {
                    mode.value: {
                        'title': get_profile(mode).title,
                        'skills': list(get_profile(mode).skills or ()) or 'all',
                        'subagents': list(get_profile(mode).subagents or ()) or 'all',
                    }
                    for mode in AgentMode
                },
            }
            print(json_mod.dumps(payload, indent=2))
            return ExitCode.DEPENDENCY_ERROR if failed else ExitCode.SUCCESS

        symbols = {OK: '✓', FAIL: '✗'}
        console.info("SYMFLUENCE agent doctor")
        console.rule()

        console.info("Runtimes (detection-priority order):")
        override = os.environ.get('SYMFLUENCE_AGENT_CLI')
        for runtime in runtimes:
            marker = '  ← active' if runtime['active'] else ''
            where = runtime['path'] or 'not installed'
            extras = []
            if runtime['installed']:
                extras.append('key set' if runtime['key_set'] else 'no key (saved login?)')
                if runtime['headless']:
                    extras.append('chat-capable')
            detail = f" ({', '.join(extras)})" if extras else ''
            console.indent(f"{runtime['name']:<8} {where}{detail}{marker}")
        if override:
            console.indent(f"SYMFLUENCE_AGENT_CLI={override} overrides detection.")

        console.info("Modes:")
        for mode in AgentMode:
            profile = get_profile(mode)
            n_skills = len(profile.skills) if profile.skills else 'all'
            n_agents = len(profile.subagents) if profile.subagents else 'all'
            console.indent(
                f"{mode.value:<6} {profile.title}: {n_skills} skill(s), "
                f"{n_agents} subagent(s)"
            )

        console.info("Checks:")
        for check in checks:
            console.indent(f"{symbols.get(check.status, '!')} {check.label}: {check.detail}")

        console.rule()
        if failed:
            failures = sum(1 for c in checks if c.status == FAIL)
            console.error(f"{failures} check(s) failed.")
            return ExitCode.DEPENDENCY_ERROR
        console.success("Agent setup looks healthy.")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def mcp(args: Namespace) -> int:
        """
        Execute: symfluence agent mcp [--mode MODE]

        Serve the SYMFLUENCE MCP server on stdio (blocks until EOF). Host CLIs
        launched by the agent verbs connect to this automatically; it can also
        be registered manually in any MCP-capable tool. ``--mode`` restricts
        the tool set to one agent mode's profile.
        """
        from symfluence.agent.mcp_server import serve

        return serve(profile=BaseCommand.get_arg(args, 'mode', None))
