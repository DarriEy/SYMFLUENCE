# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Hand ``symfluence agent`` off to an installed coding-agent CLI.

This replaces the former in-house LLM agent. The launcher detects an installed
agent CLI (Claude Code, Codex, Gemini, ...), primes it as the SYMFLUENCE agent
(skills, identity + project context, MCP server, subagents — see
:mod:`symfluence.agent.priming`), and replaces the current process with the CLI
via ``os.execvp`` so the agent owns the terminal directly.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..cli.exit_codes import ExitCode
from . import registry


def _no_cli_message() -> str:
    return (
        "No coding-agent CLI found on PATH.\n"
        "`symfluence agent` launches an installed agent CLI primed with the "
        "SYMFLUENCE skills. Install one of:\n"
        "  • Claude Code:  https://docs.claude.com/claude-code   (uses ANTHROPIC_API_KEY)\n"
        "  • Codex CLI:    https://github.com/openai/codex        (uses OPENAI_API_KEY)\n"
        "  • Gemini CLI:   https://github.com/google-gemini/gemini-cli (uses GEMINI_API_KEY)\n"
        "Then set the matching API key and re-run `symfluence agent launch`.\n"
        "Override detection with `--cli <name>` or SYMFLUENCE_AGENT_CLI=<command>."
    )


def _resolve(console, cli: str | None = None) -> registry.AgentLauncher | None:
    """Pick the launcher to use, emitting an error and returning None if none fit.

    Precedence: explicit ``cli`` argument (``--cli``), then the
    ``SYMFLUENCE_AGENT_CLI`` environment variable, then detection priority.
    """
    override = cli or os.environ.get('SYMFLUENCE_AGENT_CLI')
    if override:
        spec = registry.get(override) or registry.generic(override)
        if shutil.which(spec.binary):
            return spec
        console.error(
            f"Requested agent CLI {override!r}, but '{spec.binary}' is not on PATH."
        )
        return None

    for spec in registry.all_launchers():
        if shutil.which(spec.binary):
            return spec

    console.error(_no_cli_message())
    return None


def resolve_active(cli: str | None = None) -> registry.AgentLauncher | None:
    """The launcher `agent launch` would pick right now, or None. No output."""

    class _Quiet:
        def error(self, message: str) -> None:
            pass

    return _resolve(_Quiet(), cli)


def launch_agent(
    prompt: str | None = None,
    extra_args: list[str] | None = None,
    cli: str | None = None,
    no_skills: bool = False,
) -> int:
    """Launch the resolved agent CLI, replacing the current process on success.

    Args:
        prompt: Optional single prompt for non-interactive (one-shot) mode. When
            omitted, an interactive session is started.
        extra_args: Extra arguments forwarded verbatim to the CLI.
        cli: Launch this CLI instead of the auto-detected one (``--cli``).
        no_skills: Skip all SYMFLUENCE priming and launch the bare CLI.

    Returns:
        An exit code. On success this never actually returns — ``os.execvp``
        replaces the process — so a return only happens on failure.
    """
    from symfluence.cli.console import console

    extra_args = list(extra_args or [])

    launcher = _resolve(console, cli)
    if launcher is None:
        return int(ExitCode.DEPENDENCY_ERROR)

    if launcher.env_keys and not any(os.environ.get(k) for k in launcher.env_keys):
        console.warning(
            f"None of {', '.join(launcher.env_keys)} is set; "
            f"'{launcher.binary}' may rely on a saved login."
        )

    # Prime the CLI as the SYMFLUENCE agent (skills, identity, MCP, subagents).
    from .priming import prime_launch
    workdir = Path.cwd()
    report = prime_launch(launcher, workdir, no_skills=no_skills)
    for note in report.notes:
        console.debug(note)
    for warning in report.warnings:
        console.warning(warning)

    if prompt:
        base = launcher.oneshot_argv(prompt)
        argv = [base[0], *report.argv, *base[1:], *extra_args]
    else:
        argv = [*launcher.interactive_argv(), *report.argv, *extra_args]

    from .presentation import launch_card
    console.print(launch_card(
        launcher_name=launcher.name,
        workdir=workdir,
        interactive=prompt is None,
        layers=[name for name, active in report.layers.items() if active],
    ))
    try:
        # Deliberate process handoff to the resolved agent CLI. argv is built from
        # the launcher registry (not a shell string), so there is no shell-injection
        # surface; no shell is involved.
        os.execvp(argv[0], argv)  # nosec B606
    except OSError as e:
        console.error(f"Failed to launch '{launcher.binary}': {e}")
        return int(ExitCode.BINARY_ERROR)
    return int(ExitCode.SUCCESS)  # unreachable on a successful exec
