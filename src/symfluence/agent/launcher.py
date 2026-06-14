# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Hand ``symfluence agent`` off to an installed coding-agent CLI.

This replaces the former in-house LLM agent. The launcher detects an installed
agent CLI (Claude Code, Codex, Gemini, ...), exposes the packaged SYMFLUENCE
skills to it, and replaces the current process with the CLI via ``os.execvp`` so
the agent owns the terminal directly.
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
        "Override detection with SYMFLUENCE_AGENT_CLI=<command>."
    )


def _resolve(console) -> registry.AgentLauncher | None:
    """Pick the launcher to use, emitting an error and returning None if none fit."""
    override = os.environ.get('SYMFLUENCE_AGENT_CLI')
    if override:
        spec = registry.get(override) or registry.generic(override)
        if shutil.which(spec.binary):
            return spec
        console.error(
            f"SYMFLUENCE_AGENT_CLI={override!r} is set but '{spec.binary}' "
            f"is not on PATH."
        )
        return None

    for spec in registry.all_launchers():
        if shutil.which(spec.binary):
            return spec

    console.error(_no_cli_message())
    return None


def launch_agent(prompt: str | None = None, extra_args: list[str] | None = None) -> int:
    """Launch the resolved agent CLI, replacing the current process on success.

    Args:
        prompt: Optional single prompt for non-interactive (one-shot) mode. When
            omitted, an interactive session is started.
        extra_args: Extra arguments forwarded verbatim to the CLI.

    Returns:
        An exit code. On success this never actually returns — ``os.execvp``
        replaces the process — so a return only happens on failure.
    """
    from symfluence.cli.console import console

    extra_args = list(extra_args or [])

    launcher = _resolve(console)
    if launcher is None:
        return int(ExitCode.DEPENDENCY_ERROR)

    if launcher.env_keys and not any(os.environ.get(k) for k in launcher.env_keys):
        console.warning(
            f"None of {', '.join(launcher.env_keys)} is set; "
            f"'{launcher.binary}' may rely on a saved login."
        )

    # Materialize the SYMFLUENCE skills for this CLI.
    try:
        from ..resources import prepare_agent_context
        ctx_args, messages = prepare_agent_context(launcher.skills_mode, Path.cwd())
        for message in messages:
            console.info(message)
    except FileNotFoundError as e:  # packaged skills missing — degrade gracefully
        console.warning(f"Continuing without SYMFLUENCE skills: {e}")
        ctx_args = []

    if prompt:
        base = launcher.oneshot_argv(prompt)
        argv = [base[0], *ctx_args, *base[1:], *extra_args]
    else:
        argv = [*launcher.interactive_argv(), *ctx_args, *extra_args]

    console.info(f"Launching {launcher.name}: {' '.join(argv)}")
    try:
        # Deliberate process handoff to the resolved agent CLI. argv is built from
        # the launcher registry (not a shell string), so there is no shell-injection
        # surface; no shell is involved.
        os.execvp(argv[0], argv)  # nosec B606
    except OSError as e:
        console.error(f"Failed to launch '{launcher.binary}': {e}")
        return int(ExitCode.BINARY_ERROR)
    return int(ExitCode.SUCCESS)  # unreachable on a successful exec
