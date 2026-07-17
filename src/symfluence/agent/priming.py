# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Assemble everything that primes a host CLI as *the SYMFLUENCE agent*.

Four ingredients, each degrading gracefully if unavailable and each wired only
into CLIs whose :class:`~symfluence.agent.registry.AgentLauncher` declares the
matching flag template:

1. Skills — the packaged domain guides (``.claude/skills`` dir or ``AGENTS.md``).
2. Identity — the system-prompt block with live project context.
3. MCP — the SYMFLUENCE MCP server (structured registry/workflow access).
4. Subagents — packaged specialist definitions (calibration-debugger, ...).

Setting ``SYMFLUENCE_NO_SKILLS`` (or passing ``--no-skills``) skips all priming
and hands off to the bare CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .context import build_identity_prompt


def _fill(template: tuple[str, ...], **values: str) -> list[str]:
    """Substitute placeholder values into an argv template."""
    return [part.format(**values) for part in template]


def _parse_agent_file(path: Path) -> tuple[str, dict] | None:
    """Parse one packaged agent definition (frontmatter + prompt body)."""
    try:
        text = path.read_text(encoding='utf-8')
        _, frontmatter, body = text.split('---', 2)
        import yaml
        meta = yaml.safe_load(frontmatter)
    except Exception:  # malformed definition — skip it, don't break the launch
        return None
    if not isinstance(meta, dict) or 'description' not in meta:
        return None
    name = meta.get('name', path.stem)
    return name, {'description': str(meta['description']).strip(), 'prompt': body.strip()}


def build_agents_json() -> str | None:
    """Render the packaged subagent definitions as a CLI ``--agents`` JSON object."""
    from symfluence.resources import get_agents_dir

    try:
        agents_dir = get_agents_dir()
    except FileNotFoundError:
        return None
    agents = {}
    for path in sorted(agents_dir.glob('*.md')):
        parsed = _parse_agent_file(path)
        if parsed:
            name, spec = parsed
            agents[name] = spec
    return json.dumps(agents) if agents else None


def mcp_server_command() -> tuple[str, list[str]]:
    """The (command, args) that start the SYMFLUENCE MCP server on stdio."""
    binary = shutil.which('symfluence')
    if binary:
        return binary, ['agent', 'mcp']
    return sys.executable, ['-m', 'symfluence', 'agent', 'mcp']


def write_mcp_config(cache_root: Path) -> Path:
    """Write the MCP config file that points a host CLI at ``symfluence agent mcp``."""
    command, args = mcp_server_command()
    config = {'mcpServers': {'symfluence': {'command': command, 'args': args}}}
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / 'mcp-config.json'
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return path


def prime_launch(launcher, workdir: Path, no_skills: bool = False) -> tuple[list[str], list[str]]:
    """Build the extra argv that primes ``launcher`` as the SYMFLUENCE agent.

    Args:
        launcher: The resolved :class:`~symfluence.agent.registry.AgentLauncher`.
        workdir: Directory the CLI is launched from.
        no_skills: Skip all priming (also honoured via ``SYMFLUENCE_NO_SKILLS``).

    Returns:
        ``(extra_argv, messages)`` — argv inserted after the CLI binary, and
        info lines for the caller to log.
    """
    if no_skills or os.environ.get('SYMFLUENCE_NO_SKILLS'):
        return [], ["Agent priming disabled (SYMFLUENCE_NO_SKILLS / --no-skills)."]

    argv: list[str] = []
    messages: list[str] = []

    identity = build_identity_prompt(workdir)

    # 1. Skills. For CLIs without a system-prompt flag the identity block rides
    #    along as the AGENTS.md preamble.
    try:
        from symfluence.resources import prepare_agent_context
        preamble = None if launcher.system_prompt_args else identity
        skills_argv, skills_messages = prepare_agent_context(
            launcher.skills_mode, workdir, preamble=preamble,
        )
        argv += skills_argv
        messages += skills_messages
    except FileNotFoundError as e:
        messages.append(f"Continuing without SYMFLUENCE skills: {e}")

    # 2. Identity via the CLI's own system-prompt flag.
    if launcher.system_prompt_args:
        argv += _fill(launcher.system_prompt_args, prompt=identity)
        messages.append("Injected SYMFLUENCE agent identity and project context.")

    # 3. The SYMFLUENCE MCP server. json.dumps doubles as TOML string/array
    #    quoting for CLIs that take dotted config overrides.
    if launcher.mcp_config_args:
        try:
            from symfluence.resources import agent_cache_root
            mcp_path = write_mcp_config(agent_cache_root())
            command, args = mcp_server_command()
            argv += _fill(
                launcher.mcp_config_args,
                path=str(mcp_path),
                command_toml=json.dumps(command),
                args_toml=json.dumps(args),
            )
            messages.append("Wired in the SYMFLUENCE MCP server (registry/workflow tools).")
        except OSError as e:
            messages.append(f"Continuing without the SYMFLUENCE MCP server: {e}")
    else:
        command, args = mcp_server_command()
        messages.append(
            f"'{launcher.name}' cannot load MCP servers per-launch; to register "
            f"the SYMFLUENCE MCP tools once, add an MCP server named 'symfluence' "
            f"running: {command} {' '.join(args)}"
        )

    # 4. Specialist subagents.
    if launcher.agents_args:
        agents_json = build_agents_json()
        if agents_json:
            argv += _fill(launcher.agents_args, json=agents_json)
            names = ', '.join(sorted(json.loads(agents_json)))
            messages.append(f"Registered SYMFLUENCE subagents: {names}.")

    return argv, messages
