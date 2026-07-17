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
from dataclasses import dataclass, field
from pathlib import Path

from .context import build_identity_prompt
from .modes import AgentMode, get_profile


def _fill(template: tuple[str, ...], **values: str) -> list[str]:
    """Substitute placeholder values into an argv template."""
    return [part.format(**values) for part in template]


def _parse_agent_file(path: Path) -> tuple[str, dict] | None:
    """Parse one packaged agent definition (frontmatter + prompt body)."""
    from symfluence.resources import parse_frontmatter

    parsed = parse_frontmatter(path)
    if parsed is None:
        return None
    meta, body = parsed
    if 'description' not in meta:
        return None
    name = meta.get('name', path.stem)
    return name, {'description': str(meta['description']).strip(), 'prompt': body}


def build_agents_json(subagents: tuple[str, ...] | None = None) -> str | None:
    """Render the packaged subagent definitions as a CLI ``--agents`` JSON object.

    ``subagents`` restricts the set to the named definitions; None means all.
    """
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
            if subagents is None or name in subagents:
                agents[name] = spec
    return json.dumps(agents) if agents else None


def symfluence_invocation() -> list[str]:
    """argv prefix that reaches the symfluence CLI from this environment.

    Single source for every place that shells out to (or registers) the
    symfluence CLI, so the MCP registration and any subprocess it later runs
    resolve to the same installation.
    """
    binary = shutil.which('symfluence')
    if binary:
        return [binary]
    return [sys.executable, '-m', 'symfluence']


def mcp_server_command(mode: AgentMode | None = None) -> tuple[str, list[str]]:
    """The (command, args) that start the SYMFLUENCE MCP server on stdio.

    With ``mode``, the server is started under that mode's tool profile.
    """
    command, *rest = symfluence_invocation()
    args = [*rest, 'agent', 'mcp']
    if mode is not None:
        args += ['--mode', mode.value]
    return command, args


def write_mcp_config(cache_root: Path, mode: AgentMode | None = None) -> Path:
    """Write the MCP config file that points a host CLI at ``symfluence agent mcp``."""
    command, args = mcp_server_command(mode)
    config = {'mcpServers': {'symfluence': {'command': command, 'args': args}}}
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / 'mcp-config.json'
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return path


@dataclass
class PrimingReport:
    """What priming actually accomplished — the launcher renders this honestly.

    Attributes:
        argv: Extra argv inserted after the CLI binary.
        notes: Informational lines (debug-level).
        warnings: Degradations the user should see (warning-level).
        layers: Which priming layers are genuinely active for this launch
            (``skills`` / ``identity`` / ``mcp`` / ``subagents``).
    """

    argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    layers: dict[str, bool] = field(
        default_factory=lambda: {
            'skills': False, 'identity': False, 'mcp': False, 'subagents': False,
        }
    )


def prime_launch(
    launcher,
    workdir: Path,
    no_skills: bool = False,
    mode: AgentMode = AgentMode.CODING,
) -> PrimingReport:
    """Build the extra argv that primes ``launcher`` as the SYMFLUENCE agent.

    Args:
        launcher: The resolved :class:`~symfluence.agent.registry.AgentLauncher`.
        workdir: Directory the CLI is launched from.
        no_skills: Skip all priming (also honoured via ``SYMFLUENCE_NO_SKILLS``).
        mode: Which :class:`~symfluence.agent.modes.AgentMode` profile shapes
            the skills, identity, MCP profile, and subagents.

    Returns:
        A :class:`PrimingReport`; every degradation is recorded in
        ``warnings``/``layers`` rather than silently swallowed.
    """
    report = PrimingReport()
    if no_skills or os.environ.get('SYMFLUENCE_NO_SKILLS'):
        report.notes.append("Agent priming disabled (SYMFLUENCE_NO_SKILLS / --no-skills).")
        return report

    profile = get_profile(mode)
    identity = build_identity_prompt(workdir, mode)
    identity_in_agents_md = not launcher.system_prompt_args

    # 1. Skills. For CLIs without a system-prompt flag the identity block rides
    #    along as the AGENTS.md preamble. Any filesystem failure (missing
    #    package data, unwritable shared cache, read-only workdir) degrades to
    #    a warning — it must never abort the launch.
    try:
        from symfluence.resources import prepare_agent_context
        preamble = identity if identity_in_agents_md else None
        skills_argv, skills_messages, delivered = prepare_agent_context(
            launcher.skills_mode, workdir, preamble=preamble,
            skills=profile.skills, cache_scope=mode.value,
        )
        report.argv += skills_argv
        report.layers['skills'] = delivered
        if identity_in_agents_md:
            report.layers['identity'] = delivered
        (report.notes if delivered else report.warnings).extend(skills_messages)
    except OSError as e:
        report.warnings.append(f"Continuing without SYMFLUENCE skills: {e}")

    # 2. Identity via the CLI's own system-prompt flag.
    if launcher.system_prompt_args:
        report.argv += _fill(launcher.system_prompt_args, prompt=identity)
        report.layers['identity'] = True
        report.notes.append("Injected SYMFLUENCE agent identity and project context.")

    # 3. The SYMFLUENCE MCP server. json.dumps doubles as TOML string/array
    #    quoting for CLIs that take dotted config overrides.
    if launcher.mcp_config_args:
        try:
            from symfluence.resources import agent_cache_root
            mcp_path = write_mcp_config(agent_cache_root() / mode.value, mode)
            command, args = mcp_server_command(mode)
            report.argv += _fill(
                launcher.mcp_config_args,
                path=str(mcp_path),
                command_toml=json.dumps(command),
                args_toml=json.dumps(args),
            )
            report.layers['mcp'] = True
            report.notes.append(
                "Wired in the SYMFLUENCE MCP server (registry/workflow tools).")
        except OSError as e:
            report.warnings.append(f"Continuing without the SYMFLUENCE MCP server: {e}")
    else:
        command, args = mcp_server_command(mode)
        report.notes.append(
            f"'{launcher.name}' cannot load MCP servers per-launch; to register "
            f"the SYMFLUENCE MCP tools once, add an MCP server named 'symfluence' "
            f"running: {command} {' '.join(args)}"
        )

    # 4. Specialist subagents.
    if launcher.agents_args:
        agents_json = build_agents_json(profile.subagents)
        if agents_json:
            report.argv += _fill(launcher.agents_args, json=agents_json)
            report.layers['subagents'] = True
            names = ', '.join(sorted(json.loads(agents_json)))
            report.notes.append(f"Registered SYMFLUENCE subagents: {names}.")

    return report
