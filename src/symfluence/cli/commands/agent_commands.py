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
    try:
        text = skill_md.read_text(encoding='utf-8')
        _, frontmatter, _ = text.split('---', 2)
        import yaml
        meta = yaml.safe_load(frontmatter)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


class AgentCommands(BaseCommand):
    """Handlers for the `symfluence agent` commands."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence agent launch [PROMPT] [--cli NAME] [--no-skills]

        Launch an installed coding-agent CLI in the current project, primed as
        the SYMFLUENCE agent. With no prompt, an interactive session starts;
        with a prompt, the agent runs it once and exits.
        """
        from symfluence.agent import launch_agent

        return launch_agent(
            prompt=BaseCommand.get_arg(args, 'prompt', None),
            extra_args=BaseCommand.get_arg(args, 'extra', None),
            cli=BaseCommand.get_arg(args, 'cli', None),
            no_skills=BaseCommand.get_arg(args, 'no_skills', False),
        )

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
        from symfluence.agent.launcher import resolve_active

        console = BaseCommand._console
        failures = 0

        def check(ok: bool, label: str, detail: str, warn_only: bool = False) -> None:
            nonlocal failures
            if ok:
                console.indent(f"✓ {label}: {detail}")
            elif warn_only:
                console.indent(f"! {label}: {detail}")
            else:
                failures += 1
                console.indent(f"✗ {label}: {detail}")

        console.info("SYMFLUENCE agent doctor")
        console.rule()

        active = resolve_active()
        check(active is not None, "agent CLI",
              f"would launch '{active.name}'" if active else "none found on PATH")

        if active:
            key_ok = not active.env_keys or any(os.environ.get(k) for k in active.env_keys)
            check(key_ok, "API key",
                  f"one of {', '.join(active.env_keys)} is set" if key_ok and active.env_keys
                  else f"none of {', '.join(active.env_keys)} set (saved login may still work)",
                  warn_only=True)

        try:
            from symfluence.resources import get_skills_dir
            skills_dir = get_skills_dir()
            n = sum(1 for s in skills_dir.iterdir() if (s / 'SKILL.md').is_file())
            check(n > 0, "skills", f"{n} packaged skill(s)")
        except FileNotFoundError as e:
            check(False, "skills", str(e))

        try:
            from symfluence.resources import get_agents_dir
            agents_dir = get_agents_dir()
            n = len(list(agents_dir.glob('*.md')))
            check(n > 0, "subagents", f"{n} packaged definition(s)")
        except FileNotFoundError as e:
            check(False, "subagents", str(e))

        try:
            from symfluence.resources import agent_cache_root
            cache = agent_cache_root()
            cache.mkdir(parents=True, exist_ok=True)
            probe = cache / '.doctor-probe'
            probe.write_text('ok', encoding='utf-8')
            probe.unlink()
            check(True, "cache dir", str(cache))
        except OSError as e:
            check(False, "cache dir", f"not writable: {e}")

        try:
            from symfluence.agent.mcp_server import TOOLS, handle_message
            response = handle_message(
                {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}}
            )
            ok = bool(response and 'result' in response)
            check(ok, "MCP server", f"{len(TOOLS)} tool(s) available" if ok else "initialize failed")
        except Exception as e:
            check(False, "MCP server", f"{type(e).__name__}: {e}")

        from symfluence.agent.context import detect_project_context
        context = detect_project_context(Path.cwd())
        n_configs, n_domains = len(context['configs']), len(context['domains'])
        check(True, "project context",
              f"{n_configs} config(s), {n_domains} domain dir(s) detected here")

        console.rule()
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
