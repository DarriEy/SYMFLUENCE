# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data provider for the Agent Command Center screen.

Aggregates everything the screen renders — registered runtimes, packaged
skills/subagents, MCP tools, detected project context, and preflight
diagnostics — into one plain snapshot object. All data comes from the
``symfluence.agent`` / ``symfluence.resources`` APIs; this module holds no
Textual imports so it stays unit-testable without the TUI extra.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeStatus:
    """One registered coding-agent CLI, as shown in the runtime selector."""

    name: str
    binary: str
    path: str | None          # resolved PATH location, None if not installed
    env_keys: tuple[str, ...]
    key_set: bool
    is_default: bool          # the one auto-detection would pick

    @property
    def installed(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class AgentSnapshot:
    """Everything the Agent Command Center renders, gathered in one pass."""

    workdir: Path
    runtimes: list[RuntimeStatus] = field(default_factory=list)
    skills: list[tuple[str, str]] = field(default_factory=list)      # (name, description)
    subagents: list[tuple[str, str]] = field(default_factory=list)   # (name, description)
    mcp_tools: list[tuple[str, str]] = field(default_factory=list)   # (name, description)
    configs: list[tuple[str, dict]] = field(default_factory=list)    # (shown path, summary)
    domains: list[str] = field(default_factory=list)
    checks: list = field(default_factory=list)                       # diagnostics.Check

    @property
    def default_runtime(self) -> RuntimeStatus | None:
        return next((r for r in self.runtimes if r.is_default), None)

    @property
    def ready(self) -> bool:
        from symfluence.agent.diagnostics import FAIL
        return any(r.installed for r in self.runtimes) and not any(
            c.status == FAIL for c in self.checks
        )


def _first_sentence(text: str, limit: int = 90) -> str:
    """Compress a long frontmatter description to one display line."""
    flat = ' '.join(str(text).split())
    for stop in ('. ', ' — '):
        if stop in flat:
            flat = flat.split(stop, 1)[0]
            break
    return flat if len(flat) <= limit else flat[:limit - 3] + '...'


def _frontmatter(md_file: Path) -> dict:
    try:
        _, frontmatter, _ = md_file.read_text(encoding='utf-8').split('---', 2)
        import yaml
        meta = yaml.safe_load(frontmatter)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


class AgentService:
    """Gathers the Agent Command Center snapshot."""

    def snapshot(self, workdir: Path | None = None) -> AgentSnapshot:
        """Collect the full snapshot for ``workdir`` (default: cwd)."""
        from symfluence.agent import all_launchers
        from symfluence.agent.context import detect_project_context
        from symfluence.agent.diagnostics import run_diagnostics
        from symfluence.agent.launcher import resolve_active
        from symfluence.agent.mcp_server import TOOLS

        workdir = workdir or Path.cwd()
        active = resolve_active()

        runtimes = [
            RuntimeStatus(
                name=spec.name,
                binary=spec.binary,
                path=shutil.which(spec.binary),
                env_keys=spec.env_keys,
                key_set=any(os.environ.get(k) for k in spec.env_keys),
                is_default=bool(active and spec.name == active.name),
            )
            for spec in all_launchers()
        ]

        context = detect_project_context(workdir)
        configs = []
        for path, summary in context['configs']:
            try:
                shown = str(path.relative_to(workdir))
            except ValueError:
                shown = str(path)
            configs.append((shown, summary))

        return AgentSnapshot(
            workdir=workdir,
            runtimes=runtimes,
            skills=self._skills(),
            subagents=self._subagents(),
            mcp_tools=[
                (name, _first_sentence(spec['description']))
                for name, spec in TOOLS.items()
            ],
            configs=configs,
            domains=list(context['domains']),
            checks=run_diagnostics(workdir),
        )

    def _skills(self) -> list[tuple[str, str]]:
        from symfluence.resources import get_skills_dir

        try:
            skills_dir = get_skills_dir()
        except FileNotFoundError:
            return []
        skills = []
        for skill in sorted(skills_dir.iterdir()):
            skill_md = skill / 'SKILL.md'
            if not skill_md.is_file():
                continue
            meta = _frontmatter(skill_md)
            skills.append((skill.name, _first_sentence(meta.get('description', ''))))
        return skills

    def _subagents(self) -> list[tuple[str, str]]:
        from symfluence.resources import get_agents_dir

        try:
            agents_dir = get_agents_dir()
        except FileNotFoundError:
            return []
        agents = []
        for path in sorted(agents_dir.glob('*.md')):
            meta = _frontmatter(path)
            if 'description' in meta:
                agents.append(
                    (meta.get('name', path.stem), _first_sentence(meta['description']))
                )
        return agents
