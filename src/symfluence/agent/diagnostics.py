# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Agent-setup diagnostics shared by ``agent doctor`` and the TUI preflight panel.

Each check is returned as data so callers can render it their own way (console
symbols, TUI rows). Checks are cheap and read-only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

OK = 'ok'
WARN = 'warn'
FAIL = 'fail'


@dataclass(frozen=True)
class Check:
    """One diagnostic result."""

    status: str  # OK | WARN | FAIL
    label: str
    detail: str


def run_diagnostics(workdir: Path) -> list[Check]:
    """Run all agent-setup checks for ``workdir`` and return them as data."""
    from .launcher import resolve_active

    checks: list[Check] = []

    active = resolve_active()
    checks.append(Check(
        OK if active else FAIL, 'agent CLI',
        f"would launch '{active.name}'" if active else 'none found on PATH',
    ))

    if active:
        key_ok = not active.env_keys or any(os.environ.get(k) for k in active.env_keys)
        keys = ', '.join(active.env_keys)
        checks.append(Check(
            OK if key_ok else WARN, 'API key',
            f"one of {keys} is set" if key_ok and active.env_keys
            else f"none of {keys} set (saved login may still work)",
        ))
        checks.append(Check(
            OK if active.supports_headless else WARN, 'native chat',
            f"'{active.name}' supports headless driving"
            if active.supports_headless
            else f"'{active.name}' cannot be driven headlessly; modelling "
                 f"sessions hand off to its own UI",
        ))

    try:
        from symfluence.resources import get_skills_dir
        skills_dir = get_skills_dir()
        n = sum(1 for s in skills_dir.iterdir() if (s / 'SKILL.md').is_file())
        checks.append(Check(OK if n else FAIL, 'skills', f"{n} packaged skill(s)"))
    except FileNotFoundError as e:
        checks.append(Check(FAIL, 'skills', str(e)))

    try:
        from symfluence.resources import get_agents_dir
        agents_dir = get_agents_dir()
        n = len(list(agents_dir.glob('*.md')))
        checks.append(Check(OK if n else FAIL, 'subagents', f"{n} packaged definition(s)"))
    except FileNotFoundError as e:
        checks.append(Check(FAIL, 'subagents', str(e)))

    try:
        from symfluence.resources import agent_cache_root
        cache = agent_cache_root()
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / '.doctor-probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        checks.append(Check(OK, 'cache dir', str(cache)))
    except OSError as e:
        checks.append(Check(FAIL, 'cache dir', f"not writable: {e}"))

    try:
        from .mcp_server import TOOLS, handle_message
        response = handle_message(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}}
        )
        ok = bool(response and 'result' in response)
        checks.append(Check(
            OK if ok else FAIL, 'MCP server',
            f"{len(TOOLS)} tool(s) available" if ok else 'initialize failed',
        ))
    except Exception as e:  # noqa: BLE001 — a diagnostic must report, not raise
        checks.append(Check(FAIL, 'MCP server', f"{type(e).__name__}: {e}"))

    from .context import detect_project_context
    context = detect_project_context(workdir)
    n_configs, n_domains = len(context['configs']), len(context['domains'])
    checks.append(Check(
        OK, 'project context',
        f"{n_configs} config(s), {n_domains} domain dir(s) detected here",
    ))

    return checks
