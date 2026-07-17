# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Build the SYMFLUENCE agent identity and live project context.

``symfluence agent launch`` injects this into the host coding-agent CLI (as an
appended system prompt for Claude Code, or as the preamble of the generated
``AGENTS.md`` for other CLIs) so the session behaves as *the SYMFLUENCE agent*
rather than a bare coding assistant: it knows what platform it is embedded in,
what project it was launched into, and the house rules for operating it.

Context detection is deliberately cheap and read-only: small YAML files are
parsed, directories are globbed, nothing is imported from the heavy scientific
stack.
"""
from __future__ import annotations

import os
from pathlib import Path

# Config keys surfaced in the context block, in display order.
_CONTEXT_KEYS = (
    'DOMAIN_NAME', 'EXPERIMENT_ID', 'HYDROLOGICAL_MODEL', 'FORCING_DATASET',
)

# Never parse YAML files larger than this (config files are a few KB).
_MAX_CONFIG_BYTES = 1_000_000

# At most this many configs are parsed / listed in the context block.
_MAX_CONFIGS = 5


def _find_value(data, key: str, depth: int = 0):
    """Find ``key`` in a possibly nested config mapping (flat keys win)."""
    if not isinstance(data, dict) or depth > 3:
        return None
    if key in data and not isinstance(data[key], dict):
        return data[key]
    for value in data.values():
        if isinstance(value, dict):
            found = _find_value(value, key, depth + 1)
            if found is not None:
                return found
    return None


def _summarize_config(path: Path) -> dict[str, str] | None:
    """Return the interesting keys of one SYMFLUENCE config, or None if not one."""
    try:
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            return None
        import yaml
        data = yaml.safe_load(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:  # noqa: BLE001 — unreadable / not YAML: not a config, not an error
        return None
    if not isinstance(data, dict):
        return None
    summary = {
        key: str(value)
        for key in _CONTEXT_KEYS
        if (value := _find_value(data, key)) is not None
    }
    # A SYMFLUENCE config must identify at least a domain or a model.
    if not ('DOMAIN_NAME' in summary or 'HYDROLOGICAL_MODEL' in summary):
        return None
    return summary


def _candidate_configs(workdir: Path) -> list[Path]:
    """YAML files worth sniffing, most likely first, bounded in number."""
    candidates: list[Path] = []
    default = os.environ.get('SYMFLUENCE_DEFAULT_CONFIG')
    if default:
        candidates.append(Path(default))
    candidates.extend(sorted(workdir.glob('*.y*ml')))
    seen: set[Path] = set()
    unique = []
    for path in candidates:
        resolved = path if path.is_absolute() else workdir / path
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique[:_MAX_CONFIGS * 4]  # sniffing budget; _MAX_CONFIGS survive


def detect_project_context(workdir: Path) -> dict:
    """Detect SYMFLUENCE project state in ``workdir`` (configs, domain dirs)."""
    configs: list[tuple[Path, dict[str, str]]] = []
    for path in _candidate_configs(workdir):
        summary = _summarize_config(path)
        if summary is not None:
            configs.append((path, summary))
            if len(configs) >= _MAX_CONFIGS:
                break

    try:
        domains = sorted(
            p.name for p in workdir.iterdir()
            if p.is_dir() and p.name.startswith('domain_')
        )
    except OSError:
        domains = []

    return {'configs': configs, 'domains': domains}


def _render_context(workdir: Path, context: dict) -> list[str]:
    """Render the detected project state as prompt lines."""
    lines = [f"Working directory: {workdir}"]

    configs = context['configs']
    if configs:
        lines.append("SYMFLUENCE config files detected here:")
        for path, summary in configs:
            try:
                shown = path.relative_to(workdir)
            except ValueError:
                shown = path
            details = ", ".join(f"{k}={v}" for k, v in summary.items())
            lines.append(f"  - {shown}" + (f" ({details})" if details else ""))
    else:
        lines.append(
            "No SYMFLUENCE config file detected in this directory. To start an "
            "experiment, create one (see `symfluence list templates`) or ask the "
            "user where their config lives."
        )

    if context['domains']:
        lines.append(
            "Domain data directories here: " + ", ".join(context['domains'][:10])
        )
    return lines


_IDENTITY = """\
This session was started by `symfluence agent` and is primed with context for \
SYMFLUENCE, an open-source platform for hydrological model comparison, \
calibration, and evaluation. The work is practical: setting up experiments, \
running workflows, debugging calibrations, and extending the platform. Keep \
communication plain and technical — no persona, no self-promotion. If asked \
who or what you are, state it briefly and factually: a coding-agent session \
primed with SYMFLUENCE skills, project context, and tools."""

_HOUSE_RULES = """\
House rules for operating SYMFLUENCE:
- Discover capabilities by querying the live registry (`symfluence list`, or the \
`symfluence` MCP tools when available) — never from memory; catalogs go stale.
- Drive experiments through the `symfluence workflow` CLI (run/step/status/\
validate); do not re-implement pipeline steps by hand.
- Before any non-trivial platform task, read the matching SYMFLUENCE skill \
(explore-platform, run-workflow-locally, debug-calibration, add-model-handler, \
add-data-handler, add-optimizer).
- Model runs and calibrations can take minutes to hours: validate the config \
first, and prefer single steps over full re-runs when iterating."""


def build_identity_prompt(workdir: Path) -> str:
    """Build the complete identity + context + rules block for ``workdir``."""
    context = detect_project_context(workdir)
    parts = [
        _IDENTITY,
        "",
        "Project context (detected at launch):",
        *_render_context(workdir, context),
        "",
        _HOUSE_RULES,
    ]
    return "\n".join(parts)
