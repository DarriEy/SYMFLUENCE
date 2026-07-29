# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The two SYMFLUENCE agent modes and their priming profiles.

``symfluence agent`` offers two clearly separated experiences:

- **Modelling** (``symfluence agent model``) — a hydrologist drives experiments
  conversationally: configs, runs, calibrations, results. Primed with the
  operational skills and the MCP tools; forbidden from editing platform source.
- **Coding** (``symfluence agent code``) — extending the platform itself. Primed
  with every packaged skill and subagent; the host CLI's own permission system
  governs what it may touch.

Each mode is described by a frozen :class:`ModeProfile`, the single source of
truth for which skills, subagents, MCP tools, house rules, and headless tool
permissions that mode carries. Everything mode-aware (priming, the MCP server,
the TUI, diagnostics) resolves its behaviour from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(str, Enum):
    """The two agent experiences. Values double as CLI verb / ``--profile`` names."""

    MODELLING = 'model'
    CODING = 'code'


@dataclass(frozen=True)
class ModeProfile:
    """Everything that shapes one agent mode.

    Attributes:
        mode: The mode this profile describes.
        title: Short display name ("Modelling session").
        tagline: One calm sentence for pickers and help text.
        skills: Packaged skill names to materialize, or None for all.
        subagents: Packaged subagent names to register, or None for all.
        mcp_tools: MCP tool names the server exposes under this profile, or
            None for all registered tools.
        house_rules: Mode-specific rules appended to the identity prompt.
        allowed_tools: Host-CLI tool allowlist for headless (non-interactive)
            sessions, where no human sits behind a permission prompt. Empty
            means "no restriction is applied by SYMFLUENCE".
        disallowed_tools: Host-CLI tools explicitly denied in headless sessions.
        prefers_native_chat: Whether this mode should use the native chat
            screen when the resolved CLI supports headless driving.
    """

    mode: AgentMode
    title: str
    tagline: str
    skills: tuple[str, ...] | None
    subagents: tuple[str, ...] | None
    mcp_tools: tuple[str, ...] | None
    house_rules: str
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    prefers_native_chat: bool = False


_MODELLING_HOUSE_RULES = """\
Modelling-mode rules (this session is for a hydrologist, not a developer):
- Never modify SYMFLUENCE platform source code or installed packages. If a task \
genuinely requires a code change, say so and suggest a coding session \
(`symfluence agent code`).
- Drive every experiment through the SYMFLUENCE tools: `validate_config` before \
anything else, `workflow_status` to orient, then the workflow tools to execute. \
Do not re-implement pipeline steps by hand.
- Change experiment configuration only through the SYMFLUENCE config tools (or \
by telling the user exactly which keys to change); state the exact key changes \
before making them.
- Model runs and calibrations can take minutes to hours: confirm scope with the \
user before starting one, prefer single steps over full re-runs, and report \
progress in hydrological terms (basin, model, metric), not implementation terms.
- End every piece of work with the artifacts: where results live \
(`domain_*/...`), which figures or summaries were produced, and the headline \
metrics."""

_CODING_HOUSE_RULES = """\
Coding-mode rules:
- Before any non-trivial platform task, read the matching SYMFLUENCE skill \
(explore-platform, run-workflow-locally, debug-calibration, debug-delineation, \
add-model-handler, add-data-handler, add-optimizer).
- Discover capabilities by querying the live registry (`symfluence list`, or \
the `symfluence` MCP tools) — never from memory; catalogs go stale.
- Verify platform changes by driving the workflow CLI (run/step/status/\
validate), not only unit tests."""


PROFILES: dict[AgentMode, ModeProfile] = {
    AgentMode.MODELLING: ModeProfile(
        mode=AgentMode.MODELLING,
        title='Modelling session',
        tagline='Run experiments conversationally — configs, runs, '
                'calibrations, results.',
        skills=('explore-platform', 'run-workflow-locally', 'debug-calibration'),
        subagents=('calibration-debugger', 'platform-scout'),
        mcp_tools=None,
        house_rules=_MODELLING_HOUSE_RULES,
        allowed_tools=(
            'mcp__symfluence__*', 'Read', 'Glob', 'Grep', 'Bash(symfluence *)',
        ),
        disallowed_tools=('Write', 'Edit', 'NotebookEdit'),
        prefers_native_chat=True,
    ),
    AgentMode.CODING: ModeProfile(
        mode=AgentMode.CODING,
        title='Coding session',
        tagline='Extend the platform — models, data handlers, optimizers.',
        skills=None,
        subagents=None,
        mcp_tools=None,
        house_rules=_CODING_HOUSE_RULES,
        prefers_native_chat=False,
    ),
}


def get_profile(mode: AgentMode | str) -> ModeProfile:
    """Resolve a :class:`ModeProfile` from a mode or its string value."""
    return PROFILES[AgentMode(mode)]
