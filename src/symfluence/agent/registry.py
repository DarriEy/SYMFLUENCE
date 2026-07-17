# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Registry of external coding-agent CLIs that ``symfluence agent launch`` drives.

SYMFLUENCE does not ship its own LLM agent. Instead it hands off to an installed,
provider-agnostic coding-agent CLI (Claude Code, Codex, Gemini, ...), primed with
the SYMFLUENCE skills. Each supported CLI is described by an :class:`AgentLauncher`.
Built-ins are registered in detection-priority order; third parties may register
more via :func:`register`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentLauncher:
    """Describes how to launch one external coding-agent CLI.

    Attributes:
        name: Stable identifier (also the ``SYMFLUENCE_AGENT_CLI`` value).
        binary: Executable looked up on ``PATH`` via ``shutil.which``.
        env_keys: API-key env vars that pair naturally with this CLI. Used only to
            emit a "no key set" warning — keys are never read or forwarded by
            SYMFLUENCE; the child CLI reads its own environment.
        skills_mode: How the CLI consumes the SYMFLUENCE skills
            (``"claude_native"`` or ``"agents_md"``); see
            :func:`symfluence.resources.prepare_agent_context`.
        oneshot: argv template for a single non-interactive prompt, with a
            ``"{prompt}"`` placeholder (e.g. ``("claude", "-p", "{prompt}")``).
        system_prompt_args: argv template (``"{prompt}"`` placeholder) that
            appends a system prompt, e.g. ``("--append-system-prompt", "{prompt}")``.
            Empty when the CLI has no such flag — the identity block is then
            delivered as the ``AGENTS.md`` preamble instead.
        mcp_config_args: argv template registering the SYMFLUENCE MCP server.
            Placeholders: ``{path}`` (a written MCP JSON config file),
            ``{command_toml}`` / ``{args_toml}`` (TOML-quoted server command and
            argument list, for CLIs that take dotted config overrides). Empty
            when the CLI cannot be wired per-invocation; the launcher then just
            prints how to register ``symfluence agent mcp`` manually.
        agents_args: argv template (``"{json}"`` placeholder) that defines
            custom subagents. Empty when unsupported.
        supports_headless: Whether this CLI can be driven headlessly over a
            structured streaming protocol (the native modelling-chat screen
            requires it; other CLIs fall back to a regular handoff).
    """

    name: str
    binary: str
    env_keys: tuple[str, ...]
    skills_mode: str
    oneshot: tuple[str, ...]
    system_prompt_args: tuple[str, ...] = ()
    mcp_config_args: tuple[str, ...] = ()
    agents_args: tuple[str, ...] = ()
    supports_headless: bool = False

    def interactive_argv(self) -> list[str]:
        """argv for an interactive session."""
        return [self.binary]

    def oneshot_argv(self, prompt: str) -> list[str]:
        """argv for a single non-interactive prompt."""
        return [part.format(prompt=prompt) for part in self.oneshot]


# Built-in launchers, in detection-priority order.
_BUILTINS: tuple[AgentLauncher, ...] = (
    AgentLauncher(
        name='claude', binary='claude', env_keys=('ANTHROPIC_API_KEY',),
        skills_mode='claude_native', oneshot=('claude', '-p', '{prompt}'),
        system_prompt_args=('--append-system-prompt', '{prompt}'),
        mcp_config_args=('--mcp-config', '{path}'),
        agents_args=('--agents', '{json}'),
        supports_headless=True,
    ),
    AgentLauncher(
        name='codex', binary='codex', env_keys=('OPENAI_API_KEY',),
        skills_mode='agents_md', oneshot=('codex', 'exec', '{prompt}'),
        mcp_config_args=(
            '-c', 'mcp_servers.symfluence.command={command_toml}',
            '-c', 'mcp_servers.symfluence.args={args_toml}',
        ),
    ),
    AgentLauncher(
        name='gemini', binary='gemini', env_keys=('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
        skills_mode='agents_md', oneshot=('gemini', '-p', '{prompt}'),
    ),
)

_REGISTRY: dict[str, AgentLauncher] = {}


def register(launcher: AgentLauncher) -> None:
    """Register (or override) a launcher by name."""
    _REGISTRY[launcher.name] = launcher


def get(name: str) -> AgentLauncher | None:
    """Return the launcher registered under ``name``, or ``None``."""
    return _REGISTRY.get(name)


def all_launchers() -> list[AgentLauncher]:
    """All registered launchers, in registration (= priority) order."""
    return list(_REGISTRY.values())


def generic(name: str) -> AgentLauncher:
    """Build a best-effort launcher for an unregistered CLI named ``name``.

    Used when ``SYMFLUENCE_AGENT_CLI`` points at a CLI we don't know about: assume
    it reads ``AGENTS.md`` (the emerging cross-tool convention) and takes the prompt
    as a trailing positional argument.
    """
    return AgentLauncher(
        name=name, binary=name, env_keys=(), skills_mode='agents_md',
        oneshot=(name, '{prompt}'),
    )


for _builtin in _BUILTINS:
    register(_builtin)
