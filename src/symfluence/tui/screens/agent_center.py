# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Agent Command Center — the launch deck for the SYMFLUENCE agent.

Shows the detected runtimes (Claude Code, Codex, Gemini, ...), the mission
context (configs and domains detected in the working directory), the
capabilities that will prime the session (skills, subagents, MCP tools), and a
preflight check panel. Launching never happens inside the TUI: the screen exits
the app with an :class:`~symfluence.agent.handoff.AgentHandoff`, and the CLI
command layer performs the exec once the terminal is restored.
"""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from symfluence.agent.diagnostics import FAIL, OK
from symfluence.agent.handoff import AgentHandoff

from ..services.agent_service import AgentService, AgentSnapshot
from .path_prompt import PathPromptScreen

_STATUS_GLYPH = {OK: ('[#43d6b5]✓[/]', ''), FAIL: ('[red]✗[/]', 'red')}


class AgentCommandCenterScreen(Screen):
    """Command center for configuring and launching the SYMFLUENCE agent."""

    BINDINGS = [
        Binding("l", "launch", "Launch"),
        Binding("p", "oneshot", "One-shot Prompt"),
        Binding("k", "toggle_priming", "Toggle Priming"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._service = AgentService()
        self._snapshot: AgentSnapshot | None = None
        self._selected_runtime: str | None = None
        self._no_skills = False
        self._extra_args: list[str] = []

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Horizontal(
                Static("Runtime: -", id="agent-stat-runtime"),
                Static("Project: -", id="agent-stat-project"),
                Static("Priming: -", id="agent-stat-priming"),
                Static("Status: -", id="agent-stat-ready"),
                classes="stats-bar",
            ),
            Horizontal(
                Vertical(
                    Static("Runtime", classes="section-header"),
                    OptionList(id="agent-runtime-list", classes="agent-panel"),
                    Static("Preflight", classes="section-header"),
                    Static("", id="agent-preflight", classes="agent-panel"),
                    id="agent-left",
                ),
                Vertical(
                    Static("Mission Context", classes="section-header"),
                    Static("", id="agent-context", classes="agent-panel"),
                    Static("Capabilities", classes="section-header"),
                    Static("", id="agent-capabilities", classes="agent-panel"),
                    id="agent-right",
                ),
                id="agent-grid",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        defaults = getattr(self.app, 'agent_defaults', None) or {}
        if defaults.get('cli'):
            self._selected_runtime = defaults['cli']
        self._no_skills = bool(defaults.get('no_skills'))
        self._extra_args = list(defaults.get('extra_args') or [])
        self._refresh()

    def on_screen_resume(self) -> None:
        self._refresh()

    # ------------------------------------------------------------ rendering

    def _refresh(self) -> None:
        snapshot = self._service.snapshot(Path.cwd())
        self._snapshot = snapshot

        if self._selected_runtime is None and snapshot.default_runtime:
            self._selected_runtime = snapshot.default_runtime.name

        self._render_stats(snapshot)
        self._render_runtimes(snapshot)
        self._render_context(snapshot)
        self._render_capabilities(snapshot)
        self._render_preflight(snapshot)

    def _render_stats(self, snapshot: AgentSnapshot) -> None:
        runtime = self._selected_runtime or '-'
        self.query_one("#agent-stat-runtime", Static).update(f"Runtime: {runtime}")
        self.query_one("#agent-stat-project", Static).update(
            f"Project: {snapshot.workdir.name or snapshot.workdir}"
        )
        priming = "[yellow]off[/]" if self._no_skills else "[#43d6b5]full[/]"
        self.query_one("#agent-stat-priming", Static).update(f"Priming: {priming}")
        ready = "[#43d6b5]ready[/]" if snapshot.ready else "[red]blocked[/]"
        self.query_one("#agent-stat-ready", Static).update(f"Status: {ready}")
        self.sub_title = str(snapshot.workdir)

    def _render_runtimes(self, snapshot: AgentSnapshot) -> None:
        option_list = self.query_one("#agent-runtime-list", OptionList)
        option_list.clear_options()
        highlight = 0
        for i, runtime in enumerate(snapshot.runtimes):
            if runtime.installed:
                key = "key set" if runtime.key_set else "no key (saved login?)"
                line = (
                    f"[b]{runtime.name}[/b]"
                    f"{'  [dim]· default[/dim]' if runtime.is_default else ''}\n"
                    f"  [dim]{runtime.path} · {key}[/dim]"
                )
            else:
                line = f"[dim]{runtime.name}\n  not installed[/dim]"
            option_list.add_option(
                Option(line, id=runtime.name, disabled=not runtime.installed)
            )
            if runtime.name == self._selected_runtime:
                highlight = i
        if snapshot.runtimes:
            option_list.highlighted = highlight

    def _render_context(self, snapshot: AgentSnapshot) -> None:
        lines: list[str] = []
        if snapshot.configs:
            lines.append("[b]Configs[/b]")
            for shown, summary in snapshot.configs:
                lines.append(f"  {shown}")
                details = "  ·  ".join(f"{k} [b]{v}[/b]" for k, v in summary.items())
                if details:
                    lines.append(f"    [dim]{details}[/dim]")
        else:
            lines.append("[dim]No SYMFLUENCE config detected in this directory.[/dim]")
            lines.append("[dim]The agent can help create one from a template.[/dim]")
        if snapshot.domains:
            lines.append("")
            lines.append("[b]Domains[/b]")
            for domain in snapshot.domains[:8]:
                lines.append(f"  {domain}")
            if len(snapshot.domains) > 8:
                lines.append(f"  [dim]… and {len(snapshot.domains) - 8} more[/dim]")
        self.query_one("#agent-context", Static).update("\n".join(lines))

    def _render_capabilities(self, snapshot: AgentSnapshot) -> None:
        lines = [f"[b]Skills[/b] [dim]({len(snapshot.skills)})[/dim]"]
        lines += [f"  {name}" for name, _ in snapshot.skills]
        lines.append("")
        lines.append(f"[b]Subagents[/b] [dim]({len(snapshot.subagents)})[/dim]")
        lines += [f"  {name}" for name, _ in snapshot.subagents]
        lines.append("")
        lines.append(f"[b]MCP tools[/b] [dim]({len(snapshot.mcp_tools)})[/dim]")
        lines += [f"  {name}" for name, _ in snapshot.mcp_tools]
        if self._no_skills:
            lines.append("")
            lines.append("[yellow]Priming disabled — the CLI launches bare.[/yellow]")
        self.query_one("#agent-capabilities", Static).update("\n".join(lines))

    def _render_preflight(self, snapshot: AgentSnapshot) -> None:
        lines = []
        for check in snapshot.checks:
            glyph, _ = _STATUS_GLYPH.get(check.status, ('[yellow]![/]', ''))
            lines.append(f"{glyph} {check.label}: [dim]{check.detail}[/dim]")
        self.query_one("#agent-preflight", Static).update("\n".join(lines))

    # -------------------------------------------------------------- actions

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "agent-runtime-list" and event.option.id:
            self._selected_runtime = event.option.id
            if self._snapshot:
                self._render_stats(self._snapshot)

    def _handoff(self, prompt: str | None = None) -> None:
        snapshot = self._snapshot
        if not snapshot or not any(r.installed for r in snapshot.runtimes):
            self.app.notify(
                "No coding-agent CLI installed. See `symfluence agent doctor`.",
                severity="error",
            )
            return
        selected = next(
            (r for r in snapshot.runtimes if r.name == self._selected_runtime), None)
        if selected is not None and not selected.installed:
            self.app.notify(
                f"'{selected.name}' is not installed — pick an installed runtime.",
                severity="error",
            )
            return
        self.app.exit(
            AgentHandoff(
                cli=self._selected_runtime,
                prompt=prompt,
                no_skills=self._no_skills,
                extra_args=list(self._extra_args),
            )
        )

    def action_launch(self) -> None:
        self._handoff()

    def action_oneshot(self) -> None:
        self.app.push_screen(
            PathPromptScreen(
                title="One-shot Prompt",
                prompt_text="The agent runs this once and exits:",
                placeholder="e.g. validate my config and run the model_run step",
            ),
            self._launch_oneshot,
        )

    def _launch_oneshot(self, prompt: str | None) -> None:
        if prompt:
            self._handoff(prompt=prompt)

    def action_toggle_priming(self) -> None:
        self._no_skills = not self._no_skills
        if self._snapshot:
            self._render_stats(self._snapshot)
            self._render_capabilities(self._snapshot)

    def action_refresh(self) -> None:
        self._refresh()
