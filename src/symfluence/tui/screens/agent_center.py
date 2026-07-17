# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Agent home — pick a modelling or coding session and go.

One decision per screen: two mode cards (Model / Code) over two dim context
lines (detected config, runtime readiness). Diagnostics live behind a details
modal (``d``), not on the home screen.

Coding sessions round-trip: the TUI suspends, the real coding-agent CLI runs
full-screen in the same terminal, and the home screen returns when it exits —
never leave the app. Where the terminal cannot suspend (headless drivers,
textual-web) the screen falls back to the classic
:class:`~symfluence.agent.handoff.AgentHandoff` exec after the TUI exits.
Modelling sessions use the same round-trip with modelling priming until the
native chat screen lands.
"""
from __future__ import annotations

import subprocess  # nosec B404 — argv from build_launch_argv, no shell
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from symfluence.agent.diagnostics import FAIL, OK
from symfluence.agent.handoff import AgentHandoff
from symfluence.agent.modes import AgentMode, get_profile

from ..services.agent_service import AgentService, AgentSnapshot

_STATUS_GLYPH = {OK: '[#43d6b5]✓[/]', FAIL: '[red]✗[/]'}


class AgentDetailsScreen(ModalScreen):
    """Runtimes and preflight checks, demoted from the home screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Close"),
        Binding("d", "app.pop_screen", "Close"),
    ]

    def __init__(self, snapshot: AgentSnapshot, **kwargs):
        super().__init__(**kwargs)
        self._snapshot = snapshot

    def compose(self) -> ComposeResult:
        lines = ["[b]Runtimes[/b]"]
        for runtime in self._snapshot.runtimes:
            if runtime.installed:
                key = 'key set' if runtime.key_set else 'no key (saved login?)'
                default = '  [dim]· default[/dim]' if runtime.is_default else ''
                lines.append(f"  [b]{runtime.name}[/b]{default}")
                lines.append(f"    [dim]{runtime.path} · {key}[/dim]")
            else:
                lines.append(f"  [dim]{runtime.name} — not installed[/dim]")
        lines.append("")
        lines.append("[b]Preflight[/b]")
        for check in self._snapshot.checks:
            glyph = _STATUS_GLYPH.get(check.status, '[yellow]![/]')
            lines.append(f"  {glyph} {check.label}: [dim]{check.detail}[/dim]")
        yield VerticalScroll(
            Static("\n".join(lines), id="agent-details-body"),
            id="agent-details",
        )


class AgentHomeScreen(Screen):
    """Home screen for the SYMFLUENCE agent: pick a mode, start a session."""

    BINDINGS = [
        Binding("m", "start_model", "Model"),
        Binding("c", "start_code", "Code"),
        Binding("d", "details", "Details"),
        Binding("g", "cycle_config", "Config", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._service = AgentService()
        self._snapshot: AgentSnapshot | None = None
        self._cli: str | None = None
        self._no_skills = False
        self._extra_args: list[str] = []
        self._config_index = 0

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("What do you want to do?", id="agent-home-title"),
            OptionList(id="agent-mode-list"),
            Static("", id="agent-config-line"),
            Static("", id="agent-ready-line"),
            id="agent-home",
        )
        yield Footer()

    def on_mount(self) -> None:
        defaults = getattr(self.app, 'agent_defaults', None) or {}
        self._cli = defaults.get('cli')
        self._no_skills = bool(defaults.get('no_skills'))
        self._extra_args = list(defaults.get('extra_args') or [])

        mode_list = self.query_one("#agent-mode-list", OptionList)
        for mode, key in ((AgentMode.MODELLING, 'm'), (AgentMode.CODING, 'c')):
            profile = get_profile(mode)
            title = profile.title.split()[0]  # "Modelling" / "Coding"
            mode_list.add_option(Option(
                f"[b]▸ {title}[/b]  [dim]{key}[/dim]\n  [dim]{profile.tagline}[/dim]",
                id=mode.value,
            ))
        mode_list.highlighted = 0
        mode_list.focus()
        self._refresh()

    def on_screen_resume(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        snapshot = self._service.snapshot(Path.cwd())
        self._snapshot = snapshot
        self.sub_title = str(snapshot.workdir)
        self._render_context_lines(snapshot)

    def _render_context_lines(self, snapshot: AgentSnapshot) -> None:
        config_line = self.query_one("#agent-config-line", Static)
        if snapshot.configs:
            self._config_index %= len(snapshot.configs)
            shown, summary = snapshot.configs[self._config_index]
            details = " · ".join(str(v) for v in summary.values())
            more = (
                f"  [dim]({self._config_index + 1}/{len(snapshot.configs)}, g cycles)[/dim]"
                if len(snapshot.configs) > 1 else ""
            )
            config_line.update(f"{shown}[dim] · {details}[/dim]{more}")
        else:
            config_line.update(
                "[dim]No SYMFLUENCE config detected here — "
                "the agent can create one from a template.[/dim]"
            )

        ready_line = self.query_one("#agent-ready-line", Static)
        runtime = next(
            (r for r in snapshot.runtimes if r.name == self._cli), None
        ) or snapshot.default_runtime
        if runtime and runtime.installed:
            ready_line.update(
                f"[dim]{runtime.name} ready · {len(snapshot.skills)} skills · "
                f"{len(snapshot.mcp_tools)} tools[/dim]"
            )
        else:
            ready_line.update(
                "[red]No coding-agent CLI installed[/red][dim] — "
                "see `symfluence agent doctor`.[/dim]"
            )

    # -------------------------------------------------------------- actions

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "agent-mode-list" and event.option.id:
            self._start_session(AgentMode(event.option.id))

    def action_start_model(self) -> None:
        self._start_session(AgentMode.MODELLING)

    def action_start_code(self) -> None:
        self._start_session(AgentMode.CODING)

    def action_details(self) -> None:
        if self._snapshot:
            self.app.push_screen(AgentDetailsScreen(self._snapshot))

    def action_cycle_config(self) -> None:
        if self._snapshot and len(self._snapshot.configs) > 1:
            self._config_index += 1
            self._render_context_lines(self._snapshot)

    def _selected_config_path(self) -> Path | None:
        if not self._snapshot or not self._snapshot.configs:
            return None
        shown, _summary = self._snapshot.configs[
            self._config_index % len(self._snapshot.configs)]
        path = Path(shown)
        return path if path.is_absolute() else self._snapshot.workdir / path

    # ------------------------------------------------------------- sessions

    def _start_session(self, mode: AgentMode) -> None:
        snapshot = self._snapshot
        if not snapshot or not any(r.installed for r in snapshot.runtimes):
            self.app.notify(
                "No coding-agent CLI installed. See `symfluence agent doctor`.",
                severity="error",
            )
            return

        from symfluence.agent.launcher import resolve_active
        launcher = resolve_active(self._cli)
        if launcher is None:
            self.app.notify(
                f"Runtime {self._cli!r} is not available. See `symfluence agent doctor`.",
                severity="error",
            )
            return

        # Modelling gets the native chat when the runtime can stream headlessly;
        # other runtimes fall through to the suspend round-trip with modelling
        # priming.
        if mode is AgentMode.MODELLING and launcher.supports_headless:
            from .agent_chat import AgentChatScreen
            self.app.push_screen(AgentChatScreen(
                launcher,
                workdir=Path.cwd(),
                config_path=self._selected_config_path(),
            ))
            return

        if not self._suspend_session(launcher, mode):
            # Terminal can't suspend: classic handoff — the TUI exits and the
            # CLI command layer completes the exec.
            self.app.exit(AgentHandoff(
                cli=self._cli,
                no_skills=self._no_skills,
                extra_args=self._extra_args,
                mode=mode,
            ))

    def _suspend_session(self, launcher, mode: AgentMode) -> bool:
        """Run the session as a suspend round-trip. False = use exec handoff."""
        from textual.app import SuspendNotSupported

        from symfluence.agent import build_launch_argv

        # Probe before priming: no point assembling argv (which materializes
        # skills) when the driver cannot suspend and the exec path will prime
        # again anyway.
        driver = getattr(self.app, '_driver', None)
        if driver is None or not getattr(driver, 'can_suspend', False):
            return False

        workdir = Path.cwd()
        argv, _report = build_launch_argv(
            launcher, workdir,
            extra_args=self._extra_args,
            no_skills=self._no_skills,
            mode=mode,
        )
        try:
            with self.app.suspend():
                code = subprocess.call(argv, cwd=str(workdir))  # nosec B603
        except SuspendNotSupported:
            return False
        self._refresh()
        profile = get_profile(mode)
        if code == 0:
            self.app.notify(f"{profile.title} ended.")
        else:
            self.app.notify(
                f"{profile.title} exited with code {code}.", severity="warning",
            )
        return True
