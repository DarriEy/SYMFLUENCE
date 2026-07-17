# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Modelling chat — a native conversational surface over a headless Claude session.

The hydrologist talks; the agent drives the platform through the SYMFLUENCE
MCP tools. Assistant prose streams into the conversation; tool invocations
render as compact cards (``▸ run_workflow_step · calibrate``) that expand on
click; a run sidebar ticks independently of the agent (fed by
:class:`~symfluence.tui.services.run_monitor.RunMonitor`), so a long-blocking
tool call never makes the screen feel dead.

No single-letter bindings here — typing owns the keyboard. ``escape``
interrupts a running turn, or backs out to the Agent home when idle.
"""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Static

from symfluence.agent.headless import (
    AssistantText,
    DriverError,
    SessionInit,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnResult,
)
from symfluence.agent.modes import AgentMode
from symfluence.agent.registry import AgentLauncher

from ..services.agent_session import AgentChatSession
from ..services.run_monitor import RunMonitor, RunStatus

_MAX_CARD_OUTPUT = 2000

_EMPTY_STATE = """\
[dim]Try:
  · Validate my config
  · What's the status of this experiment?
  · Run the next pipeline step[/dim]"""


class ChatMessage(Static):
    """One conversation entry (user or agent prose)."""


class ToolCallCard(Static):
    """A tool invocation: one collapsed line, click to expand the output."""

    def __init__(self, tool_id: str, name: str, summary: str, **kwargs):
        super().__init__(classes="tool-card -running", **kwargs)
        self.tool_id = tool_id
        self.tool_name = name
        self.summary = summary
        self.output = ""
        self.failed = False
        self.done = False
        self.expanded = False
        self._render_line()

    def _render_line(self) -> None:
        if not self.done:
            glyph = "[yellow]▸[/]"
            state = " [dim]…[/dim]"
        elif self.failed:
            glyph = "[red]✗[/]"
            state = ""
        else:
            glyph = "[#43d6b5]✓[/]"
            state = ""
        line = f"{glyph} [b]{self.tool_name}[/b][dim] · {self.summary}[/dim]{state}"
        if self.expanded and self.output:
            shown = self.output[-_MAX_CARD_OUTPUT:]
            line += f"\n[dim]{shown}[/dim]"
        self.update(line)

    def resolve(self, output: str, failed: bool) -> None:
        self.output = output
        self.failed = failed
        self.done = True
        self.set_classes(f"tool-card {'-err' if failed else '-ok'}")
        self._render_line()

    def on_click(self) -> None:
        if self.output:
            self.expanded = not self.expanded
            self._render_line()


class ApprovalModal(ModalScreen):
    """One permission request from the headless agent: allow or deny."""

    BINDINGS = [
        Binding("y", "decide(True)", "Allow"),
        Binding("n", "decide(False)", "Deny"),
        Binding("escape", "decide(False)", "Deny"),
    ]

    def __init__(self, request: dict, **kwargs):
        super().__init__(**kwargs)
        self._request = request

    def compose(self) -> ComposeResult:
        tool = self._request.get('tool_name', 'unknown tool')
        args = _args_summary(self._request.get('input') or {}, limit=200)
        yield Static(
            f"[b]The agent asks to use[/b]\n\n  {tool}\n  [dim]{args}[/dim]\n\n"
            f"[b]y[/b] allow · [b]n[/b]/[b]esc[/b] deny",
            id="agent-approval",
        )

    def action_decide(self, approved: bool) -> None:
        self.dismiss((self._request['id'], approved))


class AgentChatScreen(Screen):
    """The modelling-mode chat session."""

    BINDINGS = [
        Binding("escape", "interrupt_or_back", "Interrupt / Back"),
        Binding("ctrl+s", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+e", "export_transcript", "Export"),
    ]

    def __init__(
        self,
        launcher: AgentLauncher,
        workdir: Path | None = None,
        config_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._workdir = workdir or Path.cwd()
        self._session = AgentChatSession(
            launcher, self._workdir, AgentMode.MODELLING)
        self._monitor = RunMonitor(config_path)
        self._stream_widget: ChatMessage | None = None
        self._stream_text = ""
        self._cards: dict[str, ToolCallCard] = {}
        self._session_cost = 0.0
        self._transcript: list[tuple[str, str]] = []  # (speaker, text)
        self._approval_open = False

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            VerticalScroll(id="chat-log"),
            VerticalScroll(Static("", id="agent-run-status-body"),
                           id="agent-run-status"),
            id="chat-body",
        )
        yield Input(
            placeholder="Ask about this experiment, or give the next instruction…",
            id="chat-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = (
            f"Modelling · {self._workdir.name}"
            + (f" · resumes {self._session.session_id[:8]}"
               if self._session.session_id else "")
        )
        self._append(ChatMessage(_EMPTY_STATE, classes="chat-msg-system"))
        self.query_one("#chat-input", Input).focus()
        self._refresh_sidebar()
        self.set_interval(3.0, self._refresh_sidebar)

    # ------------------------------------------------------------- sidebar

    def _refresh_sidebar(self) -> None:
        self.run_worker(self._poll_sidebar, thread=True, exclusive=True,
                        group="sidebar", exit_on_error=False)

    def _poll_sidebar(self) -> None:
        from symfluence.agent import approvals

        status = self._monitor.poll()
        pending = approvals.list_pending()
        self.app.call_from_thread(self._render_sidebar, status)
        if pending:
            self.app.call_from_thread(self._offer_approval, pending[0])

    def _offer_approval(self, request: dict) -> None:
        if self._approval_open:
            return
        self._approval_open = True
        self.app.push_screen(ApprovalModal(request), self._approval_decided)

    def _approval_decided(self, decision: tuple[str, bool] | None) -> None:
        from symfluence.agent import approvals

        self._approval_open = False
        if decision is None:
            return
        request_id, approved = decision
        approvals.reply(request_id, approved)
        note = "approved" if approved else "denied"
        self._append(ChatMessage(f"[dim]Permission {note}.[/dim]",
                                 classes="chat-msg-system"))

    def _render_sidebar(self, status: RunStatus) -> None:
        lines: list[str] = ["[b]Run status[/b]", ""]
        if status.config_name:
            lines.append(status.config_name)
        if status.domain:
            lines.append(f"[dim]domain[/dim] {status.domain}")
        calibration = status.calibration
        if calibration:
            glyph = "▶" if calibration.get('in_progress') else "✓"
            lines.append("")
            lines.append(f"{glyph} {calibration.get('algorithm') or 'calibration'}")
            if calibration.get('iterations') is not None:
                lines.append(f"  [dim]iter[/dim] {calibration['iterations']}")
            if calibration.get('best_score') is not None:
                metric = calibration.get('metric') or 'score'
                lines.append(f"  [dim]{metric}[/dim] {calibration['best_score']:.3g}")
        for job in status.jobs:
            lines.append("")
            glyph = "▶" if job['state'] == 'running' else "·"
            lines.append(f"{glyph} job {job['state']} [dim]{job['runtime_s']:.0f}s[/dim]")
            if job.get('description'):
                lines.append(f"  [dim]{job['description']}[/dim]")
        if status.last_log_line:
            lines.append("")
            lines.append(f"[dim]{status.last_log_line[-120:]}[/dim]")
        self.query_one("#agent-run-status-body", Static).update("\n".join(lines))

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#agent-run-status")
        sidebar.display = not sidebar.display

    # ----------------------------------------------------------------- turns

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self._session.busy:
            return
        event.input.value = ""
        self._append(ChatMessage(f"[b]You[/b]\n{prompt}", classes="chat-msg-user"))
        self._transcript.append(("You", prompt))
        self._set_busy(True)
        self.run_worker(self._run_turn(prompt), exclusive=True, group="turn",
                        exit_on_error=False)

    async def _run_turn(self, prompt: str) -> None:
        try:
            await self._session.send(prompt, self._on_event)
        finally:
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = busy
        input_widget.placeholder = (
            "… working (esc interrupts)" if busy
            else "Ask about this experiment, or give the next instruction…"
        )
        if not busy:
            self._stream_widget = None
            self._stream_text = ""
            input_widget.focus()

    async def _on_event(self, event: object) -> None:
        if isinstance(event, SessionInit):
            self.sub_title = f"Modelling · {self._workdir.name} · {event.session_id[:8]}"
        elif isinstance(event, TextDelta):
            self._stream_text += event.text
            if self._stream_widget is None:
                self._stream_widget = ChatMessage("", classes="chat-msg-agent")
                self._append(self._stream_widget)
            self._stream_widget.update(f"[b]Agent[/b]\n{self._stream_text}")
            self._scroll_to_end()
        elif isinstance(event, AssistantText):
            if self._stream_widget is not None:
                self._stream_widget.update(f"[b]Agent[/b]\n{event.text}")
                self._stream_widget = None
                self._stream_text = ""
            else:
                self._append(ChatMessage(f"[b]Agent[/b]\n{event.text}",
                                         classes="chat-msg-agent"))
            self._transcript.append(("Agent", event.text))
        elif isinstance(event, ToolCallStarted):
            self._stream_widget = None
            self._stream_text = ""
            card = ToolCallCard(event.tool_id, _short_tool_name(event.name),
                                _args_summary(event.arguments))
            self._cards[event.tool_id] = card
            self._append(card)
            self._transcript.append(
                ("Tool", f"{card.tool_name} · {card.summary}"))
        elif isinstance(event, ToolCallFinished):
            card = self._cards.get(event.tool_id)
            if card is not None:
                card.resolve(event.output, event.is_error)
        elif isinstance(event, TurnResult):
            if event.cost_usd:
                self._session_cost += event.cost_usd
            note = []
            if event.duration_s:
                note.append(f"{event.duration_s:.0f}s")
            if event.cost_usd:
                note.append(f"${event.cost_usd:.2f}")
            if self._session_cost:
                note.append(f"Σ ${self._session_cost:.2f}")
            if event.is_error:
                self._append(ChatMessage(
                    f"[red]Turn failed[/red][dim] — {event.text[:400]}[/dim]",
                    classes="chat-msg-system"))
                self._transcript.append(("System", f"Turn failed — {event.text}"))
            elif note:
                self._append(ChatMessage(f"[dim]{' · '.join(note)}[/dim]",
                                         classes="chat-msg-system"))
        elif isinstance(event, DriverError):
            self._append(ChatMessage(
                f"[red]Driver error[/red][dim] — {event.message}\n"
                f"Retry, or run `symfluence agent model` from a terminal.[/dim]",
                classes="chat-msg-system"))
            self._transcript.append(("System", f"Driver error — {event.message}"))

    # --------------------------------------------------------------- helpers

    def _append(self, widget: Static) -> None:
        self.query_one("#chat-log", VerticalScroll).mount(widget)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)

    def action_interrupt_or_back(self) -> None:
        if self._session.busy:
            if self._session.interrupt():
                self._append(ChatMessage("[dim]Turn interrupted.[/dim]",
                                         classes="chat-msg-system"))
        else:
            self.app.pop_screen()

    def action_export_transcript(self) -> None:
        if not self._transcript:
            self.app.notify("Nothing to export yet.", severity="warning")
            return
        import time
        path = self._workdir / f"agent-transcript-{time.strftime('%Y%m%d-%H%M%S')}.md"
        lines = [f"# SYMFLUENCE modelling session — {self._workdir.name}", ""]
        if self._session.session_id:
            lines.append(f"Session `{self._session.session_id}`")
        if self._session_cost:
            lines.append(f"Total cost ${self._session_cost:.2f}")
        lines.append("")
        for speaker, text in self._transcript:
            if speaker == "Tool":
                lines.append(f"> `{text}`")
            else:
                lines.append(f"**{speaker}**: {text}")
            lines.append("")
        try:
            path.write_text("\n".join(lines), encoding='utf-8')
        except OSError as e:
            self.app.notify(f"Export failed: {e}", severity="error")
            return
        self.app.notify(f"Transcript saved to {path.name}.")


def _short_tool_name(name: str) -> str:
    return name.removeprefix('mcp__symfluence__')


def _args_summary(arguments: dict, limit: int = 60) -> str:
    if not arguments:
        return "no arguments"
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > 24:
            text = text[:21] + '…'
        parts.append(f"{key}={text}")
    summary = ", ".join(parts)
    return summary if len(summary) <= limit else summary[:limit - 1] + '…'
