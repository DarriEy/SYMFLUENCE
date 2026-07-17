# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Textual-free orchestration of one modelling chat session.

Thin seam between the chat screen and
:class:`~symfluence.agent.headless.HeadlessClaudeDriver`: owns the busy flag,
forwards driver events to a callback, and turns unexpected driver failures
into :class:`~symfluence.agent.headless.DriverError` events instead of
exceptions. Unit-testable without the TUI extra.
"""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from symfluence.agent.headless import DriverError, HeadlessClaudeDriver
from symfluence.agent.modes import AgentMode
from symfluence.agent.registry import AgentLauncher

EventCallback = Callable[[object], Awaitable[None]]


class AgentChatSession:
    """One conversation between the chat screen and a headless driver."""

    def __init__(
        self,
        launcher: AgentLauncher,
        workdir: Path,
        mode: AgentMode = AgentMode.MODELLING,
    ):
        self.driver = HeadlessClaudeDriver(launcher, workdir, mode)
        self.busy = False

    @property
    def session_id(self) -> str | None:
        return self.driver.session_id

    async def send(self, prompt: str, on_event: EventCallback) -> None:
        """Run one turn, forwarding every event; never raises into the UI."""
        if self.busy:
            await on_event(DriverError("A turn is already running."))
            return
        self.busy = True
        try:
            async for event in self.driver.run_turn(prompt):
                await on_event(event)
        except Exception as e:  # noqa: BLE001 — the UI must get an event, not a traceback
            await on_event(DriverError(f"{type(e).__name__}: {e}"))
        finally:
            self.busy = False

    def interrupt(self) -> bool:
        """Stop the running turn (kills the turn subprocess)."""
        return self.driver.interrupt()
