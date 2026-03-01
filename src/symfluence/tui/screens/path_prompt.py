# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Reusable modal prompt for entering a filesystem path.
"""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class PathPromptScreen(ModalScreen[Optional[str]]):
    """Simple path-entry modal used by onboarding and command palette actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Submit"),
    ]

    def __init__(
        self,
        title: str,
        prompt_text: str,
        initial_value: str = "",
        placeholder: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._prompt_text = prompt_text
        self._initial_value = initial_value
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="path-prompt-dialog"):
            yield Static(self._title, classes="section-header")
            yield Static(self._prompt_text)
            yield Input(
                value=self._initial_value,
                placeholder=self._placeholder,
                id="path-prompt-input",
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Submit", id="path-prompt-submit", variant="primary")
                yield Button("Cancel", id="path-prompt-cancel")

    def on_mount(self) -> None:
        self.query_one("#path-prompt-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "path-prompt-submit":
            self.action_submit()
        elif event.button.id == "path-prompt-cancel":
            self.action_cancel()

    def action_submit(self) -> None:
        value = self.query_one("#path-prompt-input", Input).value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
