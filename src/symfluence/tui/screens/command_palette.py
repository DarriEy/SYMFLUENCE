"""
Searchable command palette modal.
"""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class CommandPaletteScreen(ModalScreen[Optional[str]]):
    """Searchable command picker returning selected command id."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("enter", "run_selected", "Run"),
    ]

    def __init__(
        self,
        commands: list[tuple[str, str, str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._commands = commands
        self._visible: list[tuple[str, str, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="command-palette-dialog"):
            yield Static("Command Palette", classes="section-header")
            yield Input(
                placeholder="Type to search commands...",
                id="command-palette-query",
            )
            yield OptionList(id="command-palette-options")
            yield Static("Enter: run, Esc: close", id="command-palette-hint")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one("#command-palette-query", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "command-palette-query":
            self._refresh_options(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option = event.option_list.get_option_at_index(event.index)
        self.dismiss(option.id if option else None)

    def action_run_selected(self) -> None:
        options = self.query_one("#command-palette-options", OptionList)
        if options.option_count == 0 or options.highlighted is None:
            return
        option = options.get_option_at_index(options.highlighted)
        self.dismiss(option.id if option else None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _refresh_options(self, query: str) -> None:
        options = self.query_one("#command-palette-options", OptionList)
        options.clear_options()

        normalized = query.strip().lower()
        self._visible = []

        for command_id, label, keywords in self._commands:
            searchable = f"{label} {keywords}".lower()
            if normalized and normalized not in searchable:
                continue
            self._visible.append((command_id, label, keywords))

        for command_id, label, _keywords in self._visible:
            options.add_option(Option(label, id=command_id))

        if options.option_count > 0:
            options.highlighted = 0
