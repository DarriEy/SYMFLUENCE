"""
In-app Help screen for TUI usage guidance.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

HELP_TEXT = """
SYMFLUENCE TUI Help

Keybindings
  1 Dashboard          2 Run Browser         3 Workflow Launcher
  4 Calibration        6 Compare             q Quit
  Ctrl+P Command Palette
  ? Help

Workflow Run Modes
  Full workflow: Execute the complete pipeline.
  Selected steps: Run only the checked steps.
  Resume failed: Run pending steps only.
  Force re-run: Available for full workflow mode.

Onboarding Flow
  On empty projects, Dashboard shows first-run actions:
    d Load Bow demo
    s Set SYMFLUENCE data directory
    o Open this Help screen

Long-Run UX
  Workflow launcher shows active task, elapsed runtime, and supports Cancel.
  For queued HPC jobs, use: symfluence job submit --config <config.yaml>

Troubleshooting
  Config path check:
    ls -lah /path/to/config.yaml
  Debug workflow execution:
    symfluence workflow run --config /path/to/config.yaml --debug
  Validate data directory:
    ls -lah /path/to/SYMFLUENCE_data

Docs
  Project README: README.md
  CLI reference: docs/source/cli_reference.rst
"""


class HelpScreen(Screen):
    """Reference screen for keybindings and common operations."""

    BINDINGS = [
        Binding("escape", "close_help", "Back"),
        Binding("q", "close_help", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="help-scroll"):
            yield Static(HELP_TEXT.strip("\n"), id="help-content")
        yield Footer()

    def action_close_help(self) -> None:
        self.app.pop_screen()
