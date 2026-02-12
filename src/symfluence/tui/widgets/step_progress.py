"""
Vertical step list widget with status icons.
"""

from typing import Dict, List, Tuple

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from ..constants import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_ICONS,
    STATUS_PENDING,
    STATUS_RUNNING,
    WORKFLOW_STEPS,
)


class StepProgressWidget(VerticalScroll):
    """Vertical list of workflow steps with status indicators."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step_widgets: Dict[str, Static] = {}

    def compose(self) -> ComposeResult:
        for cli_name, description in WORKFLOW_STEPS:
            icon = STATUS_ICONS[STATUS_PENDING]
            w = Static(f"  {icon}  {description}", id=f"step-{cli_name}")
            w.add_class("status-pending")
            self._step_widgets[cli_name] = w
            yield w

    def update_status(self, status_map: Dict[str, str]) -> None:
        """Update step icons from a {step_name: status} mapping.

        Status values: 'completed', 'failed', 'running', 'pending'.
        """
        for cli_name, widget in self._step_widgets.items():
            status = status_map.get(cli_name, STATUS_PENDING)
            icon = STATUS_ICONS.get(status, STATUS_ICONS[STATUS_PENDING])
            _, description = WORKFLOW_STEPS[self._step_index(cli_name)]
            widget.update(f"  {icon}  {description}")
            widget.remove_class(
                "status-completed", "status-failed", "status-running", "status-pending"
            )
            widget.add_class(f"status-{status}")

    def update_from_completed(self, completed: List[str], running: str = "") -> None:
        """Convenience: set status from a list of completed step names."""
        status_map = {}
        for cli_name, _ in WORKFLOW_STEPS:
            if cli_name == running:
                status_map[cli_name] = STATUS_RUNNING
            elif cli_name in completed:
                status_map[cli_name] = STATUS_COMPLETED
            else:
                status_map[cli_name] = STATUS_PENDING
        self.update_status(status_map)

    def mark_failed(self, step_name: str) -> None:
        """Mark a specific step as failed."""
        if step_name in self._step_widgets:
            w = self._step_widgets[step_name]
            icon = STATUS_ICONS[STATUS_FAILED]
            _, description = self._step_info(step_name)
            w.update(f"  {icon}  {description}")
            w.remove_class(
                "status-completed", "status-running", "status-pending"
            )
            w.add_class("status-failed")

    def _step_index(self, cli_name: str) -> int:
        for i, (name, _) in enumerate(WORKFLOW_STEPS):
            if name == cli_name:
                return i
        return 0

    def _step_info(self, cli_name: str) -> Tuple[str, str]:
        for name, desc in WORKFLOW_STEPS:
            if name == cli_name:
                return name, desc
        return cli_name, cli_name
