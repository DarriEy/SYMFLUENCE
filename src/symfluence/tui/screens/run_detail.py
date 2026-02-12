"""
Run Detail screen — drill-in view for a single run.

Shows config snapshot, completed steps, errors, and log excerpts.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane

from ..constants import STATUS_FAILED, STATUS_ICONS
from ..services.run_history import RunHistoryService, RunSummary
from ..widgets.config_tree import ConfigTreeWidget
from ..widgets.step_progress import StepProgressWidget


class RunDetailScreen(Screen):
    """Detailed view of a single workflow run."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
    ]

    def __init__(self, run: RunSummary, **kwargs):
        super().__init__(**kwargs)
        self._run = run

    def compose(self) -> ComposeResult:
        r = self._run
        title = f"Run: {r.domain} / {r.experiment_id or 'N/A'} -- {r.status}"

        # Build info text for General tab
        info_lines = [
            f"Domain:       {r.domain}",
            f"Experiment:   {r.experiment_id or 'N/A'}",
            f"Status:       {r.status}",
            f"Timestamp:    {r.timestamp.strftime('%Y-%m-%d %H:%M:%S') if r.timestamp else 'N/A'}",
            f"Duration:     {r.execution_time:.1f}s",
            f"Model:        {r.model or 'N/A'}",
            f"Algorithm:    {r.algorithm or 'N/A'}",
            f"Steps Done:   {r.total_steps}",
            f"Errors:       {r.total_errors}",
            f"Warnings:     {r.total_warnings}",
        ]

        yield Header()
        with Vertical():
            yield Static(title, classes="section-header")
            with TabbedContent():
                with TabPane("General", id="tab-general"):
                    yield Static("\n".join(info_lines))
                with TabPane("Steps", id="tab-steps"):
                    yield StepProgressWidget(id="step-progress")
                with TabPane("Config", id="tab-config"):
                    yield ConfigTreeWidget("Config", id="config-tree")
                with TabPane("Errors", id="tab-errors"):
                    yield RichLog(highlight=True, wrap=True, id="error-log")
        yield Footer()

    def on_mount(self) -> None:
        r = self._run

        # Update step progress
        step_widget = self.query_one("#step-progress", StepProgressWidget)
        step_widget.update_from_completed(r.steps_completed)

        # Load config snapshot
        config_tree = self.query_one("#config-tree", ConfigTreeWidget)
        svc = RunHistoryService(r.file_path.parent.parent)
        config_data = svc.load_config_snapshot(r)
        if config_data:
            config_tree.load_config(config_data)
        else:
            config_tree.root.add_leaf("(no config snapshot found)")

        # Populate errors
        error_log = self.query_one("#error-log", RichLog)
        if r.errors:
            for err in r.errors:
                if isinstance(err, dict):
                    step = err.get("step", "unknown")
                    msg = err.get("error", str(err))
                    error_log.write(f"[red]{STATUS_ICONS[STATUS_FAILED]}[/red] [{step}] {msg}")
                else:
                    error_log.write(f"[red]{STATUS_ICONS[STATUS_FAILED]}[/red] {err}")
        else:
            error_log.write("[green]No errors recorded.[/green]")

        if r.warnings:
            error_log.write("")
            error_log.write("[yellow]Warnings:[/yellow]")
            for w in r.warnings:
                error_log.write(f"  {w}")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
