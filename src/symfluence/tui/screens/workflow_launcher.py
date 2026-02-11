"""
Workflow Launcher screen — load config, select steps, execute.
"""

import logging

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Static,
)
from textual.worker import Worker, WorkerState

from ..services.workflow_service import WorkflowService
from ..widgets.log_panel import LogPanel, TUILogHandler
from ..widgets.step_progress import StepProgressWidget


class WorkflowLauncherScreen(Screen):
    """Load a config and run workflow steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workflow_svc = WorkflowService()
        self._log_handler = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Vertical(
                Static("Workflow Launcher", classes="section-header"),
                Static("Config file:"),
                Input(
                    placeholder="Path to config YAML...",
                    id="config-path",
                ),
                Button("Load Config", id="btn-load", variant="primary"),
                Static(""),
                Static("Run mode:"),
                RadioSet(
                    RadioButton("Full workflow", id="mode-full", value=True),
                    RadioButton("Selected steps", id="mode-steps"),
                    RadioButton("Resume failed", id="mode-resume"),
                    id="run-mode",
                ),
                Checkbox("Force re-run completed steps", id="force-rerun"),
                Static(""),
                Static("SLURM Options:", id="slurm-header"),
                Input(placeholder="Time limit (e.g. 04:00:00)", id="slurm-time"),
                Input(placeholder="Memory (e.g. 16G)", id="slurm-memory"),
                Input(placeholder="Account", id="slurm-account"),
                Input(placeholder="Partition", id="slurm-partition"),
                Static(""),
                Button("Run", id="btn-run", variant="success", disabled=True),
                Static("", id="status-text"),
                classes="launcher-left",
            ),
            Vertical(
                Static("Steps", classes="section-header"),
                StepProgressWidget(id="step-progress"),
                Static("Log", classes="section-header"),
                LogPanel(id="log-panel", classes="log-panel"),
                classes="launcher-right",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        # Pre-fill config path if provided via CLI
        if self.app.config_path:
            config_input = self.query_one("#config-path", Input)
            config_input.value = self.app.config_path
            self._do_load_config(self.app.config_path)

        # Hide SLURM options if not on HPC
        if not self.app.is_hpc:
            for widget_id in ("slurm-header", "slurm-time", "slurm-memory",
                              "slurm-account", "slurm-partition"):
                try:
                    self.query_one(f"#{widget_id}").display = False
                except Exception:
                    pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load":
            config_input = self.query_one("#config-path", Input)
            self._do_load_config(config_input.value)
        elif event.button.id == "btn-run":
            self._do_run()

    def _do_load_config(self, path: str) -> None:
        """Load config and update UI."""
        log = self.query_one("#log-panel", LogPanel)
        status = self.query_one("#status-text", Static)

        if not path:
            log.write_line("[red]No config path specified.[/red]")
            return

        log.write_line(f"Loading config: {path}")
        success = self._workflow_svc.load_config(path)

        if success:
            log.write_line("[green]Config loaded successfully.[/green]")
            domain = self._workflow_svc.get_domain_name()
            exp = self._workflow_svc.get_experiment_id()
            log.write_line(f"Domain: {domain}, Experiment: {exp}")

            # Update step progress from current status
            ws = self._workflow_svc.get_status()
            step_widget = self.query_one("#step-progress", StepProgressWidget)
            completed = [k for k, v in ws.items() if v is True]
            step_widget.update_from_completed(completed)

            self.query_one("#btn-run", Button).disabled = False
            status.update(f"Ready: {domain}")
        else:
            log.write_line("[red]Failed to load config. Check the path and try again.[/red]")
            status.update("Config load failed")

    def _do_run(self) -> None:
        """Start workflow execution in a background worker."""
        if not self._workflow_svc.is_loaded:
            return

        btn = self.query_one("#btn-run", Button)
        btn.disabled = True

        log_panel = self.query_one("#log-panel", LogPanel)
        # Attach log handler
        self._log_handler = TUILogHandler(self.app, log_panel)
        logging.getLogger("symfluence").addHandler(self._log_handler)

        log_panel.write_line("Starting workflow execution...")

        # Determine run mode
        radio_set = self.query_one("#run-mode", RadioSet)
        pressed = radio_set.pressed_button
        mode = pressed.id if pressed else "mode-full"

        if mode == "mode-full":
            self.run_worker(
                self._run_full_workflow,
                thread=True,
                name="workflow",
            )
        else:
            # For steps mode, run all steps (user can customize later)
            self.run_worker(
                self._run_full_workflow,
                thread=True,
                name="workflow",
            )

    async def _run_full_workflow(self) -> str:
        """Worker target: run full workflow."""
        force = self.query_one("#force-rerun", Checkbox).value
        try:
            self._workflow_svc.run_workflow(force_rerun=force)
            return "completed"
        except Exception as exc:
            return f"error: {exc}"

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion."""
        if event.worker.name != "workflow":
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            log_panel = self.query_one("#log-panel", LogPanel)
            if result and result.startswith("error:"):
                log_panel.write_line(f"[red]Workflow failed: {result}[/red]")
            else:
                log_panel.write_line("[green]Workflow completed successfully.[/green]")

            # Refresh step status
            ws = self._workflow_svc.get_status()
            step_widget = self.query_one("#step-progress", StepProgressWidget)
            completed = [k for k, v in ws.items() if v is True]
            step_widget.update_from_completed(completed)

            self._cleanup_logger()
            self.query_one("#btn-run", Button).disabled = False
            self.query_one("#status-text", Static).update("Workflow finished")

        elif event.state == WorkerState.ERROR:
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.write_line(f"[red]Worker error: {event.worker.error}[/red]")
            self._cleanup_logger()
            self.query_one("#btn-run", Button).disabled = False

        elif event.state == WorkerState.CANCELLED:
            self._cleanup_logger()
            self.query_one("#btn-run", Button).disabled = False

    def _cleanup_logger(self) -> None:
        """Remove the TUI log handler from the symfluence logger."""
        if self._log_handler:
            logging.getLogger("symfluence").removeHandler(self._log_handler)
            self._log_handler = None
