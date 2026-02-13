"""
Workflow Launcher screen — load config, select steps, execute.
"""

import logging
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

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
    SelectionList,
    Static,
)
from textual.worker import Worker, WorkerState

from ..constants import WORKFLOW_STEPS
from ..services.workflow_service import WorkflowService
from ..widgets.log_panel import LogPanel, TUILogHandler
from ..widgets.step_progress import StepProgressWidget


class WorkflowLauncherScreen(Screen):
    """Load a config and run workflow steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workflow_svc = WorkflowService()
        self._log_handler = None
        self._known_step_names = [step for step, _ in WORKFLOW_STEPS]
        self._workflow_worker: Worker | None = None
        self._run_started_at: datetime | None = None
        self._run_timer = None
        self._run_summary_text = ""

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
                Static("Selected steps:", id="step-select-header"),
                SelectionList(id="step-selector"),
                Static(
                    "Use `symfluence job submit` for queued SLURM submission; "
                    "this launcher executes interactively.",
                    id="launcher-help",
                ),
                Static(""),
                Horizontal(
                    Button("Run", id="btn-run", variant="success", disabled=True),
                    Button("Cancel", id="btn-cancel-run", variant="warning", disabled=True),
                    classes="run-actions",
                ),
                Static("Active: idle", id="active-task"),
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
        self._sync_config_from_app()
        self._refresh_mode_controls()

    def on_screen_resume(self) -> None:
        self._sync_config_from_app()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load":
            config_input = self.query_one("#config-path", Input)
            self._do_load_config(config_input.value)
        elif event.button.id == "btn-run":
            self._do_run()
        elif event.button.id == "btn-cancel-run":
            self._cancel_run()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "run-mode":
            self._refresh_mode_controls()

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
            self._refresh_step_selector(ws)
            step_widget = self.query_one("#step-progress", StepProgressWidget)
            step_widget.update_from_completed(self._completed_steps_from_status(ws))

            self.query_one("#btn-run", Button).disabled = False
            status.update(f"Ready: {domain}")
        else:
            self._log_actionable_error(
                log,
                action="load configuration",
                error=self._workflow_svc.last_error or "unknown error",
                config_path=path,
            )
            status.update("Config load failed")

    def _do_run(self) -> None:
        """Start workflow execution in a background worker."""
        if not self._workflow_svc.is_loaded:
            return

        log_panel = self.query_one("#log-panel", LogPanel)
        run_mode = self._current_run_mode()
        force = self.query_one("#force-rerun", Checkbox).value
        steps_to_run = self._resolve_steps_for_mode(run_mode)

        if run_mode == "mode-steps" and not steps_to_run:
            log_panel.write_line("[yellow]No steps selected.[/yellow]")
            return
        if run_mode == "mode-resume" and not steps_to_run:
            log_panel.write_line("[green]No pending steps detected. Nothing to resume.[/green]")
            return

        self._set_running_ui(True)
        # Attach log handler
        self._log_handler = TUILogHandler(
            self.app,
            log_panel,
            on_message=self._on_log_message,
        )
        logging.getLogger("symfluence").addHandler(self._log_handler)

        if run_mode == "mode-full":
            self._run_summary_text = "Running full workflow"
            log_panel.write_line("Starting full workflow...")
        elif run_mode == "mode-steps":
            self._run_summary_text = "Running selected steps"
            log_panel.write_line(f"Starting selected steps: {', '.join(steps_to_run)}")
        else:
            self._run_summary_text = "Resuming pending steps"
            log_panel.write_line(f"Resuming pending steps: {', '.join(steps_to_run)}")

        self.query_one("#active-task", Static).update(
            f"Active: {steps_to_run[0] if steps_to_run else 'workflow'}"
        )
        self._run_started_at = datetime.now()
        self._start_run_timer()

        self._workflow_worker = self.run_worker(
            partial(self._run_mode, run_mode, force, steps_to_run),
            thread=True,
            name="workflow",
            group="workflow",
            exclusive=True,
        )

    def _run_mode(self, mode: str, force: bool, steps: List[str]) -> str:
        """Worker target for full/selected/resume execution modes."""
        try:
            if mode == "mode-full":
                self._workflow_svc.run_workflow(force_rerun=force)
            else:
                self._workflow_svc.run_steps(steps)
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
                self._log_actionable_error(
                    log_panel,
                    action="run workflow",
                    error=result[len("error:"):].strip(),
                )
                self.query_one("#status-text", Static).update("Workflow failed")
            else:
                log_panel.write_line("[green]Workflow completed successfully.[/green]")
                self.query_one("#status-text", Static).update("Workflow finished")

            # Refresh step status
            ws = self._workflow_svc.get_status()
            self._refresh_step_selector(ws)
            step_widget = self.query_one("#step-progress", StepProgressWidget)
            step_widget.update_from_completed(self._completed_steps_from_status(ws))

            self._finish_run_ui()

        elif event.state == WorkerState.ERROR:
            log_panel = self.query_one("#log-panel", LogPanel)
            self._log_actionable_error(
                log_panel,
                action="run workflow worker",
                error=str(event.worker.error),
            )
            self.query_one("#status-text", Static).update("Worker failed")
            self._finish_run_ui()

        elif event.state == WorkerState.CANCELLED:
            self.query_one("#log-panel", LogPanel).write_line("[yellow]Workflow cancelled.[/yellow]")
            self.query_one("#status-text", Static).update("Workflow cancelled")
            self._finish_run_ui()

    def _cleanup_logger(self) -> None:
        """Remove the TUI log handler from the symfluence logger."""
        if self._log_handler:
            logging.getLogger("symfluence").removeHandler(self._log_handler)
            self._log_handler = None

    def on_unmount(self) -> None:
        self._stop_run_timer()
        if self._run_timer is not None:
            self._run_timer.stop()
            self._run_timer = None
        self._cleanup_logger()

    def _current_run_mode(self) -> str:
        radio_set = self.query_one("#run-mode", RadioSet)
        pressed = radio_set.pressed_button
        if pressed and pressed.id:
            return pressed.id
        return "mode-full"

    def _refresh_mode_controls(self) -> None:
        mode = self._current_run_mode()
        force = self.query_one("#force-rerun", Checkbox)
        step_header = self.query_one("#step-select-header", Static)
        selector = self.query_one("#step-selector", SelectionList)

        is_step_mode = mode == "mode-steps"
        force.disabled = mode != "mode-full"
        step_header.display = is_step_mode
        selector.display = is_step_mode
        selector.disabled = not is_step_mode

    def _sync_config_from_app(self) -> None:
        """Apply config path provided by app-level actions (CLI/demo/palette)."""
        app_config_path = self.app.config_path
        if not app_config_path:
            return
        config_input = self.query_one("#config-path", Input)
        if config_input.value != app_config_path:
            config_input.value = app_config_path
            self._do_load_config(app_config_path)

    def _refresh_step_selector(self, status: Dict[str, Any]) -> None:
        selector = self.query_one("#step-selector", SelectionList)
        selector.clear_options()

        step_names = self._workflow_svc.get_step_names()
        if step_names:
            self._known_step_names = step_names

        completed = set(self._completed_steps_from_status(status))
        step_labels = dict(WORKFLOW_STEPS)

        for step_name in self._known_step_names:
            label = step_labels.get(step_name, step_name)
            if step_name in completed:
                label = f"{label} [done]"
            selector.add_option((label, step_name))

    def _resolve_steps_for_mode(self, mode: str) -> List[str]:
        if mode == "mode-steps":
            selector = self.query_one("#step-selector", SelectionList)
            selected = {str(step) for step in selector.selected}
            return [name for name in self._known_step_names if name in selected]

        if mode == "mode-resume":
            status = self._workflow_svc.get_status()
            completed = set(self._completed_steps_from_status(status))
            return [name for name in self._known_step_names if name not in completed]

        return []

    def _set_running_ui(self, running: bool) -> None:
        self.query_one("#btn-run", Button).disabled = running
        self.query_one("#btn-load", Button).disabled = running
        self.query_one("#btn-cancel-run", Button).disabled = not running
        self.query_one("#run-mode", RadioSet).disabled = running
        self.query_one("#force-rerun", Checkbox).disabled = running or (
            self._current_run_mode() != "mode-full"
        )
        self.query_one("#step-selector", SelectionList).disabled = running or (
            self._current_run_mode() != "mode-steps"
        )

    def _start_run_timer(self) -> None:
        if self._run_timer is None:
            self._run_timer = self.set_interval(1, self._tick_run_timer)
        else:
            self._run_timer.resume()
        self._tick_run_timer()

    def _stop_run_timer(self) -> None:
        if self._run_timer is not None:
            self._run_timer.pause()
        self._run_started_at = None

    def _tick_run_timer(self) -> None:
        if not self._run_started_at:
            return
        elapsed = int((datetime.now() - self._run_started_at).total_seconds())
        mins, secs = divmod(elapsed, 60)
        self.query_one("#status-text", Static).update(
            f"{self._run_summary_text} ({mins:02d}:{secs:02d})"
        )

    def _cancel_run(self) -> None:
        if self._workflow_worker and self._workflow_worker.is_running:
            self.query_one("#log-panel", LogPanel).write_line(
                "[yellow]Cancellation requested...[/yellow]"
            )
            self.query_one("#active-task", Static).update("Active: cancelling")
            self._workflow_worker.cancel()

    def _on_log_message(self, message: str) -> None:
        if message.startswith("Executing: "):
            self.query_one("#active-task", Static).update(
                f"Active: {message.split('Executing: ', 1)[1]}"
            )
        elif "Executing step:" in message:
            self.query_one("#active-task", Static).update(
                f"Active: {message.split('Executing step:', 1)[1].strip()}"
            )

    def _finish_run_ui(self) -> None:
        self._stop_run_timer()
        self._set_running_ui(False)
        self._cleanup_logger()
        self.query_one("#active-task", Static).update("Active: idle")
        self._workflow_worker = None

    def _log_actionable_error(
        self,
        log_panel: LogPanel,
        action: str,
        error: str,
        config_path: str = "",
    ) -> None:
        err = (error or "unknown error").strip()
        cfg = config_path or (
            str(self._workflow_svc.config_path) if self._workflow_svc.config_path else ""
        )
        log_panel.write_line(f"[red]What happened:[/red] Failed to {action}.")
        log_panel.write_line(f"[red]Error:[/red] {err}")

        err_lower = err.lower()
        if "not found" in err_lower or "no such file" in err_lower:
            target = cfg or "/path/to/config.yaml"
            log_panel.write_line(
                "[yellow]Likely cause:[/yellow] Path is invalid or file is missing."
            )
            log_panel.write_line(f"[cyan]Try:[/cyan] ls -lah {target}")
            return

        if "missing required configuration keys" in err_lower:
            target = cfg or "/path/to/config.yaml"
            log_panel.write_line(
                "[yellow]Likely cause:[/yellow] Required fields are missing in config."
            )
            log_panel.write_line(
                f"[cyan]Try:[/cyan] symfluence workflow run --config {target} --debug"
            )
            return

        target = cfg or "/path/to/config.yaml"
        log_panel.write_line(
            "[yellow]Likely cause:[/yellow] Workflow/runtime setup issue."
        )
        log_panel.write_line(
            f"[cyan]Try:[/cyan] symfluence workflow run --config {target} --debug"
        )

    @staticmethod
    def _completed_steps_from_status(status: Dict[str, Any]) -> List[str]:
        """Extract completed CLI step names from workflow status payloads."""
        step_details = status.get("step_details")
        if isinstance(step_details, list):
            completed = []
            for item in step_details:
                if not isinstance(item, dict) or not item.get("complete"):
                    continue
                cli_name = item.get("cli_name") or item.get("name")
                if cli_name:
                    completed.append(str(cli_name))
            return completed

        # Backward compatibility for {step_name: bool} payloads.
        return [str(k) for k, v in status.items() if v is True]
