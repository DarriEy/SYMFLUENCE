"""
Workflow Launcher screen — load config, select steps, execute.
"""

import logging
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

        self._refresh_mode_controls()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load":
            config_input = self.query_one("#config-path", Input)
            self._do_load_config(config_input.value)
        elif event.button.id == "btn-run":
            self._do_run()

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
            log.write_line("[red]Failed to load config. Check the path and try again.[/red]")
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

        btn = self.query_one("#btn-run", Button)
        btn.disabled = True
        # Attach log handler
        self._log_handler = TUILogHandler(self.app, log_panel)
        logging.getLogger("symfluence").addHandler(self._log_handler)

        if run_mode == "mode-full":
            log_panel.write_line("Starting full workflow...")
        elif run_mode == "mode-steps":
            log_panel.write_line(f"Starting selected steps: {', '.join(steps_to_run)}")
        else:
            log_panel.write_line(f"Resuming pending steps: {', '.join(steps_to_run)}")

        self.run_worker(
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
                log_panel.write_line(f"[red]Workflow failed: {result}[/red]")
            else:
                log_panel.write_line("[green]Workflow completed successfully.[/green]")

            # Refresh step status
            ws = self._workflow_svc.get_status()
            self._refresh_step_selector(ws)
            step_widget = self.query_one("#step-progress", StepProgressWidget)
            step_widget.update_from_completed(self._completed_steps_from_status(ws))

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

    def on_unmount(self) -> None:
        self._cleanup_logger()

    def _current_run_mode(self) -> str:
        radio_set = self.query_one("#run-mode", RadioSet)
        pressed = radio_set.pressed_button
        return pressed.id if pressed else "mode-full"

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
