"""
SLURM Monitor screen — squeue wrapper for HPC job management.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from ..widgets.slurm_table import SlurmJobTable


class SlurmMonitorScreen(Screen):
    """Monitor and manage SLURM jobs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("SLURM Job Monitor", classes="section-header"),
            Horizontal(
                Button("Refresh", id="btn-refresh", variant="primary"),
                Button("Cancel Selected", id="btn-cancel", variant="error"),
                Static("", id="slurm-status"),
            ),
            SlurmJobTable(id="slurm-jobs"),
        )
        yield Footer()

    def on_mount(self) -> None:
        if not self.app.is_hpc:
            status = self.query_one("#slurm-status", Static)
            status.update("SLURM not available on this system")
            return

        self._refresh_jobs()
        self._timer = self.set_interval(60, self._refresh_jobs)

    def on_screen_resume(self) -> None:
        if self._timer is not None:
            self._timer.resume()
        if self.app.is_hpc:
            self._refresh_jobs()

    def on_screen_suspend(self) -> None:
        if self._timer is not None:
            self._timer.pause()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh":
            self._refresh_jobs()
        elif event.button.id == "btn-cancel":
            self._cancel_selected()

    def _refresh_jobs(self) -> None:
        """Query squeue and update table."""
        jobs = self.app.slurm_service.list_user_jobs()
        table = self.query_one("#slurm-jobs", SlurmJobTable)
        table.load_jobs(jobs)

        status = self.query_one("#slurm-status", Static)
        status.update(f"{len(jobs)} job(s)")

    def _cancel_selected(self) -> None:
        """Cancel the selected SLURM job."""
        table = self.query_one("#slurm-jobs", SlurmJobTable)
        job_id = table.get_selected_job_id()
        if not job_id:
            return

        status = self.query_one("#slurm-status", Static)
        success = self.app.slurm_service.cancel_job(job_id)
        if success:
            status.update(f"Cancelled job {job_id}")
            self._refresh_jobs()
        else:
            status.update(f"Failed to cancel job {job_id}")
