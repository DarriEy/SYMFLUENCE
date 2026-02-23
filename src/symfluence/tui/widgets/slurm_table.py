"""
DataTable for displaying SLURM squeue output.
"""

from typing import List

from textual.widgets import DataTable

from ..services.slurm_service import SlurmJob


class SlurmJobTable(DataTable):
    """Table of SLURM jobs for the current user."""

    def on_mount(self) -> None:
        self.add_columns("Job ID", "Name", "Status", "Partition", "Time", "Nodes")
        self.cursor_type = "row"

    def load_jobs(self, jobs: List[SlurmJob]) -> None:
        """Populate table from SlurmJob list."""
        self.clear()
        for j in jobs:
            self.add_row(
                j.job_id,
                j.name,
                j.status,
                j.partition,
                j.time,
                j.nodes,
                key=j.job_id,
            )

    def get_selected_job_id(self) -> str:
        """Return the job ID of the selected row, or empty string."""
        row_key = self.cursor_row
        if row_key is None or row_key < 0:
            return ""
        try:
            row_data = self.get_row_at(row_key)
            return str(row_data[0]) if row_data else ""
        except Exception:  # noqa: BLE001 — UI resilience
            return ""
