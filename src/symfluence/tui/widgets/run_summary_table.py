# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Filterable DataTable of run summary JSONs across domains.
"""

from typing import List, Optional

from textual.widgets import DataTable

from ..services.run_history import RunSummary


class RunSummaryTable(DataTable):
    """Table of run summaries with in-memory filtering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._all_runs: List[RunSummary] = []

    def on_mount(self) -> None:
        self.add_columns(
            "Timestamp", "Domain", "Experiment", "Status",
            "Steps", "Time (s)", "Errors", "Model",
        )
        self.cursor_type = "row"

    def load_runs(self, runs: List[RunSummary]) -> None:
        """Store runs and display all."""
        self._all_runs = list(runs)
        self._render_runs(self._all_runs)

    def filter_runs(
        self,
        domain_filter: str = "",
        status_filter: str = "all",
    ) -> None:
        """Apply domain and status filters to the stored runs."""
        filtered = self._all_runs
        if domain_filter:
            low = domain_filter.lower()
            filtered = [r for r in filtered if low in r.domain.lower()]
        if status_filter and status_filter != "all":
            filtered = [r for r in filtered if r.status == status_filter]
        self._render_runs(filtered)

    def get_selected_run(self) -> Optional[RunSummary]:
        """Return the RunSummary for the currently selected row, or None."""
        row_key = self.cursor_row
        if row_key is None or row_key < 0 or row_key >= len(self._visible_runs):
            return None
        return self._visible_runs[row_key]

    def _render_runs(self, runs: List[RunSummary]) -> None:
        """Clear and re-populate the table with given runs."""
        self.clear()
        self._visible_runs = list(runs)
        for r in runs:
            ts = r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "-"
            self.add_row(
                ts,
                r.domain,
                r.experiment_id or "-",
                r.status,
                str(r.total_steps),
                f"{r.execution_time:.0f}",
                str(r.total_errors),
                r.model or "-",
            )
