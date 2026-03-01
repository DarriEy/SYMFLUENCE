# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Run Browser screen — filterable list of all runs across domains.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Select, Static

from ..services.run_history import RunHistoryService
from ..widgets.run_summary_table import RunSummaryTable


class RunBrowserScreen(Screen):
    """Browsable list of all workflow runs across domains."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Run Browser", classes="section-header"),
            Horizontal(
                Input(placeholder="Filter by domain...", id="filter-domain"),
                Select(
                    [("All", "all"), ("Completed", "completed"),
                     ("Failed", "failed"), ("Partial", "partial")],
                    value="all",
                    id="filter-status",
                    allow_blank=False,
                ),
                classes="filter-bar",
            ),
            RunSummaryTable(id="run-table"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._load_all_runs()
        self._apply_pending_domain_filter()

    def on_screen_resume(self) -> None:
        self._load_all_runs()
        self._apply_pending_domain_filter()

    def _load_all_runs(self) -> None:
        """Load runs from all domains."""
        domains = self.app.data_dir_service.list_domains()
        all_runs = []
        for d in domains:
            svc = RunHistoryService(d.path)
            all_runs.extend(svc.list_runs())

        # Sort by timestamp descending (None timestamps sort last)
        from datetime import datetime as _dt
        _epoch = _dt.min
        all_runs.sort(key=lambda r: r.timestamp or _epoch, reverse=True)

        table = self.query_one("#run-table", RunSummaryTable)
        table.load_runs(all_runs)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-domain":
            self._apply_filters()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "filter-status":
            self._apply_filters()

    def _apply_filters(self) -> None:
        domain_input = self.query_one("#filter-domain", Input)
        status_select = self.query_one("#filter-status", Select)
        table = self.query_one("#run-table", RunSummaryTable)
        table.filter_runs(
            domain_filter=domain_input.value,
            status_filter=str(status_select.value),
        )

    def _apply_pending_domain_filter(self) -> None:
        """Apply a domain filter requested by another screen (e.g. Dashboard)."""
        pending = self.app.consume_run_browser_domain_filter()
        if not pending:
            return

        domain_input = self.query_one("#filter-domain", Input)
        domain_input.value = pending
        self._apply_filters()

    def on_data_table_row_selected(self, event) -> None:
        """Push run detail screen for the selected run."""
        table = self.query_one("#run-table", RunSummaryTable)
        run = table.get_selected_run()
        if run:
            from .run_detail import RunDetailScreen
            self.app.push_screen(RunDetailScreen(run))
