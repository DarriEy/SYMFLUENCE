# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Dashboard screen — domain overview and quick stats.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..widgets.domain_list import DomainListWidget


class DashboardScreen(Screen):
    """Home screen showing domain overview and summary statistics."""

    BINDINGS = [
        Binding("d", "load_demo", "Load Demo"),
        Binding("s", "set_data_dir", "Set Data Dir"),
        Binding("o", "open_docs", "Docs"),
        Binding("r", "refresh_dashboard", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Horizontal(
                Static("Domains: -", id="stat-domains"),
                Static("Total Runs: -", id="stat-runs"),
                Static("SLURM Jobs: -", id="stat-slurm"),
                classes="stats-bar",
            ),
            Static("Domains", classes="section-header"),
            Static("", id="onboarding-panel", classes="onboarding-panel"),
            DomainListWidget(id="domain-table"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_data()

    def on_screen_resume(self) -> None:
        self._refresh_data()

    def _refresh_data(self) -> None:
        """Load domain data and update stats."""
        app = self.app
        domains = app.data_dir_service.list_domains()

        # Update stats
        total_runs = sum(d.run_count for d in domains)
        self.query_one("#stat-domains", Static).update(f"Domains: {len(domains)}")
        self.query_one("#stat-runs", Static).update(f"Total Runs: {total_runs}")

        if app.is_hpc:
            jobs = app.slurm_service.list_user_jobs()
            self.query_one("#stat-slurm", Static).update(f"SLURM Jobs: {len(jobs)}")
        else:
            self.query_one("#stat-slurm", Static).update("SLURM: N/A")

        # Load domain table
        table = self.query_one("#domain-table", DomainListWidget)
        table.load_domains(domains)

        # Show data dir path
        data_dir = app.data_dir_service.data_dir
        if data_dir:
            self.sub_title = str(data_dir)
        else:
            self.sub_title = "No SYMFLUENCE_DATA_DIR set"

        onboarding = self.query_one("#onboarding-panel", Static)
        if not data_dir:
            onboarding.display = True
            onboarding.update(
                "First run setup:\n"
                "  d  Load Bow demo and jump to Workflow Launcher\n"
                "  s  Set your SYMFLUENCE data directory\n"
                "  o  Open in-app help and docs"
            )
        elif not domains:
            onboarding.display = True
            onboarding.update(
                "Data directory is configured but no domain_* projects were found.\n"
                "  d  Load Bow demo\n"
                "  s  Change data directory\n"
                "  o  Open help and docs"
            )
        else:
            onboarding.display = False

    def on_data_table_row_selected(self, event) -> None:
        """Navigate to run browser filtered for the selected domain."""
        row_key = event.row_key
        if row_key:
            self.app.set_run_browser_domain_filter(str(row_key))
            self.app.switch_mode("run_browser")

    def action_load_demo(self) -> None:
        if not self.app.run_demo("bow"):
            self.app.notify("Built-in demo 'bow' is not available.", severity="error")

    def action_set_data_dir(self) -> None:
        self.app.prompt_set_data_dir()

    def action_open_docs(self) -> None:
        self.app.action_show_help()

    def action_refresh_dashboard(self) -> None:
        self._refresh_data()
