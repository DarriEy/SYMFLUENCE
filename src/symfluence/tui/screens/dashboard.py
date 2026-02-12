"""
Dashboard screen — domain overview and quick stats.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..widgets.domain_list import DomainListWidget


class DashboardScreen(Screen):
    """Home screen showing domain overview and summary statistics."""

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

    def on_data_table_row_selected(self, event) -> None:
        """Navigate to run browser filtered for the selected domain."""
        row_key = event.row_key
        if row_key:
            self.app.switch_mode("run_browser")
