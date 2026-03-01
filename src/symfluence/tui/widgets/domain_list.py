# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
DataTable widget displaying domain directories from SYMFLUENCE_DATA_DIR.
"""

from textual.widgets import DataTable


class DomainListWidget(DataTable):
    """Table of discovered domain directories with run metadata."""

    def on_mount(self) -> None:
        self.add_columns("Domain", "Runs", "Last Run", "Status", "Experiments")
        self.cursor_type = "row"

    def load_domains(self, domains) -> None:
        """Populate table from a list of DomainInfo objects."""
        self.clear()
        for d in domains:
            last_run = d.last_run.strftime("%Y-%m-%d %H:%M") if d.last_run else "-"
            experiments = ", ".join(d.experiments[:3]) if d.experiments else "-"
            if len(d.experiments) > 3:
                experiments += f" (+{len(d.experiments) - 3})"
            self.add_row(
                d.name,
                str(d.run_count),
                last_run,
                d.last_status,
                experiments,
                key=d.name,
            )
