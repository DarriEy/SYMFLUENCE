# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Results Comparison screen — side-by-side metrics for multiple experiments.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Select, SelectionList, Static

from ..services.calibration_data import CalibrationDataService
from ..widgets.metrics_table import MetricsTable


class ResultsCompareScreen(Screen):
    """Compare metrics across multiple experiments."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Results Comparison", classes="section-header"),
            Horizontal(
                Vertical(
                    Static("Domain:"),
                    Select([], id="cmp-domain", allow_blank=True),
                    Static("Experiments:"),
                    SelectionList(id="cmp-experiments"),
                    Button("Compare", id="btn-compare", variant="primary"),
                    classes="compare-controls",
                ),
                Vertical(
                    Static("Metrics", classes="section-header"),
                    MetricsTable(id="cmp-metrics"),
                ),
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._populate_domains()

    def on_screen_resume(self) -> None:
        self._populate_domains()

    def _populate_domains(self) -> None:
        """Fill domain selector."""
        domains = self.app.data_dir_service.list_domains()
        options = [(d.name, str(d.path)) for d in domains]
        self.query_one("#cmp-domain", Select).set_options(options)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cmp-domain":
            self._on_domain_selected(event.value)

    def _on_domain_selected(self, domain_path) -> None:
        """Populate experiment list for selected domain."""
        if not domain_path or domain_path == Select.BLANK:
            return

        svc = CalibrationDataService(str(domain_path))
        experiments = svc.list_experiments()

        exp_list = self.query_one("#cmp-experiments", SelectionList)
        exp_list.clear_options()
        for exp_id in experiments:
            exp_list.add_option((exp_id, exp_id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-compare":
            self._do_compare()

    def _do_compare(self) -> None:
        """Calculate and display comparison metrics."""
        domain_select = self.query_one("#cmp-domain", Select)
        domain_path = domain_select.value
        if not domain_path or domain_path == Select.BLANK:
            return

        exp_list = self.query_one("#cmp-experiments", SelectionList)
        selected = list(exp_list.selected)
        if len(selected) < 2:
            return

        svc = CalibrationDataService(str(domain_path))
        all_metrics = {}
        for exp_id in selected:
            metrics = svc.calculate_metrics(str(exp_id))
            if metrics:
                all_metrics[str(exp_id)] = metrics

        metrics_table = self.query_one("#cmp-metrics", MetricsTable)
        if all_metrics:
            exp_ids = [str(e) for e in selected if str(e) in all_metrics]
            metrics_table.load_comparison(exp_ids, all_metrics)
        else:
            metrics_table.clear(columns=True)
            metrics_table.add_columns("Info")
            metrics_table.add_row("No metrics available for selected experiments")
