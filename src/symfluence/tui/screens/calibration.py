"""
Calibration Monitor screen — optimization progress and metrics.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, ProgressBar, Select, Static

from ..services.calibration_data import CalibrationDataService
from ..widgets.metrics_table import MetricsTable
from ..widgets.sparkline import SparklineWidget


class CalibrationScreen(Screen):
    """View calibration optimization progress and metrics."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Calibration Monitor", classes="section-header"),
            Horizontal(
                Select([], id="cal-domain", allow_blank=True),
                Select([], id="cal-experiment", allow_blank=True),
                classes="cal-top",
            ),
            Horizontal(
                Static("Progress:", id="cal-progress-label"),
                ProgressBar(total=100, id="cal-progress"),
                Static("Best: -", id="cal-best-score"),
            ),
            Static("Score Evolution:", classes="section-header"),
            SparklineWidget(id="cal-sparkline", classes="sparkline"),
            Static("Metrics:", classes="section-header"),
            MetricsTable(id="cal-metrics"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._populate_domains()
        self._timer = self.set_interval(30, self._auto_refresh)

    def on_screen_resume(self) -> None:
        self._populate_domains()

    def _populate_domains(self) -> None:
        """Fill domain selector from data dir."""
        domains = self.app.data_dir_service.list_domains()
        options = [(d.name, str(d.path)) for d in domains]
        domain_select = self.query_one("#cal-domain", Select)
        domain_select.set_options(options)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cal-domain":
            self._on_domain_selected(event.value)
        elif event.select.id == "cal-experiment":
            self._on_experiment_selected(event.value)

    def _on_domain_selected(self, domain_path) -> None:
        """Populate experiments for the selected domain."""
        if not domain_path or domain_path == Select.BLANK:
            return

        svc = CalibrationDataService(str(domain_path))
        experiments = svc.list_experiments()
        options = [(e, e) for e in experiments]
        exp_select = self.query_one("#cal-experiment", Select)
        exp_select.set_options(options)

    def _on_experiment_selected(self, experiment_id) -> None:
        """Load calibration data for the selected experiment."""
        if not experiment_id or experiment_id == Select.BLANK:
            return

        domain_select = self.query_one("#cal-domain", Select)
        domain_path = domain_select.value
        if not domain_path or domain_path == Select.BLANK:
            return

        self._load_calibration(str(domain_path), str(experiment_id))

    def _load_calibration(self, domain_path: str, experiment_id: str) -> None:
        """Load and display calibration data."""
        svc = CalibrationDataService(domain_path)

        # Load optimization history
        history = svc.load_optimization_history(experiment_id)
        if history is not None and len(history) > 0:
            # Update progress
            n_iter = len(history)
            progress = self.query_one("#cal-progress", ProgressBar)
            progress.update(total=n_iter, progress=n_iter)

            # Best score
            if "score" in history.columns:
                scores = history["score"].dropna().tolist()
                if scores:
                    best = max(scores)
                    self.query_one("#cal-best-score", Static).update(
                        f"Best: {best:.4f} (iter {len(scores)})"
                    )
                    # Sparkline
                    sparkline = self.query_one("#cal-sparkline", SparklineWidget)
                    sparkline.set_values(scores)
        else:
            self.query_one("#cal-best-score", Static).update("Best: No data")

        # Metrics
        metrics = svc.calculate_metrics(experiment_id)
        metrics_table = self.query_one("#cal-metrics", MetricsTable)
        if metrics:
            metrics_table.load_single(metrics)
        else:
            metrics_table.clear(columns=True)
            metrics_table.add_columns("Metric", "Value")
            metrics_table.add_row("(no data)", "-")

    def _auto_refresh(self) -> None:
        """Periodic refresh for running calibrations."""
        domain_select = self.query_one("#cal-domain", Select)
        exp_select = self.query_one("#cal-experiment", Select)
        if (domain_select.value and domain_select.value != Select.BLANK and
                exp_select.value and exp_select.value != Select.BLANK):
            self._load_calibration(str(domain_select.value), str(exp_select.value))
