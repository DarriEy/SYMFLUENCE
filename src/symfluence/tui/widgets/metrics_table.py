# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Color-coded performance metrics table widget.
"""

from typing import Dict, List

from rich.text import Text
from textual.widgets import DataTable

from ..constants import METRIC_THRESHOLDS


class MetricsTable(DataTable):
    """DataTable displaying performance metrics with color coding."""

    def on_mount(self) -> None:
        self.cursor_type = "row"

    def load_single(self, metrics: Dict[str, float]) -> None:
        """Display metrics for a single experiment."""
        self.clear(columns=True)
        self.add_columns("Metric", "Value")
        for name, value in sorted(metrics.items()):
            self.add_row(name, self._render_metric_value(name, value))

    def load_comparison(
        self,
        experiments: List[str],
        all_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """Display side-by-side metrics for multiple experiments.

        Args:
            experiments: List of experiment IDs.
            all_metrics: {experiment_id: {metric_name: value}}.
        """
        self.clear(columns=True)
        self.add_columns("Metric", *experiments, "Best")

        # Collect all metric names
        metric_names: set[str] = set()
        for m in all_metrics.values():
            metric_names.update(m.keys())

        for name in sorted(metric_names):
            values = {}
            for exp_id in experiments:
                v = all_metrics.get(exp_id, {}).get(name)
                values[exp_id] = v

            row = [name]
            best_val = None
            best_exp = ""
            higher_is_better = not self._is_lower_better(name)

            for exp_id in experiments:
                v = values[exp_id]
                if v is not None:
                    if best_val is None:
                        best_val = v
                        best_exp = exp_id
                    elif higher_is_better and v > best_val:
                        best_val = v
                        best_exp = exp_id
                    elif not higher_is_better and self._is_better_lower(name, v, best_val):
                        best_val = v
                        best_exp = exp_id

            for exp_id in experiments:
                v = values[exp_id]
                if v is not None:
                    row.append(
                        self._render_metric_value(
                            name,
                            v,
                            highlight_best=(exp_id == best_exp),
                        )
                    )
                else:
                    row.append("-")

            row.append(best_exp if best_exp else "-")
            self.add_row(*row)

    @staticmethod
    def _is_lower_better(metric_name: str) -> bool:
        upper = metric_name.upper()
        return upper in {"RMSE", "MAE", "PBIAS"}

    @staticmethod
    def _is_better_lower(metric_name: str, candidate: float, incumbent: float) -> bool:
        if metric_name.upper() == "PBIAS":
            return abs(candidate) < abs(incumbent)
        return candidate < incumbent

    def _render_metric_value(
        self,
        metric_name: str,
        value: float,
        highlight_best: bool = False,
    ) -> Text:
        quality = self._classify_metric(metric_name, value)
        style_map = {
            "good": "green",
            "poor": "red",
            "neutral": "white",
        }
        style = style_map[quality]
        if highlight_best:
            style = f"bold {style}"
        return Text(f"{value:.4f}", style=style)

    def _classify_metric(self, metric_name: str, value: float) -> str:
        thresholds = (
            METRIC_THRESHOLDS.get(metric_name)
            or METRIC_THRESHOLDS.get(metric_name.upper())
            or METRIC_THRESHOLDS.get(metric_name.lower())
        )
        if not thresholds:
            return "neutral"

        good = thresholds.get("good")
        poor = thresholds.get("poor")
        if good is None or poor is None:
            return "neutral"

        upper = metric_name.upper()
        if upper == "PBIAS":
            abs_val = abs(value)
            if abs_val <= good:
                return "good"
            if abs_val >= poor:
                return "poor"
            return "neutral"

        if self._is_lower_better(metric_name):
            if value <= good:
                return "good"
            if value >= poor:
                return "poor"
            return "neutral"

        if value >= good:
            return "good"
        if value <= poor:
            return "poor"
        return "neutral"
