"""
Color-coded performance metrics table widget.
"""

from typing import Dict, List

from textual.widgets import DataTable



class MetricsTable(DataTable):
    """DataTable displaying performance metrics with color coding."""

    def on_mount(self) -> None:
        self.cursor_type = "row"

    def load_single(self, metrics: Dict[str, float]) -> None:
        """Display metrics for a single experiment."""
        self.clear(columns=True)
        self.add_columns("Metric", "Value")
        for name, value in sorted(metrics.items()):
            self.add_row(name, f"{value:.4f}")

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
            higher_is_better = name.upper() in ("KGE", "NSE", "R", "R2")

            for exp_id in experiments:
                v = values[exp_id]
                if v is not None:
                    row.append(f"{v:.4f}")
                    if best_val is None:
                        best_val = v
                        best_exp = exp_id
                    elif higher_is_better and v > best_val:
                        best_val = v
                        best_exp = exp_id
                    elif not higher_is_better and abs(v) < abs(best_val):
                        best_val = v
                        best_exp = exp_id
                else:
                    row.append("-")

            row.append(best_exp if best_exp else "-")
            self.add_row(*row)
