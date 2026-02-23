"""Overview plot composition for model comparison reporting."""

from __future__ import annotations

from typing import Optional

from symfluence.reporting.core.base_plotter import BasePlotter


class ModelComparisonOverviewMixin:
    """Top-level overview plotting entry points."""

    @BasePlotter._plot_safe("creating model comparison overview")
    def plot_model_comparison_overview(
        self,
        experiment_id: str = "default",
        context: str = "run_model",
    ) -> Optional[str]:
        """Create a comprehensive multi-panel model-comparison overview."""
        results_df, obs_series = self._collect_model_data(experiment_id, context)

        if results_df is None or results_df.empty:
            self.logger.warning("No model data available for comparison overview")
            return None

        model_cols = self._find_discharge_columns(results_df)
        if not model_cols:
            self.logger.warning("No model discharge columns found in results")
            return None

        metrics_dict = self._calculate_all_metrics(results_df, obs_series, model_cols)

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(
            4,
            3,
            height_ratios=[0.05, 1, 1, 1],
            width_ratios=[2, 1, 1],
            hspace=0.3,
            wspace=0.3,
        )

        context_title = "Post-Calibration" if context == "calibrate_model" else "Model Run"
        fig.suptitle(
            f"Model Comparison Overview - {context_title}\n{experiment_id}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        panel_data = {
            "results_df": results_df,
            "obs_series": obs_series,
            "model_cols": model_cols,
            "metrics_dict": metrics_dict,
        }

        ax_ts = fig.add_subplot(gs[1, 0:2])
        self._ts_panel.render(ax_ts, panel_data)

        ax_metrics = fig.add_subplot(gs[1, 2])
        self._metrics_panel.render(ax_metrics, panel_data)

        ax_fdc = fig.add_subplot(gs[2, 0])
        self._fdc_panel.render(ax_fdc, panel_data)

        ax_monthly = fig.add_subplot(gs[2, 1:3])
        self._monthly_panel.render(ax_monthly, panel_data)

        n_models = len(model_cols)
        if n_models > 0:
            scatter_gs = gridspec.GridSpecFromSubplotSpec(
                1,
                min(n_models, 3),
                subplot_spec=gs[3, 0:2],
                wspace=0.3,
            )
            scatter_axes = [fig.add_subplot(scatter_gs[0, i]) for i in range(min(n_models, 3))]
            self._scatter_panel.render(scatter_axes, panel_data)

        ax_residual = fig.add_subplot(gs[3, 2])
        self._residual_panel.render(ax_residual, panel_data)

        output_dir = self._ensure_output_dir("model_comparison")
        plot_path = output_dir / f"{experiment_id}_comparison_overview.png"
        return self._save_and_close(fig, plot_path)

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Main plot method; delegates to overview plotting."""
        experiment_id = kwargs.get("experiment_id", "default")
        context = kwargs.get("context", "run_model")
        return self.plot_model_comparison_overview(experiment_id, context)
