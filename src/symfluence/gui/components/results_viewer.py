"""
Interactive results viewer component.

Provides five sub-tabs:
1. Hydrograph — observed vs simulated streamflow (Bokeh)
2. Calibration Progress — score convergence and parameter evolution
3. Metrics Summary — color-coded performance metric table
4. Flow Duration Curve — log-log exceedance plot
5. Saved Plots — legacy static image browser
"""

import logging
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np
import panel as pn
import param

from bokeh.models import ColumnDataSource, Span
from bokeh.plotting import figure

from ..data.results_loader import ResultsLoader
from ..utils.threading_utils import run_on_ui_thread

logger = logging.getLogger(__name__)
bokeh_figure = cast(Any, figure)

# Category -> subdirectory mapping (for legacy Saved Plots tab)
RESULT_CATEGORIES = {
    'Domain': 'domain',
    'Forcing': 'forcing',
    'Observations': 'observations',
    'Model Output': 'model_output',
    'Calibration': 'calibration',
    'Benchmarking': 'benchmarking',
    'Sensitivity': 'sensitivity',
    'Diagnostics': 'diagnostics',
    'All Plots': '',
}

# Primary metrics to display in the summary table
_PRIMARY_METRICS = ['KGE', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2', 'VE']


class ResultsViewer(param.Parameterized):
    """Interactive results viewer with Bokeh plots and legacy image browser."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(
        self,
        state,
        run_step_callback=None,
        run_full_callback=None,
        load_demo_callback=None,
        **kw,
    ):
        super().__init__(state=state, **kw)
        self._loader = None
        self._loader_key = None
        self._tabs = None
        self._file_list = []
        self._diagnostics_running = False
        self._run_step_callback = run_step_callback
        self._run_full_callback = run_full_callback
        self._load_demo_callback = load_demo_callback

    # ------------------------------------------------------------------
    # Loader management
    # ------------------------------------------------------------------

    def _current_experiment_id(self):
        """Resolve currently selected experiment id for result loading."""
        cfg = self.state.typed_config
        if cfg is not None:
            try:
                exp_id = cfg.domain.experiment_id
                if exp_id:
                    return exp_id
            except Exception:
                pass
        return self.state.last_completed_run

    def _get_loader(self):
        """Return (or create) a ResultsLoader for the current project."""
        loader_key = (
            self.state.project_dir,
            id(self.state.typed_config),
            self._current_experiment_id(),
        )
        if self._loader is None or self._loader_key != loader_key:
            self._loader = ResultsLoader(
                self.state.project_dir, self.state.typed_config
            )
            self._loader_key = loader_key
        return self._loader

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _run_step_action(self, step_name):
        if self._run_step_callback is None:
            self.state.append_log(f"Run action unavailable for step: {step_name}\n")
            return
        try:
            self._run_step_callback(step_name, force_rerun=False)
        except TypeError:
            self._run_step_callback(step_name)

    def _run_full_action(self):
        if self._run_full_callback is None:
            self.state.append_log("Run-full action unavailable.\n")
            return
        try:
            self._run_full_callback(force_rerun=False)
        except TypeError:
            self._run_full_callback()

    def _load_demo_action(self):
        if self._load_demo_callback is None:
            self.state.append_log("Demo action unavailable.\n")
            return
        self._load_demo_callback()

    def _actionable_alert(self, message, alert_type='info', actions=None):
        """Render a compact alert with optional one-click action buttons."""
        actions = actions or []
        buttons = []
        for spec in actions:
            label = spec.get('label', 'Action')
            callback = spec.get('callback')
            button_type = spec.get('button_type', 'light')
            disable_while_running = bool(spec.get('disable_while_running', False))
            btn = pn.widgets.Button(name=label, button_type=button_type, width=190)
            if callback is not None:
                def _invoke_callback(_event: Any, fn=callback) -> None:
                    fn()

                btn.on_click(_invoke_callback)
            if disable_while_running:
                btn.disabled = bool(self.state.is_running)
                def _sync_disabled(event: Any, button=btn) -> None:
                    setattr(button, 'disabled', bool(event.new))

                self.state.param.watch(
                    _sync_disabled,
                    'is_running',
                )
            buttons.append(btn)

        row = (
            pn.FlexBox(
                *buttons,
                sizing_mode='stretch_width',
                styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
            )
            if buttons
            else pn.Spacer(height=1)
        )
        return pn.Column(
            pn.pane.Alert(message, alert_type=alert_type, sizing_mode='stretch_width'),
            row,
            sizing_mode='stretch_width',
        )

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_hydrograph_tab(self):
        """Tab 1: interactive obs vs sim hydrograph."""
        container = pn.Column(sizing_mode='stretch_both')

        loader = self._get_loader()
        experiment_id = self._current_experiment_id()
        obs = loader.load_observed_streamflow()
        sim = loader.load_simulated_streamflow(experiment_id=experiment_id)

        if obs is None and sim is None:
            container.append(self._actionable_alert(
                "No streamflow data found. Run the model step or full workflow, then refresh results.",
                alert_type='info',
                actions=[
                    {
                        'label': 'Run run_model',
                        'callback': lambda: self._run_step_action('run_model'),
                        'button_type': 'primary',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Run Full Workflow',
                        'callback': self._run_full_action,
                        'button_type': 'success',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Load Demo',
                        'callback': self._load_demo_action,
                        'button_type': 'light',
                    },
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                ],
            ))
            return container

        if obs is None or len(obs) == 0 or sim is None or len(sim) == 0:
            container.append(self._actionable_alert(
                "Streamflow data is incomplete. Run a simulation and refresh this tab.",
                alert_type='warning',
                actions=[
                    {
                        'label': 'Run run_model',
                        'callback': lambda: self._run_step_action('run_model'),
                        'button_type': 'warning',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                ],
            ))
            return container

        # Build Bokeh figure
        p = bokeh_figure(
            title='Hydrograph',
            x_axis_type='datetime',
            tools='pan,wheel_zoom,box_zoom,reset,hover',
            active_scroll='wheel_zoom',
            sizing_mode='stretch_both',
            height=420,
        )
        p.yaxis.axis_label = 'Streamflow'
        p.xaxis.axis_label = 'Date'

        if obs is not None and len(obs) > 0:
            obs_src = ColumnDataSource(data=dict(
                x=obs.index, y=obs.values,
            ))
            p.line('x', 'y', source=obs_src, line_width=1.5,
                   color='#2c3e50', legend_label='Observed')

        if sim is not None and len(sim) > 0:
            sim_src = ColumnDataSource(data=dict(
                x=sim.index, y=sim.values,
            ))
            p.line('x', 'y', source=sim_src, line_width=1.5,
                   color='#e74c3c', legend_label='Simulated')

        p.legend.click_policy = 'hide'
        p.legend.location = 'top_right'

        container.append(pn.pane.Bokeh(p, sizing_mode='stretch_both'))

        # Date range slider if we have data
        all_dates = []
        if obs is not None and len(obs) > 0:
            all_dates.extend([obs.index.min(), obs.index.max()])
        if sim is not None and len(sim) > 0:
            all_dates.extend([sim.index.min(), sim.index.max()])

        if len(all_dates) >= 2:
            d_min = min(all_dates)
            d_max = max(all_dates)
            # Convert to python datetime for the slider
            if hasattr(d_min, 'to_pydatetime'):
                d_min = d_min.to_pydatetime()
            if hasattr(d_max, 'to_pydatetime'):
                d_max = d_max.to_pydatetime()
            # Ensure timezone-naive
            if hasattr(d_min, 'tzinfo') and d_min.tzinfo is not None:
                d_min = d_min.replace(tzinfo=None)
            if hasattr(d_max, 'tzinfo') and d_max.tzinfo is not None:
                d_max = d_max.replace(tzinfo=None)

            slider = pn.widgets.DateRangeSlider(
                name='Date Range',
                start=d_min, end=d_max,
                value=(d_min, d_max),
                sizing_mode='stretch_width',
            )

            def _on_range(event):
                start, end = event.new
                p.x_range.start = int(start.timestamp() * 1000) if hasattr(start, 'timestamp') else start
                p.x_range.end = int(end.timestamp() * 1000) if hasattr(end, 'timestamp') else end

            slider.param.watch(_on_range, 'value')
            container.append(slider)

        return container

    def _build_calibration_tab(self):
        """Tab 2: calibration convergence + parameter evolution."""
        container = pn.Column(sizing_mode='stretch_both')

        loader = self._get_loader()
        experiment_id = self._current_experiment_id()
        df = loader.load_optimization_history(experiment_id=experiment_id)

        if df is None or df.empty:
            container.append(self._actionable_alert(
                "No calibration history found. Run calibration for this experiment.",
                alert_type='info',
                actions=[
                    {
                        'label': 'Run calibrate_model',
                        'callback': lambda: self._run_step_action('calibrate_model'),
                        'button_type': 'primary',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Run Full Workflow',
                        'callback': self._run_full_action,
                        'button_type': 'success',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                ],
            ))
            return container

        # --- Top figure: score vs iteration ---
        p_score = bokeh_figure(
            title='Calibration Convergence',
            tools='pan,wheel_zoom,box_zoom,reset,hover',
            active_scroll='wheel_zoom',
            sizing_mode='stretch_both',
            height=260,
        )
        p_score.xaxis.axis_label = 'Iteration'
        p_score.yaxis.axis_label = 'Score'

        if 'iteration' in df.columns and 'score' in df.columns:
            iterations = df['iteration'].values
            scores = df['score'].values
            src = ColumnDataSource(data=dict(x=iterations, y=scores))
            p_score.line('x', 'y', source=src, line_width=1.5, color='#2980b9')
            p_score.scatter('x', 'y', source=src, size=3, color='#2980b9', alpha=0.4)

            # Best score marker
            valid_mask = ~np.isnan(scores)
            if valid_mask.any():
                best_idx = np.nanargmax(scores)
                best_score = scores[best_idx]
                p_score.add_layout(
                    Span(location=best_score, dimension='width',
                         line_color='#27ae60', line_dash='dashed', line_width=1.5)
                )
                p_score.scatter(
                    [iterations[best_idx]], [best_score],
                    size=12, color='#27ae60', marker='star',
                    legend_label=f'Best: {best_score:.4f}',
                )
                p_score.legend.location = 'bottom_right'

        container.append(pn.pane.Bokeh(p_score, sizing_mode='stretch_both'))

        # --- Bottom figure: parameter evolution ---
        # Identify parameter columns (exclude iteration, score, and other metadata)
        exclude = {'iteration', 'score', 'index', 'worker', 'trial'}
        param_cols = [c for c in df.columns if c.lower() not in exclude
                      and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if param_cols:
            # Normalize parameters to [0, 1] range for comparison
            param_selector = pn.widgets.MultiChoice(
                name='Parameters',
                options=param_cols,
                value=param_cols[:min(5, len(param_cols))],
                sizing_mode='stretch_width',
            )

            p_params = bokeh_figure(
                title='Parameter Evolution (normalized)',
                tools='pan,wheel_zoom,box_zoom,reset',
                active_scroll='wheel_zoom',
                sizing_mode='stretch_both',
                height=220,
            )
            p_params.xaxis.axis_label = 'Iteration'
            p_params.yaxis.axis_label = 'Normalized Value'

            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                      '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
            renderers = {}

            iterations = df['iteration'].values if 'iteration' in df.columns else np.arange(len(df))
            for i, col in enumerate(param_cols):
                vals = df[col].values.astype(float)
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(vals, 0.5)
                src = ColumnDataSource(data=dict(x=iterations, y=norm))
                color = colors[i % len(colors)]
                r = p_params.line('x', 'y', source=src, line_width=1.2,
                                  color=color, legend_label=col)
                renderers[col] = r

            p_params.legend.click_policy = 'hide'
            p_params.legend.location = 'top_left'
            p_params.legend.label_text_font_size = '9px'

            def _on_param_select(event):
                selected = set(event.new)
                for col, r in renderers.items():
                    r.visible = col in selected

            param_selector.param.watch(_on_param_select, 'value')
            # Set initial visibility
            initial = set(param_selector.value)
            for col, r in renderers.items():
                r.visible = col in initial

            container.append(param_selector)
            container.append(pn.pane.Bokeh(p_params, sizing_mode='stretch_both'))

        return container

    def _build_metrics_tab(self):
        """Tab 3: color-coded metrics summary table."""
        container = pn.Column(sizing_mode='stretch_both')

        loader = self._get_loader()
        experiment_id = self._current_experiment_id()
        obs = loader.load_observed_streamflow()
        sim = loader.load_simulated_streamflow(experiment_id=experiment_id)

        if obs is None or sim is None:
            container.append(self._actionable_alert(
                "Need both observed and simulated data to compute metrics.",
                alert_type='info',
                actions=[
                    {
                        'label': 'Run run_model',
                        'callback': lambda: self._run_step_action('run_model'),
                        'button_type': 'primary',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                ],
            ))
            return container

        try:
            from symfluence.reporting.core.plot_utils import align_timeseries
            obs_aligned, sim_aligned = align_timeseries(obs, sim)
        except Exception:
            # Manual alignment fallback
            common = obs.index.intersection(sim.index)
            obs_aligned = obs.loc[common]
            sim_aligned = sim.loc[common]

        if len(obs_aligned) == 0 or len(sim_aligned) == 0:
            container.append(self._actionable_alert(
                "No overlapping time period between observed and simulated series.",
                alert_type='warning',
                actions=[
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    }
                ],
            ))
            return container

        metrics = loader.calculate_metrics(obs_aligned.values, sim_aligned.values)
        if not metrics:
            container.append(self._actionable_alert(
                "Metrics calculation failed for this experiment output.",
                alert_type='danger',
                actions=[
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                    {
                        'label': 'Run postprocess_results',
                        'callback': lambda: self._run_step_action('postprocess_results'),
                        'button_type': 'warning',
                        'disable_while_running': True,
                    },
                ],
            ))
            return container

        # Try to get interpretation labels
        try:
            from symfluence.evaluation.metrics import interpret_metric
            has_interpret = True
        except ImportError:
            has_interpret = False

        # Build HTML table
        rows = []
        for name in _PRIMARY_METRICS:
            # Metric keys may be lowercase
            val = metrics.get(name) or metrics.get(name.lower())
            if val is None:
                continue
            # Determine color
            color = self._metric_color(name, val)
            interpretation = ''
            if has_interpret:
                try:
                    interpretation = interpret_metric(name, val)
                except Exception:
                    pass
            rows.append(
                f'<tr style="background-color:{color}">'
                f'<td style="padding:6px 12px;font-weight:bold">{name}</td>'
                f'<td style="padding:6px 12px;text-align:right">{val:.4f}</td>'
                f'<td style="padding:6px 12px">{interpretation}</td>'
                f'</tr>'
            )

        if rows:
            html = (
                '<table style="border-collapse:collapse;width:100%;max-width:700px">'
                '<thead><tr style="background:#2c3e50;color:white">'
                '<th style="padding:8px 12px;text-align:left">Metric</th>'
                '<th style="padding:8px 12px;text-align:right">Value</th>'
                '<th style="padding:8px 12px;text-align:left">Interpretation</th>'
                '</tr></thead><tbody>'
                + '\n'.join(rows) +
                '</tbody></table>'
            )
            container.append(pn.pane.HTML(html, sizing_mode='stretch_width'))
        else:
            container.append(pn.pane.Alert("No primary metrics available.", alert_type='info'))

        return container

    @staticmethod
    def _metric_color(name, value):
        """Return background color for metric quality."""
        name_upper = name.upper()
        # For metrics where higher is better (optimal ~1.0)
        if name_upper in ('KGE', 'NSE', 'R2', 'VE', 'KGEP', 'KGENP'):
            if value >= 0.7:
                return '#d5f5e3'  # green
            elif value >= 0.4:
                return '#fef9e7'  # yellow
            else:
                return '#fadbd8'  # red
        # For PBIAS — lower absolute value is better
        elif name_upper == 'PBIAS':
            if abs(value) < 10:
                return '#d5f5e3'
            elif abs(value) < 25:
                return '#fef9e7'
            else:
                return '#fadbd8'
        # For error metrics (lower is better) — no universal thresholds
        else:
            return '#fafafa'

    def _build_fdc_tab(self):
        """Tab 4: flow duration curve (log-log)."""
        container = pn.Column(sizing_mode='stretch_both')

        loader = self._get_loader()
        experiment_id = self._current_experiment_id()
        obs = loader.load_observed_streamflow()
        sim = loader.load_simulated_streamflow(experiment_id=experiment_id)

        if obs is None and sim is None:
            container.append(self._actionable_alert(
                "No streamflow data available for the flow duration curve.",
                alert_type='info',
                actions=[
                    {
                        'label': 'Run run_model',
                        'callback': lambda: self._run_step_action('run_model'),
                        'button_type': 'primary',
                        'disable_while_running': True,
                    },
                    {
                        'label': 'Refresh Results',
                        'callback': self.refresh,
                        'button_type': 'light',
                    },
                ],
            ))
            return container

        p = bokeh_figure(
            title='Flow Duration Curve',
            x_axis_type='log',
            y_axis_type='log',
            tools='pan,wheel_zoom,box_zoom,reset,hover',
            active_scroll='wheel_zoom',
            sizing_mode='stretch_both',
            height=420,
        )
        p.xaxis.axis_label = 'Exceedance Probability'
        p.yaxis.axis_label = 'Streamflow'

        if obs is not None and len(obs) > 0:
            exc_obs, flows_obs = loader.calculate_fdc(obs)
            if exc_obs is not None:
                src = ColumnDataSource(data=dict(x=exc_obs, y=flows_obs))
                p.line('x', 'y', source=src, line_width=2,
                       color='#2c3e50', legend_label='Observed')

        if sim is not None and len(sim) > 0:
            exc_sim, flows_sim = loader.calculate_fdc(sim)
            if exc_sim is not None:
                src = ColumnDataSource(data=dict(x=exc_sim, y=flows_sim))
                p.line('x', 'y', source=src, line_width=2,
                       color='#e74c3c', legend_label='Simulated')

        p.legend.click_policy = 'hide'
        p.legend.location = 'top_right'

        container.append(pn.pane.Bokeh(p, sizing_mode='stretch_both'))
        return container

    def _build_saved_plots_tab(self):
        """Tab 5: legacy static image browser (preserved from original)."""
        category = pn.widgets.Select(
            name='Category',
            options=list(RESULT_CATEGORIES.keys()),
            value='All Plots',
            width=200,
        )
        refresh_btn = pn.widgets.Button(name='Refresh', button_type='primary', width=90)
        diagnose_btn = pn.widgets.Button(
            name='Generate Diagnostics', button_type='success', width=170
        )

        file_select = pn.widgets.Select(name='Plot', options=[], sizing_mode='stretch_width')
        display_area = pn.Column(sizing_mode='stretch_both')
        diagnostics_status = pn.pane.Str("", sizing_mode='stretch_width')

        def _set_diagnostics_running(running):
            self._diagnostics_running = running
            diagnose_btn.disabled = running
            refresh_btn.disabled = running
            category.disabled = running
            diagnostics_status.object = "Generating diagnostics..." if running else ""

        def _refresh(event=None):
            files = self._scan_plots(category.value)
            self._file_list = files
            file_select.options = (
                {f.name: str(f) for f in files} if files else {'(no plots found)': ''}
            )
            if not files:
                display_area[:] = [self._actionable_alert(
                    "No saved diagnostic images found in reporting outputs.",
                    alert_type='info',
                    actions=[
                        {
                            'label': 'Run postprocess_results',
                            'callback': lambda: self._run_step_action('postprocess_results'),
                            'button_type': 'primary',
                            'disable_while_running': True,
                        },
                        {
                            'label': 'Generate Diagnostics',
                            'callback': lambda: _run_diagnostics(None),
                            'button_type': 'success',
                            'disable_while_running': True,
                        },
                    ],
                )]
                return
            if file_select.value not in [str(f) for f in files]:
                file_select.value = str(files[0])
            display_area[:] = [self._plot_pane_for_file(Path(file_select.value))]

        def _on_file_select(event):
            path = event.new
            if not path:
                display_area.clear()
                return
            display_area.clear()
            fp = Path(path)
            display_area.append(self._plot_pane_for_file(fp))

        def _run_diagnostics(event):
            if self.state.typed_config is None:
                self.state.append_log("Load a config first.\n")
                return
            if self._diagnostics_running:
                self.state.append_log("Diagnostics are already running.\n")
                return

            _set_diagnostics_running(True)

            def _worker():
                try:
                    sf = self.state.initialize_symfluence()
                    results = sf.run_all_diagnostics()
                    if results:
                        run_on_ui_thread(
                            lambda: self.state.append_log(
                                f"Generated {len(results)} diagnostic plot(s).\n"
                            )
                        )
                        run_on_ui_thread(_refresh)
                    else:
                        run_on_ui_thread(
                            lambda: self.state.append_log(
                                "No diagnostics generated (check that outputs exist).\n"
                            )
                        )
                except Exception as exc:
                    run_on_ui_thread(lambda e=exc: self.state.append_log(f"Diagnostics failed: {e}\n"))
                finally:
                    run_on_ui_thread(lambda: _set_diagnostics_running(False))

            threading.Thread(target=_worker, daemon=True).start()

        file_select.param.watch(_on_file_select, 'value')
        refresh_btn.on_click(_refresh)
        category.param.watch(lambda e: _refresh(), 'value')
        diagnose_btn.on_click(_run_diagnostics)
        _refresh()

        return pn.Column(
            pn.Row(category, refresh_btn, diagnose_btn),
            diagnostics_status,
            file_select,
            pn.layout.Divider(),
            display_area,
            sizing_mode='stretch_both',
        )

    @staticmethod
    def _plot_pane_for_file(path: Path):
        """Return an appropriate Panel pane for a saved plot path."""
        suffix = path.suffix.lower()
        if suffix in ('.png', '.jpg', '.jpeg'):
            return pn.pane.Image(str(path), sizing_mode='scale_both', max_height=600)
        if suffix == '.svg':
            return pn.pane.SVG(str(path), sizing_mode='scale_both', max_height=600)
        if suffix == '.pdf':
            return pn.pane.Str(f"PDF file: {path.name}\nOpen externally to view.")
        return pn.pane.Str(f"Unsupported format: {path.suffix}")

    def _scan_plots(self, category='All Plots'):
        """Scan project directory for plot files."""
        project_dir = self.state.project_dir
        if not project_dir:
            return []

        base = Path(project_dir) / 'reporting'
        subdir = RESULT_CATEGORIES.get(category, '')
        search_dir = base / subdir if subdir else base

        if not search_dir.exists():
            return []

        files: list[Path] = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg'):
            files.extend(search_dir.rglob(ext))

        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------

    def _rebuild_tabs(self):
        """Rebuild all interactive tabs (called on refresh)."""
        if self._loader:
            self._loader.clear_cache()
        if self._tabs is not None:
            self._tabs[0] = ('Hydrograph', self._build_hydrograph_tab())
            self._tabs[1] = ('Calibration', self._build_calibration_tab())
            self._tabs[2] = ('Metrics', self._build_metrics_tab())
            self._tabs[3] = ('Flow Duration Curve', self._build_fdc_tab())
            # Saved Plots tab is index 4 — not rebuilt on refresh

    def refresh(self):
        """Public API: refresh all interactive results views."""
        self._rebuild_tabs()

    # ------------------------------------------------------------------
    # Main panel
    # ------------------------------------------------------------------

    def panel(self):
        """Return the results viewer panel with sub-tabs."""
        refresh_btn = pn.widgets.Button(
            name='Refresh Results', button_type='primary', width=130
        )

        self._tabs = pn.Tabs(
            ('Hydrograph', self._build_hydrograph_tab()),
            ('Calibration', self._build_calibration_tab()),
            ('Metrics', self._build_metrics_tab()),
            ('Flow Duration Curve', self._build_fdc_tab()),
            ('Saved Plots', self._build_saved_plots_tab()),
            sizing_mode='stretch_both',
            dynamic=True,
        )

        def _on_refresh(event=None):
            self._rebuild_tabs()

        refresh_btn.on_click(_on_refresh)

        # Watch state changes for auto-refresh
        self.state.param.watch(lambda e: self._rebuild_tabs(), ['last_completed_run'])
        self.state.param.watch(
            lambda e: self._on_project_change(), ['project_dir']
        )
        self.state.param.watch(
            lambda e: self._on_config_change(), ['typed_config', 'config_path']
        )

        return pn.Column(
            "## Results",
            pn.Row(refresh_btn),
            self._tabs,
            sizing_mode='stretch_both',
        )

    def _on_project_change(self):
        """Handle project directory change — create new loader."""
        self._loader = None
        self._loader_key = None
        self._rebuild_tabs()

    def _on_config_change(self):
        """Handle config updates (including experiment-id changes)."""
        self._loader = None
        self._loader_key = None
        self._rebuild_tabs()
