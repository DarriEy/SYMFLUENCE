"""
Iterative calibration run panel.

Provides quick-config widgets for calibration parameters, a "Save & Run"
button that launches calibration in the background, and a run-history table
that shows past experiments with their scores.
"""

import logging

import panel as pn
import param

from ..data.results_loader import ResultsLoader
from ..utils.config_bridge import params_to_config_overrides
from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

# Supported algorithm / metric options
_ALGORITHMS = ['PSO', 'DDS', 'DE', 'SCE-UA', 'GLUE', 'Basin-hopping']
_METRICS = ['KGE', 'KGEp', 'KGEnp', 'NSE', 'RMSE', 'MAE', 'PBIAS']


class IterativeRunPanel(param.Parameterized):
    """Quick-config + run + experiment history for iterative calibration."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._thread = WorkflowThread(state)
        self._history_pane = None
        self._widgets = {}

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _build_widgets(self):
        """Create the quick-config widgets, pre-populated from current config."""
        defaults = self._read_config_defaults()

        w = {}
        w['algorithm'] = pn.widgets.Select(
            name='Algorithm',
            options=_ALGORITHMS,
            value=defaults.get('optimization_algorithm', 'PSO'),
            width=140,
        )
        w['metric'] = pn.widgets.Select(
            name='Metric',
            options=_METRICS,
            value=defaults.get('optimization_metric', 'KGE'),
            width=110,
        )
        w['iterations'] = pn.widgets.IntInput(
            name='Iterations',
            value=defaults.get('iterations', 1000),
            start=1, step=100, width=110,
        )
        w['population_size'] = pn.widgets.IntInput(
            name='Pop Size',
            value=defaults.get('population_size', 50),
            start=1, step=10, width=100,
        )
        w['calibration_period'] = pn.widgets.TextInput(
            name='Calibration Period',
            value=defaults.get('calibration_period', ''),
            placeholder='e.g. 2000-01-01/2005-12-31',
            width=230,
        )
        w['evaluation_period'] = pn.widgets.TextInput(
            name='Evaluation Period',
            value=defaults.get('evaluation_period', ''),
            placeholder='e.g. 2006-01-01/2010-12-31',
            width=230,
        )
        w['experiment_id'] = pn.widgets.TextInput(
            name='Experiment ID',
            value=defaults.get('experiment_id', ''),
            placeholder='e.g. run_dds_200',
            width=200,
        )

        self._widgets = w
        return w

    def _read_config_defaults(self):
        """Extract current config values for pre-populating widgets."""
        if self.state.typed_config is None:
            return {}
        try:
            from ..utils.config_bridge import config_to_params
            return config_to_params(self.state.typed_config)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def _build_history_html(self):
        """Scan optimization dir and build an HTML table of past experiments."""
        loader = ResultsLoader(self.state.project_dir, self.state.typed_config)
        experiments = loader.list_experiments()

        if not experiments:
            return '<p style="color:#888">No previous experiments found.</p>'

        rows = []
        for exp_id in experiments:
            # Read iteration results for best score
            df = loader.load_optimization_history(exp_id)
            best_score = ''
            iters = ''
            if df is not None and not df.empty:
                if 'score' in df.columns:
                    import numpy as np
                    best_score = f'{np.nanmax(df["score"].values):.4f}'
                iters = str(len(df))

            # Try to get algorithm/metric from best_params JSON
            algorithm = ''
            metric = ''
            bp = loader.load_best_params(exp_id)
            if bp and isinstance(bp, dict):
                algorithm = bp.get('algorithm', bp.get('optimization_algorithm', ''))
                metric = bp.get('metric', bp.get('optimization_metric', ''))

            rows.append(
                f'<tr>'
                f'<td style="padding:4px 8px">{exp_id}</td>'
                f'<td style="padding:4px 8px">{algorithm}</td>'
                f'<td style="padding:4px 8px">{metric}</td>'
                f'<td style="padding:4px 8px;text-align:right">{best_score}</td>'
                f'<td style="padding:4px 8px;text-align:right">{iters}</td>'
                f'</tr>'
            )

        html = (
            '<table style="border-collapse:collapse;width:100%">'
            '<thead><tr style="background:#2c3e50;color:white">'
            '<th style="padding:6px 8px;text-align:left">Experiment</th>'
            '<th style="padding:6px 8px;text-align:left">Algorithm</th>'
            '<th style="padding:6px 8px;text-align:left">Metric</th>'
            '<th style="padding:6px 8px;text-align:right">Best Score</th>'
            '<th style="padding:6px 8px;text-align:right">Iters</th>'
            '</tr></thead><tbody>'
            + '\n'.join(rows)
            + '</tbody></table>'
        )
        return html

    def _refresh_history(self):
        """Rebuild the run history table."""
        if self._history_pane is not None:
            self._history_pane.object = self._build_history_html()

    # ------------------------------------------------------------------
    # Save & Run
    # ------------------------------------------------------------------

    def _save_and_run(self, event=None):
        """Apply widget overrides to config, save, and launch calibration."""
        if self.state.typed_config is None:
            self.state.append_log("Load a config before running calibration.\n")
            return

        if self.state.is_running:
            self.state.append_log("A workflow is already running.\n")
            return

        w = self._widgets

        # Build override dict from widget values
        overrides = params_to_config_overrides({
            'optimization_algorithm': w['algorithm'].value,
            'optimization_metric': w['metric'].value,
            'iterations': w['iterations'].value,
            'population_size': w['population_size'].value,
            'calibration_period': w['calibration_period'].value,
            'evaluation_period': w['evaluation_period'].value,
            'experiment_id': w['experiment_id'].value,
        })

        # Apply overrides to the typed config
        try:
            from symfluence.core.config.models import SymfluenceConfig
            current = self.state.typed_config.to_dict(flatten=True)
            current.update(overrides)
            self.state.typed_config = SymfluenceConfig(**current)
        except Exception as exc:
            self.state.append_log(f"Config update failed: {exc}\n")
            return

        # Save and invalidate
        self.state.save_config()
        self.state.invalidate_symfluence()

        # Launch calibration
        self.state.append_log(
            f"Starting calibration: {w['algorithm'].value}, "
            f"{w['metric'].value}, {w['iterations'].value} iters, "
            f"experiment={w['experiment_id'].value}\n"
        )
        self._thread.run_steps(['calibrate_model'], force_rerun=True)

    # ------------------------------------------------------------------
    # Load selected config
    # ------------------------------------------------------------------

    def _load_selected_experiment(self, event=None):
        """Load config from a selected past experiment into the widgets."""
        loader = ResultsLoader(self.state.project_dir, self.state.typed_config)
        experiments = loader.list_experiments()
        if not experiments:
            self.state.append_log("No experiments to load.\n")
            return

        # Use the experiment selector value
        exp_id = self._exp_selector.value if hasattr(self, '_exp_selector') else None
        if not exp_id:
            self.state.append_log("Select an experiment first.\n")
            return

        bp = loader.load_best_params(exp_id)
        w = self._widgets

        if bp and isinstance(bp, dict):
            alg = bp.get('algorithm', bp.get('optimization_algorithm', ''))
            if alg and alg in _ALGORITHMS:
                w['algorithm'].value = alg
            met = bp.get('metric', bp.get('optimization_metric', ''))
            if met and met in _METRICS:
                w['metric'].value = met

        w['experiment_id'].value = exp_id
        self.state.append_log(f"Loaded config from experiment: {exp_id}\n")

    # ------------------------------------------------------------------
    # Main panel
    # ------------------------------------------------------------------

    def panel(self):
        """Build and return the iterative calibration panel."""
        w = self._build_widgets()

        run_btn = pn.widgets.Button(
            name='Save & Run', button_type='success', width=120,
        )
        run_btn.on_click(self._save_and_run)

        # Disable button while running
        self.state.param.watch(
            lambda e: setattr(run_btn, 'disabled', e.new), ['is_running']
        )

        config_section = pn.Column(
            '### Quick Calibration Config',
            pn.Row(w['algorithm'], w['metric'], w['iterations'], w['population_size']),
            pn.Row(w['calibration_period'], w['evaluation_period']),
            pn.Row(w['experiment_id'], run_btn),
            sizing_mode='stretch_width',
        )

        # Run history section
        self._history_pane = pn.pane.HTML(
            self._build_history_html(), sizing_mode='stretch_width'
        )

        # Experiment selector for loading past configs
        loader = ResultsLoader(self.state.project_dir, self.state.typed_config)
        experiments = loader.list_experiments()
        self._exp_selector = pn.widgets.Select(
            name='Experiment',
            options=experiments if experiments else [],
            width=200,
        )
        load_btn = pn.widgets.Button(
            name='Load Selected Config', button_type='default', width=160,
        )
        load_btn.on_click(self._load_selected_experiment)

        history_section = pn.Column(
            '### Run History',
            self._history_pane,
            pn.Row(self._exp_selector, load_btn),
            sizing_mode='stretch_width',
        )

        # Auto-refresh when a run completes
        self.state.param.watch(
            lambda e: self._on_run_complete(), ['last_completed_run']
        )
        # Re-populate widgets when config changes
        self.state.param.watch(
            lambda e: self._on_config_change(), ['config_path']
        )

        return pn.Column(
            "## Calibrate",
            config_section,
            pn.layout.Divider(),
            history_section,
            sizing_mode='stretch_both',
        )

    def _on_run_complete(self):
        """Handle completed run — refresh history and experiment selector."""
        self._refresh_history()
        loader = ResultsLoader(self.state.project_dir, self.state.typed_config)
        experiments = loader.list_experiments()
        if hasattr(self, '_exp_selector'):
            self._exp_selector.options = experiments if experiments else []

    def _on_config_change(self):
        """Re-populate widgets from newly loaded config."""
        defaults = self._read_config_defaults()
        w = self._widgets
        if not w:
            return
        for key in ['algorithm', 'metric', 'iterations', 'population_size',
                     'calibration_period', 'evaluation_period', 'experiment_id']:
            config_key = {
                'algorithm': 'optimization_algorithm',
                'metric': 'optimization_metric',
            }.get(key, key)
            val = defaults.get(config_key)
            if val is not None and key in w:
                try:
                    w[key].value = val
                except Exception:
                    pass
