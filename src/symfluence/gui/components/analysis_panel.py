"""
Analysis panel for benchmarking, decision analysis, and sensitivity analysis.

Visible after model run completes (gui_phase >= 'model_ready').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_ANALYSIS_OPTIONS = ['benchmarking', 'decision', 'sensitivity']

# Map analysis names to workflow step names
_ANALYSIS_STEP_MAP = {
    'benchmarking': 'run_benchmarking',
    'decision': 'run_decision_analysis',
    'sensitivity': 'run_sensitivity_analysis',
}


class AnalysisPanel(param.Parameterized):
    """Sidebar panel for running post-calibration analyses."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        self._analyses = pn.widgets.MultiChoice(
            name='Analyses',
            options=_ANALYSIS_OPTIONS,
            value=['benchmarking'],
            **_WIDGET_KW,
        )
        self._run_btn = pn.widgets.Button(
            name='Run Analyses',
            button_type='primary',
            **_BTN_KW,
        )
        self._run_btn.on_click(self._on_run_analyses)

        # Phase and running-state sync
        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------
    # Phase visibility
    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        phase = event.new
        self._panel_card.visible = phase in (
            'model_ready', 'calibrated', 'analyzed',
        )

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _sync_running(self, event):
        self._run_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------
    # Action handler
    # ------------------------------------------------------------------

    def _on_run_analyses(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        selected = self._analyses.value or []
        if not selected:
            self.state.append_log("ERROR: Select at least one analysis.\n")
            return

        # Update evaluation.analyses via model_copy chain
        new_eval = cfg.evaluation.model_copy(update={'analyses': selected})
        self.state.typed_config = cfg.model_copy(update={'evaluation': new_eval})

        if self.state.config_path:
            self.state.save_config()
        self.state.invalidate_symfluence()

        # Build step list from selection
        step_list = [
            _ANALYSIS_STEP_MAP[a] for a in selected if a in _ANALYSIS_STEP_MAP
        ]

        self.state.append_log(f"Running analyses: {', '.join(selected)}...\n")
        self._wt.run_steps(step_list)

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._analyses,
            pn.layout.Divider(),
            self._run_btn,
            title='Analysis',
            collapsed=True,
            visible=self.state.gui_phase in (
                'model_ready', 'calibrated', 'analyzed',
            ),
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
