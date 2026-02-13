"""
Workflow runner component.

Displays all 16 workflow steps as cards with status icons.
Provides Run Full / Run Step / Force-rerun controls.
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

# Step descriptions (mirrors WorkflowCommands.WORKFLOW_STEPS)
STEP_INFO = {
    'setup_project': 'Initialize project directory structure and shapefiles',
    'create_pour_point': 'Create pour point shapefile from coordinates',
    'acquire_attributes': 'Download and process geospatial attributes',
    'define_domain': 'Define hydrological domain boundaries and river basins',
    'discretize_domain': 'Discretize domain into HRUs or modeling units',
    'process_observed_data': 'Process observational data (streamflow, etc.)',
    'acquire_forcings': 'Acquire meteorological forcing data',
    'model_agnostic_preprocessing': 'Model-agnostic preprocessing of forcing and attribute data',
    'model_specific_preprocessing': 'Setup model-specific input files and configuration',
    'run_model': 'Execute the hydrological model simulation',
    'calibrate_model': 'Run model calibration and parameter optimization',
    'run_emulation': 'Run emulation-based optimization if configured',
    'run_benchmarking': 'Run benchmarking analysis against observations',
    'run_decision_analysis': 'Run decision analysis for model comparison',
    'run_sensitivity_analysis': 'Run sensitivity analysis on model parameters',
    'postprocess_results': 'Postprocess and finalize model results',
}

STEP_ORDER = list(STEP_INFO.keys())

# Status icons
_ICONS = {
    'pending': '\u25CB',   # empty circle
    'complete': '\u25CF',  # filled circle (green via CSS)
    'running': '\u25D4',   # half circle
    'failed': '\u2716',    # cross
}


class WorkflowRunner(param.Parameterized):
    """Step-by-step workflow runner with background execution."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._thread = WorkflowThread(state)
        self._step_buttons = {}

        # Watch for status refreshes
        self.state.param.watch(self._on_status_change, ['workflow_status', 'is_running'])

    def _get_step_status(self, step_name):
        """Return 'complete', 'running', or 'pending' for a step."""
        if self.state.is_running and self.state.running_step == step_name:
            return 'running'
        status = self.state.workflow_status
        if not status:
            return 'pending'
        for detail in status.get('step_details', []):
            if detail.get('name') == step_name or detail.get('cli_name') == step_name:
                return 'complete' if detail.get('complete') else 'pending'
        return 'pending'

    def _on_status_change(self, *events):
        """Update step card styling when workflow_status changes."""
        for name, btn in self._step_buttons.items():
            status = self._get_step_status(name)
            if status == 'complete':
                btn.button_type = 'success'
                btn.name = f"{_ICONS['complete']}  {name}"
            elif status == 'running':
                btn.button_type = 'warning'
                btn.name = f"{_ICONS['running']}  {name}"
            else:
                btn.button_type = 'default'
                btn.name = f"{_ICONS['pending']}  {name}"

    def run_step(self, step_name, force_rerun=False):
        """Public API: run one workflow step."""
        if self.state.typed_config is None:
            self.state.append_log("Load a config first.\n")
            return False
        self._thread.run_steps([step_name], force_rerun=force_rerun)
        return True

    def run_full(self, force_rerun=False):
        """Public API: run full workflow."""
        if self.state.typed_config is None:
            self.state.append_log("Load a config first.\n")
            return False
        self._thread.run_workflow(force_rerun=force_rerun)
        return True

    def panel(self):
        """Build the workflow runner panel."""
        # Controls
        step_selector = pn.widgets.Select(
            name='Step', options=STEP_ORDER, value=STEP_ORDER[0], width=250,
        )
        force_rerun = pn.widgets.Checkbox(name='Force re-run', value=False)

        run_step_btn = pn.widgets.Button(name='Run Step', button_type='primary', width=100)
        run_all_btn = pn.widgets.Button(name='Run Full Workflow', button_type='success', width=160)

        def _run_step(event):
            self.run_step(step_selector.value, force_rerun=force_rerun.value)

        def _run_all(event):
            self.run_full(force_rerun=force_rerun.value)

        run_step_btn.on_click(_run_step)
        run_all_btn.on_click(_run_all)

        def _set_controls_running(is_running):
            step_selector.disabled = is_running
            force_rerun.disabled = is_running
            run_step_btn.disabled = is_running
            run_all_btn.disabled = is_running

        self.state.param.watch(lambda event: _set_controls_running(bool(event.new)), 'is_running')
        _set_controls_running(bool(self.state.is_running))

        controls = pn.Row(step_selector, run_step_btn, run_all_btn, force_rerun)

        # Status summary
        def _status_text():
            ws = self.state.workflow_status
            if not ws:
                return "No status available — load a config and refresh."
            total = ws.get('total_steps', 0)
            done = ws.get('completed_steps', 0)
            return f"Progress: {done} / {total} steps completed"

        status_pane = pn.pane.Str(
            pn.bind(
                lambda ws, running, step: (
                    f"Running: {step or 'workflow'}"
                    if running else _status_text()
                ),
                self.state.param.workflow_status,
                self.state.param.is_running,
                self.state.param.running_step,
            ),
            sizing_mode='stretch_width',
            styles={'font-weight': 'bold', 'font-size': '14px'},
        )

        # Step cards
        step_column = pn.Column(sizing_mode='stretch_width')
        for name in STEP_ORDER:
            desc = STEP_INFO[name]
            btn = pn.widgets.Button(
                name=f"{_ICONS['pending']}  {name}",
                button_type='default',
                sizing_mode='stretch_width',
            )
            self._step_buttons[name] = btn

            # Click individual step button to run it
            def _make_click_handler(step_name):
                def handler(event):
                    self.run_step(step_name, force_rerun=force_rerun.value)
                return handler

            btn.on_click(_make_click_handler(name))

            tooltip = pn.pane.Str(desc, styles={'color': '#666', 'font-size': '11px'})
            step_column.append(pn.Column(btn, tooltip, margin=(2, 0)))

        # Force initial status update
        self._on_status_change()

        return pn.Column(
            "## Workflow Steps",
            controls,
            status_pane,
            pn.layout.Divider(),
            step_column,
            sizing_mode='stretch_both',
            scroll=True,
        )
