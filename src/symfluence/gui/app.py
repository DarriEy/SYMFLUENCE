"""
Main application: assembles a guided modeling workspace.
"""

import logging
from pathlib import Path
from typing import Any

import panel as pn

from .components.config_editor import ConfigEditor
from .components.gauge_setup_panel import GaugeSetupPanel
from .components.iterative_run_panel import IterativeRunPanel
from .components.log_viewer import LogViewer
from .components.map_view import MapView
from .components.results_viewer import ResultsViewer
from .components.workflow_runner import STEP_ORDER, WorkflowRunner
from .models.workflow_state import WorkflowState
from .utils.telemetry import UXTelemetry

logger = logging.getLogger(__name__)

# Design tokens + component styling for the v1 guided experience.
GUI_THEME_CSS = """
:root {
  --sf-bg: linear-gradient(170deg, #eef5fb 0%, #f8fbff 45%, #f2f7f9 100%);
  --sf-surface: #ffffff;
  --sf-surface-soft: #f7fbff;
  --sf-ink: #1b2f45;
  --sf-muted: #6e7f91;
  --sf-accent: #0f80d8;
  --sf-accent-soft: #dff0fe;
  --sf-line: #d7e2eb;
  --sf-success-soft: #e8f8ee;
  --sf-warning-soft: #fff6de;
  --sf-font: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
}
html, body {
  background: var(--sf-bg);
  color: var(--sf-ink);
  font-family: var(--sf-font);
}
.bk-root {
  font-family: var(--sf-font);
}
.sf-hero-title {
  font-weight: 700;
  letter-spacing: 0.2px;
  color: var(--sf-ink);
  font-size: 1.6rem;
}
.sf-hero-subtitle {
  color: var(--sf-muted);
  font-size: 0.98rem;
}
.sf-stage-caption {
  color: var(--sf-muted);
  font-size: 0.88rem;
}
.sf-status-row {
  color: var(--sf-muted);
  font-size: 0.9rem;
}
.sf-command-title {
  color: var(--sf-ink);
  font-weight: 600;
  font-size: 0.93rem;
}
.sf-artifact-title {
  font-size: 0.82rem;
  color: var(--sf-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.sf-artifact-count {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--sf-ink);
}
.sf-wizard-list {
  margin: 0;
  padding-left: 1.1rem;
  color: var(--sf-ink);
}
.sf-wizard-list li {
  margin-bottom: 2px;
}
"""


_STAGE_ORDER = ['Domain', 'Forcing', 'Model', 'Run', 'Analyze']
_STAGE_ICONS = {
    'pending': '\u25CB',
    'running': '\u25D4',
    'done': '\u25CF',
}

_STAGE_STEP_MAP = {
    'Domain': [
        'setup_project',
        'create_pour_point',
        'acquire_attributes',
        'define_domain',
        'discretize_domain',
    ],
    'Forcing': [
        'process_observed_data',
        'acquire_forcings',
        'model_agnostic_preprocessing',
    ],
    'Model': [
        'model_specific_preprocessing',
    ],
    'Run': [
        'run_model',
        'calibrate_model',
        'run_emulation',
    ],
    'Analyze': [
        'run_benchmarking',
        'run_decision_analysis',
        'run_sensitivity_analysis',
        'postprocess_results',
    ],
}


class SymfluenceApp:
    """Assembles all GUI components into a guided BootstrapTemplate."""

    def __init__(self, config_path=None, demo=None):
        self.state = WorkflowState()
        self.telemetry = UXTelemetry(enabled=False)
        self._last_run_step = None
        self._last_log_len = 0

        # Shared-state components.
        self.config_editor = ConfigEditor(self.state)
        self.map_view = MapView(self.state)
        self.workflow_runner = WorkflowRunner(self.state)
        self.log_viewer = LogViewer(self.state)
        self.results_viewer = ResultsViewer(
            self.state,
            run_step_callback=self.workflow_runner.run_step,
            run_full_callback=self.workflow_runner.run_full,
            load_demo_callback=lambda: self._load_demo('bow'),
        )
        self.iterative_panel = IterativeRunPanel(self.state)
        self.gauge_setup = GaugeSetupPanel(self.state)

        # Auto-load config if provided via CLI.
        if config_path:
            try:
                self.state.load_config(config_path)
            except Exception as exc:
                self.state.append_log(f"Failed to load initial config: {exc}\\n")

        # Demo mode (runs after components are wired up).
        if demo:
            self._load_demo(demo)

        self.state.param.watch(self._on_run_state_change, 'is_running')
        self.state.param.watch(self._on_log_change, 'log_text')
        self.state.param.watch(self._on_results_synced, 'last_completed_run')

    def build(self):
        """Build and return the guided BootstrapTemplate."""
        template = pn.template.BootstrapTemplate(
            title='SYMFLUENCE',
            sidebar_width=0,
            header_background='#1b2f45',
        )

        stage_views = self._build_stage_views()
        stage_names = list(stage_views.keys())
        active = {'idx': 0}
        selector_guard = {'active': False}

        hero = pn.Column(
            pn.pane.HTML("<div class='sf-hero-title'>SYMFLUENCE v1 Workspace</div>"),
            pn.pane.HTML(
                "<div class='sf-hero-subtitle'>"
                "A guided left-to-right modeling narrative: Domain -> Forcing -> Model -> Run -> Analyze."
                "</div>"
            ),
            sizing_mode='stretch_width',
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '14px 18px',
                'margin-bottom': '10px',
            },
        )

        command_palette = self._build_command_palette()
        stage_caption = pn.pane.HTML(
            "<div class='sf-stage-caption'>Step 1 of 5</div>",
            sizing_mode='stretch_width',
        )
        stage_content = pn.Column(sizing_mode='stretch_both', min_height=760)

        stage_buttons: dict[str, Any] = {}
        stage_selector = pn.widgets.Select(
            name='Stage',
            options={f"{idx + 1}. {name}": idx for idx, name in enumerate(stage_names)},
            value=0,
            width=240,
        )

        def _button_type_for(state, is_active):
            if is_active:
                return {
                    'pending': 'primary',
                    'running': 'warning',
                    'done': 'success',
                }[state]
            return {
                'pending': 'light',
                'running': 'warning',
                'done': 'success',
            }[state]

        def _refresh_stage_rail():
            stage_states = self._compute_stage_states()
            current_stage = stage_names[active['idx']]
            current_state = stage_states[current_stage]
            symbol = _STAGE_ICONS[current_state]
            stage_caption.object = (
                "<div class='sf-stage-caption'>"
                f"Step {active['idx'] + 1} of {len(stage_names)} - {symbol} {current_stage} ({current_state})"
                "</div>"
            )
            for idx, name in enumerate(stage_names):
                state = stage_states[name]
                btn = stage_buttons[name]
                btn.name = f"{idx + 1}. {_STAGE_ICONS[state]} {name} [{state}]"
                btn.button_type = _button_type_for(state, idx == active['idx'])
            refresh_wizard(stage_states)
            refresh_guidance(stage_states)
            refresh_artifacts(stage_states)

        def _activate_stage(idx, source='rail'):
            active['idx'] = idx
            stage_content[:] = [stage_views[stage_names[idx]]]
            selector_guard['active'] = True
            stage_selector.value = idx
            selector_guard['active'] = False
            self._record_telemetry(
                'stage_activated',
                stage=stage_names[idx],
                source=source,
            )
            _refresh_stage_rail()

        quickstart, refresh_wizard = self._build_quickstart_wizard(stage_names, _activate_stage)
        guidance_panel, refresh_guidance = self._build_guidance_panel(stage_names, _activate_stage)
        artifact_strip, refresh_artifacts = self._build_artifact_strip(stage_names, _activate_stage)

        for idx, name in enumerate(stage_names):
            btn = pn.widgets.Button(
                name=f"{idx + 1}. {name}",
                button_type='light',
                width=170,
                height=42,
            )
            btn.on_click(lambda event, i=idx: _activate_stage(i, source='stage_rail'))
            stage_buttons[name] = btn

        stage_rail = pn.FlexBox(
            *[stage_buttons[name] for name in stage_names],
            sizing_mode='stretch_width',
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '10px',
                'column-gap': '8px',
                'row-gap': '8px',
                'flex-wrap': 'wrap',
            },
        )

        nav_back = pn.widgets.Button(name='Back', button_type='light', width=110)
        nav_next = pn.widgets.Button(name='Next', button_type='success', width=130)

        def _go_back(event):
            if active['idx'] > 0:
                _activate_stage(active['idx'] - 1, source='back_button')

        def _go_next(event):
            if active['idx'] < len(stage_names) - 1:
                _activate_stage(active['idx'] + 1, source='next_button')

        nav_back.on_click(_go_back)
        nav_next.on_click(_go_next)

        def _on_stage_select(event):
            if selector_guard['active']:
                return
            _activate_stage(int(event.new), source='stage_selector')

        stage_selector.param.watch(_on_stage_select, 'value')

        nav_row = pn.FlexBox(
            stage_caption,
            pn.Spacer(sizing_mode='stretch_width'),
            stage_selector,
            nav_back,
            nav_next,
            sizing_mode='stretch_width',
            styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
        )

        status_bar = pn.pane.HTML(
            pn.bind(
                self._build_status_html,
                self.state.param.config_path,
                self.state.param.project_dir,
                self.state.param.workflow_status,
                self.state.param.is_running,
                self.state.param.config_dirty,
            ),
            sizing_mode='stretch_width',
            styles={
                'background': 'var(--sf-surface-soft)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '12px',
                'padding': '10px 14px',
                'margin-top': '10px',
            },
        )

        shell = pn.Column(
            hero,
            quickstart,
            command_palette,
            stage_rail,
            artifact_strip,
            guidance_panel,
            nav_row,
            stage_content,
            status_bar,
            sizing_mode='stretch_both',
        )
        template.main.append(shell)

        self.state.param.watch(
            lambda event: _refresh_stage_rail(),
            [
                'workflow_status',
                'is_running',
                'running_step',
                'config_path',
                'project_dir',
                'last_completed_run',
                'typed_config',
            ],
        )

        _activate_stage(0)

        return template

    def _build_command_palette(self):
        """Build compact expert controls for frequent operations."""
        step_selector = pn.widgets.Select(
            name='Step',
            options=STEP_ORDER,
            value=STEP_ORDER[0],
            width=210,
        )
        force_rerun = pn.widgets.Checkbox(name='Force', value=False, width=70)
        run_step_btn = pn.widgets.Button(name='Run Step', button_type='primary', width=120)

        demo_selector = pn.widgets.Select(
            name='Demo',
            options=['bow'],
            value='bow',
            width=110,
        )
        load_demo_btn = pn.widgets.Button(name='Load Demo', button_type='light', width=120)
        refresh_results_btn = pn.widgets.Button(name='Refresh Results', button_type='success', width=140)
        telemetry_toggle = pn.widgets.Checkbox(
            name='Telemetry',
            value=bool(self.telemetry.enabled),
            width=120,
        )

        def _run_step(event):
            self.workflow_runner.run_step(step_selector.value, force_rerun=force_rerun.value)
            self._record_telemetry(
                'command_palette_run_step',
                step=step_selector.value,
                force=bool(force_rerun.value),
            )

        def _load_demo(event):
            self._load_demo(demo_selector.value)
            self._record_telemetry('command_palette_load_demo', demo=demo_selector.value)

        def _refresh_results(event):
            self.results_viewer.refresh()
            self._record_telemetry('command_palette_refresh_results')

        def _toggle_telemetry(event):
            self.telemetry.set_enabled(bool(event.new))
            status = 'enabled' if event.new else 'disabled'
            self.state.append_log(
                f"GUI telemetry {status}: {self.telemetry.path}\n"
            )

        run_step_btn.on_click(_run_step)
        load_demo_btn.on_click(_load_demo)
        refresh_results_btn.on_click(_refresh_results)
        telemetry_toggle.param.watch(_toggle_telemetry, 'value')

        def _sync_run_enabled(*events):
            run_step_btn.disabled = bool(self.state.is_running) or self.state.typed_config is None
            refresh_results_btn.disabled = bool(self.state.is_running)
            load_demo_btn.disabled = bool(self.state.is_running)
            telemetry_toggle.disabled = bool(self.state.is_running)

        self.state.param.watch(lambda event: _sync_run_enabled(), 'is_running')
        self.state.param.watch(lambda event: _sync_run_enabled(), 'typed_config')
        _sync_run_enabled()

        return pn.Column(
            pn.pane.HTML("<div class='sf-command-title'>Command Palette</div>"),
            pn.FlexBox(
                step_selector,
                force_rerun,
                run_step_btn,
                pn.Spacer(width=16),
                demo_selector,
                load_demo_btn,
                refresh_results_btn,
                telemetry_toggle,
                sizing_mode='stretch_width',
                styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
            ),
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '10px 14px',
                'margin': '0 0 10px 0',
            },
            sizing_mode='stretch_width',
        )

    def _build_stage_views(self):
        """Assemble stage-specific workspace panels."""
        domain_fields = [
            'domain_name',
            'pour_point_coords',
            'bounding_box_coords',
            'definition_method',
            'discretization',
            'stream_threshold',
        ]
        forcing_fields = [
            'time_start',
            'time_end',
            'forcing_dataset',
            'calibration_period',
            'evaluation_period',
        ]
        model_fields = [
            'hydrological_model',
            'experiment_id',
            'optimization_algorithm',
            'optimization_metric',
            'iterations',
            'population_size',
        ]

        domain_map = self._surface_card(
            "Domain Map",
            "Select a pour point, inspect catchment geometry, and load gauge context.",
            self.map_view.panel(),
        )
        domain_settings = self._surface_card(
            "Domain Settings",
            "Fine-tune basin definition controls before running domain setup.",
            self.config_editor.stage_panel(
                title='Domain',
                description='Only domain-shaping fields are shown here.',
                parameters=domain_fields,
            ),
        )
        gauge_domain_setup = self._surface_card(
            "Gauge To Domain",
            "Select a gauge and generate a domain-ready config in one action.",
            self.gauge_setup.panel(),
        )
        domain_stage = pn.Row(
            domain_map,
            pn.Column(gauge_domain_setup, domain_settings, width=430, sizing_mode='stretch_height'),
            sizing_mode='stretch_both',
        )

        forcing_stage = self._surface_card(
            "Forcing",
            "Load a config file and set forcing/time windows for acquisition and preprocessing.",
            self.config_editor.stage_panel(
                title='Forcing',
                description='Forcing and period controls for data preparation.',
                parameters=forcing_fields,
                include_file_controls=True,
            ),
        )

        model_stage = pn.Row(
            self._surface_card(
                "Model Setup",
                "Choose model and optimization defaults used by upcoming runs.",
                self.config_editor.stage_panel(
                    title='Model',
                    description='Model identity and optimization controls only.',
                    parameters=model_fields,
                ),
            ),
            self._surface_card(
                "Calibration Workspace",
                "Fast iteration controls and experiment history for model tuning.",
                self.iterative_panel.panel(),
            ),
            sizing_mode='stretch_both',
        )

        run_stage = pn.Row(
            self._surface_card(
                "Run",
                "Launch full workflows or selected steps with explicit execution state.",
                self.workflow_runner.panel(),
            ),
            self._surface_card(
                "Runtime Logs",
                "Monitor execution output and diagnostics as the workflow runs.",
                self.log_viewer.panel(),
            ),
            sizing_mode='stretch_both',
        )

        analyze_stage = self._surface_card(
            "Analyze",
            "Review hydrographs, calibration trajectories, metrics, and diagnostics.",
            self.results_viewer.panel(),
        )

        return {
            'Domain': domain_stage,
            'Forcing': forcing_stage,
            'Model': model_stage,
            'Run': run_stage,
            'Analyze': analyze_stage,
        }

    def _build_quickstart_wizard(self, stage_names, activate_stage):
        """Build first-run guided quickstart with recommended next action."""
        status_pane = pn.pane.HTML(sizing_mode='stretch_width')
        primary_btn = pn.widgets.Button(name='Open Domain', button_type='primary', width=190)
        secondary_btn = pn.widgets.Button(name='Load Demo', button_type='light', width=160)
        action_state = {'primary': ('open_stage', 0), 'secondary': ('load_demo', None)}

        def _apply_action(key):
            action, payload = action_state[key]
            if action == 'open_stage':
                activate_stage(int(payload), source='quickstart')
            elif action == 'run_step':
                self.workflow_runner.run_step(str(payload), force_rerun=False)
            elif action == 'run_full':
                self.workflow_runner.run_full(force_rerun=False)
            elif action == 'refresh_results':
                self.results_viewer.refresh()
            elif action == 'load_demo':
                self._load_demo('bow')
            self._record_telemetry('quickstart_action', action=action, payload=payload)

        primary_btn.on_click(lambda event: _apply_action('primary'))
        secondary_btn.on_click(lambda event: _apply_action('secondary'))

        def _refresh(stage_states):
            list_items = ''.join(
                f"<li>{_STAGE_ICONS[stage_states[name]]} <b>{name}</b> - {stage_states[name]}</li>"
                for name in stage_names
            )
            status_pane.object = (
                "<div class='sf-command-title' style='margin-bottom:6px'>Quickstart Wizard</div>"
                "<div class='sf-stage-caption' style='margin-bottom:8px'>"
                "Follow the narrative once, then use the command palette for expert speed."
                "</div>"
                f"<ol class='sf-wizard-list'>{list_items}</ol>"
            )

            next_idx = next((i for i, name in enumerate(stage_names) if stage_states[name] != 'done'), None)
            if next_idx is None:
                primary_btn.name = 'Open Analyze'
                primary_btn.button_type = 'success'
                action_state['primary'] = ('open_stage', stage_names.index('Analyze'))
            else:
                stage_name = stage_names[next_idx]
                primary_btn.name = f"Open {stage_name}"
                primary_btn.button_type = 'primary'
                action_state['primary'] = ('open_stage', next_idx)

            if self.state.typed_config is None:
                secondary_btn.name = 'Load Demo'
                secondary_btn.button_type = 'light'
                secondary_btn.disabled = bool(self.state.is_running)
                action_state['secondary'] = ('load_demo', None)
                return

            next_stage = stage_names[next_idx] if next_idx is not None else 'Analyze'
            next_step = self._recommended_step_for_stage(next_stage)
            if next_step and not self.state.is_running:
                secondary_btn.name = f"Run {next_step}"
                secondary_btn.button_type = 'success'
                secondary_btn.disabled = False
                action_state['secondary'] = ('run_step', next_step)
            else:
                secondary_btn.name = 'Refresh Results'
                secondary_btn.button_type = 'light'
                secondary_btn.disabled = False
                action_state['secondary'] = ('refresh_results', None)

        panel = pn.Column(
            status_pane,
            pn.FlexBox(
                primary_btn,
                secondary_btn,
                sizing_mode='stretch_width',
                styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
            ),
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '10px 14px',
                'margin': '0 0 10px 0',
            },
            sizing_mode='stretch_width',
        )
        return panel, _refresh

    def _build_guidance_panel(self, stage_names, activate_stage):
        """Build global actionable empty/error state guidance."""
        alert = pn.pane.Alert("", alert_type='info', sizing_mode='stretch_width')
        primary_btn = pn.widgets.Button(name='Load Demo', button_type='primary', width=170)
        secondary_btn = pn.widgets.Button(name='Open Domain', button_type='light', width=170)
        action_state = {'primary': ('load_demo', None), 'secondary': ('open_stage', 0)}

        def _run_action(key):
            action, payload = action_state[key]
            if action == 'load_demo':
                self._load_demo('bow')
            elif action == 'open_stage':
                activate_stage(int(payload), source='guidance')
            elif action == 'run_step':
                self.workflow_runner.run_step(str(payload), force_rerun=False)
            elif action == 'run_full':
                self.workflow_runner.run_full(force_rerun=False)
            elif action == 'refresh_results':
                self.results_viewer.refresh()
            self._record_telemetry('guidance_action', action=action, payload=payload)

        primary_btn.on_click(lambda event: _run_action('primary'))
        secondary_btn.on_click(lambda event: _run_action('secondary'))

        def _refresh(stage_states):
            report_count = self._artifact_count('Analyze')
            if self.state.typed_config is None:
                alert.object = "Start by loading a configuration. Quickest path: load the Bow demo."
                alert.alert_type = 'info'
                primary_btn.name = 'Load Demo'
                secondary_btn.name = 'Open Domain'
                primary_btn.button_type = 'primary'
                action_state['primary'] = ('load_demo', None)
                action_state['secondary'] = ('open_stage', stage_names.index('Domain'))
                secondary_btn.disabled = False
                return

            if self.state.is_running:
                running = self.state.running_step or 'workflow'
                alert.object = f"Workflow is running ({running}). Controls are locked until completion."
                alert.alert_type = 'warning'
                primary_btn.name = 'Open Run'
                primary_btn.button_type = 'warning'
                secondary_btn.name = 'Open Logs'
                secondary_btn.button_type = 'light'
                action_state['primary'] = ('open_stage', stage_names.index('Run'))
                action_state['secondary'] = ('open_stage', stage_names.index('Run'))
                secondary_btn.disabled = False
                return

            if self.state.last_completed_run is None:
                alert.object = (
                    "No completed simulation yet. Run at least one model step to unlock results and diagnostics."
                )
                alert.alert_type = 'info'
                primary_btn.name = 'Run Full Workflow'
                primary_btn.button_type = 'success'
                secondary_btn.name = 'Open Run'
                secondary_btn.button_type = 'light'
                action_state['primary'] = ('run_full', None)
                action_state['secondary'] = ('open_stage', stage_names.index('Run'))
                secondary_btn.disabled = False
                return

            if report_count == 0:
                alert.object = (
                    "Simulation is complete but no report plots were found. Run post-processing or refresh results."
                )
                alert.alert_type = 'warning'
                primary_btn.name = 'Run postprocess_results'
                primary_btn.button_type = 'warning'
                secondary_btn.name = 'Refresh Results'
                secondary_btn.button_type = 'light'
                action_state['primary'] = ('run_step', 'postprocess_results')
                action_state['secondary'] = ('refresh_results', None)
                secondary_btn.disabled = False
                return

            alert.object = (
                f"Results are synced for experiment `{self.state.last_completed_run}`. "
                "Continue exploring diagnostics in Analyze."
            )
            alert.alert_type = 'success'
            primary_btn.name = 'Open Analyze'
            primary_btn.button_type = 'success'
            secondary_btn.name = 'Refresh Results'
            secondary_btn.button_type = 'light'
            action_state['primary'] = ('open_stage', stage_names.index('Analyze'))
            action_state['secondary'] = ('refresh_results', None)
            secondary_btn.disabled = False

        panel = pn.Column(
            alert,
            pn.FlexBox(
                primary_btn,
                secondary_btn,
                sizing_mode='stretch_width',
                styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
            ),
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '10px 14px',
                'margin': '0 0 10px 0',
            },
            sizing_mode='stretch_width',
        )
        return panel, _refresh

    def _build_artifact_strip(self, stage_names, activate_stage):
        """Build compact stage artifact summary row."""
        cards = {}
        open_buttons = {}
        stage_columns = []
        for idx, name in enumerate(stage_names):
            cards[name] = pn.pane.HTML(sizing_mode='stretch_width')
            open_btn = pn.widgets.Button(name=f'Open {name}', button_type='light', width=130)
            open_btn.on_click(
                lambda event, i=idx, stage=name: self._open_stage_from_artifacts(
                    activate_stage,
                    i,
                    stage,
                )
            )
            open_buttons[name] = open_btn
            stage_columns.append(
                pn.Column(
                    cards[name],
                    open_btn,
                    styles={
                        'background': 'var(--sf-surface-soft)',
                        'border': '1px solid var(--sf-line)',
                        'border-radius': '12px',
                        'padding': '8px 10px',
                        'min-width': '170px',
                    },
                )
            )

        def _refresh(stage_states):
            artifacts = self._collect_stage_artifacts()
            for name in stage_names:
                state = stage_states[name]
                count = artifacts.get(name, 0)
                cards[name].object = (
                    f"<div class='sf-artifact-title'>{_STAGE_ICONS[state]} {name}</div>"
                    f"<div class='sf-artifact-count'>{count}</div>"
                    "<div class='sf-stage-caption'>artifact files</div>"
                )
                open_buttons[name].button_type = (
                    'success' if state == 'done'
                    else ('warning' if state == 'running' else 'light')
                )

        panel = pn.Column(
            pn.pane.HTML(
                "<div class='sf-command-title'>Stage Artifacts</div>",
                sizing_mode='stretch_width',
            ),
            pn.FlexBox(
                *stage_columns,
                sizing_mode='stretch_width',
                styles={'column-gap': '8px', 'row-gap': '8px', 'flex-wrap': 'wrap'},
            ),
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '10px 14px',
                'margin': '0 0 10px 0',
            },
            sizing_mode='stretch_width',
        )
        return panel, _refresh

    def _open_stage_from_artifacts(self, activate_stage, idx, stage_name):
        """Handle artifact strip stage opening with telemetry."""
        activate_stage(int(idx), source='artifact_strip')
        self._record_telemetry('artifact_open_stage', stage=stage_name)

    def _collect_stage_artifacts(self):
        """Collect rough artifact counts by stage from common output folders."""
        project_dir = self.state.project_dir
        if not project_dir:
            return {name: 0 for name in _STAGE_ORDER}

        root = Path(project_dir)
        return {
            'Domain': self._count_artifacts(root / 'shapefiles', ('*.shp',)),
            'Forcing': self._count_artifacts(root / 'forcing', ('*.nc', '*.nc4', '*.csv')),
            'Model': self._count_artifacts(
                root / 'settings',
                ('*.txt', '*.nc', '*.csv'),
            ),
            'Run': self._count_artifacts(
                root / 'simulations',
                ('*.txt', '*.nc', '*.csv', '*.out'),
            ),
            'Analyze': self._count_artifacts(
                root / 'reporting',
                ('*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf'),
            ),
        }

    def _artifact_count(self, stage_name):
        """Convenience helper for a single stage artifact count."""
        return self._collect_stage_artifacts().get(stage_name, 0)

    @staticmethod
    def _count_artifacts(base_path: Path, patterns, limit=500):
        """Count files matching glob patterns with an upper bound."""
        if not base_path.exists():
            return 0
        count = 0
        for pattern in patterns:
            for _ in base_path.rglob(pattern):
                count += 1
                if count >= limit:
                    return count
        return count

    def _compute_stage_states(self):
        """Compute pending/running/done stage states from workflow + config state."""
        step_done = self._step_completion_lookup()

        states = {stage: 'pending' for stage in _STAGE_ORDER}
        for stage in _STAGE_ORDER:
            if self._stage_done(stage, step_done):
                states[stage] = 'done'

        running_stage = self._running_stage(step_done)
        if running_stage in states:
            states[running_stage] = 'running'

        return states

    def _step_completion_lookup(self):
        """Build a lookup of CLI step -> completion bool."""
        lookup = {}
        status = self.state.workflow_status or {}
        for detail in status.get('step_details', []):
            key = detail.get('cli_name') or detail.get('name')
            if key:
                lookup[key] = bool(detail.get('complete'))
        return lookup

    def _stage_done(self, stage_name, step_done):
        """Return True if a stage should be visually considered complete."""
        mapped_steps = _STAGE_STEP_MAP.get(stage_name, [])
        any_step_done = any(step_done.get(step, False) for step in mapped_steps)

        if stage_name == 'Domain':
            has_domain_context = bool(self.state.config_path) and (
                bool(self.state.project_dir)
                or (self.state.pour_point_lat is not None and self.state.pour_point_lon is not None)
            )
            return has_domain_context or any_step_done

        if stage_name == 'Forcing':
            return any_step_done

        if stage_name == 'Model':
            return any_step_done

        if stage_name == 'Run':
            return any_step_done

        if stage_name == 'Analyze':
            return any_step_done or bool(self.state.last_completed_run)

        return any_step_done

    def _running_stage(self, step_done):
        """Infer which stage should be marked running."""
        if not self.state.is_running:
            return None

        running_step = self.state.running_step
        if running_step and running_step != 'full_workflow':
            for stage_name, stage_steps in _STAGE_STEP_MAP.items():
                if running_step in stage_steps:
                    return stage_name

        if running_step == 'full_workflow':
            for stage_name in _STAGE_ORDER:
                if not self._stage_done(stage_name, step_done):
                    return stage_name
            return 'Analyze'

        return 'Run'

    def _recommended_step_for_stage(self, stage_name):
        """Return next recommended CLI step for a given stage."""
        step_done = self._step_completion_lookup()
        for step in _STAGE_STEP_MAP.get(stage_name, []):
            if not step_done.get(step, False):
                return step
        mapped = _STAGE_STEP_MAP.get(stage_name, [])
        return mapped[-1] if mapped else None

    @staticmethod
    def _surface_card(title, subtitle, body):
        """Render consistent section styling for each stage workspace."""
        header = pn.Column(
            pn.pane.Markdown(f"### {title}", margin=(0, 0, 2, 0)),
            pn.pane.HTML(
                f"<div class='sf-stage-caption'>{subtitle}</div>",
                margin=(0, 0, 8, 0),
            ),
            sizing_mode='stretch_width',
        )
        return pn.Column(
            header,
            body,
            sizing_mode='stretch_both',
            styles={
                'background': 'var(--sf-surface)',
                'border': '1px solid var(--sf-line)',
                'border-radius': '14px',
                'padding': '12px 14px',
            },
        )

    def _build_status_html(self, config_path, project_dir, workflow_status, is_running, config_dirty):
        """Generate the status bar HTML string."""
        parts = []

        if config_path:
            parts.append(f"<b>Config:</b> {config_path}")
        else:
            parts.append("<b>Config:</b> <i>none</i>")

        if project_dir:
            parts.append(f"<b>Project:</b> {project_dir}")

        if workflow_status:
            total = workflow_status.get('total_steps', 0)
            done = workflow_status.get('completed_steps', 0)
            parts.append(f"<b>Steps:</b> {done}/{total}")

        if is_running:
            step = self.state.running_step or 'unknown'
            parts.append(f"<b>Running:</b> {step}")

        dirty = " [unsaved changes]" if config_dirty else ""
        return f"<div class='sf-status-row'>{' &nbsp;|&nbsp; '.join(parts)}{dirty}</div>"

    def _record_telemetry(self, event_name, **payload):
        """Record telemetry event if opt-in is enabled."""
        self.telemetry.record(event_name, **payload)

    def _on_run_state_change(self, event):
        """Telemetry hook for run starts/completions."""
        if bool(event.new):
            self._last_run_step = self.state.running_step
            self._record_telemetry('run_started', step=self._last_run_step)
        else:
            self._record_telemetry(
                'run_finished',
                step=self._last_run_step,
                last_completed_run=self.state.last_completed_run,
            )
            self._last_run_step = None

    def _on_results_synced(self, event):
        """Telemetry hook for experiment/result synchronization."""
        if event.new:
            self._record_telemetry('results_experiment_synced', experiment_id=event.new)

    def _on_log_change(self, event):
        """Capture workflow errors from log stream."""
        old_len = self._last_log_len
        new_text = event.new or ""
        self._last_log_len = len(new_text)
        if old_len >= len(new_text):
            return
        delta = new_text[old_len:]
        for line in delta.splitlines():
            if 'ERROR:' in line or 'failed' in line.lower():
                self._record_telemetry('workflow_error', message=line.strip()[:220])
                break

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------

    def _load_demo(self, demo_name: str):
        """Load a built-in demo configuration."""
        demo_name = demo_name.lower().strip()
        if demo_name == 'bow':
            self._load_bow_demo()
        else:
            self.state.append_log(f"Unknown demo: {demo_name}. Available: bow\\n")
            self._record_telemetry('unknown_demo_requested', demo=demo_name)

    def _load_bow_demo(self):
        """Load the Bow at Banff demo config with dynamic path resolution."""
        import os

        # Resolve SYMFLUENCE_CODE_DIR
        code_dir = os.environ.get('SYMFLUENCE_CODE_DIR')
        if not code_dir:
            # Infer from package location (gui/ -> symfluence/ -> src/ -> repo root)
            code_dir = str(Path(__file__).resolve().parents[3])

        config_path = (
            Path(code_dir)
            / 'examples' / '02_watershed_modelling' / 'configs'
            / 'config_bow_summa_optimization.yaml'
        )

        if not config_path.exists():
            self.state.append_log(
                f"Bow demo config not found at {config_path}\\n"
                "Set SYMFLUENCE_CODE_DIR or pass --config explicitly.\\n"
            )
            return

        # Resolve SYMFLUENCE_DATA_DIR
        data_dir = os.environ.get('SYMFLUENCE_DATA_DIR')
        if not data_dir:
            data_dir = str(Path(code_dir).parent / 'SYMFLUENCE_data')

        try:
            # Load config - ConfigEditor auto-refreshes via config_path watcher
            self.state.load_config(str(config_path))

            # Override project dir to point to existing Bow domain data
            bow_domain = Path(data_dir) / 'domain_bow_at_banff_lumped_era5'
            if bow_domain.is_dir():
                self.state.project_dir = str(bow_domain)
                self.state.append_log(f"Demo: project dir -> {bow_domain}\\n")
                # Auto-load shapefiles
                self.map_view.load_layers()
            else:
                self.state.append_log(
                    f"Bow domain data not found at {bow_domain} - "
                    "map layers not loaded.\\n"
                )

            self.state.append_log("Bow at Banff demo loaded.\\n")
            self._record_telemetry('demo_loaded', demo='bow', project_dir=str(self.state.project_dir))
        except Exception as exc:
            self.state.append_log(f"Failed to load Bow demo: {exc}\\n")
            self._record_telemetry('demo_load_failed', demo='bow', error=str(exc))
