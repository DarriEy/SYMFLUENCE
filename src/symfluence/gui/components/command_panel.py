"""
Left-sidebar command panel for project setup (Phase 1).

Handles domain name, experiment, pour point, bounding box,
model/forcing selection, and "Initialize & Acquire" button.
Domain definition (delineation/discretization) lives in DomainPanel.
"""

import logging
import os
from pathlib import Path

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

# Model and forcing choices
HYDRO_MODELS = [
    'SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSys', 'NGEN', 'LSTM',
]
FORCING_DATASETS = [
    'ERA5', 'RDRS', 'CARRA', 'CERRA', 'MSWEP', 'AORC', 'CONUS404',
]

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))


class CommandPanel(param.Parameterized):
    """Left sidebar with project setup controls."""

    state = param.Parameter(doc="WorkflowState instance")
    map_view = param.Parameter(default=None, doc="MapView instance for layer loading")

    def __init__(self, state, map_view=None, **kw):
        super().__init__(state=state, map_view=map_view, **kw)
        self._wt = WorkflowThread(state)

        # --- Phase 1 widgets (greyed-out placeholders; auto-filled on click) ---
        self._domain_name = pn.widgets.TextInput(
            name='Domain Name', placeholder='bow_at_banff', **_WIDGET_KW,
        )
        self._experiment_id = pn.widgets.TextInput(
            name='Experiment ID', placeholder='run_001', **_WIDGET_KW,
        )
        self._pour_point = pn.widgets.TextInput(
            name='Pour Point (lat/lon)', placeholder='51.1722/-115.5717', **_WIDGET_KW,
        )
        self._bounding_box = pn.widgets.TextInput(
            name='Bounding Box (north/west/south/east)',
            placeholder='51.76/-116.55/50.95/-115.5',
            **_WIDGET_KW,
        )
        self._model_select = pn.widgets.Select(
            name='Hydrological Model', options=HYDRO_MODELS, value='SUMMA', **_WIDGET_KW,
        )
        self._forcing_select = pn.widgets.Select(
            name='Forcing Dataset', options=FORCING_DATASETS, value='ERA5', **_WIDGET_KW,
        )
        self._time_start = pn.widgets.TextInput(
            name='Time Start', placeholder='2008-01-01 00:00', **_WIDGET_KW,
        )
        self._time_end = pn.widgets.TextInput(
            name='Time End', placeholder='2008-12-31 23:00', **_WIDGET_KW,
        )
        self._init_btn = pn.widgets.Button(
            name='Initialize & Acquire Data',
            button_type='primary',
            **_BTN_KW,
        )
        self._init_btn.on_click(self._on_initialize)

        # Bidirectional sync: pour point text <-> state
        self._pour_point.param.watch(self._on_pour_point_text, 'value')
        state.param.watch(self._on_pour_point_state, ['pour_point_lat', 'pour_point_lon'])

        # Bidirectional sync: bounding box text <-> state
        self._bounding_box.param.watch(self._on_bbox_text, 'value')
        state.param.watch(self._on_bbox_state, ['bounding_box_coords'])

        # Phase transitions
        state.param.watch(self._check_phase_transition, ['workflow_status', 'is_running'])

        # Disable button while running
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------
    # Bidirectional sync
    # ------------------------------------------------------------------

    def _on_pour_point_text(self, event):
        """User typed in pour point field -> update state."""
        val = (event.new or '').strip()
        if not val:
            return
        try:
            parts = val.split('/')
            if len(parts) == 2:
                lat, lon = float(parts[0]), float(parts[1])
                self.state.pour_point_lat = lat
                self.state.pour_point_lon = lon
        except (ValueError, TypeError):
            pass

    def _on_pour_point_state(self, *events):
        """State pour point changed (map tap) -> update text field."""
        lat = self.state.pour_point_lat
        lon = self.state.pour_point_lon
        if lat is not None and lon is not None:
            new_val = f"{lat:.6f}/{lon:.6f}"
            if self._pour_point.value != new_val:
                self._pour_point.value = new_val

    def _on_bbox_text(self, event):
        """User typed in bounding box field -> update state."""
        val = (event.new or '').strip()
        if val and val != self.state.bounding_box_coords:
            self.state.bounding_box_coords = val

    def _on_bbox_state(self, event):
        """State bounding box changed (map draw) -> update text field."""
        val = event.new or ''
        if val and self._bounding_box.value != val:
            self._bounding_box.value = val

    # ------------------------------------------------------------------
    # Phase transition logic
    # ------------------------------------------------------------------

    def _check_phase_transition(self, *events):
        """Monitor workflow_status for phase advancement."""
        if self.state.is_running:
            return  # wait until step finishes

        status = self.state.workflow_status or {}
        step_done = {}
        for detail in status.get('step_details', []):
            key = detail.get('cli_name') or detail.get('name')
            if key:
                step_done[key] = bool(detail.get('complete'))

        # Check phases top-down (most advanced first)
        if step_done.get('run_benchmarking') or step_done.get('run_decision_analysis') or step_done.get('run_sensitivity_analysis'):
            if self.state.gui_phase != 'analyzed':
                self.state.gui_phase = 'analyzed'
        elif step_done.get('calibrate_model'):
            if self.state.gui_phase != 'calibrated':
                self.state.gui_phase = 'calibrated'
        elif step_done.get('run_model') or step_done.get('postprocess_results'):
            if self.state.gui_phase != 'model_ready':
                self.state.gui_phase = 'model_ready'
        elif step_done.get('build_model_ready_store'):
            if self.state.gui_phase != 'data_ready':
                self.state.gui_phase = 'data_ready'
        elif step_done.get('discretize_domain'):
            if self.state.gui_phase != 'discretized':
                self.state.gui_phase = 'discretized'
        elif step_done.get('define_domain'):
            if self.state.gui_phase not in ('domain_defined', 'discretized'):
                self.state.gui_phase = 'domain_defined'
        elif step_done.get('acquire_attributes'):
            if self.state.gui_phase in ('init', 'project_created'):
                self.state.gui_phase = 'attributes_loaded'
                # Auto-load raster layers
                if self.map_view is not None:
                    self.map_view.load_attribute_rasters()
        elif step_done.get('create_pour_point'):
            if self.state.gui_phase == 'init':
                self.state.gui_phase = 'project_created'

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _sync_running(self, event):
        """Disable action button while workflow is running."""
        self._init_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _val_or_placeholder(widget):
        """Return widget value if set, otherwise fall back to placeholder."""
        val = (widget.value or '').strip()
        if val:
            return val
        return (getattr(widget, 'placeholder', '') or '').strip()

    def _populate_defaults(self):
        """Fill empty fields with their placeholder values so the user sees them."""
        for w in (self._domain_name, self._experiment_id, self._pour_point,
                  self._bounding_box, self._time_start, self._time_end):
            if not (w.value or '').strip():
                w.value = getattr(w, 'placeholder', '') or ''

    def _on_initialize(self, event):
        """Validate inputs, create config, save YAML, and run initial steps."""
        # Auto-populate empty fields from placeholders
        self._populate_defaults()

        domain_name = self._val_or_placeholder(self._domain_name)
        if not domain_name:
            self.state.append_log("ERROR: Domain name is required.\n")
            return

        experiment_id = self._val_or_placeholder(self._experiment_id) or 'default'
        model = self._model_select.value
        forcing = self._forcing_select.value
        pour_point = self._val_or_placeholder(self._pour_point)
        bbox = self._val_or_placeholder(self._bounding_box)
        time_start = self._val_or_placeholder(self._time_start)
        time_end = self._val_or_placeholder(self._time_end)

        if not time_start or not time_end:
            self.state.append_log(
                "ERROR: Time Start and Time End are required.\n"
            )
            return

        # Build overrides
        overrides = {
            'EXPERIMENT_ID': experiment_id,
            'time_start': time_start,
            'time_end': time_end,
        }
        if pour_point:
            overrides['pour_point_coords'] = pour_point
            overrides['DELINEATE_BY_POURPOINT'] = True
        if bbox:
            overrides['BOUNDING_BOX_COORDS'] = bbox

        self.state.append_log(
            f"Initializing domain '{domain_name}' with {model}/{forcing}...\n"
        )

        try:
            from symfluence.core.config.models import SymfluenceConfig

            config = SymfluenceConfig.from_minimal(
                domain_name,
                model=model,
                forcing_dataset=forcing,
                **overrides,
            )

            # Save config YAML to project root
            code_dir = os.environ.get('SYMFLUENCE_CODE_DIR')
            if not code_dir:
                code_dir = str(Path(__file__).resolve().parents[4])

            config_dir = Path(code_dir) / '0_config_files'
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"config_{domain_name}.yaml"
            config_dict = config.to_dict(flatten=True)

            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

            self.state.append_log(f"Config saved: {config_path}\n")

            # Load into state (skip refresh — run_steps will create the
            # SYMFLUENCE instance once and refresh status when done)
            self.state.load_config(str(config_path), refresh=False)

            # Run project setup steps (attribute acquisition is a separate card)
            self._wt.run_steps(['setup_project', 'create_pour_point'])

        except Exception as exc:
            self.state.append_log(f"ERROR during initialization: {exc}\n")
            logger.exception("Initialize failed")

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        """Build and return the sidebar Column."""
        phase1_card = pn.Card(
            self._domain_name,
            self._experiment_id,
            self._pour_point,
            self._bounding_box,
            self._model_select,
            self._forcing_select,
            self._time_start,
            self._time_end,
            pn.layout.Divider(),
            self._init_btn,
            title='Project Setup',
            collapsed=False,
            sizing_mode='stretch_width',
            styles={'margin-bottom': '8px'},
        )

        return pn.Column(
            pn.pane.HTML(
                '<div style="font-size:16px; font-weight:700; padding:4px 5px 8px;">SYMFLUENCE</div>',
                sizing_mode='stretch_width',
            ),
            phase1_card,
            sizing_mode='stretch_width',
            scroll=True,
        )
