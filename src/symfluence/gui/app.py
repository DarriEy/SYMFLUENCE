"""
Main application: assembles layout and tabs using BootstrapTemplate.

Layout:
    Sidebar  — config file picker + basic config form + gauge setup + advanced toggle
    Main     — Tabs: Map | Workflow | Results | Logs
    Footer   — Status bar: config path, project dir, step progress
"""

import logging
from pathlib import Path

import panel as pn

from .models.workflow_state import WorkflowState
from .components.map_view import MapView
from .components.config_editor import ConfigEditor
from .components.workflow_runner import WorkflowRunner
from .components.log_viewer import LogViewer
from .components.results_viewer import ResultsViewer
from .components.iterative_run_panel import IterativeRunPanel
from .components.gauge_setup_panel import GaugeSetupPanel

logger = logging.getLogger(__name__)


class SymfluenceApp:
    """Assembles all GUI components into a Panel BootstrapTemplate."""

    def __init__(self, config_path=None, demo=None):
        self.state = WorkflowState()

        # Components — all receive the shared state object.
        # ConfigEditor watches state.config_path and auto-refreshes its
        # widgets, so any code that calls state.load_config() will
        # automatically propagate to the editor (no explicit
        # load_from_state() needed).
        self.config_editor = ConfigEditor(self.state)
        self.map_view = MapView(self.state)
        self.workflow_runner = WorkflowRunner(self.state)
        self.log_viewer = LogViewer(self.state)
        self.results_viewer = ResultsViewer(self.state)
        self.iterative_panel = IterativeRunPanel(self.state)
        self.gauge_setup = GaugeSetupPanel(self.state)

        # Auto-load config if provided via CLI
        if config_path:
            try:
                self.state.load_config(config_path)
            except Exception as exc:
                self.state.append_log(f"Failed to load initial config: {exc}\n")

        # Demo mode (runs after components are wired up)
        if demo:
            self._load_demo(demo)

    def build(self):
        """Build and return the BootstrapTemplate."""
        template = pn.template.BootstrapTemplate(
            title='SYMFLUENCE',
            sidebar_width=380,
            header_background='#2c3e50',
        )

        # Sidebar: config editor + gauge setup panel
        template.sidebar.append(self.config_editor.panel())
        template.sidebar.append(self.gauge_setup.panel())

        # Main: tabbed panels
        tabs = pn.Tabs(
            ('Map', self.map_view.panel()),
            ('Workflow', self.workflow_runner.panel()),
            ('Calibrate', self.iterative_panel.panel()),
            ('Results', self.results_viewer.panel()),
            ('Logs', self.log_viewer.panel()),
            sizing_mode='stretch_both',
            dynamic=True,
        )
        template.main.append(tabs)

        # Footer: status bar (reactive via pn.bind)
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
            styles={'background': '#ecf0f1', 'padding': '4px 12px', 'font-size': '12px'},
        )
        template.main.append(status_bar)

        return template

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
        return " &nbsp;|&nbsp; ".join(parts) + dirty

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------

    def _load_demo(self, demo_name: str):
        """Load a built-in demo configuration."""
        demo_name = demo_name.lower().strip()
        if demo_name == 'bow':
            self._load_bow_demo()
        else:
            self.state.append_log(f"Unknown demo: {demo_name}. Available: bow\n")

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
                f"Bow demo config not found at {config_path}\n"
                "Set SYMFLUENCE_CODE_DIR or pass --config explicitly.\n"
            )
            return

        # Resolve SYMFLUENCE_DATA_DIR
        data_dir = os.environ.get('SYMFLUENCE_DATA_DIR')
        if not data_dir:
            data_dir = str(Path(code_dir).parent / 'SYMFLUENCE_data')

        try:
            # Load config — ConfigEditor auto-refreshes via config_path watcher
            self.state.load_config(str(config_path))

            # Override project dir to point to existing Bow domain data
            bow_domain = Path(data_dir) / 'domain_bow_at_banff_lumped_era5'
            if bow_domain.is_dir():
                self.state.project_dir = str(bow_domain)
                self.state.append_log(f"Demo: project dir \u2192 {bow_domain}\n")
                # Auto-load shapefiles
                self.map_view.load_layers()
            else:
                self.state.append_log(
                    f"Bow domain data not found at {bow_domain} \u2014 "
                    "map layers not loaded.\n"
                )

            self.state.append_log("Bow at Banff demo loaded.\n")
        except Exception as exc:
            self.state.append_log(f"Failed to load Bow demo: {exc}\n")
