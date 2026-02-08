"""
Main application: assembles layout and tabs using BootstrapTemplate.

Layout:
    Sidebar  — config file picker + basic config form + advanced toggle
    Main     — Tabs: Map | Workflow | Results | Logs
    Footer   — Status bar: config path, project dir, step progress
"""

import panel as pn

from .models.workflow_state import WorkflowState
from .components.map_view import MapView
from .components.config_editor import ConfigEditor
from .components.workflow_runner import WorkflowRunner
from .components.log_viewer import LogViewer
from .components.results_viewer import ResultsViewer


class SymfluenceApp:
    """Assembles all GUI components into a Panel BootstrapTemplate."""

    def __init__(self, config_path=None):
        self.state = WorkflowState()
        self.config_editor = ConfigEditor(self.state)
        self.map_view = MapView(self.state)
        self.workflow_runner = WorkflowRunner(self.state)
        self.log_viewer = LogViewer(self.state)
        self.results_viewer = ResultsViewer(self.state)

        # Auto-load config if provided
        if config_path:
            try:
                self.state.load_config(config_path)
                self.config_editor.load_from_state()
            except Exception as exc:
                self.state.append_log(f"Failed to load initial config: {exc}\n")

    def build(self):
        """Build and return the BootstrapTemplate."""
        template = pn.template.BootstrapTemplate(
            title='SYMFLUENCE',
            sidebar_width=380,
            header_background='#2c3e50',
        )

        # Sidebar: config editor
        template.sidebar.append(self.config_editor.panel())

        # Main: tabbed panels
        tabs = pn.Tabs(
            ('Map', self.map_view.panel()),
            ('Workflow', self.workflow_runner.panel()),
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
