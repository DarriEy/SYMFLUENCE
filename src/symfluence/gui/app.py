"""
Main application: map-centric interface with left sidebar command panel.
"""

import logging

import panel as pn

from .components.analysis_panel import AnalysisPanel
from .components.attributes_panel import AttributesPanel
from .components.calibration_panel import CalibrationPanel
from .components.command_panel import CommandPanel
from .components.data_panel import DataPanel
from .components.discretization_panel import DiscretizationPanel
from .components.domain_browser import DomainBrowser
from .components.domain_panel import DomainPanel
from .components.forcings_panel import ForcingsPanel
from .components.log_viewer import LogViewer
from .components.map_view import MapView
from .components.model_panel import ModelPanel
from .components.progress_panel import ProgressPanel
from .components.results_viewer import ResultsViewer
from .models.workflow_state import WorkflowState
from .utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

GUI_THEME_CSS = """
:root {
  --sf-bg: #f8fbff;
  --sf-surface: #ffffff;
  --sf-ink: #1b2f45;
  --sf-muted: #6e7f91;
  --sf-accent: #0f80d8;
  --sf-line: #d7e2eb;
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
/* Card styling */
.card {
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  border: 1px solid var(--sf-line);
}
/* Sidebar background */
.sidenav {
  background: var(--sf-bg);
}
/* Widget labels */
.bk-input-group label {
  font-size: 12px;
  color: var(--sf-muted);
  font-weight: 500;
}
/* Button radius */
.bk-btn {
  border-radius: 6px;
}
"""


class SymfluenceApp:
    """Map-centric GUI with sidebar command panel."""

    def __init__(self, config_path=None, demo=None):
        self.state = WorkflowState()
        self.map_view = MapView(self.state)
        self.command_panel = CommandPanel(self.state, map_view=self.map_view)
        self.attributes_panel = AttributesPanel(self.state, map_view=self.map_view)
        self.domain_panel = DomainPanel(self.state, map_view=self.map_view)
        self.discretization_panel = DiscretizationPanel(self.state)
        self.data_panel = DataPanel(self.state)
        self.forcings_panel = ForcingsPanel(self.state)
        self.model_panel = ModelPanel(self.state)
        self.calibration_panel = CalibrationPanel(self.state)
        self.analysis_panel = AnalysisPanel(self.state)
        self.log_viewer = LogViewer(self.state)
        self.progress_panel = ProgressPanel(self.state)
        self._results_wt = WorkflowThread(self.state)
        self.results_viewer = ResultsViewer(
            self.state,
            run_step_callback=lambda step, **kw: self._results_wt.run_steps([step], **kw),
            run_full_callback=lambda **kw: self._results_wt.run_workflow(**kw),
        )
        self.domain_browser = DomainBrowser(self.state, self.map_view)

        if config_path:
            try:
                self.state.load_config(config_path)
            except Exception as exc:
                self.state.append_log(f"Failed to load initial config: {exc}\n")

        # Reload map layers after steps complete
        self.state.param.watch(self._on_status_change, 'workflow_status')

    def build(self):
        """Build and return the BootstrapTemplate."""
        template = pn.template.BootstrapTemplate(
            title='SYMFLUENCE',
            sidebar_width=380,
            header_background='#1b2f45',
        )

        # Sidebar: workflow panels in progression order
        template.sidebar.append(self.command_panel.panel())
        template.sidebar.append(self.attributes_panel.panel())
        template.sidebar.append(self.domain_panel.panel())
        template.sidebar.append(self.discretization_panel.panel())
        template.sidebar.append(self.data_panel.panel())
        template.sidebar.append(self.forcings_panel.panel())
        template.sidebar.append(self.model_panel.panel())
        template.sidebar.append(self.calibration_panel.panel())
        template.sidebar.append(self.analysis_panel.panel())

        # Main area: map + results tabs + progress + console
        console_card = pn.Card(
            self.log_viewer.panel(),
            title='Console Output',
            collapsed=True,
            sizing_mode='stretch_width',
            height=280,
            header_background='#1b2f45',
            header_color='white',
            styles={'border-radius': '8px 8px 0 0'},
        )

        map_with_browser = pn.Row(
            self.map_view.panel(),
            self.domain_browser.panel(),
            sizing_mode='stretch_both',
        )

        main_tabs = pn.Tabs(
            ('Map', map_with_browser),
            ('Results', self.results_viewer.panel()),
            sizing_mode='stretch_both',
            dynamic=True,
        )

        main_area = pn.Column(
            main_tabs,
            self.progress_panel.panel(),
            console_card,
            sizing_mode='stretch_both',
        )
        template.main.append(main_area)

        return template

    def _on_status_change(self, event):
        """Reload map layers when domain definition steps complete."""
        if self.state.is_running:
            return
        if not self.state.project_dir:
            return

        status = self.state.workflow_status or {}
        step_done = {}
        for detail in status.get('step_details', []):
            key = detail.get('cli_name') or detail.get('name')
            if key:
                step_done[key] = bool(detail.get('complete'))

        if step_done.get('define_domain') or step_done.get('discretize_domain'):
            self.map_view.load_layers()
