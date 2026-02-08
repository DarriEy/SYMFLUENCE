"""
Results viewer component.

Scans the project reporting directory for generated plots and displays
them, organized by category.  Also provides a button to run diagnostics
on existing outputs.
"""

import logging
from pathlib import Path

import panel as pn
import param

logger = logging.getLogger(__name__)

# Category -> subdirectory mapping
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


class ResultsViewer(param.Parameterized):
    """Browse and display output plots from the reporting directory."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._plot_pane = pn.pane.PNG(None, sizing_mode='scale_both', max_height=600)
        self._file_list = []

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

        files: list = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg'):
            files.extend(search_dir.rglob(ext))

        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def panel(self):
        """Return the results viewer panel."""
        category = pn.widgets.Select(
            name='Category',
            options=list(RESULT_CATEGORIES.keys()),
            value='All Plots',
            width=200,
        )
        refresh_btn = pn.widgets.Button(name='Refresh', button_type='primary', width=90)
        diagnose_btn = pn.widgets.Button(name='Generate Diagnostics', button_type='success', width=170)

        file_select = pn.widgets.Select(name='Plot', options=[], sizing_mode='stretch_width')
        display_area = pn.Column(sizing_mode='stretch_both')

        def _refresh(event=None):
            files = self._scan_plots(category.value)
            self._file_list = files
            file_select.options = {f.name: str(f) for f in files} if files else {'(no plots found)': ''}

        def _on_file_select(event):
            path = event.new
            if not path:
                display_area.clear()
                return
            display_area.clear()
            p = Path(path)
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                display_area.append(pn.pane.PNG(str(p), sizing_mode='scale_both', max_height=600))
            elif p.suffix.lower() == '.svg':
                display_area.append(pn.pane.SVG(str(p), sizing_mode='scale_both', max_height=600))
            elif p.suffix.lower() == '.pdf':
                display_area.append(pn.pane.Str(f"PDF file: {p.name}\nOpen externally to view."))
            else:
                display_area.append(pn.pane.Str(f"Unsupported format: {p.suffix}"))

        def _run_diagnostics(event):
            if self.state.typed_config is None:
                self.state.append_log("Load a config first.\n")
                return
            try:
                sf = self.state.initialize_symfluence()
                results = sf.run_all_diagnostics()
                if results:
                    self.state.append_log(f"Generated {len(results)} diagnostic plot(s).\n")
                    _refresh()
                else:
                    self.state.append_log("No diagnostics generated (check that outputs exist).\n")
            except Exception as exc:
                self.state.append_log(f"Diagnostics failed: {exc}\n")

        file_select.param.watch(_on_file_select, 'value')
        refresh_btn.on_click(_refresh)
        category.param.watch(lambda e: _refresh(), 'value')
        diagnose_btn.on_click(_run_diagnostics)

        # Initial scan
        _refresh()

        return pn.Column(
            "## Results",
            pn.Row(category, refresh_btn, diagnose_btn),
            file_select,
            pn.layout.Divider(),
            display_area,
            sizing_mode='stretch_both',
        )
