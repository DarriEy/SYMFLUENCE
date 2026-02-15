"""
Acquire Attributes panel for downloading geospatial attribute data.

Visible after project setup completes (gui_phase >= 'project_created').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_VISIBLE_PHASES = (
    'project_created', 'attributes_loaded', 'domain_defined', 'discretized',
    'data_ready', 'model_ready', 'calibrated', 'analyzed',
)


class AttributesPanel(param.Parameterized):
    """Sidebar card for geospatial attribute acquisition."""

    state = param.Parameter(doc="WorkflowState instance")
    map_view = param.Parameter(default=None, doc="MapView instance for layer loading")

    def __init__(self, state, map_view=None, **kw):
        super().__init__(state=state, map_view=map_view, **kw)
        self._wt = WorkflowThread(state)

        self._acquire_btn = pn.widgets.Button(
            name='Acquire Attributes',
            button_type='primary',
            **_BTN_KW,
        )
        self._acquire_btn.on_click(self._on_acquire)

        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        self._panel_card.visible = event.new in _VISIBLE_PHASES

    def _sync_running(self, event):
        self._acquire_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------

    def _on_acquire(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded. Initialize first.\n")
            return

        self.state.append_log("Acquiring geospatial attributes...\n")
        self._wt.run_steps(['acquire_attributes'])

    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._acquire_btn,
            title='Acquire Attributes',
            collapsed=False,
            visible=self.state.gui_phase in _VISIBLE_PHASES,
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
