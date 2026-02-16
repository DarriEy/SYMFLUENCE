"""
Discretization panel for HRU sub-grid methods.

Visible after domain delineation completes (gui_phase >= 'domain_defined').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_VISIBLE_PHASES = (
    'domain_defined', 'discretized',
    'data_ready', 'model_ready', 'calibrated', 'analyzed',
)


class DiscretizationPanel(param.Parameterized):
    """Sidebar card for sub-grid discretization."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        self._discretization = pn.widgets.MultiChoice(
            name='Discretization Methods',
            options=['grus', 'elevation', 'soilclass', 'landclass', 'aspect', 'radiation'],
            value=['grus'],
            **_WIDGET_KW,
        )
        self._elevation_band_size = pn.widgets.FloatInput(
            name='Elevation Band Size (m)',
            value=200.0,
            step=50.0,
            visible=False,
            **_WIDGET_KW,
        )
        self._aspect_class_number = pn.widgets.IntInput(
            name='Aspect Class Number',
            value=1,
            visible=False,
            **_WIDGET_KW,
        )
        self._radiation_class_number = pn.widgets.IntInput(
            name='Radiation Class Number',
            value=1,
            visible=False,
            **_WIDGET_KW,
        )
        self._min_hru_size = pn.widgets.FloatInput(
            name='Min HRU Size (km\u00b2)',
            value=0.0,
            step=0.5,
            **_WIDGET_KW,
        )
        self._discretize_btn = pn.widgets.Button(
            name='Run Discretization',
            button_type='primary',
            **_BTN_KW,
        )
        self._discretize_btn.on_click(self._on_discretize)

        self._discretization.param.watch(
            self._on_discretization_change, 'value',
        )

        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------

    def _on_discretization_change(self, event):
        selected = event.new or []
        self._elevation_band_size.visible = 'elevation' in selected
        self._aspect_class_number.visible = 'aspect' in selected
        self._radiation_class_number.visible = 'radiation' in selected

    def _on_phase_change(self, event):
        self._panel_card.visible = event.new in _VISIBLE_PHASES

    def _sync_running(self, event):
        self._discretize_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------

    def _on_discretize(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded. Initialize first.\n")
            return

        domain_updates = {
            'discretization': ','.join(self._discretization.value),
            'elevation_band_size': self._elevation_band_size.value,
            'aspect_class_number': self._aspect_class_number.value,
            'radiation_class_number': self._radiation_class_number.value,
            'min_hru_size': self._min_hru_size.value,
        }

        new_domain = cfg.domain.model_copy(update=domain_updates)
        self.state.typed_config = cfg.model_copy(update={'domain': new_domain})

        if self.state.config_path:
            self.state.save_config()

        self.state.invalidate_symfluence()
        self.state.append_log("Running discretization...\n")
        self._wt.run_steps(['discretize_domain'])

    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._discretization,
            self._elevation_band_size,
            self._aspect_class_number,
            self._radiation_class_number,
            self._min_hru_size,
            pn.layout.Divider(),
            self._discretize_btn,
            title='Discretization',
            collapsed=True,
            visible=self.state.gui_phase in _VISIBLE_PHASES,
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
