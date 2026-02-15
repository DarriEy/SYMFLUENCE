"""
Observation processing panel for streamflow data acquisition.

Visible after discretization completes (gui_phase >= 'discretized').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_STREAMFLOW_PROVIDERS = ['USGS', 'WSC', 'SMHI', 'LAMAH-ICE', 'local']

_VISIBLE_PHASES = ('discretized', 'data_ready', 'model_ready', 'calibrated', 'analyzed')


class DataPanel(param.Parameterized):
    """Sidebar card for observation data acquisition."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        cfg = state.typed_config

        obs_provider = 'WSC'
        obs_station = ''
        if cfg is not None:
            try:
                obs_provider = cfg.evaluation.streamflow.data_provider or 'WSC'
            except (AttributeError, TypeError):
                pass
            try:
                obs_station = cfg.evaluation.streamflow.station_id or ''
            except (AttributeError, TypeError):
                pass

        self._streamflow_provider = pn.widgets.Select(
            name='Streamflow Provider',
            options=_STREAMFLOW_PROVIDERS,
            value=obs_provider if obs_provider in _STREAMFLOW_PROVIDERS else 'WSC',
            **_WIDGET_KW,
        )
        self._station_id = pn.widgets.TextInput(
            name='Station ID',
            placeholder='05BB001',
            value=obs_station,
            **_WIDGET_KW,
        )
        self._obs_btn = pn.widgets.Button(
            name='Process Observations',
            button_type='primary',
            **_BTN_KW,
        )
        self._obs_btn.on_click(self._on_acquire_obs)

        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        self._panel_card.visible = event.new in _VISIBLE_PHASES

    def _sync_running(self, event):
        self._obs_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------

    def _on_acquire_obs(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        provider = self._streamflow_provider.value
        station = self._station_id.value.strip() if self._station_id.value else ''

        if provider in ('WSC', 'USGS') and not station:
            self.state.append_log(
                f"ERROR: Station ID is required for {provider} data "
                f"(e.g. 05BB001 for WSC, 01013500 for USGS).\n"
            )
            return

        new_streamflow = cfg.evaluation.streamflow.model_copy(update={
            'data_provider': provider,
            'station_id': station or None,
            'download_wsc': provider == 'WSC',
            'download_usgs': provider == 'USGS',
        })
        new_eval = cfg.evaluation.model_copy(update={'streamflow': new_streamflow})
        # Also update data section — data_manager dispatches on data.streamflow_data_provider
        new_data = cfg.data.model_copy(update={'streamflow_data_provider': provider})
        self.state.typed_config = cfg.model_copy(update={
            'evaluation': new_eval,
            'data': new_data,
        })

        if self.state.config_path:
            self.state.save_config()
        self.state.invalidate_symfluence()
        self.state.append_log("Processing observations...\n")
        self._wt.run_steps(['process_observed_data'])

    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._streamflow_provider,
            self._station_id,
            pn.layout.Divider(),
            self._obs_btn,
            title='Process Observations',
            collapsed=False,
            visible=self.state.gui_phase in _VISIBLE_PHASES,
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
