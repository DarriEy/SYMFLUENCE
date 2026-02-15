"""
Forcing acquisition panel for meteorological data download, model-agnostic
preprocessing, and building the model-ready datastore.

Visible after discretization completes (gui_phase >= 'discretized').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_PET_METHODS = ['oudin', 'hargreaves', 'priestley_taylor', 'penman', 'fao56']

_VISIBLE_PHASES = ('discretized', 'data_ready', 'model_ready', 'calibrated', 'analyzed')


class ForcingsPanel(param.Parameterized):
    """Sidebar card for forcing acquisition, preprocessing, and datastore."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        cfg = state.typed_config

        forcing_dataset = ''
        pet_method = 'oudin'
        if cfg is not None:
            try:
                forcing_dataset = cfg.forcing.dataset or ''
            except (AttributeError, TypeError):
                pass
            try:
                pet_method = cfg.forcing.pet_method or 'oudin'
            except (AttributeError, TypeError):
                pass

        self._forcing_dataset = pn.widgets.Select(
            name='Forcing Dataset',
            options=[forcing_dataset] if forcing_dataset else ['ERA5'],
            value=forcing_dataset or 'ERA5',
            disabled=True,
            **_WIDGET_KW,
        )
        self._pet_method = pn.widgets.Select(
            name='PET Method',
            options=_PET_METHODS,
            value=pet_method if pet_method in _PET_METHODS else 'oudin',
            **_WIDGET_KW,
        )
        self._forcing_btn = pn.widgets.Button(
            name='Acquire & Process Forcings',
            button_type='default',
            **_BTN_KW,
        )
        self._forcing_btn.on_click(self._on_acquire_forcings)

        self._datastore_btn = pn.widgets.Button(
            name='Build Model-Ready Store',
            button_type='default',
            **_BTN_KW,
        )
        self._datastore_btn.on_click(self._on_build_store)

        self._run_all_btn = pn.widgets.Button(
            name='Run All Forcing Steps',
            button_type='primary',
            **_BTN_KW,
        )
        self._run_all_btn.on_click(self._on_run_all)

        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        self._panel_card.visible = event.new in _VISIBLE_PHASES

    def _sync_running(self, event):
        running = bool(event.new)
        self._forcing_btn.disabled = running
        self._datastore_btn.disabled = running
        self._run_all_btn.disabled = running

    # ------------------------------------------------------------------

    def _on_acquire_forcings(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        new_forcing = cfg.forcing.model_copy(update={
            'pet_method': self._pet_method.value,
        })
        self.state.typed_config = cfg.model_copy(update={'forcing': new_forcing})

        if self.state.config_path:
            self.state.save_config()
        self.state.invalidate_symfluence()
        self.state.append_log("Acquiring and processing forcings...\n")
        self._wt.run_steps(['acquire_forcings', 'model_agnostic_preprocessing'])

    def _on_build_store(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        self.state.append_log("Building model-ready store...\n")
        self._wt.run_steps(['build_model_ready_store'])

    def _on_run_all(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        new_forcing = cfg.forcing.model_copy(update={
            'pet_method': self._pet_method.value,
        })
        self.state.typed_config = cfg.model_copy(update={'forcing': new_forcing})

        if self.state.config_path:
            self.state.save_config()
        self.state.invalidate_symfluence()
        self.state.append_log("Running all forcing steps...\n")
        self._wt.run_steps([
            'acquire_forcings',
            'model_agnostic_preprocessing',
            'build_model_ready_store',
        ])

    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._forcing_dataset,
            self._pet_method,
            self._forcing_btn,
            pn.layout.Divider(),
            self._datastore_btn,
            pn.layout.Divider(),
            self._run_all_btn,
            title='Process Forcings',
            collapsed=False,
            visible=self.state.gui_phase in _VISIBLE_PHASES,
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
