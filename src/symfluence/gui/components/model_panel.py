# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model initiation panel for model-specific preprocessing, model run,
and post-processing.

Visible after data processing completes (gui_phase >= 'data_ready').
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_HYDRO_MODELS = [
    'SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSys', 'NGEN', 'LSTM',
]
_ROUTING_MODELS = ['None', 'MIZUROUTE', 'DROUTE', 'TROUTE']


class ModelPanel(param.Parameterized):
    """Sidebar panel for model preprocessing, execution, and post-processing."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        cfg = state.typed_config

        # Read-only display of selected hydrological model
        hydro_model = ''
        routing_model = 'None'
        if cfg is not None:
            try:
                hydro_model = cfg.model.hydrological_model or ''
                if isinstance(hydro_model, list):
                    hydro_model = hydro_model[0] if hydro_model else ''  # type: ignore[index]
            except (AttributeError, TypeError):
                pass
            try:
                routing_model = cfg.model.routing_model or 'None'
            except (AttributeError, TypeError):
                pass

        self._hydrological_model = pn.widgets.Select(
            name='Hydrological Model',
            options=_HYDRO_MODELS,
            value=hydro_model if hydro_model in _HYDRO_MODELS else 'SUMMA',
            **_WIDGET_KW,
        )
        self._routing_model = pn.widgets.Select(
            name='Routing Model',
            options=_ROUTING_MODELS,
            value=routing_model if routing_model in _ROUTING_MODELS else 'None',
            **_WIDGET_KW,
        )

        self._preprocess_btn = pn.widgets.Button(
            name='Run Preprocessing',
            button_type='default',
            **_BTN_KW,
        )
        self._preprocess_btn.on_click(self._on_preprocess)

        self._run_model_btn = pn.widgets.Button(
            name='Run Model',
            button_type='default',
            **_BTN_KW,
        )
        self._run_model_btn.on_click(self._on_run_model)

        self._postprocess_btn = pn.widgets.Button(
            name='Run Post-processing',
            button_type='default',
            **_BTN_KW,
        )
        self._postprocess_btn.on_click(self._on_postprocess)

        self._run_all_btn = pn.widgets.Button(
            name='Run All Model Steps',
            button_type='primary',
            **_BTN_KW,
        )
        self._run_all_btn.on_click(self._on_run_all)

        # Phase and running-state sync
        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------
    # Phase visibility
    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        phase = event.new
        self._panel_card.visible = phase in (
            'data_ready', 'model_ready', 'calibrated', 'analyzed',
        )

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _sync_running(self, event):
        running = bool(event.new)
        self._preprocess_btn.disabled = running
        self._run_model_btn.disabled = running
        self._postprocess_btn.disabled = running
        self._run_all_btn.disabled = running

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _update_model_config(self):
        """Apply hydrological model and routing model selections to config."""
        cfg = self.state.typed_config
        if cfg is None:
            return
        routing = self._routing_model.value
        routing_val = None if routing == 'None' else routing
        new_model = cfg.model.model_copy(update={
            'hydrological_model': self._hydrological_model.value,
            'routing_model': routing_val,
        })
        self.state.typed_config = cfg.model_copy(update={'model': new_model})
        if self.state.config_path:
            self.state.save_config()
        self.state.invalidate_symfluence()

    def _on_preprocess(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        self._update_model_config()
        self.state.append_log("Running model-specific preprocessing...\n")
        self._wt.run_steps(['model_specific_preprocessing'])

    def _on_run_model(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        self.state.append_log("Running model...\n")
        self._wt.run_steps(['run_model'])

    def _on_postprocess(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        self.state.append_log("Running post-processing...\n")
        self._wt.run_steps(['postprocess_results'])

    def _on_run_all(self, event):
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        self._update_model_config()
        self.state.append_log("Running all model steps...\n")
        self._wt.run_steps([
            'model_specific_preprocessing',
            'run_model',
            'postprocess_results',
        ])

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._hydrological_model,
            self._routing_model,
            pn.layout.Divider(),
            self._preprocess_btn,
            self._run_model_btn,
            self._postprocess_btn,
            pn.layout.Divider(),
            self._run_all_btn,
            title='Model Initiation',
            collapsed=True,
            visible=self.state.gui_phase in (
                'data_ready', 'model_ready', 'calibrated', 'analyzed',
            ),
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
