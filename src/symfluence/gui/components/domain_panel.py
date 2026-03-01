# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Domain definition panel for delineation settings.

Handles delineation (lumped / semidistributed / distributed) with
progressive disclosure driven by WorkflowState.gui_phase.
Discretization lives in DiscretizationPanel.
"""

import logging

import panel as pn
import param

from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_VISIBLE_PHASES = (
    'attributes_loaded', 'domain_defined', 'discretized',
    'data_ready', 'model_ready', 'calibrated', 'analyzed',
)


class DomainPanel(param.Parameterized):
    """Sidebar panel for domain delineation."""

    state = param.Parameter(doc="WorkflowState instance")
    map_view = param.Parameter(default=None, doc="MapView instance for layer loading")

    def __init__(self, state, map_view=None, **kw):
        super().__init__(state=state, map_view=map_view, **kw)
        self._wt = WorkflowThread(state)

        # ==============================================================
        # Delineation widgets
        # ==============================================================

        # Always visible
        self._definition_method = pn.widgets.Select(
            name='Definition Method',
            options=['point', 'lumped', 'semidistributed', 'distributed'],
            value='lumped',
            **_WIDGET_KW,
        )
        self._geofabric_type = pn.widgets.Select(
            name='Geofabric Type',
            options=['na', 'merit', 'tdx', 'hydrosheds', 'nws'],
            value='na',
            **_WIDGET_KW,
        )
        self._subset_from_geofabric = pn.widgets.Checkbox(
            name='Subset from Geofabric',
            value=False,
            **_WIDGET_KW,
        )
        self._move_outlets_max_distance = pn.widgets.FloatInput(
            name='Move Outlets Max Distance (m)',
            value=200.0,
            step=50.0,
            **_WIDGET_KW,
        )

        # Lumped-only
        self._lumped_watershed_method = pn.widgets.Select(
            name='Lumped Watershed Method',
            options=['TauDEM', 'pysheds'],
            value='TauDEM',
            **_WIDGET_KW,
        )
        self._lumped_opts = pn.Column(
            self._lumped_watershed_method,
            sizing_mode='stretch_width',
            visible=True,  # lumped is the default
        )

        # Semidistributed-only
        self._delineation_method = pn.widgets.Select(
            name='Delineation Method',
            options=['stream_threshold', 'multi_scale', 'drop_analysis'],
            value='stream_threshold',
            **_WIDGET_KW,
        )
        self._stream_threshold = pn.widgets.FloatInput(
            name='Stream Threshold',
            value=5000.0,
            step=500.0,
            **_WIDGET_KW,
        )
        self._semidist_opts = pn.Column(
            self._delineation_method,
            self._stream_threshold,
            sizing_mode='stretch_width',
            visible=False,
        )

        # Distributed-only
        self._grid_source = pn.widgets.Select(
            name='Grid Source',
            options=['generate', 'native'],
            value='generate',
            **_WIDGET_KW,
        )
        self._grid_cell_size = pn.widgets.FloatInput(
            name='Grid Cell Size (m)',
            value=1000.0,
            step=100.0,
            **_WIDGET_KW,
        )
        self._dist_opts = pn.Column(
            self._grid_source,
            self._grid_cell_size,
            sizing_mode='stretch_width',
            visible=False,
        )

        self._delineate_btn = pn.widgets.Button(
            name='Run Delineation',
            button_type='primary',
            **_BTN_KW,
        )
        self._delineate_btn.on_click(self._on_delineate)

        # Watch definition method changes
        self._definition_method.param.watch(
            self._on_definition_method_change, 'value',
        )

        # Phase transitions and running-state sync
        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------
    # Conditional visibility
    # ------------------------------------------------------------------

    def _on_definition_method_change(self, event):
        """Show/hide method-specific widget groups."""
        method = event.new
        self._lumped_opts.visible = (method == 'lumped')
        self._semidist_opts.visible = (method == 'semidistributed')
        self._dist_opts.visible = (method == 'distributed')

    # ------------------------------------------------------------------
    # Phase visibility
    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        """Toggle panel visibility based on gui_phase."""
        self._panel_card.visible = event.new in _VISIBLE_PHASES

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _sync_running(self, event):
        """Disable action buttons while workflow is running."""
        self._delineate_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_delineate(self, event):
        """Update config with delineation settings and run define_domain."""
        cfg = self.state.typed_config
        if cfg is None:
            self.state.append_log("ERROR: No config loaded. Initialize first.\n")
            return

        method = self._definition_method.value

        # Build updates for the nested DelineationConfig
        delineation_updates = {
            'geofabric_type': self._geofabric_type.value,
            'move_outlets_max_distance': self._move_outlets_max_distance.value,
        }
        if method == 'lumped':
            delineation_updates['lumped_watershed_method'] = self._lumped_watershed_method.value
        elif method == 'semidistributed':
            delineation_updates['method'] = self._delineation_method.value
            delineation_updates['stream_threshold'] = self._stream_threshold.value

        # Build updates for DomainConfig
        domain_updates = {
            'definition_method': method,
            'subset_from_geofabric': self._subset_from_geofabric.value,
            'delineation': cfg.domain.delineation.model_copy(update=delineation_updates),
        }
        if method == 'distributed':
            domain_updates['grid_source'] = self._grid_source.value
            domain_updates['grid_cell_size'] = self._grid_cell_size.value

        # Rebuild frozen config via model_copy
        new_domain = cfg.domain.model_copy(update=domain_updates)
        self.state.typed_config = cfg.model_copy(update={'domain': new_domain})

        if self.state.config_path:
            self.state.save_config()

        self.state.invalidate_symfluence()
        self.state.append_log("Running delineation...\n")
        self._wt.run_steps(['define_domain'])

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        """Build and return the domain definition card."""
        self._panel_card = pn.Card(
            self._definition_method,
            self._lumped_opts,
            self._semidist_opts,
            self._dist_opts,
            self._geofabric_type,
            self._subset_from_geofabric,
            self._move_outlets_max_distance,
            pn.layout.Divider(),
            self._delineate_btn,
            title='Domain Definition',
            collapsed=True,
            visible=self.state.gui_phase in _VISIBLE_PHASES,
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )

        return self._panel_card
