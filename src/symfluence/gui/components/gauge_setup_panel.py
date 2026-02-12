"""
Collapsible sidebar panel for creating a new domain from a selected gauge.

Appears when a gauge marker is clicked on the map; provides a form for
configuring the new domain and calls ``setup_pour_point_workflow()`` on submit,
then patches the resulting config with user-chosen model/forcing/time values.
"""

import logging
import re

import panel as pn
import param
import yaml

logger = logging.getLogger(__name__)

# Choices mirrored from argument_parser.py
_MODELS = ['SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSys', 'NGEN', 'LSTM']
_FORCING = ['ERA5', 'RDRS', 'CARRA', 'CERRA', 'MSWEP', 'AORC', 'CONUS404']
_DISCRETIZATION = ['lumped', 'point', 'subset', 'delineate']

# Map network → streamflow data provider name used in configs
_NETWORK_TO_PROVIDER = {
    'WSC': 'WSC',
    'USGS': 'USGS',
    'SMHI': 'SMHI',
    'LamaH-ICE': 'LamaH',
}


def _sanitize_name(text: str) -> str:
    """Turn a gauge name into a safe domain name slug."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_').lower()
    return slug[:60] if slug else 'new_domain'


class GaugeSetupPanel(param.Parameterized):
    """Sidebar component for domain creation from a selected gauge station."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)

        # Info display
        self._info_pane = pn.pane.HTML("", styles={'font-size': '12px'})

        # Form widgets
        self._domain_name = pn.widgets.TextInput(name='Domain Name', value='', width=320)
        self._model = pn.widgets.Select(name='Model', options=_MODELS, value='SUMMA', width=320)
        self._forcing = pn.widgets.Select(name='Forcing', options=_FORCING, value='ERA5', width=320)
        self._discretization = pn.widgets.Select(
            name='Discretization', options=_DISCRETIZATION, value='lumped', width=320,
        )
        self._time_start = pn.widgets.TextInput(name='Start', value='2015-01-01 00:00', width=155)
        self._time_end = pn.widgets.TextInput(name='End', value='2019-12-31 23:00', width=155)

        self._create_btn = pn.widgets.Button(
            name='Create Domain Config', button_type='success', width=320,
        )
        self._create_btn.on_click(self._on_create)

        self._status_pane = pn.pane.HTML("", styles={'font-size': '12px'})

        # Build the collapsible card
        self._card = pn.Card(
            self._info_pane,
            self._domain_name,
            self._model,
            self._forcing,
            self._discretization,
            pn.Row(self._time_start, self._time_end),
            self._create_btn,
            self._status_pane,
            title='Gauge \u2192 Domain Setup',
            collapsed=True,
            width=350,
            styles={'margin': '6px 0'},
        )

        # React to gauge selection
        state.param.watch(self._on_gauge_selected, ['selected_gauge'])

    def _on_gauge_selected(self, event):
        gauge = event.new
        if not gauge:
            self._card.collapsed = True
            self.state.domain_creation_active = False
            return

        self._card.collapsed = False
        self.state.domain_creation_active = True

        self._info_pane.object = (
            f"<b>{gauge.get('name', '')}</b><br>"
            f"ID: {gauge.get('station_id', '')} &nbsp; Network: {gauge.get('network', '')}<br>"
            f"Lat: {gauge.get('lat', '')}, Lon: {gauge.get('lon', '')}<br>"
            f"River: {gauge.get('river', '')}"
        )

        # Auto-generate domain name
        self._domain_name.value = _sanitize_name(
            gauge.get('name', gauge.get('station_id', 'domain'))
        )
        self._status_pane.object = ""

    def _on_create(self, event):
        """Create a domain config from the selected gauge + form values."""
        gauge = self.state.selected_gauge
        if not gauge:
            self._status_pane.object = "<span style='color:red'>No gauge selected.</span>"
            return

        from symfluence.project.pour_point_workflow import setup_pour_point_workflow

        lat = gauge['lat']
        lon = gauge['lon']
        coordinates = f"{lat}/{lon}"
        domain_name = self._domain_name.value or _sanitize_name(gauge.get('name', 'domain'))

        self._create_btn.disabled = True
        self._status_pane.object = "Creating config\u2026"

        try:
            # Step 1: Generate base config via pour-point workflow
            result = setup_pour_point_workflow(
                coordinates=coordinates,
                domain_def_method=self._discretization.value,
                domain_name=domain_name,
            )
            config_path = result.config_file

            # Step 2: Patch with form values (model, forcing, time, station)
            self._patch_config(config_path, gauge)

            # Step 3: Load into GUI state
            self.state.load_config(str(config_path))

            self._status_pane.object = (
                f"<span style='color:green'>Config created: {config_path}</span>"
            )
            self.state.append_log(
                f"Domain config created from gauge {gauge['station_id']}: {config_path}\n"
            )
        except Exception as exc:
            self._status_pane.object = f"<span style='color:red'>Error: {exc}</span>"
            self.state.append_log(f"Domain creation failed: {exc}\n")
        finally:
            self._create_btn.disabled = False

    def _patch_config(self, config_path, gauge):
        """Patch the generated YAML with user-selected form values."""
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        # Model and forcing
        config['HYDROLOGICAL_MODEL'] = self._model.value
        config['FORCING_DATASET'] = self._forcing.value

        # Time period
        config['EXPERIMENT_TIME_START'] = self._time_start.value
        config['EXPERIMENT_TIME_END'] = self._time_end.value

        # Station info from gauge
        config['STATION_ID'] = str(gauge.get('station_id', ''))
        provider = _NETWORK_TO_PROVIDER.get(gauge.get('network', ''), '')
        if provider:
            config['STREAMFLOW_DATA_PROVIDER'] = provider

        with open(config_path, 'w') as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    def panel(self):
        """Return the Panel component for embedding in the sidebar."""
        return self._card
