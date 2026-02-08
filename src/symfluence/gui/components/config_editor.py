"""
Configuration editor component.

Basic mode: auto-generated widgets from BasicConfigParams.
Advanced mode: Ace code editor with YAML syntax highlighting.
"""

import logging

import panel as pn
import param

from ..models.config_params import BasicConfigParams
from ..utils.config_bridge import config_to_params, params_to_config_overrides

logger = logging.getLogger(__name__)


class ConfigEditor(param.Parameterized):
    """Sidebar config editor with basic (widgets) and advanced (YAML) modes."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._basic_params = BasicConfigParams()
        self._yaml_editor = pn.widgets.CodeEditor(
            value='# Load a config file to edit\n',
            language='yaml',
            theme='monokai',
            sizing_mode='stretch_both',
            min_height=300,
        )
        self._advanced_mode = False

        # Sync pour point from map clicks
        self.state.param.watch(self._on_pour_point_change, ['pour_point_lat', 'pour_point_lon'])

    def _on_pour_point_change(self, *events):
        lat = self.state.pour_point_lat
        lon = self.state.pour_point_lon
        if lat is not None and lon is not None:
            self._basic_params.pour_point_coords = f"{lat}/{lon}"

    def load_from_state(self):
        """Populate widgets from the current state.typed_config."""
        config = self.state.typed_config
        if config is None:
            return

        # Basic mode: populate param widgets
        values = config_to_params(config)
        for key, val in values.items():
            if hasattr(self._basic_params, key):
                try:
                    setattr(self._basic_params, key, val)
                except Exception:
                    pass

        # Advanced mode: show full YAML
        try:
            import yaml
            config_dict = config.to_dict(flatten=True)
            self._yaml_editor.value = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            logger.warning(f"Could not serialize config to YAML: {exc}")

    def save_to_state(self):
        """Apply widget values back to state config and save."""
        if self.state.typed_config is None:
            self.state.append_log("No config loaded to save.\n")
            return

        if self._advanced_mode:
            self._save_from_yaml()
        else:
            self._save_from_widgets()

    def _save_from_widgets(self):
        """Merge widget overrides into existing config, write to disk."""
        overrides = params_to_config_overrides(self._basic_params)

        # Also sync pour point from map if set
        lat = self.state.pour_point_lat
        lon = self.state.pour_point_lon
        if lat is not None and lon is not None:
            overrides['POUR_POINT_COORDS'] = f"{lat}/{lon}"

        try:
            from symfluence.core.config.models import SymfluenceConfig
            base_dict = self.state.typed_config.to_dict(flatten=True)
            base_dict.update(overrides)
            new_config = SymfluenceConfig(**base_dict)
            self.state.typed_config = new_config
            self.state.invalidate_symfluence()
            self.state.save_config()
        except Exception as exc:
            self.state.append_log(f"Save failed: {exc}\n")

    def _save_from_yaml(self):
        """Parse the YAML editor content and replace the config."""
        import yaml
        try:
            raw = yaml.safe_load(self._yaml_editor.value)
            if not isinstance(raw, dict):
                self.state.append_log("YAML must be a dictionary.\n")
                return

            from symfluence.core.config.models import SymfluenceConfig
            new_config = SymfluenceConfig(**raw)
            self.state.typed_config = new_config
            self.state.invalidate_symfluence()
            self.state.save_config()
        except Exception as exc:
            self.state.append_log(f"YAML save failed: {exc}\n")

    def panel(self):
        """Return the Panel layout for embedding in the sidebar."""
        # File picker row
        file_input = pn.widgets.TextInput(
            name='Config File',
            placeholder='Path to config YAML...',
            value=self.state.config_path or '',
            sizing_mode='stretch_width',
        )
        load_btn = pn.widgets.Button(name='Load', button_type='primary', width=70)
        save_btn = pn.widgets.Button(name='Save', button_type='success', width=70)
        mode_toggle = pn.widgets.Toggle(name='Advanced YAML', value=False, width=130)

        def _on_load(event):
            path = file_input.value.strip()
            if path:
                try:
                    self.state.load_config(path)
                    self.load_from_state()
                except Exception as exc:
                    self.state.append_log(f"Load failed: {exc}\n")

        def _on_save(event):
            self.save_to_state()

        def _on_mode_toggle(event):
            self._advanced_mode = event.new

        load_btn.on_click(_on_load)
        save_btn.on_click(_on_save)
        mode_toggle.param.watch(_on_mode_toggle, 'value')

        file_row = pn.Row(file_input, load_btn, save_btn, sizing_mode='stretch_width')

        # Basic param widgets
        basic_panel = pn.Param(
            self._basic_params,
            sizing_mode='stretch_width',
            show_name=False,
            widgets={
                'domain_name': pn.widgets.TextInput,
                'experiment_id': pn.widgets.TextInput,
                'time_start': pn.widgets.TextInput,
                'time_end': pn.widgets.TextInput,
                'calibration_period': pn.widgets.TextInput,
                'evaluation_period': pn.widgets.TextInput,
                'pour_point_coords': pn.widgets.TextInput,
                'bounding_box_coords': pn.widgets.TextInput,
                'discretization': pn.widgets.TextInput,
                'stream_threshold': pn.widgets.FloatInput,
                'iterations': pn.widgets.IntInput,
                'population_size': pn.widgets.IntInput,
            },
        )

        # Switch between basic and advanced
        editor_view = pn.Column(sizing_mode='stretch_both')

        def _update_editor(event=None):
            advanced = mode_toggle.value
            editor_view.clear()
            if advanced:
                editor_view.append(self._yaml_editor)
            else:
                editor_view.append(basic_panel)

        mode_toggle.param.watch(_update_editor, 'value')
        _update_editor()  # initial

        return pn.Column(
            file_row,
            mode_toggle,
            editor_view,
            sizing_mode='stretch_both',
        )
