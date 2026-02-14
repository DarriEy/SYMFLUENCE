"""
Central reactive state shared by all GUI components.

WorkflowState is a param.Parameterized class that holds the current config,
workflow progress, log output, and map coordinates. All components watch
these parameters and react to changes.
"""

import logging
import threading
from pathlib import Path

import param

logger = logging.getLogger(__name__)


class WorkflowState(param.Parameterized):
    """Central reactive state for the SYMFLUENCE GUI."""

    # Config state
    config_path = param.String(default=None, allow_None=True, doc="Path to loaded YAML config")
    typed_config = param.Parameter(default=None, doc="SymfluenceConfig instance")
    config_dirty = param.Boolean(default=False, doc="True if config has unsaved changes")

    # Workflow execution state
    is_running = param.Boolean(default=False, doc="True while a step/workflow is executing")
    running_step = param.String(default=None, allow_None=True, doc="Name of currently running step")
    workflow_status = param.Dict(default={}, doc="Dict from get_workflow_status()")

    # Log capture
    log_text = param.String(default="", doc="Accumulated log output for the terminal widget")

    # Map / pour point
    pour_point_lat = param.Number(default=None, allow_None=True, doc="Pour point latitude")
    pour_point_lon = param.Number(default=None, allow_None=True, doc="Pour point longitude")

    # Gauge selection
    selected_gauge = param.Dict(default=None, allow_None=True, doc="Currently selected gauge info dict")
    domain_creation_active = param.Boolean(default=False, doc="True while gauge setup panel is shown")

    # Results refresh trigger
    last_completed_run = param.String(default=None, allow_None=True,
                                      doc="Experiment ID of last completed run")

    # Project directory (derived from config)
    project_dir = param.String(default=None, allow_None=True, doc="Resolved project directory path")

    # SYMFLUENCE instance (lazy)
    _symfluence = param.Parameter(default=None, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._run_lock = threading.Lock()

    def load_config(self, path):
        """Load a SymfluenceConfig from a YAML file and update state."""
        from symfluence.core.config.models import SymfluenceConfig

        path = str(path)
        config = SymfluenceConfig.from_file(Path(path))
        self.typed_config = config
        self.config_path = path
        self.config_dirty = False

        # Derive project dir
        try:
            root = str(config.system.data_dir)
            domain_name = config.domain.name
            self.project_dir = str(Path(root) / f"domain_{domain_name}")
        except Exception:
            self.project_dir = None

        # Extract pour point if present
        if config.domain.pour_point_coords:
            try:
                lat, lon = config.domain.pour_point_coords.split('/')
                self.pour_point_lat = float(lat)
                self.pour_point_lon = float(lon)
            except (ValueError, AttributeError):
                pass

        self._symfluence = None  # reset cached instance
        self.append_log(f"Config loaded: {path}\n")
        self.refresh_status()

    def save_config(self, path=None):
        """Write the current config state back to YAML."""
        import yaml

        path = path or self.config_path
        if not path or not self.typed_config:
            return

        config_dict = self.typed_config.to_dict(flatten=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.config_path = str(path)
        self.config_dirty = False
        self.append_log(f"Config saved: {path}\n")

    def initialize_symfluence(self):
        """Create (or return cached) SYMFLUENCE instance from current config."""
        if self._symfluence is not None:
            return self._symfluence

        if self.typed_config is None:
            raise RuntimeError("No configuration loaded")

        from symfluence.core.system import SYMFLUENCE

        config_input = self.config_path or self.typed_config
        self._symfluence = SYMFLUENCE(config_input)
        return self._symfluence

    def refresh_status(self):
        """Query workflow status from an initialized SYMFLUENCE instance."""
        try:
            sf = self.initialize_symfluence()
            self.workflow_status = sf.get_workflow_status()
        except Exception as exc:
            logger.debug(f"Could not refresh status: {exc}")
            self.workflow_status = {}

    def invalidate_symfluence(self):
        """Force re-creation of SYMFLUENCE instance on next use."""
        self._symfluence = None

    def try_begin_run(self, step_name):
        """Atomically mark workflow execution as active.

        Returns:
            True if run was started, False if another run is already active.
        """
        with self._run_lock:
            if self.is_running:
                return False
            self.is_running = True
            self.running_step = step_name
            return True

    def end_run(self):
        """Clear workflow execution flags."""
        with self._run_lock:
            self.is_running = False
            self.running_step = None

    def append_log(self, text):
        """Thread-safe log append (call via pn.state.execute from threads)."""
        self.log_text += text
