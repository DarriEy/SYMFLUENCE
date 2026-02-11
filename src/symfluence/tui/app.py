"""
Root Textual application for SYMFLUENCE TUI.

Manages screen modes, key bindings, and shared services.
"""

from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from .screens.calibration import CalibrationScreen
from .screens.dashboard import DashboardScreen
from .screens.results_compare import ResultsCompareScreen
from .screens.run_browser import RunBrowserScreen
from .screens.slurm_monitor import SlurmMonitorScreen
from .screens.workflow_launcher import WorkflowLauncherScreen
from .services.data_dir import DataDirService
from .services.slurm_service import SlurmService


class SymfluenceTUI(App):
    """SYMFLUENCE interactive terminal application."""

    TITLE = "SYMFLUENCE"
    SUB_TITLE = "Hydrological Modeling Framework"
    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("1", "switch_mode('dashboard')", "Dashboard", priority=True),
        Binding("2", "switch_mode('run_browser')", "Runs", priority=True),
        Binding("3", "switch_mode('workflow')", "Workflow", priority=True),
        Binding("4", "switch_mode('calibration')", "Calibration", priority=True),
        Binding("5", "switch_mode('slurm')", "SLURM", show=False, priority=True),
        Binding("6", "switch_mode('compare')", "Compare", priority=True),
        Binding("q", "quit", "Quit"),
    ]

    MODES = {
        "dashboard": DashboardScreen,
        "run_browser": RunBrowserScreen,
        "workflow": WorkflowLauncherScreen,
        "calibration": CalibrationScreen,
        "slurm": SlurmMonitorScreen,
        "compare": ResultsCompareScreen,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        demo: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._config_path = config_path
        self._demo = demo
        self.data_dir_service = DataDirService()
        self.slurm_service = SlurmService()
        self._is_hpc = False

    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @property
    def demo(self) -> Optional[str]:
        return self._demo

    @property
    def is_hpc(self) -> bool:
        return self._is_hpc

    def on_mount(self) -> None:
        """Initialize app state on startup."""
        # Detect HPC environment
        self._is_hpc = self.slurm_service.is_hpc()

        # Resolve demo config path
        if self._demo and not self._config_path:
            self._config_path = self._resolve_demo_config(self._demo)

        # Start on dashboard
        self.switch_mode("dashboard")

    def _resolve_demo_config(self, demo_name: str) -> Optional[str]:
        """Resolve a demo name to a config file path."""
        try:
            from importlib.resources import files
            data_pkg = files("symfluence.data")
            demo_dir = data_pkg / "configs" / "demos"
            candidates = [
                demo_dir / f"{demo_name}.yaml",
                demo_dir / f"{demo_name}.yml",
                demo_dir / f"config_{demo_name}.yaml",
            ]
            for c in candidates:
                p = Path(str(c))
                if p.is_file():
                    return str(p)
        except Exception:
            pass
        return None
