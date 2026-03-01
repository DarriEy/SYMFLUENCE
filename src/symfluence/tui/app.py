# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Root Textual application for SYMFLUENCE TUI.

Manages screen modes, key bindings, and shared services.
"""

import logging
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from .screens.calibration import CalibrationScreen
from .screens.command_palette import CommandPaletteScreen
from .screens.dashboard import DashboardScreen
from .screens.help import HelpScreen
from .screens.path_prompt import PathPromptScreen
from .screens.results_compare import ResultsCompareScreen
from .screens.run_browser import RunBrowserScreen
from .screens.slurm_monitor import SlurmMonitorScreen
from .screens.workflow_launcher import WorkflowLauncherScreen
from .services.data_dir import DataDirService
from .services.slurm_service import SlurmService

logger = logging.getLogger(__name__)


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
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("?", "show_help", "Help", priority=True),
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
        self._pending_run_browser_domain_filter: Optional[str] = None

    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @property
    def demo(self) -> Optional[str]:
        return self._demo

    @property
    def is_hpc(self) -> bool:
        return self._is_hpc

    def set_config_path(self, config_path: Optional[str]) -> None:
        """Set a config path to preload in the workflow screen."""
        self._config_path = config_path

    def resolve_demo_config(self, demo_name: str) -> Optional[str]:
        """Public wrapper around demo config resolution."""
        return self._resolve_demo_config(demo_name)

    def set_run_browser_domain_filter(self, domain_name: str) -> None:
        """Queue a domain filter to apply when Run Browser is shown."""
        self._pending_run_browser_domain_filter = domain_name

    def consume_run_browser_domain_filter(self) -> Optional[str]:
        """Fetch and clear any queued Run Browser domain filter."""
        pending = self._pending_run_browser_domain_filter
        self._pending_run_browser_domain_filter = None
        return pending

    def on_mount(self) -> None:
        """Initialize app state on startup."""
        # Detect HPC environment
        self._is_hpc = self.slurm_service.is_hpc()

        # Resolve demo config path
        if self._demo and not self._config_path:
            self._config_path = self._resolve_demo_config(self._demo)

        # Start on dashboard
        self.switch_mode("dashboard")

    def action_show_help(self) -> None:
        """Open in-app help with keybindings and workflows."""
        self.push_screen(HelpScreen())

    def action_command_palette(self) -> None:
        """Open searchable command palette."""
        self.push_screen(
            CommandPaletteScreen(self._command_palette_items()),
            self._execute_palette_command,
        )

    def prompt_set_data_dir(self) -> None:
        """Prompt user for a data directory path and apply it."""
        initial = str(self.data_dir_service.data_dir) if self.data_dir_service.data_dir else ""
        self.push_screen(
            PathPromptScreen(
                title="Set SYMFLUENCE Data Directory",
                prompt_text="Enter path to SYMFLUENCE_DATA_DIR:",
                initial_value=initial,
                placeholder="/path/to/SYMFLUENCE_data",
            ),
            self._apply_prompted_data_dir,
        )

    def set_data_dir(self, data_dir: str) -> bool:
        """Set active data directory; returns True if path is valid."""
        if not data_dir:
            return False
        candidate = Path(data_dir).expanduser()
        if not candidate.is_dir():
            return False
        self.data_dir_service = DataDirService(str(candidate))
        return True

    def run_demo(self, demo_name: str) -> bool:
        """Resolve a demo config and open workflow launcher with it."""
        config_path = self.resolve_demo_config(demo_name)
        if not config_path:
            return False
        self.set_config_path(config_path)
        self.switch_mode("workflow")
        return True

    def refresh_active_screen(self) -> None:
        """Best-effort refresh for currently visible screen."""
        handler = getattr(self.screen, "on_screen_resume", None)
        if callable(handler):
            handler()

    def _command_palette_items(self) -> list[tuple[str, str, str]]:
        """Return command palette entries as (id, label, keywords)."""
        items = [
            ("mode:dashboard", "Go to Dashboard", "home domains overview"),
            ("mode:run_browser", "Go to Run Browser", "runs history filter"),
            ("mode:workflow", "Go to Workflow Launcher", "workflow run steps"),
            ("mode:calibration", "Go to Calibration Monitor", "calibration metrics"),
            ("mode:compare", "Go to Results Comparison", "compare metrics experiments"),
            ("app:load_demo_bow", "Load Demo: Bow at Banff", "demo sample onboarding"),
            ("app:set_data_dir", "Set Data Directory", "data dir path configure"),
            ("app:open_help", "Open Help", "help docs keybindings"),
            ("app:refresh", "Refresh Current Screen", "reload refresh"),
            ("app:quit", "Quit SYMFLUENCE TUI", "quit exit"),
        ]
        if self.is_hpc:
            items.insert(
                5,
                ("mode:slurm", "Go to SLURM Monitor", "slurm jobs hpc scheduler"),
            )
        return items

    def _execute_palette_command(self, command_id: Optional[str]) -> None:
        """Execute command selected from palette."""
        if not command_id:
            return
        if command_id.startswith("mode:"):
            self.switch_mode(command_id.split(":", 1)[1])
            return
        if command_id == "app:load_demo_bow":
            if not self.run_demo("bow"):
                self.notify("Demo config 'bow' not found.", severity="error")
            return
        if command_id == "app:set_data_dir":
            self.prompt_set_data_dir()
            return
        if command_id == "app:open_help":
            self.action_show_help()
            return
        if command_id == "app:refresh":
            self.refresh_active_screen()
            return
        if command_id == "app:quit":
            self.exit()

    def _apply_prompted_data_dir(self, selected_path: Optional[str]) -> None:
        """Handle data-dir prompt result."""
        if not selected_path:
            return
        if self.set_data_dir(selected_path):
            self.notify(f"Data directory set: {selected_path}")
            self.refresh_active_screen()
        else:
            self.notify(
                f"Invalid directory: {selected_path}",
                severity="error",
            )

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
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            logger.debug("Failed to resolve demo config '%s': %s", demo_name, exc)
        return None
