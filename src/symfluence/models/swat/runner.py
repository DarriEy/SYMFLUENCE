"""
SWAT Model Runner.

Executes the SWAT model using prepared TxtInOut input files.
SWAT runs from within its TxtInOut directory and produces output.rch
as its primary output file.
"""

from pathlib import Path
from typing import Optional, List

from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_runner('SWAT')
class SWATRunner(BaseModelRunner):
    """Runner for the SWAT model."""

    MODEL_NAME = "SWAT"

    def _setup_model_specific_paths(self) -> None:
        """Set up SWAT-specific paths."""
        txtinout_name = self._get_config_value(
            lambda: self.config.model.swat.txtinout_dir,
            default='TxtInOut',
        )
        self.txtinout_dir = self.project_dir / "SWAT_input" / txtinout_name

        self.swat_exe = self.get_model_executable(
            install_path_key='SWAT_INSTALL_PATH',
            default_install_subpath='installs/swat',
            default_exe_name='swat_rel.exe',
            typed_exe_accessor=lambda: (
                self.config.model.swat.exe
                if self.config.model and self.config.model.swat
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_output_dir(self) -> Path:
        """SWAT writes output into TxtInOut directory."""
        return self.txtinout_dir

    def _build_run_command(self) -> Optional[List[str]]:
        """Build SWAT execution command.

        SWAT reads file.cio from its working directory (TxtInOut).
        """
        return [str(self.swat_exe)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from TxtInOut directory."""
        return self.txtinout_dir

    def _get_expected_outputs(self) -> List[str]:
        """Expect output.rch in TxtInOut."""
        return ['output.rch']

    def _get_run_timeout(self) -> int:
        """SWAT timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.swat.timeout,
            default=3600,
        )
